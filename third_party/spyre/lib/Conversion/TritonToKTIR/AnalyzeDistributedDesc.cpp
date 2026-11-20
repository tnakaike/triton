//===- AnalyzeDistributedDesc.cpp - Who reads a distributed view ----------===//
//
// For each pinned read through a tts.make_distributed_descriptor, finds which
// tiles perform it, the block each reads, and where that block sits in the
// composed destination, and writes the answer as `tts.dist_landing` on the
// landing `tts.pin` marker.
//
// The design (docs/inter-tile-lowering-to-mem-view.md, §3 phase 3 and "Checks
// before the analysis") identifies the destination from the Triton program
// rather than from an annotation: the `if` around the read says which tiles
// receive, the coordinates say what each receives, and the pin says that the
// value lands. All three are answerable only if they are constants on every
// tile, so this pass evaluates them once per tile, with `tt.get_program_id x`
// replaced by the tile id, and refuses what it cannot evaluate at the read's
// Triton line. Its result is stated in the IR so that it can be tested on its
// own, which is what the design asks of it.
//
// It runs before LowerDistributedDesc, which consumes the attribute, and
// changes nothing else.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <map>
#include <set>

using namespace mlir;
namespace tts = mlir::triton::tts;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_ANALYZEDISTRIBUTEDDESC
#include "Conversion/TritonToKTIR/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

//===----------------------------------------------------------------------===//
// Evaluating an integer value on one tile
//===----------------------------------------------------------------------===//

/// Why a value has no constant on a tile: the value to point at, and what to
/// say there.
struct NotConstant {
  Value culprit;
  std::string why;
};

/// Evaluates i1/iN/index values built from `tt.get_program_id x`,
/// `tt.get_num_programs x`, constants and integer arith, on one tile.
///
/// Values are kept as int64 sign-extended from their type's width, so a wider
/// type and a narrower one agree on what a bit pattern means; an op that reads
/// its operands unsigned masks them back to their width first. Arithmetic
/// wraps, as the IR's does.
class TileEvaluator {
public:
  TileEvaluator(int64_t tile, int64_t tiles) : tile(tile), tiles(tiles) {}

  FailureOr<int64_t> eval(Value v, NotConstant &fail) {
    if (auto it = cache.find(v); it != cache.end())
      return it->second;
    FailureOr<int64_t> r = compute(v, fail);
    if (succeeded(r))
      cache[v] = *r;
    return r;
  }

private:
  static unsigned widthOf(Type t) {
    if (t.isIndex())
      return 64;
    return t.getIntOrFloatBitWidth();
  }

  /// `v` as a value of `t`: sign-extended from its width.
  static int64_t norm(int64_t v, Type t) {
    unsigned w = widthOf(t);
    return w >= 64 ? v : llvm::SignExtend64(v, w);
  }

  /// `v`, a value of `t`, read as unsigned.
  static uint64_t asUnsigned(int64_t v, Type t) {
    unsigned w = widthOf(t);
    return w >= 64 ? (uint64_t)v
                   : ((uint64_t)v & llvm::maskTrailingOnes<uint64_t>(w));
  }

  FailureOr<int64_t> notConstant(NotConstant &fail, Value v, std::string why) {
    fail = {v, std::move(why)};
    return failure();
  }

  FailureOr<int64_t> compute(Value v, NotConstant &fail) {
    static constexpr const char *kNotTileFunction =
        "this value is not a function of tl.program_id(0) and constants";
    Type type = v.getType();
    if (!type.isIntOrIndex())
      return notConstant(fail, v, "this value is not an integer");
    Operation *def = v.getDefiningOp();
    // An op this evaluator does not model is the culprit itself, not whatever
    // feeds it: a load's pointer is not where the value stops being a constant.
    if (!def ||
        !isa<arith::ConstantOp, triton::GetProgramIdOp,
             triton::GetNumProgramsOp, arith::AddIOp, arith::SubIOp,
             arith::MulIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
             arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp,
             arith::SelectOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
             arith::IndexCastOp, arith::IndexCastUIOp, arith::CmpIOp,
             arith::DivSIOp, arith::RemSIOp, arith::FloorDivSIOp,
             arith::CeilDivSIOp, arith::DivUIOp, arith::RemUIOp,
             arith::CeilDivUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(
            def))
      return notConstant(fail, v, kNotTileFunction);

    // Operands first, so a culprit deeper in the expression is the one named.
    SmallVector<int64_t> in;
    if (!isa<arith::ConstantOp>(def))
      for (Value operand : def->getOperands()) {
        FailureOr<int64_t> x = eval(operand, fail);
        if (failed(x))
          return failure();
        in.push_back(*x);
      }
    Type inTy = def->getNumOperands() ? def->getOperand(0).getType() : type;
    auto u = [&](unsigned i) { return asUnsigned(in[i], inTy); };
    auto wrap = [&](uint64_t r) { return norm((int64_t)r, type); };

    std::optional<int64_t> r =
        llvm::TypeSwitch<Operation *, std::optional<int64_t>>(def)
            .Case([&](arith::ConstantOp op) -> std::optional<int64_t> {
              if (auto i = dyn_cast<IntegerAttr>(op.getValue()))
                return norm(i.getValue().getSExtValue(), type);
              return std::nullopt;
            })
            .Case([&](triton::GetProgramIdOp op) -> std::optional<int64_t> {
              if (op.getAxisAsInt() != 0)
                return std::nullopt;
              return norm(tile, type);
            })
            .Case([&](triton::GetNumProgramsOp op) -> std::optional<int64_t> {
              if (op.getAxisAsInt() != 0)
                return std::nullopt;
              return norm(tiles, type);
            })
            .Case([&](arith::AddIOp) { return wrap(u(0) + u(1)); })
            .Case([&](arith::SubIOp) { return wrap(u(0) - u(1)); })
            .Case([&](arith::MulIOp) { return wrap(u(0) * u(1)); })
            .Case([&](arith::AndIOp) { return wrap(u(0) & u(1)); })
            .Case([&](arith::OrIOp) { return wrap(u(0) | u(1)); })
            .Case([&](arith::XOrIOp) { return wrap(u(0) ^ u(1)); })
            .Case([&](arith::MaxSIOp) { return std::max(in[0], in[1]); })
            .Case([&](arith::MinSIOp) { return std::min(in[0], in[1]); })
            .Case([&](arith::MaxUIOp) { return wrap(std::max(u(0), u(1))); })
            .Case([&](arith::MinUIOp) { return wrap(std::min(u(0), u(1))); })
            .Case([&](arith::SelectOp) { return in[0] ? in[1] : in[2]; })
            .Case([&](arith::ExtSIOp) { return in[0]; })
            .Case([&](arith::ExtUIOp) { return (int64_t)u(0); })
            .Case([&](arith::TruncIOp) { return norm(in[0], type); })
            .Case([&](arith::IndexCastOp) { return norm(in[0], type); })
            .Case(
                [&](arith::IndexCastUIOp) { return norm((int64_t)u(0), type); })
            .Case([&](arith::CmpIOp op) -> std::optional<int64_t> {
              int64_t a = in[0], b = in[1];
              uint64_t ua = u(0), ub = u(1);
              bool c = false;
              switch (op.getPredicate()) {
              case arith::CmpIPredicate::eq:
                c = a == b;
                break;
              case arith::CmpIPredicate::ne:
                c = a != b;
                break;
              case arith::CmpIPredicate::slt:
                c = a < b;
                break;
              case arith::CmpIPredicate::sle:
                c = a <= b;
                break;
              case arith::CmpIPredicate::sgt:
                c = a > b;
                break;
              case arith::CmpIPredicate::sge:
                c = a >= b;
                break;
              case arith::CmpIPredicate::ult:
                c = ua < ub;
                break;
              case arith::CmpIPredicate::ule:
                c = ua <= ub;
                break;
              case arith::CmpIPredicate::ugt:
                c = ua > ub;
                break;
              case arith::CmpIPredicate::uge:
                c = ua >= ub;
                break;
              }
              return norm(c ? 1 : 0, type);
            })
            .Default([](Operation *) { return std::nullopt; });
    if (r)
      return *r;

    // The divisions and shifts, which can be undefined on a tile rather than
    // merely unknown -- so they say which.
    std::optional<int64_t> d =
        llvm::TypeSwitch<Operation *, std::optional<int64_t>>(def)
            .Case<arith::DivSIOp, arith::RemSIOp, arith::FloorDivSIOp,
                  arith::CeilDivSIOp, arith::DivUIOp, arith::RemUIOp,
                  arith::CeilDivUIOp>(
                [&](Operation *op) -> std::optional<int64_t> {
                  if (in[1] == 0)
                    return std::nullopt;
                  int64_t a = in[0], b = in[1];
                  if (isa<arith::DivUIOp>(op))
                    return wrap(u(0) / u(1));
                  if (isa<arith::RemUIOp>(op))
                    return wrap(u(0) % u(1));
                  if (isa<arith::CeilDivUIOp>(op))
                    return wrap(u(0) / u(1) + (u(0) % u(1) != 0));
                  // The one signed overflow, INT_MIN / -1, is poison in arith.
                  if (b == -1 && a == norm(llvm::minIntN(widthOf(type)), type))
                    return std::nullopt;
                  if (isa<arith::DivSIOp>(op))
                    return a / b;
                  if (isa<arith::RemSIOp>(op))
                    return a % b;
                  if (isa<arith::FloorDivSIOp>(op))
                    return llvm::divideFloorSigned(a, b);
                  return llvm::divideCeilSigned(a, b);
                })
            .Case<arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(
                [&](Operation *op) -> std::optional<int64_t> {
                  if (u(1) >= widthOf(type))
                    return std::nullopt;
                  if (isa<arith::ShLIOp>(op))
                    return wrap(u(0) << u(1));
                  if (isa<arith::ShRSIOp>(op))
                    return in[0] >> u(1);
                  return wrap(u(0) >> u(1));
                })
            .Default([](Operation *) { return std::nullopt; });
    if (d)
      return *d;
    if (isa<arith::DivSIOp, arith::RemSIOp, arith::FloorDivSIOp,
            arith::CeilDivSIOp, arith::DivUIOp, arith::RemUIOp,
            arith::CeilDivUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(
            def))
      return notConstant(fail, v,
                         "this value is undefined on tile " +
                             std::to_string(tile) +
                             " (a division by zero, an overflowing division, "
                             "or a shift by the width or more)");
    return notConstant(fail, v, kNotTileFunction);
  }

  int64_t tile, tiles;
  llvm::DenseMap<Value, int64_t> cache;
};

//===----------------------------------------------------------------------===//
// One pinned read
//===----------------------------------------------------------------------===//

/// The composed source's extent: a share's extent times the slice count on a
/// divided dimension, the share's own on an undivided one.
static FailureOr<SmallVector<int64_t>>
composedShape(tts::MakeDistributedDescriptorOp op) {
  SmallVector<std::optional<tts::WorkSliceEntry>> entries;
  llvm::MapVector<StringRef, int64_t> sliceCounts;
  if (failed(tts::readWorkSliceTable(op.getWorkSlices(), entries, sliceCounts,
                                     [&]() { return op.emitError(); })))
    return failure();
  auto shareShape =
      cast<RankedTensorType>(op.getPartial().getType()).getShape();
  SmallVector<int64_t> whole(shareShape.begin(), shareShape.end());
  for (auto [d, attr] : llvm::enumerate(op.getAxes())) {
    StringRef key = cast<StringAttr>(attr).getValue();
    if (!key.empty())
      whole[d] *= sliceCounts.find(key)->second;
  }
  return whole;
}

static std::string formatCoords(ArrayRef<int64_t> c) {
  std::string s = "[";
  llvm::raw_string_ostream os(s);
  llvm::interleaveComma(c, os);
  os << "]";
  return s;
}

/// One guard on the read: an `scf.if` around it, and which side the read is on.
struct Guard {
  scf::IfOp op;
  bool thenSide;
};

static LogicalResult analyzeRead(triton::DescriptorLoadOp load,
                                 tts::MakeDistributedDescriptorOp desc,
                                 tts::PinOp landing, int64_t tiles) {
  // --- where the read sits ---------------------------------------------------
  // Only `scf.if` may enclose it. Any other region op -- a loop above all --
  // would run the read a number of times this analysis cannot count, so it
  // would land more than once into one buffer.
  SmallVector<Guard> guards;
  for (Operation *child = load, *parent = load->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    if (isa<FunctionOpInterface>(parent))
      break;
    if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
      guards.push_back(
          {ifOp, ifOp.getThenRegion().isAncestor(child->getParentRegion())});
      continue;
    }
    InFlightDiagnostic diag =
        load.emitError()
        << "a read through a distributed descriptor whose value is pinned may "
           "be guarded by an if, but not run inside '"
        << parent->getName().getStringRef()
        << "': it would land more than once";
    diag.attachNote(parent->getLoc()) << "it is inside this";
    return diag;
  }
  // Outermost first, so a tile stops at the first guard it fails and an inner
  // condition is never evaluated where it does not run.
  std::reverse(guards.begin(), guards.end());

  // The descriptor is a collective: every tile makes it. And the transfer is
  // emitted ahead of these `if`s, which needs the composed source made there.
  // So a descriptor made inside an `if` that guards a read through it is
  // refused.
  for (Guard &g : guards)
    if (g.op->isAncestor(desc)) {
      InFlightDiagnostic diag =
          load.emitError()
          << "the distributed descriptor this reads through is made inside an "
             "if that guards the read; every tile makes the descriptor, so "
             "make it outside";
      diag.attachNote(g.op.getLoc()) << "this if";
      return diag;
    }

  ArrayRef<int64_t> block = desc.getBlockShape();
  unsigned rank = block.size();
  FailureOr<SmallVector<int64_t>> whole = composedShape(desc);
  if (failed(whole))
    return failure();

  // --- per tile: does it read, and what --------------------------------------
  SmallVector<std::optional<SmallVector<int64_t>>> src(tiles);
  for (int64_t t = 0; t < tiles; ++t) {
    TileEvaluator ev(t, tiles);
    NotConstant fail;
    bool reads = true;
    for (Guard &g : guards) {
      FailureOr<int64_t> c = ev.eval(g.op.getCondition(), fail);
      if (failed(c)) {
        InFlightDiagnostic diag =
            load.emitError()
            << "which tiles perform this read is decided by an if whose "
               "condition must be a constant on every tile, and on tile "
            << t << " it is not";
        diag.attachNote(fail.culprit.getLoc()) << fail.why;
        return diag;
      }
      if ((*c != 0) != g.thenSide) {
        reads = false;
        break;
      }
    }
    if (!reads)
      continue;

    SmallVector<int64_t> coords;
    for (auto [d, index] : llvm::enumerate(load.getIndices())) {
      FailureOr<int64_t> c = ev.eval(index, fail);
      if (failed(c)) {
        InFlightDiagnostic diag =
            load.emitError()
            << "the coordinate this read passes on dimension " << d
            << " must be a constant on every tile, and on tile " << t
            << " it is not";
        diag.attachNote(fail.culprit.getLoc()) << fail.why;
        return diag;
      }
      // The block lies inside the composed source, or it names memory that no
      // partition holds.
      if (*c < 0 || *c + block[d] > (*whole)[d])
        return load.emitError()
               << "on tile " << t << " this read takes " << block[d]
               << " elements from " << *c << " on dimension " << d
               << ", outside the composed source's extent " << (*whole)[d];
      coords.push_back(*c);
    }
    src[t] = std::move(coords);
  }

  // --- the destination -------------------------------------------------------
  SmallVector<int64_t> readers;
  for (int64_t t = 0; t < tiles; ++t)
    if (src[t])
      readers.push_back(t);
  if (readers.empty())
    return load.emitError()
           << "no tile performs this read: on every tile an if around it "
              "skips it, so the value it pins never lands";

  // TODO: several receivers of one box, which the design admits (§3 phase 3:
  // the copies are laid apart along a further dimension, in ascending tile id)
  // and this pass does not yet place in the destination.
  std::map<SmallVector<int64_t>, int64_t> firstReader;
  for (int64_t t : readers) {
    auto [it, inserted] = firstReader.insert({*src[t], t});
    if (!inserted)
      return load.emitError()
             << "tiles " << it->second << " and " << t
             << " both read the block at " << formatCoords(*src[t])
             << ": several tiles receiving one block is not supported yet";
  }

  // The destination is the union of the blocks, so they must tile a box: on
  // each dimension the distinct starts sit one block apart, and every
  // combination of them is read. Distinct blocks and a count equal to the
  // product of the per-dimension counts make that a one-to-one cover.
  SmallVector<int64_t> lo(rank), dstShape(rank);
  int64_t cells = 1;
  for (unsigned d = 0; d < rank; ++d) {
    std::set<int64_t> starts;
    for (int64_t t : readers)
      starts.insert((*src[t])[d]);
    lo[d] = *starts.begin();
    int64_t k = 0;
    for (int64_t s : starts) {
      if (s != lo[d] + k * block[d]) {
        SmallVector<int64_t> list(starts.begin(), starts.end());
        return load.emitError()
               << "on dimension " << d << " the tiles that read start at "
               << formatCoords(list) << ": the blocks must sit side by side, "
               << block[d] << " apart, for the destination to be contiguous";
      }
      ++k;
    }
    dstShape[d] = k * block[d];
    cells *= k;
  }
  if (cells != (int64_t)readers.size())
    return load.emitError()
           << readers.size() << " tiles read, and their blocks span " << cells
           << " places: every place in the box they span must be read for the "
              "destination to be whole";

  // --- the attribute ---------------------------------------------------------
  Builder b(load.getContext());
  SmallVector<Attribute> srcList, dstList;
  for (int64_t t = 0; t < tiles; ++t) {
    if (!src[t]) {
      srcList.push_back(b.getUnitAttr());
      dstList.push_back(b.getUnitAttr());
      continue;
    }
    SmallVector<int64_t> dst(rank);
    for (unsigned d = 0; d < rank; ++d)
      dst[d] = (*src[t])[d] - lo[d];
    srcList.push_back(b.getDenseI64ArrayAttr(*src[t]));
    dstList.push_back(b.getDenseI64ArrayAttr(dst));
  }
  landing->setAttr(tts::TTSDialect::kDistLandingAttrName,
                   b.getDictionaryAttr({
                       b.getNamedAttr(tts::TTSDialect::kSrcCoordinatesName,
                                      b.getArrayAttr(srcList)),
                       b.getNamedAttr(tts::TTSDialect::kDstShapeName,
                                      b.getDenseI64ArrayAttr(dstShape)),
                       b.getNamedAttr(tts::TTSDialect::kDstCoordinatesName,
                                      b.getArrayAttr(dstList)),
                   }));
  return success();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct AnalyzeDistributedDescPass
    : public mlir::triton::spyre::impl::AnalyzeDistributedDescBase<
          AnalyzeDistributedDescPass> {

  using AnalyzeDistributedDescBase::AnalyzeDistributedDescBase;

  AnalyzeDistributedDescPass(ArrayRef<int64_t> gridShape) { grid = gridShape; }

  void runOnOperation() override {
    // Same default as DistributeWork's and LowerDistributedDesc's.
    SmallVector<int64_t> gridShape(grid.begin(), grid.end());
    if (gridShape.empty())
      gridShape.push_back(32);
    int64_t tiles = 1;
    for (int64_t g : gridShape)
      tiles *= g;

    bool failedAny = false;
    getOperation().walk([&](triton::DescriptorLoadOp load) {
      auto desc =
          load.getDesc().getDefiningOp<tts::MakeDistributedDescriptorOp>();
      if (!desc)
        return;
      // The landing marker is among the read's users; the marker driver refuses
      // a second one later, so the first is the one.
      tts::PinOp landing;
      for (Operation *user : load.getResult().getUsers())
        if ((landing = dyn_cast<tts::PinOp>(user)))
          break;
      // TODO: refuse an unpinned COPY. The design rejects a copy whose landing
      // is not pinned (§10: a value received over the interconnect has to be
      // resident in the scratchpad before a compute unit reads it) and exempts
      // only a reduction (§7), whose folded value feeds the next computation
      // directly. Telling the two apart belongs with the reduction's lowering,
      // so until then every unpinned read is skipped, a copy included.
      if (!landing)
        return;
      // A tile is `tt.get_program_id x` only on a one-axis grid. On more, the
      // tile id is DistributeWork's linearization of every axis, which this
      // evaluator does not model.
      if (gridShape.size() != 1) {
        load.emitError() << "a pinned read through a distributed descriptor "
                            "needs a one-axis grid, and this one has "
                         << gridShape.size() << " axes";
        failedAny = true;
        return;
      }
      if (failed(analyzeRead(load, desc, landing, tiles)))
        failedAny = true;
    });
    if (failedAny)
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>>
createAnalyzeDistributedDescPass(ArrayRef<int64_t> grid) {
  return std::make_unique<AnalyzeDistributedDescPass>(grid);
}

} // namespace mlir::triton::spyre
