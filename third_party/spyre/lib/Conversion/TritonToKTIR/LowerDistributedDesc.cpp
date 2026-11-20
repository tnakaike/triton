//===- LowerDistributedDesc.cpp - Lower distributed descriptors to KTDP --===//
//
// Lowers each tts.make_distributed_descriptor, and the tt.descriptor_load ops
// reading its result, to one ktdp.construct_memory_view per partition, a
// ktdp.construct_distributed_memory_view over them, and for each read an access
// tile plus a ktdp.load -- followed, for a read whose value a pin lands, by a
// ktdp.store into a composed dst view.
//
// The src side was moved here from LowerInterTile.cpp, which #217 removed
// together with tl.inter_tile; it was that pass's second mode.
//
//===----------------------------------------------------------------------===//

#include "Conversion/TritonToKTIR/Passes.h"
#include "Dialect/KTDP/Utils/Utility.h"
#include "Dialect/TTS/IR/Dialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <map>

using namespace mlir;

namespace mlir::triton::spyre {
#define GEN_PASS_DEF_LOWERDISTRIBUTEDDESC
#include "Conversion/TritonToKTIR/Passes.h.inc"
} // namespace mlir::triton::spyre

namespace {

//===----------------------------------------------------------------------===//
// tts.make_distributed_descriptor — the compose, and the reads through it
//
// Composes a redistribution, then performs it.
//
// This is design §3's three phases in one function, and the phases are worth
// keeping visible because each has a different reason to be here:
//
//   Phase 1  the source view: one `construct_memory_view` per partition,
//            differing ONLY in `coordinate_set` and `ct_id`, composed by
//            `construct_distributed_memory_view`.
//   Phase 2  the access: a `construct_access_tile` at the coordinates this
//            instance passed to `.load()`. Nothing has moved yet -- both view
//            ops are `Pure`, so a core may NAME another core's scratchpad
//            without anything crossing a core boundary.
//   Phase 3  the transfer: one `ktdp.load`, which IS the transfer. For a read
//            whose value is pinned, the landing is a `ktdp.store` into the
//            composed dst view, emitted here from what AnalyzeDistributedDesc
//            stated (see lowerLandedRead); a reduction's read has none.
//
// Where each share LIVES is READ, not recovered: the `tts.pin` marker among the
// share's users carries the memory space and the address, and this pass runs before
// `LowerTTSMarkers`, so the marker is still standing.
//
// It deliberately does not look for a buffer. An earlier revision walked
// `ktdp.load -> construct_access_tile -> construct_memory_view`, because the pin was
// materialized in this stage; it is materialized in `spyrecode` now, so by the time a
// buffer exists this pass has long finished. Nothing is lost: a share's address is
// what the pin SAID, and the marker says it directly.
//
// Three passes have to agree here and nothing local to any of them makes it obvious,
// so it is stated once.
//
// This pass erases the compose, which is the share's only use besides the marker. So
// from here until `MaterializePinnedBuffers` runs, the MARKER is the only thing
// holding the value: it is not memory-effect-free, so DCE leaves it, and while it
// stands it uses the value. That is why `LowerTTSMarkers` converts it to an attribute
// only after this stage's closing canonicalize -- converting earlier would let that
// canonicalize delete a share nobody was left using, pin and all.
//
// Then `MaterializePinnedBuffers`, first in `spyrecode`, inserts the share's store
// immediately after its producer -- ahead of the compose and the load emitted here,
// which sit further down -- so a share is resident before any peer reads it. By then
// the pinned value has no uses at all, and that pass still builds the view and the
// store, which is precisely what makes the share readable from another core.
//===----------------------------------------------------------------------===//

/// The slice index `entry` gives `key`, or nullopt if it carries none.
static std::optional<int64_t> sliceIndexOf(const triton::tts::WorkSliceEntry &entry,
                                           StringRef key) {
  for (auto &[k, index] : entry)
    if (k == key)
      return index;
  return std::nullopt;
}

/// Row-major strides of a dense buffer of `shape`.
static SmallVector<int64_t> rowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

/// The value on dimension `d` of each dst tile's entry in `table`, as one
/// `index` that is right on every tile: a constant where all dst tiles agree,
/// else a chain of selects on the tile id. A tile that is not a dst tile gets
/// the first dst tile's value, which keeps its access in bounds; what it does
/// with the access is nothing (see lowerLandedRead).
static Value perTileIndex(OpBuilder &b, Location loc, Value tileId,
                          ArrayRef<std::optional<SmallVector<int64_t>>> table,
                          unsigned d) {
  std::optional<int64_t> first;
  for (const auto &entry : table)
    if (entry) {
      first = (*entry)[d];
      break;
    }
  Value v = arith::ConstantIndexOp::create(b, loc, *first);
  for (auto [t, entry] : llvm::enumerate(table)) {
    if (!entry || (*entry)[d] == *first)
      continue;
    Value isT = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::eq, tileId,
        arith::ConstantIndexOp::create(b, loc, (int64_t)t));
    v = arith::SelectOp::create(
        b, loc, isT, arith::ConstantIndexOp::create(b, loc, (*entry)[d]), v);
  }
  return v;
}

/// The integer constant `v` is, if it is one.
static std::optional<int64_t> constantOf(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
      return i.getValue().getSExtValue();
  return std::nullopt;
}

/// Whether `v` is quasi-affine in the program id: built from integer constants,
/// `tt.get_program_id x`, `tt.get_num_programs x`, addition, subtraction,
/// multiplication by a constant, division and remainder by a positive constant,
/// and integer casts. A divisor is a constant, so the expression is defined on
/// every tile, including the tiles that do not receive.
static bool isQuasiAffine(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<arith::ConstantOp>(def))
    return constantOf(v).has_value();
  if (auto pid = dyn_cast<triton::GetProgramIdOp>(def))
    return pid.getAxisAsInt() == 0;
  if (auto num = dyn_cast<triton::GetNumProgramsOp>(def))
    return num.getAxisAsInt() == 0;
  if (isa<arith::AddIOp, arith::SubIOp>(def))
    return isQuasiAffine(def->getOperand(0)) &&
           isQuasiAffine(def->getOperand(1));
  if (isa<arith::MulIOp>(def))
    return (constantOf(def->getOperand(0)) &&
            isQuasiAffine(def->getOperand(1))) ||
           (constantOf(def->getOperand(1)) &&
            isQuasiAffine(def->getOperand(0)));
  if (isa<arith::DivSIOp, arith::FloorDivSIOp, arith::RemSIOp, arith::DivUIOp,
          arith::RemUIOp>(def)) {
    std::optional<int64_t> divisor = constantOf(def->getOperand(1));
    return divisor && *divisor > 0 && isQuasiAffine(def->getOperand(0));
  }
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
          arith::IndexCastUIOp>(def))
    return isQuasiAffine(def->getOperand(0));
  return false;
}

/// `v`, available ahead of `hoist`: the ops computing it that sit inside
/// `hoist` are cloned at the builder's insertion point, the rest are reused.
static Value cloneAhead(OpBuilder &b, Value v, Operation *hoist,
                        IRMapping &mapping) {
  if (Value m = mapping.lookupOrNull(v))
    return m;
  Operation *def = v.getDefiningOp();
  if (!hoist->isAncestor(def))
    return v;
  for (Value operand : def->getOperands())
    cloneAhead(b, operand, hoist, mapping);
  return b.clone(*def, mapping)->getResult(0);
}

/// Lower a read whose value lands in a pin, from what AnalyzeDistributedDesc
/// stated on the landing marker (design §3 phases 3-4).
///
/// The `if` that chose the dst tiles was read by the analysis and does not
/// reach the transfer. The transfer is emitted unguarded ahead of the outermost
/// `if` around the read: a load from the src view and a store into the dst
/// view, each at the coordinates this tile has in `tts.dist_landing`. A store
/// through a distributed view is performed only by the holder of the part it
/// lands in (§4), so on a tile that is not a dst tile the store does nothing
/// and the load, whose only use it is, moves nothing.
///
/// The dst view composes the dst tiles' landings: one view per dst tile, in
/// that tile's scratchpad at the pin's offset, of the landing's shape.
///
/// The read itself is replaced by a load from this tile's own landing, so its
/// uses, which stay inside the `if`, read the value where it landed (§2). The
/// marker is consumed: the store above is its landing, and leaving it would
/// have MaterializePinnedBuffers store the value a second time.
static LogicalResult lowerLandedRead(triton::DescriptorLoadOp load,
                                     mlir::triton::tts::PinOp landingPin,
                                     Value whole, ArrayRef<int64_t> blockShape,
                                     IRRewriter &rewriter, int64_t cores) {
  MLIRContext *ctx = load.getContext();
  Location loc = load.getLoc();
  auto emitError = [&]() { return landingPin.emitError(); };
  triton::tts::DistLanding landing;
  if (failed(triton::tts::readDistLanding(
          landingPin->getAttr(triton::tts::TTSDialect::kDistLandingAttrName),
          landing, emitError)))
    return failure();
  if ((int64_t)landing.dstCoordinates.size() != cores)
    return landingPin.emitError()
           << "tts.dist_landing has " << landing.dstCoordinates.size()
           << " entries for a grid of " << cores << ": it needs one per tile";
  auto offsetAttr = landingPin.getOffsetAttr();
  if (!offsetAttr)
    return landingPin.emitError()
           << "the landing's pin states no offset, so the dst view has none";

  auto valueTy = cast<RankedTensorType>(load.getResult().getType());
  ArrayRef<int64_t> landingShape = valueTy.getShape();
  Type elemTy = valueTy.getElementType();
  SmallVector<int64_t> landingStrides = rowMajorStrides(landingShape);
  unsigned rank = landingShape.size();

  // The outermost `if` around the read, or the read itself.
  // AnalyzeDistributedDesc has refused every other enclosing region and a
  // descriptor made inside one of these `if`s, so this sits in the
  // descriptor's block, after it.
  Operation *hoist = load;
  for (Operation *p = load->getParentOp(); p && !isa<FunctionOpInterface>(p);
       p = p->getParentOp())
    if (isa<scf::IfOp>(p))
      hoist = p;
  rewriter.setInsertionPoint(hoist);

  // --- the dst view ----------------------------------------------------------
  Value offset =
      arith::ConstantIndexOp::create(rewriter, loc, offsetAttr.getInt());
  SmallVector<Value> landings;
  for (auto [t, dst] : llvm::enumerate(landing.dstCoordinates)) {
    if (!dst)
      continue;
    SmallVector<int64_t> his(rank);
    for (unsigned d = 0; d < rank; ++d)
      his[d] = (*dst)[d] + landingShape[d];
    auto space = mlir::ktdp::MemorySpaceAttr::get(
        ctx, mlir::ktdp::MemorySpaceKind::ct_local, /*ct_id=*/(int32_t)t);
    landings.push_back(mlir::triton::ktdp::buildMemoryView(
        rewriter, loc, offset, landingShape, landingStrides, /*dynSizes=*/{},
        /*dynStrides=*/{}, elemTy, space,
        mlir::triton::ktdp::buildBoxSetND(ctx, *dst, his),
        /*spaceInResultType=*/true));
  }
  Value dstView =
      mlir::ktdp::ConstructDistributedMemoryViewOp::create(
          rewriter, loc, MemRefType::get(landing.dstShape, elemTy), landings)
          .getResult();

  // --- the transfer, unguarded -----------------------------------------------
  // The coordinates on every tile. Where the read's own coordinate is
  // quasi-affine in the program id it is cloned ahead of the `if` and used on
  // every tile, the dst coordinate being it minus the dst view's origin in the
  // src view. On a tile that does not receive, its value may fall outside the
  // src view or inside a part some other tile holds; either way the transfer
  // does nothing there, because a distributed view moves only what a holder
  // holds and a store lands only in its own tile's part (design §4). That is a
  // premise on the consumer of the views, stated here because the code relies
  // on it.
  //
  // Otherwise the coordinates come from tts.dist_landing, as a select chain on
  // the tile id. The read's expression is not cloned then: a division by a
  // value that depends on the program id may be by zero on a tile that does
  // not receive.
  Value tileId;
  auto getTileId = [&]() {
    // A fresh `tt.get_program_id`; DistributeWork replaces it later.
    if (!tileId)
      tileId = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(),
          triton::GetProgramIdOp::create(rewriter, loc, 0));
    return tileId;
  };
  // One chain per distinct column of per-tile values: where the src and dst
  // coordinates agree on a dimension both accesses share it.
  std::map<SmallVector<int64_t>, Value> chains;
  auto chainFor = [&](ArrayRef<std::optional<SmallVector<int64_t>>> table,
                      unsigned d) {
    SmallVector<int64_t> column;
    for (const auto &entry : table)
      column.push_back(entry ? (*entry)[d] : -1);
    Value &v = chains[column];
    if (!v)
      v = perTileIndex(rewriter, loc, getTileId(), table, d);
    return v;
  };

  // The dst view's origin in the src view, from any dst tile.
  SmallVector<int64_t> origin(rank);
  for (auto [src, dst] :
       llvm::zip_equal(landing.srcCoordinates, landing.dstCoordinates))
    if (src) {
      for (unsigned d = 0; d < rank; ++d)
        origin[d] = (*src)[d] - (*dst)[d];
      break;
    }

  IRMapping cloned;
  SmallVector<Value> srcIdx, dstIdx;
  for (auto [d, coord] : llvm::enumerate(load.getIndices())) {
    if (!isQuasiAffine(coord)) {
      srcIdx.push_back(chainFor(landing.srcCoordinates, d));
      dstIdx.push_back(chainFor(landing.dstCoordinates, d));
      continue;
    }
    Value src = cloneAhead(rewriter, coord, hoist, cloned);
    if (!src.getType().isIndex())
      src = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                       src);
    srcIdx.push_back(src);
    Value dst = src;
    if (origin[d] != 0) {
      Value o = arith::ConstantIndexOp::create(rewriter, loc, origin[d]);
      dst = arith::SubIOp::create(rewriter, loc, src, o);
    }
    dstIdx.push_back(dst);
  }
  Value srcTile = mlir::triton::ktdp::buildAccessTile(rewriter, loc, whole,
                                                      blockShape, srcIdx);
  Value moved =
      mlir::ktdp::LoadOp::create(rewriter, loc, valueTy, srcTile).getResult();
  Value dstTile = mlir::triton::ktdp::buildAccessTile(rewriter, loc, dstView,
                                                      landingShape, dstIdx);
  mlir::ktdp::StoreOp::create(rewriter, loc, moved, dstTile);

  // This tile's own landing, for the read-back. Built here with the transfer,
  // ahead of the `if`, so it precedes every op that uses it.
  auto ownSpace = mlir::ktdp::MemorySpaceAttr::get(
      ctx, mlir::ktdp::MemorySpaceKind::ct_local, /*ct_id=*/-1);
  Value own = mlir::triton::ktdp::buildMemoryView(
      rewriter, loc, offset, landingShape, landingStrides, /*dynSizes=*/{},
      /*dynStrides=*/{}, elemTy, ownSpace);

  // --- the read, as a read of the landing ------------------------------------
  rewriter.setInsertionPoint(load);
  SmallVector<Value> zeros(
      rank, arith::ConstantIndexOp::create(rewriter, loc, 0).getResult());
  Value ownTile = mlir::triton::ktdp::buildAccessTile(rewriter, loc, own,
                                                      landingShape, zeros);
  Value readBack =
      mlir::ktdp::LoadOp::create(rewriter, loc, valueTy, ownTile).getResult();
  rewriter.eraseOp(landingPin);
  rewriter.replaceOp(load, readBack);
  return success();
}

/// Lower one `tts.make_distributed_descriptor`, and the reads through its result.
static LogicalResult
lowerDistributedDescriptor(triton::tts::MakeDistributedDescriptorOp op,
                           IRRewriter &rewriter, int64_t cores) {
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  auto shareTy = cast<RankedTensorType>(op.getPartial().getType());
  ArrayRef<int64_t> shareShape = shareTy.getShape();
  unsigned rank = shareShape.size();

  // --- the work-slice list, through the op's own reader ----------------------
  SmallVector<std::optional<triton::tts::WorkSliceEntry>> entries;
  llvm::MapVector<StringRef, int64_t> sliceCounts;
  auto emitError = [&]() { return op.emitError(); };
  if (failed(triton::tts::readWorkSliceTable(op.getWorkSlices(), entries, sliceCounts,
                                     emitError)))
    return failure();

  // One entry per tile, so the holder is the entry's position (§1). A tile that
  // holds nothing is `unit` and contributes no partition. The length is a claim
  // about the launch grid, which the op cannot see, so it is checked here.
  if ((int64_t)entries.size() != cores)
    return op.emitError()
           << "work_slices has " << entries.size() << " entries for a grid of "
           << cores
           << ": it needs one per tile, None where the tile holds nothing";

  // --- where the share lives, read off the pin -------------------------------
  // The MARKER OP, not the `tts.pin` attribute. This pass runs before
  // LowerTTSMarkers, which is the `ktir` stage's last, so the attribute does not
  // exist yet -- and that order is deliberate rather than incidental: the marker is
  // what holds the share alive once this pass erases the compose, which is the
  // share's only other use. See LowerTTSMarkers' contract in Passes.td.
  //
  // So the pin is found among the value's USERS. It is unique: the generic marker
  // driver refuses a second pin on one value, because `setAttr` would resolve that
  // collision by overwriting and lose the first silently.
  mlir::triton::tts::PinOp pin;
  for (Operation *user : op.getPartial().getUsers())
    if (auto candidate = dyn_cast<mlir::triton::tts::PinOp>(user)) {
      pin = candidate;
      break;
    }
  if (!pin)
    return op.emitError()
           << "cannot tell where the share lives: nothing pinned it. Pin it with "
              "tl.spyre_pin, which is what supplies a partition's offset";

  // A name, not `#ktdp.memory_space`: the pin is authored during tracing, where
  // loading ktdp would load `func` and abort the `ttir` stage's Inliner, so the op
  // carries a `MemorySpaceKind` name and each consumer symbolizes it.
  //
  // The op's verifier has already refused everything but `ct_local` and has already
  // required an offset, so neither is re-derived here -- what is left is to read
  // them. Checked anyway rather than asserted, since this pass is invocable on
  // hand-written IR in its own lit tests.
  std::optional<mlir::ktdp::MemorySpaceKind> shareKind =
      mlir::ktdp::symbolizeMemorySpaceKind(pin.getMemorySpace());
  if (!shareKind || *shareKind != mlir::ktdp::MemorySpaceKind::ct_local)
    return op.emitError()
           << "the share lives in '" << pin.getMemorySpace()
           << "', and a redistribution composes scratchpad shares: pin it in "
              "'ct_local'";
  auto offsetAttr = pin.getOffsetAttr();
  if (!offsetAttr)
    return op.emitError() << "the share's pin states no offset, so there is no "
                             "partition offset to compose from";

  // --- phase 1: one view per partition, then the compose ---------------------
  // Offsets, sizes and strides are identical across the partitions -- §3 says the
  // coordinate set and the holder are "the entire content of the distribution" --
  // so they are computed once, out of the loop. The offset is a fresh constant
  // rather than a value borrowed from elsewhere in the function, which is what a
  // pin's offset being an attribute buys: nothing has to be in scope here.
  rewriter.setInsertionPoint(op);
  Value offset =
      arith::ConstantIndexOp::create(rewriter, loc, offsetAttr.getInt());
  SmallVector<int64_t> strides(rank, 1);
  for (int i = (int)rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shareShape[i + 1];

  // The composed extents: a share's extent on a divided dimension is the whole
  // extent divided by the slice count, so the whole is the share's times that
  // count. An undivided dimension composes to itself.
  SmallVector<int64_t> wholeShape(shareShape.begin(), shareShape.end());
  SmallVector<StringRef> dimKey(rank);
  for (auto [d, attr] : llvm::enumerate(op.getAxes())) {
    StringRef key = cast<StringAttr>(attr).getValue();
    dimKey[d] = key;
    if (!key.empty())
      wholeShape[d] = shareShape[d] * sliceCounts.find(key)->second;
  }

  SmallVector<Value> partitions;
  for (auto [id, maybeEntry] : llvm::enumerate(entries)) {
    if (!maybeEntry)
      continue;
    const triton::tts::WorkSliceEntry &entry = *maybeEntry;
    // The region this partition owns, in the composed tensor's index space.
    SmallVector<int64_t> los(rank, 0), his(wholeShape.begin(), wholeShape.end());
    for (unsigned d = 0; d < rank; ++d) {
      if (dimKey[d].empty())
        continue;
      int64_t index = *sliceIndexOf(entry, dimKey[d]);
      los[d] = index * shareShape[d];
      his[d] = los[d] + shareShape[d];
    }

    auto space = mlir::ktdp::MemorySpaceAttr::get(
        ctx, mlir::ktdp::MemorySpaceKind::ct_local, /*ct_id=*/(int32_t)id);
    partitions.push_back(mlir::triton::ktdp::buildMemoryView(
        rewriter, loc, offset, shareShape, strides, /*dynSizes=*/{},
        /*dynStrides=*/{}, shareTy.getElementType(), space,
        mlir::triton::ktdp::buildBoxSetND(ctx, los, his),
        /*spaceInResultType=*/true));
  }

  auto wholeTy = MemRefType::get(wholeShape, shareTy.getElementType());
  Value whole = mlir::ktdp::ConstructDistributedMemoryViewOp::create(
                    rewriter, loc, wholeTy, partitions)
                    .getResult();

  // --- phases 2 and 3: the reads through the composed view -------------------
  // Collected first: each is erased as it is rewritten.
  SmallVector<Operation *> readers;
  for (Operation *user : op.getResult().getUsers())
    readers.push_back(user);

  ArrayRef<int64_t> blockShape = op.getBlockShape();
  for (Operation *user : readers) {
    auto load = dyn_cast<triton::DescriptorLoadOp>(user);
    if (!load) {
      // The dst view is composed from the landing pin, not authored: a store
      // through a distributed descriptor in the kernel names no dst tile the
      // analysis could check. Refused rather than lowered, so the restriction
      // is the design's and not a silent gap.
      return op.emitError()
             << "a distributed descriptor may only be read: '"
             << user->getName().getStringRef()
             << "' is not a tt.descriptor_load";
    }

    // A read whose value lands: the analysis has said who receives and where.
    mlir::triton::tts::PinOp landingPin;
    for (Operation *u : load.getResult().getUsers())
      if ((landingPin = dyn_cast<mlir::triton::tts::PinOp>(u)))
        break;
    if (landingPin) {
      if (!landingPin->hasAttr(triton::tts::TTSDialect::kDistLandingAttrName))
        return landingPin.emitError()
               << "this pin lands a read through a distributed descriptor, "
                  "but carries no tts.dist_landing: run "
                  "analyze-distributed-desc first";
      if (failed(lowerLandedRead(load, landingPin, whole, blockShape, rewriter,
                                 cores)))
        return failure();
      continue;
    }

    // A read whose value does not land (a reduction's, §7): read in place.
    rewriter.setInsertionPoint(load);
    Value tile = mlir::triton::ktdp::buildAccessTile(
        rewriter, load.getLoc(), whole, blockShape, load.getIndices());
    auto loaded = mlir::ktdp::LoadOp::create(
        rewriter, load.getLoc(),
        cast<RankedTensorType>(load.getResult().getType()), tile);
    rewriter.replaceOp(load, loaded.getResult());
  }

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct LowerDistributedDescPass
    : public mlir::triton::spyre::impl::LowerDistributedDescBase<
          LowerDistributedDescPass> {

  using LowerDistributedDescBase::LowerDistributedDescBase;

  LowerDistributedDescPass(ArrayRef<int64_t> gridShape) { grid = gridShape; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    // Same default as DistributeWork's, and for the same reason: a ListOption
    // takes no tablegen default, and [32] is the 1D-on-full-hardware case. The
    // grid is needed because a partition's HOLDER is its index in the table only
    // when the table has one entry per tile, which is a claim about the grid.
    SmallVector<int64_t> gridShape(grid.begin(), grid.end());
    if (gridShape.empty())
      gridShape.push_back(32);
    int64_t cores = 1;
    for (int64_t g : gridShape)
      cores *= g;

    SmallVector<triton::tts::MakeDistributedDescriptorOp> composes;
    mod.walk([&](triton::tts::MakeDistributedDescriptorOp op) { composes.push_back(op); });

    for (auto op : composes) {
      if (failed(lowerDistributedDescriptor(op, rewriter, cores)))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton::spyre {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerDistributedDescPass(ArrayRef<int64_t> grid) {
  return std::make_unique<LowerDistributedDescPass>(grid);
}

} // namespace mlir::triton::spyre
