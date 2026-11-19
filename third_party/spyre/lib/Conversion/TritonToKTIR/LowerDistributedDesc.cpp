//===- LowerDistributedDesc.cpp - Lower distributed descriptors to KTDP --===//
//
// Lowers each tts.make_distributed_descriptor, and the tt.descriptor_load ops
// reading its result, to one ktdp.construct_memory_view per partition, a
// ktdp.construct_distributed_memory_view over them, and an access tile plus a
// ktdp.load for each read.
//
// Moved here unchanged from LowerInterTile.cpp, which #217 removed together with
// tl.inter_tile; this file holds only what was that pass's second mode.
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

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
//   Phase 2  the access: a `construct_access_tile` at the offsets this instance
//            passed to `.load()`. Nothing has moved yet -- both view ops are
//            `Pure`, so a core may NAME another core's scratchpad without
//            anything crossing a core boundary.
//   Phase 3  the transfer: one `ktdp.load`, which IS the transfer. The landing
//            store the design also requires is not emitted here: it is a
//            `tts.pin` on the loaded value, and `MaterializePinnedBuffers` builds
//            it later, in the `spyrecode` stage.
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

/// Lower one `tts.make_distributed_descriptor`, and the reads through its result.
static LogicalResult
lowerDistributedDescriptor(triton::tts::MakeDistributedDescriptorOp op,
                           IRRewriter &rewriter, int64_t cores) {
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  auto shareTy = cast<RankedTensorType>(op.getPartial().getType());
  ArrayRef<int64_t> shareShape = shareTy.getShape();
  unsigned rank = shareShape.size();

  // --- the partition table, through the op's own reader -----------------------
  SmallVector<triton::tts::WorkSliceEntry> entries;
  llvm::MapVector<StringRef, int64_t> sliceCounts;
  auto emitError = [&]() { return op.emitError(); };
  if (failed(triton::tts::readWorkSliceTable(op.getWorkSlices(), entries, sliceCounts,
                                     emitError)))
    return failure();

  // Holder = the partition's index in the table. §1 resolves a holder by matching
  // the partition's coordinate dict against the tile -> coordinate table, which
  // is a second table this op does not carry; the two coincide exactly when the
  // table has one entry per tile, and then the match is the identity. Anything
  // else -- a tensor held by 8 of 32 cores, or a broadcast source of 1 -- needs
  // that second table, and is refused here rather than resolved by a guess about
  // which cores were meant.
  if ((int64_t)entries.size() != cores)
    return op.emitError()
           << "work_slices has " << entries.size() << " partitions for a grid of "
           << cores
           << ": only one partition per tile is supported, since the holder is "
              "then the partition's own index. A shorter table needs the tile "
              "coordinate table to resolve holders against";

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
  partitions.reserve(entries.size());
  for (auto [id, entry] : llvm::enumerate(entries)) {
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
      // A store through a distributed view is not forbidden by KTDP, and this
      // design composes no destination view (assumption 4): under the pull model
      // each destination holder writes into its own scratchpad, so there is
      // nothing on that side to compose. Refused rather than lowered, so the
      // restriction is the design's and not a silent gap.
      return op.emitError()
             << "a distributed descriptor may only be read: '"
             << user->getName().getStringRef()
             << "' is not a tt.descriptor_load";
    }

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
