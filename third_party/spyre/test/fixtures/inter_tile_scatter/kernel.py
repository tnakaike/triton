"""Inter-tile scatter: one row of one region split 8 ways across the receivers.

The block is **smaller than a region** — that is what makes this a scatter. It is
also the only one of the four fixtures with no guard on either side: every tile
holds a source region and every tile holds a destination region.

`work_slices` is a dense positional list: entry k is the region held by tile k.
This record's holders really are `0 .. 31`, so nothing is deviated from here.

DRAFT, written for discussion. It does not run and the suite does not discover it:
``tl.make_distributed_descriptor`` does not exist, ``tl.spyre_tensor_layout`` has no
memory-space argument yet, and there is no ``meta.py``, so ``conftest.py`` never
imports this file. See README.md for the fixture and the assumptions.

Sibling of ``inter_tile_broadcast``, ``inter_tile_gather``, ``inter_tile_all_to_all``
— four spellings of one constructor, differing only in the block each instance asks
for. ``inter_tile_reduce`` is the fold family instead of the copy family.

Every size is written as a literal rather than a parameter, on purpose — the point
is to read the shape of the expression, not to sweep it. Real code generation would
inject these constants anyway.

  extents          {mb: 512, out: 4096}  src == dst
  source tile      {mb: 64, out: 1024}   32 tiles, an 8 x 4 grid
  source           region k on tile k       every tile holds one
  destination tile {mb: 1, out: 128}     32 tiles: mb sliced to row 511, out
                                         scattered from one source tile to 8
                                         destination tiles
  grid             [32]

"tile" is what one core holds; "region" is a distinct piece of the tensor. The two
coincide throughout this fixture -- one holder per region on both sides -- so
nothing is replicated. Where they diverge is where a region has several holders,
as in ``inter_tile_broadcast``: 8 regions but 32 destination tiles.

Only the four regions covering row 511 are ever read; the other 28 tiles hold a
region that no instance asks for. Nothing declares that -- it falls out of the
offsets, and the arithmetic below says which tile supplies which.
"""

import triton
import triton.language as tl

# work_slices: entry k is the region held by tile k. Two axes are divided -- mb into
# eight row blocks of 64, out into four slabs of 1024 -- so tile k holds row block
# k // 4 and slab k % 4. Written out rather than comprehended, since the mapping is
# the thing under discussion.
SRC_WORK_SLICES = [
    {"mb": 0, "out": 0}, {"mb": 0, "out": 1}, {"mb": 0, "out": 2}, {"mb": 0, "out": 3},
    {"mb": 1, "out": 0}, {"mb": 1, "out": 1}, {"mb": 1, "out": 2}, {"mb": 1, "out": 3},
    {"mb": 2, "out": 0}, {"mb": 2, "out": 1}, {"mb": 2, "out": 2}, {"mb": 2, "out": 3},
    {"mb": 3, "out": 0}, {"mb": 3, "out": 1}, {"mb": 3, "out": 2}, {"mb": 3, "out": 3},
    {"mb": 4, "out": 0}, {"mb": 4, "out": 1}, {"mb": 4, "out": 2}, {"mb": 4, "out": 3},
    {"mb": 5, "out": 0}, {"mb": 5, "out": 1}, {"mb": 5, "out": 2}, {"mb": 5, "out": 3},
    {"mb": 6, "out": 0}, {"mb": 6, "out": 1}, {"mb": 6, "out": 2}, {"mb": 6, "out": 3},
    {"mb": 7, "out": 0}, {"mb": 7, "out": 1}, {"mb": 7, "out": 2}, {"mb": 7, "out": 3},
]


@triton.jit
def inter_tile_scatter(x_ptr, out_ptr, SRC_SLICES: tl.constexpr):
    """Split row 511 into 32 chunks of 128 columns, one per tile."""
    pid = tl.program_id(0)  # grid is [32]: 32 source regions, 32 destination regions

    # --- my share, resident in my own scratchpad --------------------------------
    # A ct_local descriptor is per-tile shaped: 64 x 1024, one region, not the
    # 512 x 4096 whole. The layout is logical [mb, out] with the stick on out and
    # stick size 64, so physically [out // 64, mb, out % 64].
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[64, 1024], strides=[1024, 1], block_shape=[64, 1024],
    )
    tl.spyre_tensor_layout(
        x_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )

    # No guard: all 32 tiles hold a source region, so all 32 load one. An absent
    # guard on this side is how "every instance holds a share" is stated.
    partial = x_desc.load([0, 0])                # [64, 1024] fp16 -- one whole region

    # --- compose the 32 regions into the whole tensor ---------------------------
    # Both dims are divided, so both are named. len(axes) == rank(partial), so no
    # axis is created. The compose grows dim 0 from 64 to 8 * 64 == 512 and dim 1
    # from 1024 to 4 * 1024 == 4096 -- each dim's index count times the share's
    # extent, which is why no extents are passed.
    whole = tl.make_distributed_descriptor(
        partial,
        work_slices=SRC_SLICES,
        axes=["mb", "out"],          # composed: [512, 4096]
        block_shape=[1, 128],        # a 128-column chunk of one row
    )

    # --- read the chunk I end up holding ---------------------------------------
    # block_shape is 1/512 of a row block and 1/8 of a slab, so this load lands
    # strictly inside one region: that the block is smaller than a region is the
    # entire difference from the gather, where it was larger.
    #
    # Which tile supplies me is not declared anywhere -- it is implied. Row 511 is
    # in row block 511 // 64 == 7, and column 128 * pid is in slab
    # (128 * pid) // 1024 == pid // 8, so my source is tile 4 * 7 + pid // 8, that
    # is 28 + pid // 8. Tiles 0..27 hold a region that nobody asks for.
    mine = whole.load([511, 128 * pid])          # [1, 128] fp16 -- inside one region

    # --- land it in my own scratchpad ------------------------------------------
    # Also ct_local and per-tile shaped, so every tile stores at local [0, 0] and
    # those are 32 distinct places. No guard: all 32 tiles hold a chunk, and an
    # absent guard is how "every instance" is stated.
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[1, 128], strides=[128, 1], block_shape=[1, 128],
    )
    tl.spyre_tensor_layout(
        out_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )
    out_desc.store([0, 0], mine)
