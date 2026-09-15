"""Assembling delivery: the block spans several source regions, so a load gathers.

Two kernels, and **both blocks span more than one region**. What separates them is
the offset pattern across instances, which is the other of the two knobs the surface
has:

  inter_tile_gather             one instance loads         -> a single holder
  inter_tile_gather_replicated  4 instances share an offset -> four holders each

The second is the combination: assemble *and* replicate, which is what half the
measured records need and what neither of the other directories shows. See README.md,
"Why both kernels".

So this file is one row of the taxonomy and the sibling directories are the others:
``inter_tile_broadcast`` (block equals a region -- broadcast and permutation),
``inter_tile_scatter`` (block inside one region), ``inter_tile_all_to_all`` (block
spans all of them). ``inter_tile_reduce`` is the fold family instead of the copy
family.

`work_slices` is a dense positional list in both — entry k is the region held by
tile k. Both records' holders are the *even* tiles, which a list cannot express;
see README.md, "Deviation from the record".

DRAFT, written for discussion. It does not run and the suite does not discover it:
``tl.make_distributed_descriptor`` does not exist, ``tl.spyre_tensor_layout`` has no
memory-space argument yet, and there is no ``meta.py``, so ``conftest.py`` never
imports this file. See README.md for the fixtures and the assumptions.

Every size is written as a literal rather than a parameter, on purpose — the point
is to read the shape of the expression, not to sweep it. Real code generation would
inject these constants anyway.

  inter_tile_gather

  extents          {mb: 8, out: 128}     src == dst
  source tile      {mb: 1, out: 64}      16 tiles: 0..15 hold, 16..31 hold nothing
  source           region l on tile l
  destination tile {mb: 8, out: 128}     1 tile: all 16 source tiles assembled on
                                         tile 0, so the destination tile is the
                                         whole tensor
  grid             [32]

  inter_tile_gather_replicated

  extents          {mb: 512, in: 4096}   src == dst
  source tile      {mb: 32, in: 4096}    16 tiles: 0..15 hold, 16..31 hold nothing
  source           region l on tile l
  destination tile {mb: 64, in: 4096}    32 tiles: 8 assembled regions of two source
                                         tiles each, replicated to tiles 4g..4g+3
  grid             [32]

"tile" is what one core holds; "region" is a distinct piece of the tensor. In
``inter_tile_gather`` the two coincide -- one holder per region on both sides. In
``inter_tile_gather_replicated`` they diverge on the destination side: 8 assembled
regions but 32 destination tiles, because each is replicated to four. So a tile count
can exceed a region count, and `work_slices` -- which pairs each region with its
holder -- describes the source side only.
"""

import triton
import triton.language as tl

# work_slices: entry l is the region held by tile l. Two axes are divided, so each
# entry names two dim ids -- mb slowest, out fastest, which is the odometer order
# the region count implies. Written out rather than comprehended, since the mapping
# is the thing under discussion.
SRC_WORK_SLICES = [
    {"mb": 0, "out": 0}, {"mb": 0, "out": 1},
    {"mb": 1, "out": 0}, {"mb": 1, "out": 1},
    {"mb": 2, "out": 0}, {"mb": 2, "out": 1},
    {"mb": 3, "out": 0}, {"mb": 3, "out": 1},
    {"mb": 4, "out": 0}, {"mb": 4, "out": 1},
    {"mb": 5, "out": 0}, {"mb": 5, "out": 1},
    {"mb": 6, "out": 0}, {"mb": 6, "out": 1},
    {"mb": 7, "out": 0}, {"mb": 7, "out": 1},
]


@triton.jit
def inter_tile_gather(x_ptr, out_ptr, SRC_SLICES: tl.constexpr):
    """Assemble 16 regions of {mb: 1, out: 64} into one {mb: 8, out: 128} on tile 0."""
    pid = tl.program_id(0)  # grid is [32]: 16 source regions, 1 destination holder

    # --- my share, resident in my own scratchpad --------------------------------
    # A ct_local descriptor is per-tile shaped: 1 x 64, one region, not the 8 x 128
    # whole. The layout is logical [mb, out] with the stick on out and stick size
    # 64, so physically [out // 64, mb, out % 64] -- and a share is exactly one
    # stick wide.
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[1, 64], strides=[64, 1], block_shape=[1, 64],
    )
    tl.spyre_tensor_layout(
        x_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )

    # The destination is one region on one tile, so its descriptor is the whole
    # tensor. Built unconditionally: only the transfer below is guarded.
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[8, 128], strides=[128, 1], block_shape=[8, 128],
    )
    tl.spyre_tensor_layout(
        out_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )

    # Only tiles 0..15 hold a source region. The others must still reach the
    # constructor with a value, and theirs is never read, since SRC_SLICES has
    # sixteen entries and so describes tiles 0..15 only.
    partial = tl.zeros([1, 64], dtype=tl.float16)
    if pid < 16:
        partial = x_desc.load([0, 0])            # [1, 64] fp16 -- one whole region

    # --- compose the 16 regions into the whole tensor ---------------------------
    # Both dims are divided, so both are named: dim 0 along "mb", dim 1 along "out".
    # len(axes) == rank(partial), so no axis is created. The compose grows dim 0
    # from 1 to 8 * 1 and dim 1 from 64 to 2 * 64 -- each dim's index count times
    # the share's extent, which is why no extents are passed.
    #
    # This is *outside* the guard on purpose. The constructor is collective: its
    # operand is a per-instance value, so every instance has to reach it, including
    # the 16 that hold nothing. Only the load below belongs to the destination set.
    whole = tl.make_distributed_descriptor(
        partial,
        work_slices=SRC_SLICES,
        axes=["mb", "out"],          # composed: [8, 128]
        block_shape=[8, 128],        # the whole tensor -- all 16 regions per load
    )

    # --- assemble, on the one tile that holds the result ------------------------
    # block_shape is the whole composed tensor, so this single load draws from all
    # 16 regions: 15 remote reads and one local. That the block spans several
    # regions is the entire difference from the broadcast, where it spanned one.
    #
    # The guard *is* the destination set -- a set of one here. There is no table
    # for it, and nothing at this level checks that the guarded tiles cover the
    # tensor.
    if pid == 0:
        assembled = whole.load([0, 0])           # [8, 128] fp16 -- all 16 regions
        out_desc.store([0, 0], assembled)


# work_slices for the grouped gather: entry l is the region held by tile l. One axis
# is divided -- mb into sixteen shards of 32 -- so a single dim id per entry. Written
# out rather than comprehended, since the mapping is the thing under discussion.
GROUPED_WORK_SLICES = [
    {"mb":  0}, {"mb":  1}, {"mb":  2}, {"mb":  3},
    {"mb":  4}, {"mb":  5}, {"mb":  6}, {"mb":  7},
    {"mb":  8}, {"mb":  9}, {"mb": 10}, {"mb": 11},
    {"mb": 12}, {"mb": 13}, {"mb": 14}, {"mb": 15},
]


@triton.jit
def inter_tile_gather_replicated(x_ptr, out_ptr, GROUPED_SLICES: tl.constexpr):
    """Assemble 2 regions per group, then hold each result on four tiles."""
    pid = tl.program_id(0)  # grid is [32]: 16 source regions, 8 results, 4 holders each

    # --- my share, resident in my own scratchpad --------------------------------
    # A ct_local descriptor is per-tile shaped: 32 x 4096, one region, not the
    # 512 x 4096 whole. The layout is logical [mb, in] with the stick on in and stick
    # size 64, so physically [in // 64, mb, in % 64].
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[32, 4096], strides=[4096, 1], block_shape=[32, 4096],
    )
    tl.spyre_tensor_layout(
        x_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )

    # Only tiles 0..15 hold a source region. The others must still reach the
    # constructor with a value, and theirs is never read, since GROUPED_SLICES has
    # sixteen entries and so describes tiles 0..15 only.
    partial = tl.zeros([32, 4096], dtype=tl.float16)
    if pid < 16:
        partial = x_desc.load([0, 0])            # [32, 4096] fp16 -- one whole region

    # --- compose the 16 regions into the whole tensor ---------------------------
    # Only dim 0 is divided; dim 1 is whole, so it is named None. len(axes) ==
    # rank(partial), so no axis is created. The compose grows dim 0 from 32 to
    # 16 * 32 == 512 -- the index count times the share's extent, which is why no
    # extents are passed.
    whole = tl.make_distributed_descriptor(
        partial,
        work_slices=GROUPED_SLICES,
        axes=["mb", None],           # composed: [512, 4096]
        block_shape=[64, 4096],      # two regions per load
    )

    # --- read the assembled region I end up holding -----------------------------
    # This is the two knobs used together, and it is the only fixture where both are
    # non-trivial at once:
    #
    #   block spans 2 regions  -> P = 2, so the load assembles       (gather)
    #   4 instances share g    -> each result has four holders       (replication)
    #
    # inter_tile_gather above has the first without the second (one holder); the
    # broadcast has the second without the first (block = one region). Half the
    # measured records need both.
    g = pid // 4
    mine = whole.load([g * 64, 0])               # [64, 4096] fp16 -- two regions

    # --- land it in my own scratchpad ------------------------------------------
    # Also ct_local and per-tile shaped, so every tile stores at local [0, 0] and
    # those are 32 distinct places. No guard: all 32 tiles hold an assembled region,
    # and an absent guard is how "every instance" is stated.
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[64, 4096], strides=[4096, 1], block_shape=[64, 4096],
    )
    tl.spyre_tensor_layout(
        out_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )
    out_desc.store([0, 0], mine)
