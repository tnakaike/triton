"""Inter-tile all-to-all: row-sharded to column-sharded, every tile to every tile.

The block is **larger than a region along the divided axis and smaller along
another** — that is what makes this an all-to-all rather than a gather or a
scatter. Each instance's block spans all 32 regions and takes 128 of each
region's 4096 columns, so every tile sends a distinct chunk to every tile:
32 x 32 = 1024 chunks, none of them whole and none of them shared.

`work_slices` is a dense positional list: entry k is the region held by tile k.
Holders are `0 .. 31`, so nothing is deviated from on that axis — but the routing
itself is not one of the catalogued records. See README.md.

DRAFT, written for discussion. It does not run and the suite does not discover it:
``tl.make_distributed_descriptor`` does not exist, ``tl.spyre_tensor_layout`` has no
memory-space argument yet, and there is no ``meta.py``, so ``conftest.py`` never
imports this file. See README.md for the fixture and the assumptions.

Sibling of ``inter_tile_broadcast``, ``inter_tile_gather``, ``inter_tile_scatter``
— four spellings of one constructor, differing only in the block each instance asks
for. ``inter_tile_reduce`` is the fold family instead of the copy family.

Every size is written as a literal rather than a parameter, on purpose — the point
is to read the shape of the expression, not to sweep it. Real code generation would
inject these constants anyway.

  extents          {mb: 512, out: 4096}  src == dst
  source tile      {mb: 16, out: 4096}   32 tiles, rows only
  source           region k on tile k       every tile holds one row shard
  destination tile {mb: 512, out: 128}   32 tiles: out sliced, mb assembled from
                                         all 32 source tiles
  grid             [32]

"tile" is what one core holds; "region" is a distinct piece of the tensor. The two
coincide throughout this fixture -- one holder per region on both sides -- so
nothing is replicated. Where they diverge is where a region has several holders,
as in ``inter_tile_broadcast``: 8 regions but 32 destination tiles.

Source and destination divide **different** axes, and that is the whole content of
an all-to-all: the source cuts `mb`, the destination cuts `out`, so no chunk of the
exchange is a whole region on either side.
"""

import triton
import triton.language as tl

# work_slices: entry k is the region held by tile k. One axis is divided -- mb into
# 32 shards of 16 rows -- so a single dim id per entry. Written out rather than
# comprehended, since the mapping is the thing under discussion.
SRC_WORK_SLICES = [
    {"mb":  0}, {"mb":  1}, {"mb":  2}, {"mb":  3},
    {"mb":  4}, {"mb":  5}, {"mb":  6}, {"mb":  7},
    {"mb":  8}, {"mb":  9}, {"mb": 10}, {"mb": 11},
    {"mb": 12}, {"mb": 13}, {"mb": 14}, {"mb": 15},
    {"mb": 16}, {"mb": 17}, {"mb": 18}, {"mb": 19},
    {"mb": 20}, {"mb": 21}, {"mb": 22}, {"mb": 23},
    {"mb": 24}, {"mb": 25}, {"mb": 26}, {"mb": 27},
    {"mb": 28}, {"mb": 29}, {"mb": 30}, {"mb": 31},
]


@triton.jit
def inter_tile_all_to_all(x_ptr, out_ptr, SRC_SLICES: tl.constexpr):
    """Exchange row shards for column shards: 16 x 4096 in, 512 x 128 out."""
    pid = tl.program_id(0)  # grid is [32]: 32 row shards in, 32 column shards out

    # --- my share, resident in my own scratchpad --------------------------------
    # A ct_local descriptor is per-tile shaped: 16 x 4096, one region, not the
    # 512 x 4096 whole. The layout is logical [mb, out] with the stick on out and
    # stick size 64, so physically [out // 64, mb, out % 64].
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[16, 4096], strides=[4096, 1], block_shape=[16, 4096],
    )
    tl.spyre_tensor_layout(
        x_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )

    # No guard: all 32 tiles hold a row shard, so all 32 load one.
    partial = x_desc.load([0, 0])                # [16, 4096] fp16 -- one whole region

    # --- compose the 32 row shards into the whole tensor ------------------------
    # Only dim 0 is divided; dim 1 is whole, so it is named None. len(axes) ==
    # rank(partial), so no axis is created. The compose grows dim 0 from 16 to
    # 32 * 16 == 512 -- the index count times the share's extent, which is why no
    # extents are passed -- and leaves dim 1 at 4096.
    whole = tl.make_distributed_descriptor(
        partial,
        work_slices=SRC_SLICES,
        axes=["mb", None],           # composed: [512, 4096]
        block_shape=[512, 128],      # all rows, 128 columns
    )

    # --- read the column shard I end up holding --------------------------------
    # The block is the full extent of the divided axis, so it spans every one of the
    # 32 regions; and it is 128 of 4096 columns, so it takes a strict slice of each.
    # Both at once is the all-to-all: 32 sources x 32 destinations, each pair
    # exchanging one [16, 128] chunk. A gather has only the first property, a
    # scatter only the second.
    #
    # Nothing here names the 32 partners. The offset does it: column 128 * pid of
    # every row shard, and the shards are exactly what the compose enumerated.
    mine = whole.load([0, 128 * pid])            # [512, 128] fp16 -- 32 x [16, 128]

    # --- land it in my own scratchpad ------------------------------------------
    # Also ct_local and per-tile shaped, so every tile stores at local [0, 0] and
    # those are 32 distinct places. No guard: all 32 tiles hold a column shard, and
    # an absent guard is how "every instance" is stated.
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[512, 128], strides=[128, 1], block_shape=[512, 128],
    )
    tl.spyre_tensor_layout(
        out_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )
    out_desc.store([0, 0], mine)
