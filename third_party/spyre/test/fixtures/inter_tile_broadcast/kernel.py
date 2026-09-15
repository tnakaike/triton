"""Whole-region delivery: the block equals a source region, so nothing is cut.

Two kernels, and the **block shape is the same in both** -- one whole region. What
separates them is the offset pattern across instances, which is the other of the two
knobs the surface has:

  inter_tile_broadcast     several instances pass the same offset -> replication
  inter_tile_permutation   every instance passes a distinct offset -> a bijection

So this file is one row of the taxonomy and the sibling directories are the others:
``inter_tile_gather`` (block spans several regions), ``inter_tile_scatter`` (block
inside one region), ``inter_tile_all_to_all`` (block spans all of them).
``inter_tile_reduce`` is the fold family instead of the copy family.

`work_slices` is a **dense positional list** in both — entry k is the region held by
tile k, so the position is the holder and there are no keys. This is the form the
current Triton support already takes, so it is the one to prototype against.

`kernel_map.py` carries the broadcast alone, with a map keyed by tile id. That form
is the one that can express holders which are not `0 .. N-1`, and the choice between
the two is list-only or map-only rather than both — see README.md.

DRAFT, written for discussion. It does not run and the suite does not discover it:
``tl.make_distributed_descriptor`` does not exist, ``tl.spyre_tensor_layout`` has no
memory-space argument yet, and there is no ``meta.py``, so ``conftest.py`` never
imports this file. See README.md for the fixtures and the assumptions.

Every size is written as a literal rather than a parameter, on purpose — the point
is to read the shape of the expression, not to sweep it. Real code generation would
inject these constants anyway.

  inter_tile_broadcast

  extents          {out: 512, x: 64}    src == dst
  source tile      {out: 64, x: 64}     8 tiles: 0..7 hold, 8..31 hold nothing
  source           region k on tile k
  destination tile {out: 64, x: 64}     32 tiles: region k replicated to tiles
                                        4k..4k+3, four holders each
  grid             [32]

  inter_tile_permutation

  extents          {j: 8, mb: 512, out: 128}    src == dst
  source tile      {j: 2, mb: 64, out: 128}     32 tiles: every tile holds one
  source           region (J, M) on tile 8J + M     J = j // 2, M = mb // 64
  destination tile {j: 2, mb: 64, out: 128}     32 tiles: the same region on
                                                tile J + 4M -- one holder each
  grid             [32]

"tile" is what one core holds; "region" is a distinct piece of the tensor. The
broadcast is where they diverge: 8 source regions with one holder each, but 32
destination tiles, because each region is replicated to four. So a tile count can
exceed a region count, and `work_slices` -- which pairs each region with its
holder -- describes the source side only. In the permutation they coincide on both
sides, and no region is replicated.
"""

import triton
import triton.language as tl

# work_slices: entry k is the region held by tile k. Dense from tile 0, one entry per
# region, and the position is the holder. Written out rather than comprehended, since
# the mapping is the thing under discussion.
SRC_WORK_SLICES = [
    {"out": 0}, {"out": 1}, {"out": 2}, {"out": 3},
    {"out": 4}, {"out": 5}, {"out": 6}, {"out": 7},
]


@triton.jit
def inter_tile_broadcast(x_ptr, out_ptr, SRC_SLICES: tl.constexpr):
    """Broadcast region ``pid // 4`` to tile ``pid``, composed over 8 source regions."""
    pid = tl.program_id(0)  # grid is [32]: 8 source regions, 4 holders each

    # --- my share, resident in my own scratchpad --------------------------------
    # A ct_local descriptor is per-tile shaped: 64 x 64, one region, not the 512 x 64
    # whole. The layout is the fixture's own -- logical [out, x], stick on x, stick
    # size 64 -- so physically [x // 64, out, x % 64].
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[64, 64], strides=[64, 1], block_shape=[64, 64],
    )
    tl.spyre_tensor_layout(
        x_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local"
    )

    # Only tiles 0..7 hold a source region. The others must still reach the
    # constructor with a value, and theirs is never read, since SRC_SLICES has
    # eight entries and so describes tiles 0..7 only.
    partial = tl.zeros([64, 64], dtype=tl.float16)
    if pid < 8:
        partial = x_desc.load([0, 0])         # [64, 64] fp16 -- one whole region

    # --- compose the 8 regions into the whole tensor ----------------------------
    # axes is positional over the composed tensor: dim 0 is divided along "out",
    # dim 1 is whole. len(axes) == rank(partial), so no axis is created -- the
    # compose grows dim 0 from 64 to 8 * 64 == 512.
    whole = tl.make_distributed_descriptor(
        partial,
        work_slices=SRC_SLICES,
        axes=["out", None],          # composed: [512, 64]
        block_shape=[64, 64],        # one region per load
    )

    # --- read the region I end up holding --------------------------------------
    # Four tiles share each value of g, and that is the entire expression of the
    # broadcast: replication is several instances passing the same offset, not
    # anything the descriptor holds. This load *is* the transfer -- scratchpad to
    # scratchpad, remote for every tile but 0.
    g = pid // 4
    mine = whole.load([g * 64, 0])            # [64, 64] fp16 -- one whole region

    # --- land it in my own scratchpad ------------------------------------------
    # Also ct_local and per-tile shaped, so every tile stores at local [0, 0] and
    # those are 32 distinct places. No guard: all 32 tiles hold a copy, and an
    # absent guard is how "every instance" is stated.
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[64, 64], strides=[64, 1], block_shape=[64, 64],
    )
    tl.spyre_tensor_layout(
        out_desc,
        [(1, "floordiv", 64), 0, (1, "mod", 64)],
        memory_space="ct_local",
    )
    out_desc.store([0, 0], mine)


# work_slices for the permutation: entry k is the region held by tile k. Two axes are
# divided -- j into four pairs, mb into eight blocks of 64 -- and the measured source
# owner is k = 8J + M, so tile k holds j-pair k // 8 and row block k % 8. Written out
# rather than comprehended, since the mapping is the thing under discussion.
PERM_WORK_SLICES = [
    {"j": 0, "mb": 0}, {"j": 0, "mb": 1}, {"j": 0, "mb": 2}, {"j": 0, "mb": 3},
    {"j": 0, "mb": 4}, {"j": 0, "mb": 5}, {"j": 0, "mb": 6}, {"j": 0, "mb": 7},
    {"j": 1, "mb": 0}, {"j": 1, "mb": 1}, {"j": 1, "mb": 2}, {"j": 1, "mb": 3},
    {"j": 1, "mb": 4}, {"j": 1, "mb": 5}, {"j": 1, "mb": 6}, {"j": 1, "mb": 7},
    {"j": 2, "mb": 0}, {"j": 2, "mb": 1}, {"j": 2, "mb": 2}, {"j": 2, "mb": 3},
    {"j": 2, "mb": 4}, {"j": 2, "mb": 5}, {"j": 2, "mb": 6}, {"j": 2, "mb": 7},
    {"j": 3, "mb": 0}, {"j": 3, "mb": 1}, {"j": 3, "mb": 2}, {"j": 3, "mb": 3},
    {"j": 3, "mb": 4}, {"j": 3, "mb": 5}, {"j": 3, "mb": 6}, {"j": 3, "mb": 7},
]


@triton.jit
def inter_tile_permutation(x_ptr, out_ptr, PERM_SLICES: tl.constexpr):
    """Relabel 32 whole regions: source owner 8J + M becomes destination owner J + 4M."""
    pid = tl.program_id(0)  # grid is [32]: 32 regions, one holder each on both sides

    # --- my share, resident in my own scratchpad --------------------------------
    # A ct_local descriptor is per-tile shaped: 2 x 64 x 128, one region, not the
    # 8 x 512 x 128 whole. The layout is logical [j, mb, out] with the stick on out
    # and stick size 64, so physically [out // 64, j, mb, out % 64] -- out is 128,
    # that is two sticks.
    x_desc = tl.make_tensor_descriptor(
        x_ptr, shape=[2, 64, 128], strides=[8192, 128, 1], block_shape=[2, 64, 128],
    )
    tl.spyre_tensor_layout(
        x_desc,
        [(2, "floordiv", 64), 0, 1, (2, "mod", 64)],
        memory_space="ct_local",
    )

    # No guard: all 32 tiles hold a source region, so all 32 load one.
    partial = x_desc.load([0, 0, 0])         # [2, 64, 128] fp16 -- one whole region

    # --- compose the 32 regions into the whole tensor ---------------------------
    # Two dims are divided, so both are named; out is whole, so it is None.
    # len(axes) == rank(partial), so no axis is created. The compose grows dim 0
    # from 2 to 4 * 2 == 8 and dim 1 from 64 to 8 * 64 == 512 -- each dim's index
    # count times the share's extent, which is why no extents are passed.
    whole = tl.make_distributed_descriptor(
        partial,
        work_slices=PERM_SLICES,
        axes=["j", "mb", None],          # composed: [8, 512, 128]
        block_shape=[2, 64, 128],        # exactly one region, as in the broadcast
    )

    # --- read the region I end up holding --------------------------------------
    # block_shape equals a region, so nothing is cut -- identical to the broadcast on
    # that knob. The difference is here: every instance passes a *distinct* offset,
    # so the 32 regions land one per tile instead of four tiles sharing each.
    #
    # I am destination owner m == pid, and the measured map is m = J + 4M, so I hold
    # the region with J = m % 4 and M = m // 4. That region's source owner is
    # 8J + M == 8 * (pid % 4) + pid // 4, which is a bijection of 0..31 -- so this is
    # a relabelling. It has exactly two fixed points, tiles 0 and 31 (8J + M == J + 4M
    # reduces to 7J == 3M), so 30 reads are remote and two are local.
    mine = whole.load([2 * (pid % 4), 64 * (pid // 4), 0])
    # [2, 64, 128] fp16 -- one whole region

    # --- land it in my own scratchpad ------------------------------------------
    # Also ct_local and per-tile shaped, so every tile stores at local [0, 0, 0] and
    # those are 32 distinct places. In the record the destination address equals the
    # source address, so the permutation moves no local offset at all -- only which
    # core holds which region.
    out_desc = tl.make_tensor_descriptor(
        out_ptr, shape=[2, 64, 128], strides=[8192, 128, 1], block_shape=[2, 64, 128],
    )
    tl.spyre_tensor_layout(
        out_desc,
        [(2, "floordiv", 64), 0, 1, (2, "mod", 64)],
        memory_space="ct_local",
    )
    out_desc.store([0, 0, 0], mine)
