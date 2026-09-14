"""Inter-tile broadcast: one tile's region read by four, via a distributed descriptor.

DRAFT, written for discussion. It does not run and the suite does not discover it:
``tl.make_distributed_descriptor`` does not exist, ``tl.spyre_tensor_layout`` has no
memory-space argument yet, and there is no ``meta.py``, so ``conftest.py`` never
imports this file. See README.md for the fixture and the assumptions.

Sibling of ``inter_tile_reduce``: the copy family rather than the fold family.

Every size is written as a literal rather than a parameter, on purpose — the point
is to read the shape of the expression, not to sweep it. Real code generation would
inject these constants anyway.

  extents        {out: 512, x: 64}     src == dst
  per-tile share {out: 64,  x: 64}     8 regions over 32 tiles
  source         region k on tile k       tiles 0..7 hold, 8..31 hold nothing
  destination    region k on tiles 4k..4k+3   four holders each
"""

import triton
import triton.language as tl

# work_slices: tile id -> the region that tile holds. One entry per region, and the
# key is the holder. Written out rather than comprehended, since the mapping is the
# thing under discussion.
SRC_WORK_SLICES = {
    0: {"out": 0}, 1: {"out": 1}, 2: {"out": 2}, 3: {"out": 3},
    4: {"out": 4}, 5: {"out": 5}, 6: {"out": 6}, 7: {"out": 7},
}


@triton.jit
def inter_tile_broadcast(x_ptr, out_ptr, SRC_SLICES: tl.constexpr):
    """Broadcast region ``pid // 4`` to tile ``pid``, composed over 8 source regions."""
    pid = tl.program_id(0)

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
        memory_space="ct_local",
    )

    # Only tiles 0..7 hold a source region. The others must still reach the
    # constructor with a value, and theirs is never read, since SRC_SLICES keys
    # only tiles 0..7.
    partial = tl.zeros([64, 64], dtype=tl.float16)
    if pid < 8:
        partial = x_desc.load([0, 0])

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
    mine = whole.load([g * 64, 0])

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
