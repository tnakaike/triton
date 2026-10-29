# dist-store: both sides composed

One of two spellings of the same 35 movements. The other is
[`../if-store/`](../if-store/), and [the parent README](../README.md) carries the
provenance, the SDSC mapping and the table of 35. **Neither spelling is decided.**

## What one file contains

| | |
|---|---|
| the source distribution | one `ktdp.construct_memory_view` per source piece, composed by `ktdp.construct_distributed_memory_view` |
| the destination distribution | the same, one view per destination *(piece, owner)* pair, composed the same way |
| the movement | `ktdp.get_compute_tile_id` and the index arithmetic for this core's box, then one `ktdp.construct_access_tile` + `ktdp.load` on the source and one tile + `ktdp.store` into this core's slot of the destination |
| participation | an `scf.if` on the tile id around the load and the store, in the 10 files where not every core owns a slot |

**Participation is an `scf.if`, as in `if-store`.** The function is SPMD, so a
store outside a guard runs on every core. A core that owns no slot would compute
an index into another core's slot and store there: a remote write. The guard
keeps it from doing so, and the load goes inside with the store, since the load
is the transfer. Checked over all 35 and every core: the cores the guard admits
are exactly the slot owners, and each stores into its own slot only.

**The composed destination is a global view of disjoint slots.** Every
*(piece, owner)* pair is a slot of the result and the slots of one piece stack on
physical dimension 0, so C002 -- the whole tensor, physically `[1, 64, 1, 64]`,
on 16 cores -- composes to `memref<16x64x1x64xf16>`, its 16 views covering
`y = 0`, `y = 1`, ...
[The parent README](../README.md) states the rule and the other case it decides,
C035, whose destination covers one row of 512.

## What composing the destination buys

**The file states its own movement.** A piece with several owners is that many
views, one slot each, which is exactly what replication is. C001's destination is
one piece held by all 32 cores, and here that is 32 views -- the breadth is in the
IR rather than in the launch.

**No coordinate has two writers.** The 16 copies of C002 are 16 slots, each
written by its own owner, and a reader of the IR can say which core wrote which
bytes.

**Ownership stays readable at the destination.** A store names the slot it
lands, so the IR says which coordinates a core holds; `if-store` moves that into
the load's base indices and leaves the landing anonymous.

## What it costs

**It is verbose where replication is wide.** The destination side is one view per
*(piece, owner)* pair, so C008 has 32 source views and 32 destination views, and a
reader checks a 32-operand compose rather than an arithmetic expression.

**The index arithmetic is not saved.** Both spellings derive the box from the
tile id, and this one adds the replica index on top.
