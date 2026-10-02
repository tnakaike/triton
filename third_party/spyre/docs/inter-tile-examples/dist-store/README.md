# dist-store: both sides composed

One of two spellings of the same 35 movements. The other is
[`../if-store/`](../if-store/), and [the parent README](../README.md) carries the
provenance, the SDSC mapping and the table of 35. **Neither spelling is decided.**

## What one file contains

| | |
|---|---|
| the source distribution | one `ktdp.construct_memory_view` per source piece, composed by `ktdp.construct_distributed_memory_view` |
| the destination distribution | the same, one view per destination *(piece, owner)* pair, composed the same way |
| the movement | one `ktdp.construct_access_tile` + `ktdp.load` on the source, one tile + `ktdp.store` on the destination, whole-tensor on both sides |

There is no tile id in these files and no control flow. Every core runs the same
two ops, and what varies between cores is carried entirely by the `ct_id`s in the
two composed types.

## What composing the destination buys

**The file states its own movement.** A piece with several owners is that many
views with the **same** `coordinate_set` and a different `ct_id`, which is exactly
what replication is. C001's destination is one piece held by all 32 cores, and
here that is 32 views -- the breadth is in the IR rather than in the launch.

**It takes no side on transport.** With both distributions named, *which* side
does the transferring is a lowering's choice: pull, where each destination holder
reads its share, or push. `if-store` has pull in its shape.

**The movement is two ops regardless of geometry.** A 32-to-32 permutation reads
the same as a 1-to-1 remap; nothing about the owner map reaches the movement.

## What it costs

**It diverges from the design.**
[§4](../../inter-tile-lowering-to-mem-view.md) composes only the source and calls
a destination view "meaningless where destinations are replicated", on the grounds
that it would have "two writers for one coordinate with nothing saying which
wins". That argument does not hold for a broadcast, where the writers agree and
every holder receives the same bytes -- but it is unanswered in general, and the
op's own documentation leaves overlapping coordinate sets "unspecified unless ...
constrained by ... program semantics". The semantics is here a comment, not a
check.

**A written distributed view implies remote writes.** §4's other objection is
that cross-core *reads* are what the interconnect is described as supporting while
remote writes are unverified. Storing through a composed destination asks for the
unverified direction, and nothing in this tree can yet say whether a lowering
would.

**It is verbose where replication is wide.** The destination side is one view per
*(piece, owner)* pair, so C008 has 32 source views and 32 destination views, and a
reader checks a 32-operand compose rather than an arithmetic expression.
