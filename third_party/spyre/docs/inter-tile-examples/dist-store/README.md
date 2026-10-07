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

**There is no control flow in these files.** Every core runs the same two
transfers, and which of them is the slot's holder is carried by the `ct_id` in
the composed destination's type rather than by a guard -- see the parent README's
participation row. Two files have no tile id either (C010, C019: one piece, one
owner, so there is one slot and nothing to select).

**The composed destination is the whole of what its views hold.** Every
*(piece, owner)* pair is a slot of the result and the slots of one piece stack on
dimension 0, so C002 -- the whole 4096-element tensor on 16 cores -- composes to
`memref<65536x1x1xf16>`. [The parent README](../README.md) states the rule and
the other case it decides, C035, whose destination covers one row of 512.

## What composing the destination buys

**The file states its own movement.** A piece with several owners is that many
views with the **same** `coordinate_set` and a different `ct_id`, which is exactly
what replication is. C001's destination is one piece held by all 32 cores, and
here that is 32 views -- the breadth is in the IR rather than in the launch.

**It answers §4's objection rather than setting it aside.**
[§4](../../inter-tile-lowering-to-mem-view.md) calls a destination view
"meaningless where destinations are replicated", because it would have "two
writers for one coordinate with nothing saying which wins". Under the rule above
there are no two writers: the 16 copies of C002 are 16 slots, each written by its
own owner, and a reader of the IR can say which core wrote which bytes. The
objection holds against a composed destination typed as the coordinates once,
which is what an earlier revision of these files emitted.

**Ownership stays readable at the destination.** A store names the global box it
lands, so the IR says which coordinates a core holds; `if-store` moves that into
the load's base indices and leaves the landing anonymous.

**It takes no side on transport.** With both distributions named and no guard on
the store, *which* side does the transferring is a lowering's choice: pull, where
the slot's holder is the one that moves it, or push. A guard on "am I the holder"
would have chosen pull in the IR, which is `if-store`'s shape and is the reason
there is no `scf.if` here.

## What it costs

**A written distributed view implies remote writes.** §4's other objection
stands: cross-core *reads* are what the interconnect is described as supporting,
and storing through a composed destination asks for the unverified direction.
Nothing in this tree can yet say whether a lowering would -- though with the
slots distinct, a lowering is free to realize each store locally, which is
`if-store`'s shape arrived at by analysis rather than by spelling.

**It asks more of the backend.** Participation is derivable rather than stated,
so a lowering has to fold the index expression against the tile id and compare
the result with the slot's `ct_id`. `if-store` asks nothing: the guard is there
to read. These files are evidence for the first reading being enough, not proof
-- nothing in tree lowers either form yet.

**The index arithmetic is not saved.** An earlier revision of these files had no
tile id at all, because every core stored the whole tensor through a view typed
as the coordinates once. That was the thing the rule above rejects, so the saving
went with it: both spellings now derive the box from the tile id.

**A written distributed view implies remote writes.** §4's other objection is
that cross-core *reads* are what the interconnect is described as supporting while
remote writes are unverified. Storing through a composed destination asks for the
unverified direction, and nothing in this tree can yet say whether a lowering
would.

**It is verbose where replication is wide.** The destination side is one view per
*(piece, owner)* pair, so C008 has 32 source views and 32 destination views, and a
reader checks a 32-operand compose rather than an arithmetic expression.
