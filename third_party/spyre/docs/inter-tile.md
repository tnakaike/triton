# Expression of Inter-Tile Communications and Lowering to Distributed Memory Views

`ktdp.construct_distributed_memory_view` is an existing KTIR feature. It composes
per-core partitions into a single memref whose domain is the union of their coordinate
sets, and its lowering is expected to resolve a global index into "selection of the
appropriate underlying memref and a local coordinate computation, potentially producing
explicit address calculation **and communication** when required by the target
architecture."

This document is about the two things that do not exist yet: **how a Triton kernel
expresses an inter-tile communication** (§2), and **how that expression lowers onto the
view** (§3). The view itself is the target, not the subject — where the design would
change it, §8 says so.

The communication in question is a **scratchpad relayout**: a tensor moving between
two ownership arrangements while it stays resident in the scratchpad. Both sides are
scratchpad-allocated and the
two work divisions differ — a producer and a consumer disagree about which core holds
which region, and the data has to be redistributed without a round trip through global
memory. That is why the kernel body needs scratchpad descriptors even when its entry inputs do
not.

Scope is the copy family: gather, scatter, all-to-all, relocation and broadcast.
Cross-core reduction is tracked separately in #20; §7 states the boundary and where it
may dissolve.

State of play: the view has **no producer and no lowering** today — every reference to it
is the dialect definition, the README, and Dialect round-trip tests. That makes it cheap
to change, which is what §8's second decision turns on.

## 1. Definitions and assumptions

A **partition** is one core's share of a tensor: a coordinate region, plus the core
whose scratchpad holds it. In KTDP it is a `ktdp.construct_memory_view` carrying a
`coordinate_set` — the region in the tensor's global index space — and a `memory_space`
naming the holder.

```mlir
%p3 = ktdp.construct_memory_view %off, sizes: [64, 64], strides: [64, 1] {
    coordinate_set = #region_3,
    memory_space   = #ktdp.memory_space<ct_local, ct_id = 3>
} : memref<64x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
```

Two things about that type are load-bearing. The `ct_id` is part of it, so two
partitions in different cores' scratchpads have *different types* — that difference is the only
thing distinguishing otherwise identical operands. And the shape is static, so
partitions of differing shape cannot share one SSA value.

A **distributed view** composes partitions into a single memref whose domain is the
union of their coordinate sets. It moves nothing; it establishes a map from a global
coordinate to the partition holding it, plus the local coordinate within that partition —
the behaviour quoted at the top of this document.

A **work slice table** is a list with one element per tile, indexed by tile id. Each
element is a dict holding that tile's grid coordinate: one key per dimension the work was
divided along, mapping to that tile's slice index on it. Every element carries the same
keys. It is what `tl.inter_tile` already accepts as `work_slices`.

The slice count along a dimension is not declared alongside the table; it is one more than
the largest index appearing for that key anywhere in the list. That count is what turns a
coordinate into a region.

Eight tiles, work divided four ways on `x` and two ways on `n`:

```python
work_slices = [{"x": t // 2, "n": t % 2} for t in range(8)]

# tile 0 -> {"x": 0, "n": 0}     tile 4 -> {"x": 2, "n": 0}
# tile 1 -> {"x": 0, "n": 1}     tile 5 -> {"x": 2, "n": 1}
# tile 2 -> {"x": 1, "n": 0}     tile 6 -> {"x": 3, "n": 0}
# tile 3 -> {"x": 1, "n": 1}     tile 7 -> {"x": 3, "n": 1}
```

`x` has four slices and `n` has two — one more than the largest index under each key, and
neither count written down. With extents `{x: 512, n: 128}` the slices are 128 and 64 wide,
so tile 5, at `{x: 2, n: 1}`, owns `x[256:384]` by `n[64:128]`. That projection is all the
lowering needs from the table, and §3 does exactly it.

A redistribution needs two such tables, one per side, because it is precisely the
difference between them.

The **stick** is the hardware's contiguous innermost unit, 128 bytes, so `S = 128 /
itemsize` — 64 for fp16. Written `S` throughout, as in
[spyre-tensor-layouts.md](spyre-tensor-layouts.md).

### Assumptions this design relies on

1. **A kernel is one fused unit and its entry inputs live in global memory.** This is a
   property of where a fused region can begin. It says nothing about the kernel *body*,
   which is where a relayout appears — see §2.
2. **The lowering is pull.** §4.
3. **A source view's regions are disjoint or identical.** A slice table cannot express a
   partial overlap; identical means the holders are replicas. §5.
4. **There is no destination view.** Each destination holder writes into its own
   scratchpad, so nothing on that side is composed. §4.
5. **A reduction expressed over a distributed view keeps its order-freedom.** §7, and
   the one assumption still under discussion.
6. **Scratchpad offsets reach the kernel as metadata**, not as arguments and not as
   constants: a distributed view can involve up to 32 distinct indices, so an argument per
   index is impractical.

## 2. What the kernel provides

Three pieces, and the split is what keeps the Triton surface per-instance.

| Piece | Where it lives | What it carries |
|---|---|---|
| partition **identity** | `work_slices`, one table per side | which core owns which slice |
| memory-space **kind** | `tl.spyre_tensor_layout` | `global` \| `ct_local` |
| the N **views** | the lowering | `construct_memory_view` + `ct_id`, then the compose |

The kernel body needs **no** `ct_id` and no knowledge of the grid. Holder identities are
derived during lowering from the slice tables, which is what keeps the global vantage out
of a kernel that only knows its own `program_id`.

`tl.inter_tile` takes one slice table today. A copy pattern needs **two**, because source
and destination ownership differ — that difference is the relayout.

### Memory space

`tl.spyre_tensor_layout` gains a memory-space attribute. The op is already a result-less
marker "consumed and erased by the lowering", carrying three parallel `DenseI64ArrayAttr`s
for the physical layout, so a memory space is one more attribute on an op whose job is
already to carry device-side facts about a descriptor:

```
let arguments = (ins
  TT_TensorDescType:$desc,
  DenseI64ArrayAttr:$phys_src,
  DenseI64ArrayAttr:$phys_op,
  DenseI64ArrayAttr:$phys_arg,
  <enum>:$memory_space          // new
);
```

The vocabulary is `global` and `ct_local`, matching `Ktdp_MemorySpaceKind`, which already
has exactly those two kinds plus an optional `ct_id`. It does **not** go on
`make_tensor_descriptor`, which is upstream generic.

This is needed from the start, not deferred. A relayout is scratchpad on both sides by
definition, so a kernel expressing one needs scratchpad descriptors immediately; assumption 1
constrains the signature, not the body.

## 3. Phases

The lowering runs in three phases, stated as a contract per phase. Its inputs are the two
slice tables, the tensor's extents, and the memory-space attribute; its output is the
redistributed data resident in each consumer's own scratchpad.

**Phase 1 — build the source view.**
Input: the source-side slice table, the extents, and the memory space.
Lowering: turn each tile's coordinate into the region that tile owns, exactly as the §1
example does — slice width is the extent divided by the slice count, and the coordinate
picks which slice. Emit one `construct_memory_view` per tile with that region as
`coordinate_set` and the tile as `ct_id`, then compose them with
`construct_distributed_memory_view`.
Output: one distributed view whose domain is the whole tensor, and which knows for every
coordinate which tile holds it.

The emitted views differ **only** in `coordinate_set` and `ct_id`; offsets, sizes and
strides are identical across them. Those two attributes are the entire content of the
distribution.

**Phase 2 — place the access.**
Input: the distributed view, and the destination-side slice table.
Lowering: project the executing tile's destination coordinate the same way, giving the
region this tile needs, and build a `construct_access_tile` on the view anchored there.
Output: an access tile naming the coordinates this tile will consume — and nothing has
moved yet. `construct_memory_view` and `construct_access_tile` are `Pure`: they
materialize addressing, not access, so a core may name another core's scratchpad without
anything crossing a core boundary.

**Phase 3 — transfer and land.**
Input: the access tile.
Lowering: one `ktdp.load`, which **is** the transfer. Where the access tile's region spans
several partitions the lowering resolves it into per-partition transfers, so fan-in does
not multiply the load count — it stays one load. Then a `ktdp.store` into a plain
`ct_local` view with no `ct_id`, meaning the executing tile's own scratchpad.
Output: the data in place for the next operation to read locally.

Worked example for phase 1. Eight tiles, `out` divided eight ways, `x` uncut:

```
source slice table    [{"out": 0}, {"out": 1}, ..., {"out": 7}]
extents               {out: 512, x: 64}
slice width           512 / 8 = 64 on out; x is whole
=> tile 3 owns        out[192:256] by x[0:64]
```

### Where the phases live

`LowerInterTile` gains a second mode. The existing path — `tt.inter_tile_reduce` to
`ktdp.inter_tile_produce` plus a delivery op — is not removed.

## 4. The pull model

A distributed view is composed over the **source** partitions. Consumers read; producers
do not write remotely.

This is not merely a convention. Under pull a destination holder writes only into its own
scratchpad, so the destination side needs no distributed view and nothing has to be
composed there. One consequence worth naming: replication needs no expression of its own,
because a region needed by several consumers is just several reads, and reads do not
conflict.

A destination view remains *possible* where destination regions are disjoint, which would
make a relayout a view-to-view copy. It is never *necessary*, and it is impossible where
destinations are replicated — a view that is written cannot have two writers for one
coordinate, whereas a view that is only read tolerates replicas because either holder
returns the same bytes.

Push would need a core to write into another core's scratchpad. Cross-core **reads** are
what the interconnect is described as supporting; remote writes are unverified. If they
exist, pull versus push becomes a performance question rather than an expressiveness one.

## 5. Several holders, and why that is welcome

A region may be held by more than one tile. That is expressible, it occurs, and it should
**not** be forbidden — it is information the backend can use.

**What the input format can and cannot produce.** Each tile gets one slice index per
divided dimension, so its region is one box. For any two tiles the boxes are therefore
either **identical** — every index equal, which is replication — or **disjoint**, because
differing slices do not intersect. There is no third case: a partially overlapping pair
would need a tile to own something other than a whole slice, or to own two of them, and a
slice table can express neither.

That is a stronger guarantee than a measurement. The dangerous shape is partial overlap,
where two holders share only some coordinates and could disagree about those. A slice
table **cannot describe it**, so the lowering cannot emit it, whatever future schedules
look like. The only overlap reachable is exact replication.

**Replication is a scheduling asset.** With a region held by two tiles and two consumers
needing it, the backend can pair them off:

```
region held by tiles 0 and 1;  tiles 2 and 3 both need it

  one holder only        free choice
  2 <- 0                 2 <- 0   ┐ two different source ports,
  3 <- 0   serialised    3 <- 1   ┘ transfers in parallel
```

So source selection should stay the backend's, and a canonical rule — "always read from
the lowest tile id" — would be actively worse than no rule: it funnels every reader onto
one holder and discards the parallelism the replication provides.

Note this is the opposite answer from `reduce_to_one`'s result owner (§8, decision 1), and
for the opposite reason. A reader has several valid holders and the choice is worth
keeping; a fold's result has none, so it has to be stipulated.

**What replication does not carry.** A slice table records ownership and cannot assert
*agreement*. Replicas and copies that have diverged have byte-identical declarations, so
"read from any holder" is sound exactly while the holders agree — which holds when a
region is written once by its holders and thereafter only read. It would stop holding if
something mutated one copy in place.

For reference, in the catalog pinned by torch-spyre#4300 every coordinate has exactly one
holder on the read side: no source region lists more than one owner, and no two distinct
regions overlap. Replication appears on the destination side — 97 of 130 records, up to 32
holders — where each holder writes and reads its own copy locally.

## 6. Why the slice tables stay tables

What the measured patterns rule out is deriving ownership from **axis counts** — an axis
name and a slice count, which is the form `tl.inter_tile`'s `axis` parameter has. Ownership
is frequently strided rather than contiguous. In the measured records the cores feeding one
destination region sit two apart (cores 0 and 2, then 4 and 6, and so on), or eight apart
(0, 8, 16 and 24), or are drawn only from the even-numbered cores. No axis count reproduces
those, because the core-to-region assignment is not a function of how many pieces an axis
was cut into.

A closed form *does* reach them — with `mod` and `floordiv`, affine sets express strides of
that kind directly, and a survey finds a closed form for the consumer-to-source map in all
130 records. The table is kept anyway for two reasons: it is **always** expressible, so an
irregular pattern that no closed form covers still has somewhere to go; and it needs **no
inference**, where recovering a formula from 32 elements can fail, or over-generalise
silently into one that is right for the cases inspected and wrong for one that was not.

A table is also enough for what the lowering needs, which is one *concrete*
`coordinate_set` per partition — a box — not a parameterized family.

## 7. The boundary with cross-core reduction

#20 tracks cross-core reduction. The division is by whether data is **combined on the way**
or only **moved**: a fold (`combine = fold`) versus bytes arriving as they left
(`combine = none`).

**The surface commits to no fold order.** Naming an operation to fold with says *what* to
compute, not in what sequence, so choosing the schedule is the backend's.

**Whether that survives lowering** has two parts. Order-freedom is assumed to carry over
(assumption 5), conditional on the lowering emitting a reduction op over the distributed
axis rather than a written-out accumulation — see §8, decision 3. The **result location**
does not carry over: ownership records who holds each input, never who holds the output, so
`reduce_to_one`'s final holder must be assumed.

**Where the boundary may dissolve.** `reduce_scatter` can be expressed as a reduction whose
input comes from the distributed view: core `j` reduces over the axis spanning the `P`
partials of region `j`, with `linalg.generic` + `linalg.addf`. Nothing says the gathered
values are materialized first, so the backend keeps the choice of how to realize the
combination, and no inter-tile delivery op appears. If `reduce_to_one` also gets a rule,
#20's surface can lower through this machinery rather than needing its own.

Note that reductions are absent from the pinned catalog only because it records scratchpad
relayouts specifically. They exist in the workload: an OpSpec can carry a matmul or
reduction that splits the reduction dimension.

## 8. Open decisions

1. **`reduce_to_one`'s result owner.** Proposed: the smallest `ct_id`, as a **language
   rule**. A distributed view cannot otherwise pin the final holder — it can only be
   inferred backwards from a following store, and for an intermediate not at all. As a
   backend convention it would be worse than useless: a kernel tuned around "core 0 holds
   it" would expire silently when the lowering changed, with no diff to review.
2. **Whether the N partition operands can be replaced.** Enumerating every share is the
   cost that stands out. Two candidates. An **ownership relation** — an affine set relating
   coordinates to holding tiles — reaches the measured patterns, keeps holder and shape in
   the type, and stays statically checkable, at the cost of new attribute machinery. A
   **dynamic `ct_id`** is a far smaller change, but `ct_id` is part of the memref type, so
   partitions of different cores would share one type and the operand list would stop
   carrying ownership. Neither changes the Triton surface.
3. **May a kernel specify the reduction order?** Proposed: **no**. Then the lowering must
   emit the order-free form and must not emit a written-out accumulation:
   `for k: acc += load(dview[k])` fixes a left-to-right fold, and since fp16 addition is
   not associative the backend can no longer legally re-associate it. This is easy to get
   wrong by accident — an accumulation loop is the obvious way to write "read from all
   partitions and sum" — and the mistake is silent, correct results at permanently worse
   traffic. One case "no" does not cover: order-freedom is associativity *plus*
   commutativity, and while `add` / `max` / `mul` have both, a custom reducer region need
   not, and **argmax** is associative while its result depends on traversal order on ties.

## 9. Designs rejected

**Exposing a distributed-view interface in Triton.** The op takes memref *values*, so all
N partitions must exist as SSA values in one function — a global vantage forced by the
operand list, which collides with SPMD. Keeping the enumeration inside the lowering puts
that vantage where the grid knowledge already is.

**Emulating scratchpad-to-scratchpad through global memory.** Functionally it works, and the
descriptor-layout machinery would carry the form change. But if every intermediate goes
through global memory there is nothing left for a relayout to do — that round trip is
exactly what this exists to remove. It would pass expression and numerical checks while
failing on emitted accesses, with every number green.

**Deriving ownership from an axis and a count.** §6.

## 10. Rejected inputs

The lowering should refuse, rather than guess:

- **A slice table whose length does not match the launch grid.** `prod(grid) ==
  len(work_slices)` is a launch obligation the kernel body cannot enforce.
- **Slice tables with differing key sets** between the two sides, or within one table —
  every element of a `work_slices` list must have identical keys.
- **A `ct_id` on a `global` memory space.** `ct_id` is meaningful only for `ct_local`.
- **A written-out accumulation over a distributed view**, if decision 3 resolves to "no".
  Emitting it silently forfeits the backend's schedule freedom, so it should be a
  diagnostic rather than a quiet pessimization.
