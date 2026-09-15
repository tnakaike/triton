# inter_tile_gather

Assembling delivery — the block spans several source regions, so a load gathers. Written against
the design in
[`inter-tile-lowering-to-mem-view.md`](../../../docs/inter-tile-lowering-to-mem-view.md).

**Two kernels in one `kernel.py`.** `inter_tile_gather` assembles onto a single holder;
`inter_tile_gather_replicated` assembles and then holds each result on four tiles. The second is the
largest class in the catalog and the one no other fixture reaches — see
[Why both kernels](#why-both-kernels).

**Status: draft, deliberately inert.** `tl.make_distributed_descriptor` does not exist, and there
is no `meta.py`, so `conftest.py::_load_examples` — which globs `fixtures/*/meta.py` — never
discovers this folder and never imports `kernel.py`. Nothing runs and nothing fails.

## What makes it a gather

Every copy fixture calls the same constructor over the same kind of `work_slices`. The class is
decided by **two independent knobs**, and by nothing else:

- **the block relative to a source region** — inside one, equal to one, spanning several, spanning
  all. This is `block_shape`, and it says whether a load splits, copies, or assembles.
- **the offsets across instances** — one instance, several sharing an offset, all distinct. This is
  the argument each instance passes to `load`, and it says how the result is held.

Their product is the taxonomy, and every named class is one cell:

| Block vs. a source region | one instance | several share an offset | all distinct |
|---|---|---|---|
| **inside one region** | — | — | `scatter` |
| **equal to a region** | — | `broadcast` | `permutation` |
| **spans several** | — | `gather_replicated` | — |
| **spans all** | `gather` | all-gather to every tile | `all_to_all` |

The `all_to_all` cell is worth reading carefully: its block relation is the **same** as `gather`'s.
Only the offset knob separates them. An earlier draft of this table claimed the all-to-all was
distinguished by being "smaller on another axis" — that is wrong, because the axis in question is
undivided on the source side, so a narrower block there is an ordinary partial read rather than a
distribution property.

This directory is the two "spans" rows. `inter_tile_gather`'s block is the whole composed tensor
with one instance loading; `inter_tile_gather_replicated`'s spans two regions with four instances
sharing each offset. The other rows are [`inter_tile_broadcast`](../inter_tile_broadcast/) (block
equals a region — broadcast and permutation), [`inter_tile_scatter`](../inter_tile_scatter/) and
[`inter_tile_all_to_all`](../inter_tile_all_to_all/).

## Fixture — `inter_tile_gather`

GR-PF-121, `relayouts[120]` of the ownership catalog pinned by torch-spyre#4300 — the KV-cache
scatter's input, route class `all_gather`.

```
extents          {mb: 8, out: 128}       src == dst
work division    mb 8 ways, out 2 ways   16 regions
per-tile share   {mb: 1, out: 64}        one stick
word length      2 (fp16)

Region (m, s) is mb[m : m+1] x out[64s : 64s+64].
```

| Side | Ownership |
|---|---|
| source | 16 regions, **one owner each** |
| destination | the whole 8 × 128 tensor, **core 0 alone** |

So `P = 16`: sixteen fragments per destination piece, one destination piece per source fragment.
Both axes grow — `mb` 1 → 8 and `out` 64 → 128, and 8 × 2 = 16 — which is why each entry of
`work_slices` names two dim ids rather than one.

The odometer order is `mb` slowest, `out` fastest, and it is visible in the table: entries 0 and 1
share `mb = 0` and differ in `out`. Reversing it would be a different assembly of the same regions
with the same shape and the same element count.

Fifteen of the sixteen reads are remote; the odd tiles are neither producer nor consumer.

## Deviation from the record — `inter_tile_gather`

**One, and it is the holders.** In the record the source owners are the **even** tiles —
`0, 2, 4 … 30`, with the odd tiles holding nothing. A dense positional list cannot say that: its
position *is* the holder, so sixteen entries claim `0 .. 15`. The fixture therefore places region
`l` on tile `l`, which is the measured geometry on a relabelled grid.

Everything the fixture is about survives the relabelling — the region count, the two-axis
division, `P = 16`, the single destination holder, the block spanning every region. What is lost is
the holder identity, and the map form of `work_slices`
([`inter_tile_broadcast/kernel_map.py`](../inter_tile_broadcast/kernel_map.py)) is exactly what
restores it: `{2 * l: {...}}` instead of a list.

**One open item, not a deviation.** The fixture sticks `out`, a real 64-element axis. This record's
own layout fields were not read; the KT transcription of a sibling record has an extent-1 `y`
sticked instead, which would make a share mostly padding. Which of the two the fixture should carry
is open. It does not change what the fixture is about — the compose sees logical extents.

## Why both kernels

`inter_tile_gather` uses one knob and `inter_tile_gather_replicated` uses both, and that difference
is the largest coverage gap in the fixture set:

| | assembles | replicates | route class | records |
|---|:---:|:---:|---|---:|
| `inter_tile_broadcast` | no | yes | replication of whole regions | part of 32 |
| `inter_tile_gather` | yes | no | `all_gather` | 26 |
| `inter_tile_gather_replicated` | **yes** | **yes** | `grouped_all_gather_with_replication` | **65** |

Half the catalog needs both at once, and neither of the other two shows it. Written together the
combination is legible as a product rather than as a third thing: `block_shape` spanning two regions
is the gather, four instances sharing `g = pid // 4` is the replication, and they compose without
interacting.

## Fixture — `inter_tile_gather_replicated`

GR-PF-002, `relayouts[1]` of the same catalog, route class
**`grouped_all_gather_with_replication`**. The tensor is `mean-LayerNormNorm_out`, input 0 of
`mm-BMM_1`.

```
extents          {mb: 512, in: 4096}      src == dst
work division    mb 16 ways on the source, 8 ways on the destination
source tile      {mb: 32, in: 4096}       16 regions
destination tile {mb: 64, in: 4096}       8 assembled regions, four holders each
word length      2 (fp16)

Source region l is mb[32l : 32l+32] x in[0:4096].
```

| Side | Ownership |
|---|---|
| source | 16 regions, **one owner each** |
| destination | 8 regions of `{mb: 64}`, **four owners each** — group `g` is tiles `4g .. 4g+3` |

So `P = 2` and fan-out is 4: each group assembles two source regions and four tiles hold the
result. `remote_destination_bytes` is 12 MiB.

**The same relayout is recorded three times** — `relayouts[1]`, `[8]`, `[16]`, for consumers
`mm-BMM_1`, `mm_1-BMM_1`, `mm_2-BMM_1`. One layer-norm output feeding Q, K and V projections; the
three records are identical in extents, pieces, owners, route class and fragment counts. The
catalog is indexed by `(consumer, input)`, so a reader that walks records one at a time emits three
deliveries and moves 36 MiB where 12 MiB suffices — **exactly 3×**. Eight of the catalog's 120
distinct tensors are shared this way. Nothing in this fixture prevents that mistake; it is a
reason the *lowering* must key on the tensor rather than on the record.

**Deviation from the record:** the same holder relabelling as `inter_tile_gather` — measured source
owners are the **even** tiles (`{4g, 4g+2}` per group, so `0, 2 … 30` overall), which a dense
positional list cannot express, so region `l` sits on tile `l`. This is also the record whose
non-adjacent owners are why the KTIR spec defines a producer's position `l` as a position rather
than a tile id.

One sizing point arrives with execution: a source region is 32 × 4096 fp16 = 256 KiB and the
assembled block is 512 KiB, both as whole Triton blocks. A scaled-down variant may be the one that
runs first.

## Operation sequence

```
Load (own scratchpad, guarded pid < 16) -> Compose (16 regions) -> Load (distributed, guarded pid == 0) -> Store
```

Note where the two guards are. The compose sits **outside** both, because it is a collective
constructor: its operand is a per-instance value, so every instance has to reach it — including the
sixteen that hold nothing. Only the distributed load belongs to the destination set, and the guard
on it is the whole statement of that set.

## Target KTIR

What the lowering should produce, and the shape a `test/Conversion` lit test would pin once the
lowering exists:

- sixteen `ktdp.construct_memory_view`, differing **only** in `coordinate_set` and `ct_id = 0..15`
  — those two attributes are the entire content of the distribution
- one `ktdp.construct_distributed_memory_view` composing them → `memref<8x128xf16>`
- `construct_access_tile %src[%c0, %c0]` with an 8 × 128 tile — the whole domain
- one `ktdp.load` inside an `scf.if` on tile id 0. **Sixteen remote reads collapse into one op**;
  fan-in does not multiply it, just as fan-out did not in the broadcast
- `ktdp.store` into a plain `ct_local` view with **no** `ct_id`, meaning the executing tile

The asymmetry is the same as the broadcast's, with the roles swapped. Source ownership is in the
`ct_id`s; the destination *holder* is the `scf.if`; the destination *region* is the whole tensor, so
the access tile's offsets are zero. Neither the store nor the load alone gives the map.

## Assumptions

Each is settled elsewhere; they are recorded here as what the example rests on rather than as
things it decides.

**1. `work_slices` enumerates the source regions and names their holders.** One entry per region —
sixteen for a 32-tile grid, so dense over the tensor and sparse over the grid. The entry count
gives the composed domain, since no extents are passed: eight `mb` indices × share extent 1, two
`out` indices × share extent 64. §1 of the design document, #153.

**2. There is no destination table.** The destination *set* is stated by a guard — `pid == 0` here
— and the *arrangement* is the offset each instance passes, `[0, 0]`. The set and the offset do not reach
the compose; the destination *extent* does, as `block_shape` — so the destination side is three
facts, of which exactly one is an operand. Because it is a single constexpr rather than a
per-instance value, **every destination tile must have the same shape**: uniform cardinality holds
structurally here instead of needing a verifier, at the cost of making unequal chunk sizes
inexpressible rather than merely unchecked. Nothing on this side is verified at the Triton level. §2 by #153.

**3. A memory-space attribute exists on `tl.spyre_tensor_layout`.** It does not yet — the one
assumption the example is ahead of. The relayout is scratchpad on both sides, and expressing that
directly is what makes the shape discussable. Tracked in #137.

## What it exercises, and what it cannot check yet

Exercises: a block **larger than a region**, so one load draws from sixteen sources; a two-axis
division, so each entry names two dim ids and the odometer order is observable; a destination set of
exactly one, stated by a guard; a collective constructor reached by instances that hold nothing; and
`ct_local` descriptors on both sides so the transfer is scratchpad to scratchpad throughout.

Cannot check: anything numerical, or that the lowering produces the target KTIR. There is no
lowering, and the KTDP view has no producer. The first provable milestone is a lit round-trip on the
emitted TTIR, once `tl.make_distributed_descriptor` exists.

## When `meta.py` is added

It needs `SIGNATURE`, `VARIANTS` with `params` and `constexpr`, a NumPy `reference` oracle and an
`inputs` generator — all of which presuppose execution, which is why it is omitted now. Add
`__init__.py` at the same time; `_import_meta` imports the folder as a package so `from . import
kernel` resolves. Setting `grid` to `[32]` matches the fixture.

The kernel also gains a prologue and an epilogue, since a runnable test has to supply its input from
the host and let the host read the result. The prologue is the producer this example assumes: load
from a global descriptor into a tensor, then store that tensor to the `ct_local` descriptor, and the
share is resident. The epilogue reverses it on core 0. The relayout between them — compose, then
`whole.load` — is unchanged, so the two files diverge only at the boundaries.

That does not remove the dependency on #137: the staging store needs a `ct_local` descriptor too.
