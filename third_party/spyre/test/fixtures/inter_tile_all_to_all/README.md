# inter_tile_all_to_all

Exchanging delivery — every tile sends a distinct chunk to every tile. Written against the design
in [`inter-tile-lowering-to-mem-view.md`](../../../docs/inter-tile-lowering-to-mem-view.md).

**Read the next section before the fixture.** Unlike every sibling, this one is not a transcription
of a catalogued record, because the catalog contains no all-to-all.

## The catalog has no all-to-all, and that is the finding

The 130 records pinned by torch-spyre#4300 fall into five route classes, and **not one of them is a
strict all-to-all**:

| Route class | Records | What it is |
|---|---:|---|
| `grouped_all_gather_with_replication` | 65 | gather, then several holders per result |
| `replicate_or_owner_remap` | 32 | copy or relabel, whole regions |
| `all_gather` | 26 | gather, one or few holders |
| `permutation` | 6 | a bijection of **whole** regions |
| `general_relayout` | 1 | not transcribed; the only possible candidate |

The nearest measured thing is `permutation` — GR-PF-052, `relayouts[51]`, a transpose of a 4 × 8
index grid where source owner `k = 8J + M` and destination owner `m = J + 4M`. Every tile sends and
every tile receives, and the ownership genuinely moves, so SDSC's capability table calls this shape a
"uniform one-to-one all-to-all shuffle". But each tile sends its region **whole, to exactly one
partner**. That is a permutation, and it is already covered by the broadcast's expression with a
bijective offset — nothing new is exercised.

A strict all-to-all is the case where each *pair* exchanges a *sub-chunk*: source and destination
divide **different** axes, so no chunk is whole on either side. That is the class the catalog does
not exercise, and it is also the class the KTIR delivery ops cannot express — `scatter` rejects more
than one producer per group, `gather` and `consume` move whole partials, so the only route today is
an N-way decomposition. The gap is recorded independently of this fixture.

**So this fixture is measured in its extents and synthesized in its routing**, and the split is
exactly:

- **measured** — the tensor and its extents, `{mb: 512, out: 4096}` fp16, `src == dst`: that is
  `mean_80-LayerNormNorm_out`, the tensor of GR-PF-055, and the 32-tile grid
- **synthesized** — the division on each side. Row-sharded to column-sharded is the canonical
  all-to-all and is what SDSC lists as v1-supported, but no record in the catalog has it.

Whether that is worth having as a fixture is a question for review. The argument for: it is the one
class the surface has not been shown to express, and the four fixtures together are the coverage
claim. The argument against: a fixture with no record behind it cannot be checked against measured
data, only against the design.

## What makes it an all-to-all

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

Here the block is `[512, 128]` against a region of `[16, 4096]`. It is the *full* extent of `mb`, so
it spans all 32 regions — the same block relation the gather has. What makes it an exchange rather
than a gather is the offset: every instance passes a distinct `[0, 128 * pid]`, so each takes a
different 128-column slice of every region and no chunk is shared. The narrowness on `out` is what
carries the destination division, but `out` is undivided on the source side, so it is not a
block-versus-region property.

## Fixture

```
extents          {mb: 512, out: 4096}      src == dst
source division  mb 32 ways, out uncut     32 regions of {mb: 16, out: 4096}
dest division    out 32 ways, mb uncut     32 regions of {mb: 512, out: 128}
word length      2 (fp16)

Source region k is mb[16k : 16k + 16] x out[0 : 4096], on tile k.
Destination region d is mb[0 : 512] x out[128d : 128d + 128], on tile d.
```

| Side | Ownership |
|---|---|
| source | 32 row shards, **one owner each**, holders `0 .. 31` |
| destination | 32 column shards, **one owner each**, holders `0 .. 31` |

Every (source, destination) pair exchanges one `[16, 128]` chunk, so the exchange is
32 × 32 = **1024 chunks**, none of them whole and none of them shared. One of each tile's 32 chunks
is local; 31 are remote.

Only the source division reaches `work_slices`. The destination division is not declared anywhere —
it is the offset `[0, 128 * pid]` that each instance passes, and the shard width is `block_shape`.

## Deviation from the record

**The routing, as above.** The extents and grid are GR-PF-055's; both divisions are synthesized.
Holders are `0 .. 31` on both sides, so nothing is lost to the dense positional list.

**The layout, and it is measured rather than open.** A real SDSC run of this exact logical shape —
512 × 4096 × 1, fp16 — reports `layoutDimOrder_ = ["mb", "out", "y"]`, `stickDimOrder_ = ["y"]`,
`stickSize_ = 64` and `device_size = [1, 4096, 512, 64]`, so the sticked axis is the extent-1 `y` at
64 padded lanes and the physical form is `[1, out, mb, 64]`. The fixture sticks `out` and drops `y`,
which is a definite deviation. The compose sees logical extents, so it does not change what the
fixture is about.

## Operation sequence

```
Load (own scratchpad) -> Compose (32 regions) -> Load (distributed) -> Store
```

No guard on either side: every tile holds a row shard and every tile holds a column shard, and an
absent guard is how "every instance" is stated. Structurally identical to the scatter's sequence —
the two differ only in the block shape and the offset, which is the point being made.

## Target KTIR

What the lowering should produce, and the shape a `test/Conversion` lit test would pin once the
lowering exists:

- thirty-two `ktdp.construct_memory_view`, differing **only** in `coordinate_set` and
  `ct_id = 0..31` — those two attributes are the entire content of the distribution
- one `ktdp.construct_distributed_memory_view` composing them → `memref<512x4096xf16>`
- `%col = tid * 128`, then `construct_access_tile %src[%c0, %col]` with a 512 × 128 tile
- one `ktdp.load` — this **is** the transfer, and it is the case where a single access tile spans
  every composed view. Whether the backend emits 32 point-to-point reads, a ring, or a staged
  transpose is a route decision the IR leaves free
- `ktdp.store` into a plain `ct_local` view with **no** `ct_id`, meaning the executing tile

This is the fixture where the "one `ktdp.load` per transfer" property is doing the most work: 1024
chunks, one op. It is also where the freedom that buys is largest, and where a wrong route costs the
most — an all-to-all routed as 32 sequential gathers is 32 rounds where a shuffle is one.

## Assumptions

Each is settled elsewhere; they are recorded here as what the example rests on rather than as
things it decides.

**1. `work_slices` enumerates the source regions and names their holders.** One entry per region —
thirty-two here, so dense over both the tensor and the grid. Only one axis is divided, so each entry
names one dim id. The entry count gives the composed domain, since no extents are passed: 32 `mb`
indices × share extent 16 = 512, and `out` stays 4096. §1 of the design document, #153.

**2. There is no destination table.** The destination *set* is stated by a guard — absent here — and
the *arrangement* is the offset each instance passes, `[0, 128 * pid]`. So the destination division,
which is the *other half* of what makes this an all-to-all, is carried entirely by an offset and a
block shape. That is a stronger reliance on #153's §2 than the other fixtures place, and it is the
thing to look at if the destination side turns out to need a form after all.

Note that the shard *width* does reach the compose, as `block_shape`. So the destination side is
three facts — set, offset, extent — of which exactly one is an operand, and because it is a single
constexpr every destination tile must have the same shape. Here that is benign; it is what rules
out an all-to-all-v with unequal chunks.

**3. A memory-space attribute exists on `tl.spyre_tensor_layout`.** It does not yet — the one
assumption the example is ahead of. The relayout is scratchpad on both sides, and expressing that
directly is what makes the shape discussable. Tracked in #137.

## What it exercises, and what it cannot check yet

Exercises: a block that is **larger than a region on the divided axis and smaller on another**, so
one load draws a distinct slice from every source; a source and destination that divide different
axes, with only the source's division declared; `axes` with one named entry and one `None`; and
`ct_local` descriptors on both sides so the transfer is scratchpad to scratchpad throughout.

Cannot check: anything numerical, that the lowering produces the target KTIR, or — unlike its three
siblings — that the geometry matches a measured record, since none does. There is no lowering and the
KTDP view has no producer. The first provable milestone is a lit round-trip on the emitted TTIR, once
`tl.make_distributed_descriptor` exists.

## When `meta.py` is added

It needs `SIGNATURE`, `VARIANTS` with `params` and `constexpr`, a NumPy `reference` oracle and an
`inputs` generator — all of which presuppose execution, which is why it is omitted now. Add
`__init__.py` at the same time; `_import_meta` imports the folder as a package so `from . import
kernel` resolves. Setting `grid` to `[32]` matches the fixture.

The kernel also gains a prologue and an epilogue, since a runnable test has to supply its input from
the host and let the host read the result. The prologue is the producer this example assumes: load
from a global descriptor into a tensor, then store that tensor to the `ct_local` descriptor, and the
share is resident. The epilogue reverses it. The relayout between them — compose, then
`whole.load` — is unchanged, so the two files diverge only at the boundaries.

A sizing point arrives with execution and not before: both shares are 16 × 4096 and 512 × 128 fp16 =
128 KiB, and the Triton blocks are those whole shares. A scaled-down variant may be the one that runs
first, with the measured extents kept for the lit test. This does not remove the dependency on #137:
the staging store needs a `ct_local` descriptor too.

This is also the fixture whose numerical oracle is worth the most. Every one of the 1024 chunks has a
distinct (source, destination) pair, so a per-coordinate value encoding source tile and position
distinguishes a correct exchange from a transposed, rotated, or partially-dropped one — which shapes
and element counts do not.
