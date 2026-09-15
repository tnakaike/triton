# inter_tile_scatter

Splitting delivery — the block lands strictly inside one source region, so a load takes a slice.
One row of one region divided eight ways across the receivers. Written against the design in
[`inter-tile-lowering-to-mem-view.md`](../../../docs/inter-tile-lowering-to-mem-view.md).

Chosen because its holders are `0 .. 31` on both sides, so no relabelling is needed, and because it
is the case where the source set is not declared but **implied**. The arithmetic in the kernel is
what says which tile supplies whom.

**Status: draft, deliberately inert.** `tl.make_distributed_descriptor` does not exist, and there
is no `meta.py`, so `conftest.py::_load_examples` — which globs `fixtures/*/meta.py` — never
discovers this folder and never imports `kernel.py`. Nothing runs and nothing fails.

## What makes it a scatter

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

Here the block is `[1, 128]` against a region of `[64, 1024]` — 1/512 of a row block and 1/8 of a
slab — so it lands strictly inside one region.

## Fixture

GR-PF-055, `relayouts[54]` of the ownership catalog pinned by torch-spyre#4300. Consumer
`slice_161-Stcdp` input 0, on the tensor `mean_80-LayerNormNorm_out`; `remote_required` true.

**The catalog labels this record `route_class = permutation`**, which describes the *pre-select*
pair — 32 pieces to 32 pieces, one fragment each — and is not a statement about the transfer. The
split below is what the *post-select* pair is. So this fixture is the sixth of the six `permutation`
records, the other five being the whole-region bijection in
[`inter_tile_broadcast`](../inter_tile_broadcast/); between them that route class is fully covered.

```
extents          {mb: 512, out: 4096}      src == dst
work division    mb 8 ways, out 4 ways     32 regions, an 8 x 4 grid
per-tile share   {mb: 64, out: 1024}
word length      2 (fp16)

Source region k is mb[64 * (k // 4) : +64] x out[1024 * (k % 4) : +1024], on tile k.
Destination region d is mb[511 : 512] x out[128d : 128d + 128], on tile d.
```

| Side | Ownership |
|---|---|
| source | 32 regions, **one owner each**, holders `0 .. 31` |
| destination | 32 single-row chunks of 128 columns, **one owner each**, holders `0 .. 31` |

Every destination wants row 511, which lies in the last row block, `mb` 448..511. Only four tiles
hold that block — 28, 29, 30, 31 — in four consecutive slabs of 1024 columns. A destination is 128
columns wide, so each slab covers eight consecutive destinations:

```
tile 28  out    0..1023  ->  d =  0.. 7        tile 30  out 2048..3071  ->  d = 16..23
tile 29  out 1024..2047  ->  d =  8..15        tile 31  out 3072..4095  ->  d = 24..31
```

**Nothing in the kernel declares that.** It falls out of the offsets: row 511 is row block
`511 // 64 == 7`, column `128 * pid` is slab `pid // 8`, so the supplier is tile
`4 * 7 + pid // 8`, that is `28 + pid // 8`. The catalog states the same relation as "source
`28 + floor(d/8)` supplies destination `d`", and records
`destination_pieces_per_source_piece = 0` for pieces 0..27.

So **28 of the 32 tiles hold a region that no instance ever asks for**, and that is expressible
without being stated — the enumeration covers the tensor, and the offsets decide what is read. It
is also indistinguishable, at this level, from those tiles being idle.

## Deviation from the record

**None on the holders.** Both sides own `0 .. 31`, so the dense positional list carries this record
exactly. It is the case that shows the list form is not always the lossy one.

**One on the layout, and for this record it is measured rather than open.** A real SDSC run of
this exact logical shape — 512 × 4096 × 1, fp16 — reports `layoutDimOrder_ = ["mb", "out", "y"]`,
`stickDimOrder_ = ["y"]`, `stickSize_ = 64` and `device_size = [1, 4096, 512, 64]`. So the sticked
axis is the extent-1 `y`, carried as 64 padded lanes, and the physical form is `[1, out, mb, 64]`.
The fixture sticks `out` instead and drops `y`, which is a definite deviation, not an unknown. It
does not change what the fixture is about — the compose sees logical extents — but a fixture that
has to match emitted addresses will need the measured form.

## Operation sequence

```
Load (own scratchpad) -> Compose (32 regions) -> Load (distributed) -> Store
```

**No guard anywhere** — the only one of the four fixtures with none on either side. Every tile holds
a source region and every tile holds a destination chunk, and an absent guard is how "every
instance" is stated. The four-tile source set is not a guard; it is a consequence of the offsets.

## Target KTIR

What the lowering should produce, and the shape a `test/Conversion` lit test would pin once the
lowering exists:

- thirty-two `ktdp.construct_memory_view`, differing **only** in `coordinate_set` and
  `ct_id = 0..31` — those two attributes are the entire content of the distribution
- one `ktdp.construct_distributed_memory_view` composing them → `memref<512x4096xf16>`
- `%col = tid * 128`, then `construct_access_tile %src[%c511, %col]` with a 1 × 128 tile
- one `ktdp.load` — this **is** the transfer
- `ktdp.store` into a plain `ct_local` view with **no** `ct_id`, meaning the executing tile

A lowering is free to notice that 28 of the composed views are never touched by any access tile and
drop them. Whether it must is a question this fixture raises and does not answer: dropping them is a
transfer-count optimisation, keeping them is what makes the composed domain the full tensor.

## Assumptions

Each is settled elsewhere; they are recorded here as what the example rests on rather than as
things it decides.

**1. `work_slices` enumerates the source regions and names their holders.** One entry per region —
thirty-two here, so dense over both the tensor and the grid. The entry count gives the composed
domain, since no extents are passed: eight `mb` indices × share extent 64 = 512, four `out` indices
× share extent 1024 = 4096. §1 of the design document, #153.

**2. There is no destination table.** The destination *set* is stated by a guard — absent here,
since all 32 tiles hold a chunk — and the *arrangement* is the offset each instance passes,
`[511, 128 * pid]`. The set and the offset do not reach the compose; the destination
*extent* does, as `block_shape` — so the destination side is three facts, of which exactly one is an
operand. Because it is a single constexpr rather than a per-instance value, **every destination tile
must have the same shape**: uniform cardinality holds structurally here instead of needing a
verifier, at the cost of making unequal chunk sizes inexpressible rather than merely unchecked.
Nothing on this side is verified at the Triton level. §2 by #153.

**3. A memory-space attribute exists on `tl.spyre_tensor_layout`.** It does not yet — the one
assumption the example is ahead of. The relayout is scratchpad on both sides, and expressing that
directly is what makes the shape discussable. Tracked in #137.

## What it exercises, and what it cannot check yet

Exercises: a block **smaller than a region**, so a load lands strictly inside one; a source set that
is implied by the offsets rather than declared, including 28 regions nobody reads; a two-axis
division with a `work_slices` entry per axis; a fixture with no guard on either side; and `ct_local`
descriptors on both sides so the transfer is scratchpad to scratchpad throughout.

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
share is resident. The epilogue reverses it. The relayout between them — compose, then
`whole.load` — is unchanged, so the two files diverge only at the boundaries.

One sizing point arrives with execution and not before. A source region is 64 × 1024 fp16 = 128 KiB
and the Triton block is that whole region, which is a real scratchpad claim; the destination chunk
is 256 bytes, exactly two sticks. So a scaled-down variant may be the one that runs first, with the
measured extents kept for the lit test. This does not remove the dependency on #137: the staging
store needs a `ct_local` descriptor too.
