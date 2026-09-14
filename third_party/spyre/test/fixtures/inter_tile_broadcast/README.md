# inter_tile_broadcast

Sibling of `inter_tile_reduce` — the copy family rather than the fold family. First Triton
expression of a catalogued scratchpad relayout, written against the design in
[`inter-tile-lowering-to-mem-view.md`](../../../docs/inter-tile-lowering-to-mem-view.md).

Chosen as the simplest measured geometry: one division, one owner per source region, and a
plain 1 → 4 replication — no multi-axis split and no concatenation. So it exercises the
surface without confounds, and its target KTIR is short enough to state in full below.

**Status: draft, deliberately inert.** `tl.make_distributed_descriptor` does not exist, and
there is no `meta.py`, so `conftest.py::_load_examples` — which globs `fixtures/*/meta.py` —
never discovers this folder and never imports `kernel.py`. Nothing runs and nothing fails.

## Fixture

A measured scratchpad relayout from the ownership catalog pinned by torch-spyre#4300: a
512 × 64 fp16 tensor moving from 8 single-owner regions to the same 8 regions held by four
cores each — so ownership changes and the regions do not.

```
extents          {out: 512, x: 64}      src == dst
work division    8 slices on `out`, `x` uncut
per-core tile    {out: 64, x: 64}
layout order     [out, x];  stick on `x`, stick size 64
word length      2 (fp16)
local addresses  input 0, output 256 bytes -> element offsets 0 and 128

Region k is out[64k : 64k+64] x x[0:64].
```

| Side | Ownership |
|---|---|
| source | region k on core k **alone** — cores 0..7 hold, 8..31 hold nothing |
| destination | region k on cores 4k..4k+3 — **four holders each** |

Eight regions, 1 → 4 per region, 32-core grid. Both sides divide `out` eight ways, so the
regions are identical on each side and only *ownership* moves — the catalog-wide fact that
`source_extents == destination_extents` in 130 of 130 records.

Per-core view: core `c` ends up holding region `c // 4`, read from core `c // 4`. Only core 0
is local; 31 remote reads, fan-out 4 per source.

## Operation sequence

```
Load (own scratchpad, guarded pid < 8) -> Compose (8 regions) -> Load (distributed) -> Store
```

The guard produces the share; the compose is pure addressing and moves nothing; the distributed
load **is** the transfer, scratchpad to scratchpad and remote for every tile but 0; the store is
the landing, into the executing tile's own scratchpad.

## Target KTIR

What the lowering should produce for this fixture, and the shape a `test/Conversion` lit test
would pin once the lowering exists:

- eight `ktdp.construct_memory_view`, differing **only** in `coordinate_set` and
  `ct_id = 0..7` — those two attributes are the entire content of the distribution
- one `ktdp.construct_distributed_memory_view` composing them → `memref<512x64xf16>`
- `%g = tid / 4`, `%base = %g * 64`, then `construct_access_tile %src[%base, %c0]`
- one `ktdp.load` — this **is** the transfer, and fan-in does not multiply it
- `ktdp.store` into a plain `ct_local` view with **no** `ct_id`, meaning the executing tile

Note the asymmetry. Source ownership is in the `ct_id`s; destination *holder* is the absence
of a `ct_id` on the store; destination *region* appears only in `%base`. Neither the store nor
the load alone gives the map.

## Assumptions

All three were open questions when this fixture was drafted; each is now settled elsewhere, so
they are recorded here as what the example rests on rather than as things it decides.

**1. `work_slices` is a map keyed by tile id.** `{k: {"out": k} for k in range(8)}` — one entry
per region, the key naming the tile that holds it. Region k lives on tile k here, so the keys
happen to run `0 .. 7`, but nothing depends on that: they need not be contiguous, and a measured
case has sixteen regions on the even tiles alone. Settled in §1 of the design document by #153.

**2. There is no destination table.** The destination *set* is stated by a guard — absent here,
since all 32 tiles hold a copy — and the *arrangement* is the offset each instance passes,
`pid // 4` in this fixture. Neither reaches the compose and neither is verified at the Triton
level. Replication is not expressed at all: the factor *is* the number of instances passing the
same offset, four per region here. Settled in §2 by #153.

**3. A memory-space attribute exists on `tl.spyre_tensor_layout`.** It does not yet — this is
the one assumption the example is ahead of. Writing it this way is deliberate: the relayout is
scratchpad on both sides, and expressing that directly is what makes the shape discussable.
Tracked in #137.

Two things follow from it that the design document does not currently state. A `ct_local`
descriptor is **per-tile shaped** — `[64, 64]`, one region — and the 512 × 64 whole exists only
as the composed domain. And because each tile addresses its own scratchpad, every tile stores at
local `[0, 0]` and those are 32 distinct places, so `src == dst` extents are preserved exactly as
the record has them.

The alternative was to emulate scratchpad-to-scratchpad with global descriptors at the
boundaries, which does work — the transfer itself would still be scratchpad to scratchpad — but
it forces the output to carry 32 copies at distinct global offsets, breaking `src == dst`, and it
reads as a workaround rather than as the design.

## What it exercises, and what it cannot check yet

Exercises: a `work_slices` map with fewer entries than the grid — eight regions across 32 tiles
— holders named by key rather than by position, replication as coincident offsets rather than as a
construct, `axes` with `len(axes) == rank(partial)` so no axis is created, and `ct_local`
descriptors on both sides so the transfer is scratchpad to scratchpad throughout.

Cannot check: anything numerical, or that the lowering produces the target KTIR. There is no
lowering, and the KTDP view has no producer. The first provable milestone is a lit round-trip on
the emitted TTIR, once `tl.make_distributed_descriptor` exists; comparing against the target
KTIR arrives with the lowering.

## When `meta.py` is added

It needs `SIGNATURE`, `VARIANTS` with `params` and `constexpr`, a NumPy `reference` oracle
and an `inputs` generator — all of which presuppose execution, which is why it is omitted
now. Add `__init__.py` at the same time; `_import_meta` imports the folder as a package so
`from . import kernel` resolves. Setting `grid` to `[32]` matches the fixture.

The kernel also gains a prologue and an epilogue, since a runnable test has to supply its input
from the host and let the host read the result. The prologue is the producer this example
assumes: load from a global descriptor into a tensor, then store that tensor to the `ct_local`
descriptor, and the share is resident. The epilogue reverses it on the received tile. The
relayout between them — compose, then `whole.load` — is unchanged, so the two files diverge only
at the boundaries.

That does not remove the dependency on #137. The staging store needs a `ct_local` descriptor, so
memory space on `tl.spyre_tensor_layout` has to exist before the fixture can run either; making
the producer explicit only means the fixture needs no harness support beyond it.
