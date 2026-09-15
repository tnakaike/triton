# inter_tile_broadcast

Whole-region delivery — the block equals a source region, so no load ever cuts or assembles one.
Sibling of `inter_tile_reduce`, which is the fold family rather than the copy family. Written
against the design in
[`inter-tile-lowering-to-mem-view.md`](../../../docs/inter-tile-lowering-to-mem-view.md).

**Status: draft, deliberately inert.** `tl.make_distributed_descriptor` does not exist, and
there is no `meta.py`, so `conftest.py::_load_examples` — which globs `fixtures/*/meta.py` —
never discovers this folder and never imports `kernel.py`. Nothing runs and nothing fails.

## Two kernels, and what separates them

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
| **equal to a region** | — | **`broadcast`** | **`permutation`** |
| **spans several** | — | `gather_replicated` | — |
| **spans all** | `gather` | all-gather to every tile | `all_to_all` |

This directory is the middle row: `inter_tile_broadcast` and `inter_tile_permutation` in the same
`kernel.py`, with `block_shape` equal to a region in both. The only difference between them is the
offset — shared by four instances, or distinct for all 32. The other rows are
[`inter_tile_gather`](../inter_tile_gather/), [`inter_tile_scatter`](../inter_tile_scatter/) and
[`inter_tile_all_to_all`](../inter_tile_all_to_all/).

`kernel_map.py` carries the broadcast alone, spelling `work_slices` as a map keyed by tile id
instead of as a dense positional list. It differs in that one declaration and nowhere else. See
[How `work_slices` is spelled](#how-work_slices-is-spelled).

## Fixture — `inter_tile_broadcast`

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

**1. `work_slices` enumerates the source regions and names their holders.** One entry per
region — eight entries for a 32-tile grid, so it is dense over the tensor and sparse over the
grid. `kernel.py` names the holders by position, which works because they run `0 .. 7`;
`kernel_map.py` names them by key, which does not need that. Either way the entry count gives
the composed domain, since no extents are passed. §1 of the design document, #153.

**2. There is no destination table.** The destination *set* is stated by a guard — absent here,
since all 32 tiles hold a copy — and the *arrangement* is the offset each instance passes,
`pid // 4` in this fixture. The set and the offset do not reach the compose; the destination
*extent* does, as `block_shape` — so the destination side is three facts, of which exactly one is an
operand. Because it is a single constexpr rather than a per-instance value, **every destination tile
must have the same shape**: uniform cardinality holds structurally here instead of needing a
verifier, at the cost of making unequal chunk sizes inexpressible rather than merely unchecked.
Nothing on this side is verified at the Triton level. Replication is not expressed at all: the factor *is*
the number of instances passing the same offset, four per region here. Settled in §2 by #153.

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

## Fixture — `inter_tile_permutation`

GR-PF-052, `relayouts[51]` of the same catalog. Consumer
`bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify` input 0, route class **`permutation`**.

```
extents          {j: 8, mb: 512, out: 128}    src == dst
work division    j 4 ways, mb 8 ways          32 regions, a 4 x 8 index grid
per-core tile    {j: 2, mb: 64, out: 128}     the same size on both sides
word length      2 (fp16)

Region (J, M) is j[2J : 2J+2] x mb[64M : 64M+64] x out[0:128],  J in [0,4), M in [0,8).
```

| Side | Ownership |
|---|---|
| source | owner `k = 8J + M` — `M` varies fastest, row-major in `(J, M)` |
| destination | owner `m = J + 4M` — `J` varies fastest, column-major |

Both sides own `0 .. 31`, one region each, `P = 1`, and
`destination_pieces_per_source_piece = 1` — a **bijection**. So the map is
`pi(k) = (k div 8) + 4 * (k mod 8)`, and the kernel passes its inverse as the offset:
instance `m` holds `J = m mod 4`, `M = m div 4`, hence `[2 * (pid % 4), 64 * (pid // 4), 0]`.

The permutation has **exactly two fixed points**, tiles 0 and 31: `8J + M == J + 4M` reduces to
`7J == 3M`, which over `J < 4`, `M < 8` holds only at `(0, 0)` and `(3, 7)`. So 30 of the 32 reads
are remote and two are local — worth knowing for a numerical oracle, since a fixed point is where a
dropped transfer still yields the right value.

That is a transpose of a 4 × 8 index grid — the catalog's row-major-versus-column-major core order.
Four sibling records share the identical geometry and owner map (`relayouts[52]`, `[53]`, and the
decode counterparts `[118]`, `[119]`), so one kernel covers five of the six `permutation` records;
the sixth is the one [`inter_tile_scatter`](../inter_tile_scatter/) transcribes.

**Deviation from the record:** the extent-1 `x` and `y` axes are dropped, so the fixture is rank 3
rather than rank 5. Holders need no relabelling — this is the second of the two records that the
dense positional list carries exactly.

**Why it belongs beside the broadcast.** Identical `block_shape`, identical operation sequence,
and no guard on either side. The whole diff is the offset expression: `[g * 64, 0]` with `g = pid //
4`, versus `[2 * (pid % 4), 64 * (pid // 4), 0]`. Reading them together is what shows that
replication is not a construct but a *coincidence of offsets* — remove the coincidence and the same
call becomes a permutation.

Its target KTIR is the broadcast's with one change: `%j = (tid % 4) * 2` and
`%mb = (tid / 4) * 64` in place of `%base`, and 32 `construct_memory_view` rather than 8.

## How `work_slices` is spelled

The two files differ in one declaration:

```python
# kernel.py -- position is the holder
SRC_WORK_SLICES = [{"out": 0}, {"out": 1}, ..., {"out": 7}]

# kernel_map.py -- the key is the holder
SRC_WORK_SLICES = {0: {"out": 0}, 1: {"out": 1}, ..., 7: {"out": 7}}
```

This fixture reads the same under both, because its holders are exactly `0 .. 7`. It is
therefore not the case that decides between them — the deciding case is one whose holders are
not `0 .. N-1`, and a measured `all_gather` record is one: sixteen regions, one owner each, on
the **even** tiles `0, 2 … 30`, with the odd tiles holding nothing. A sixteen-entry list claims
holders `0 .. 15` there, wrong for every entry but the first; a thirty-two-entry list would need
sixteen holes, and there is no spelling for "holds nothing".

The intent is to support **one** of the two, not both: a list where a map is needed is silently
wrong rather than rejected, so accepting both makes the sparse case a footgun. `kernel.py` leads
because the list is what the current Triton support already accepts, which makes it the cheaper
thing to prototype against; `kernel_map.py` is kept because going back to the map is a
one-declaration change here and the simpler implementation there.

## What it exercises, and what it cannot check yet

Exercises: a `work_slices` enumeration with fewer entries than the grid — eight regions across
32 tiles — replication as coincident offsets rather than as a construct, `axes` with
`len(axes) == rank(partial)` so no axis is created, and `ct_local` descriptors on both sides so
the transfer is scratchpad to scratchpad throughout.

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
