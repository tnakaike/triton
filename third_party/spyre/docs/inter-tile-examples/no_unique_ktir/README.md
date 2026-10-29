# no_unique_ktir: one Triton kernel, two correct KTIRs

[`no_unique_ktir.py`](no_unique_ktir.py) is a gather onto several tiles. [`ktir_1.mlir`](ktir_1.mlir) and [`ktir_2.mlir`](ktir_2.mlir) are two KTIRs for it, and both are correct. The point is that **the destination of this kernel is not determined by the kernel**: a lowering that composes the destination has to choose between them, and nothing in the kernel makes one choice right.

## The movement

| | |
|---|---|
| tensor | `[mb, out] = [4, 64]`, fp16, stick on `out`; physically `(out // 64, mb, out % 64) = [1, 4, 64]` |
| source | 4 regions of `[1, 64]`, region `k` on tile `k` |
| receivers | tiles 0 and 2, the whole tensor each |
| grid | `[4]` |

## What the kernel states, and what it does not

The kernel states:

- **which tiles receive**: the `if pid % 2 == 0`;
- **what each receives**: the load's coordinates `[0, 0]` and its block `[4, 64]`, the whole tensor;
- **where it lands in the receiving tile's own scratchpad**: the pin, at `DST_OFF`.

It does not state where the two copies sit in a composed destination. No access in the kernel names the destination as a global view, and the two receivers load the same box, so their copies have no coordinate that tells them apart.

## The two KTIRs

Both are in physical coordinates, `(out // 64, mb, out % 64)`.

| | copies laid out along | destination | tile 0 stores at | tile 2 stores at |
|---|---|---|---|---|
| [`ktir_1.mlir`](ktir_1.mlir) | `mb` (dimension 1) | `[1, 8, 64]` | `[0, 0, 0]` | `[0, 4, 0]` |
| [`ktir_2.mlir`](ktir_2.mlir) | `out // 64` (dimension 0) | `[2, 4, 64]` | `[0, 0, 0]` | `[1, 0, 0]` |

They agree on everything the kernel says: the same source views and their compose, the same guard, the same load, and on each receiving tile the same `[1, 4, 64]` buffer at the same offset. They differ only in the part the kernel leaves unsaid:

- the second destination view's coordinate set, `#dst1`;
- the composed destination type, `1x8x64` against `2x4x64`;
- the store's coordinates.

Both pass the checks a composed view has to meet: in each, the destination views are disjoint and tile the composed type exactly, the guard admits exactly tiles 0 and 2, and each stores into its own slot only. Both parse and verify with `spyre-triton-opt`.

## Why it matters

Where every receiver loads a different box, the destination is determined: its gather dims are the dims on which the receivers' boxes differ, and those come from evaluating the load coordinates per tile. Here two receivers load the same box, and that is where the determination stops. The same holds for every replicated destination in [`../dist-store/`](../dist-store/), 22 of the 35, and the SDSC they come from records each such destination as one region with several holders, not as copies laid out along any dimension.

## Status

`no_unique_ktir.py` is a draft for discussion and does not run: `tl.make_distributed_descriptor` taking a pinned share is triton#207, which is open, and nothing lowers this kernel yet. The KTIR files are examples, not tests.
