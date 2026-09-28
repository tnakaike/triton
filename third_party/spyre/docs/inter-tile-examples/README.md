# KTIR examples: inter-tile communication

Thirty-five relayouts a real model performs, each written as KTIR. They are the
corpus for [inter-tile-lowering-to-mem-view.md](../inter-tile-lowering-to-mem-view.md):
that document states the design and works small examples by hand, while these are
the movements actually recorded in a Granite run, at their real extents and with
their real owner maps.

They are **examples, not tests**. Nothing lowers
`ktdp.construct_distributed_memory_view` in tree yet and dbo-opt's legality check
rejects it, so the bar these meet is that they parse and verify. One lit test,
`test/docs/inter-tile-examples-parse.mlir`, keeps them at that bar.

The counterpart set is `test/fixtures/inter_tile_*/`, which approaches the same
subject from the other end: hand-written Triton kernels for a movement family, each
with a *Target KTIR* section saying what its lowering should produce. Those are
chosen to isolate one question at a time; these are whatever a real model did. The
KTIR spelling is the same in both, deliberately.

## What one file contains

The whole movement, as two distributions and a copy between them:

| | |
|---|---|
| the source distribution | one `ktdp.construct_memory_view` per source piece, composed by `ktdp.construct_distributed_memory_view` |
| the destination distribution | the same, one view per destination *(piece, owner)* pair, composed the same way |
| the movement | one `ktdp.construct_access_tile` + `ktdp.load` on the source, one tile + `ktdp.store` on the destination, whole-tensor on both sides |

**Both sides are composed, and that is the point.** An earlier revision wrote
each example from one tile's point of view: the source composed, the destination
a plain `ct_local` landing with no `ct_id`. That form cannot state its own
movement. C001 lands the whole tensor on all 32 cores, and in the one-tile form
the only "32" in the file was a comment -- the breadth lived in the launch, so
the IR was equally consistent with one core or all of them.

Composing the destination puts it back in the IR. A piece with several owners is
that many views with the **same** `coordinate_set` and different `ct_id`, which is
exactly what replication is. Nothing in the dialect objects: the verifier checks
element type and rank, and the op's own documentation leaves overlapping
coordinate sets "unspecified unless ... constrained by ... program semantics" --
a broadcast is such a semantics, since every holder receives the same bytes.

It also stops the examples from taking a side on transport. With both
distributions named, *which* side does the transferring is a lowering's choice:
pull, where each destination holder reads its share, or push. The one-tile form
had pull baked into its shape.

This is a deliberate divergence from
[inter-tile-lowering-to-mem-view.md](../inter-tile-lowering-to-mem-view.md) §4,
which composes only the source and calls a destination view "meaningless where
destinations are replicated", on the grounds that it would have "two writers for
one coordinate with nothing saying which wins". That argument holds for two
*different* values racing; it does not hold for a broadcast, where the writers
agree. §4's other objection -- that remote writes are unverified where remote
reads are supported -- is about lowering, and survives.

## How the SDSC becomes KTIR

| Source record | KTIR |
|---|---|
| a piece's `start` and `size` | the `coordinate_set`, as the box `[start, start + size)` in the tensor's global index space |
| each of that piece's `owners` | one view per owner, `memory_space = #ktdp.memory_space<ct_local, ct_id = N>`, repeated in the result memref type |
| the union of a side's pieces | that side's composed result shape |
| `word_length` | the element type: 2 bytes, so `f16` throughout |

Source and destination extents are identical in all 35, and each side's boxes
union to exactly those extents, so both composed views have the same type and the
copy between them needs no reshaping.

Two conventions the source record does not dictate:

**One `%src` and one `%dst`.** The design requires the views on a side to differ
*only* in `coordinate_set` and `ct_id`, which means one address, at the same place
in each core's scratchpad. The capture carries no LX addresses, so these are
`index` arguments rather than invented constants.

**`access_tile_set` is a range over the block shape**, not the box in global
coordinates, because that is what `buildAccessTile` emits
(`lib/Dialect/KTDP/Utils/Utility.cpp`).

## What the set does and does not cover

Ranks run 3 to 6. Fan-in runs from 1 source piece (C020, a broadcast, where the
source compose has a single partition) to 32.

**Core ids are never arbitrary.** Across all 130 relayouts in the package -- not
just these 35 -- every owner set is either a single core or an exact arithmetic
progression: 3109 single, 542 contiguous, and 195 strided by 2, 4, 8 or 16. Zero
irregular sets out of 3846 checked. Non-contiguous is common (C013's destinations
are `0, 4, 8, ...`; C014's sources are just `0, 16`), but a stride and an offset
always describe one. That is what a grid coordinate projects to -- 32 cores as 8
groups of 4 gives each group `{g, g+8, g+16, g+24}` -- so a holder is derivable
from a work-slice coordinate rather than needing an arbitrary table. The format
could express an irregular set, so a verifier should not *rely* on this; what the
evidence supports is that a coordinate is sufficient for real workloads.

Every **source** piece has exactly one owner, so a region held by several tiles on
the *source* side does not appear here at all. Destination replication is
everywhere, up to all 32 cores holding the whole tensor.

## The 35

| ID | Phase | Route class | Pieces | Rank | Consumer | Tensor |
|---|---|---|---:|---:|---|---|
| [C001](C001__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__25-to-1.mlir) | decode | `grouped_all_gather_with_replication` | 25 -> 1 | 3 | mm_13-BMM_1 | mul_263_out |
| [C002](C002__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_1-BMM_1 | mean-LayerNormNorm_out |
| [C003](C003__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_11-BMM_1 | mean_3-LayerNormNorm_out |
| [C004](C004__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_280-BMM_1 | mean_80-LayerNormNorm_out |
| [C005](C005__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm-BMM_1 | mean-LayerNormNorm_out |
| [C006](C006__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-8.mlir) | decode | `grouped_all_gather_with_replication` | 32 -> 8 | 5 | bmm_2-BMM_1 | bmm_2-actAttnHeadBreak-VirtualReshape_out |
| [C007](C007__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-16.mlir) | decode | `grouped_all_gather_with_replication` | 32 -> 16 | 5 | bmm_3-BMM_1 | bmm_3-actAttnHeadBreak-VirtualReshape_out |
| [C008](C008__decode__BatchMatMulV2__input1__all_gather__32-to-32.mlir) | decode | `all_gather` | 32 -> 32 | 5 | bmm-BMM_1 | mul_114_out |
| [C009](C009__decode__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-16.mlir) | decode | `grouped_all_gather_with_replication` | 32 -> 16 | 5 | bmm_3-BMM_1 | bmm_3-wtAttnHeadBreak-VirtualReshape_out |
| [C010](C010__decode__Exx2__input0__all_gather__32-to-1.mlir) | decode | `all_gather` | 32 -> 1 | 3 | mean-Exx2 | mul_out |
| [C011](C011__decode__LayerNormNorm__input1__replicate_or_owner_remap__1-to-1.mlir) | decode | `replicate_or_owner_remap` | 1 -> 1 | 3 | mean-LayerNormNorm | mean-Exx2_out |
| [C012](C012__decode__LayerNormNorm__input2__replicate_or_owner_remap__1-to-1.mlir) | decode | `replicate_or_owner_remap` | 1 -> 1 | 3 | mean-LayerNormNorm | mean-LayerNormScale_out |
| [C013](C013__decode__Max__input0__all_gather__32-to-8.mlir) | decode | `all_gather` | 32 -> 8 | 4 | _safe_softmax-Max | _safe_softmax-Where3-inp0Masking_out |
| [C014](C014__decode__Mul__input1__grouped_all_gather_with_replication__2-to-1.mlir) | decode | `grouped_all_gather_with_replication` | 2 -> 1 | 6 | mul_138-mul_1 | mul_138-split_2-Ds0_out |
| [C015](C015__decode__Mul__input1__grouped_all_gather_with_replication__2-to-1.mlir) | decode | `grouped_all_gather_with_replication` | 2 -> 1 | 6 | mul_137-mul_1 | mul_137-split_2-Ds0_out |
| [C016](C016__decode__Mul__input1__replicate_or_owner_remap__8-to-8.mlir) | decode | `replicate_or_owner_remap` | 8 -> 8 | 4 | _safe_softmax_1-Mul | _safe_softmax_1-Reciprocal_out |
| [C017](C017__decode__Restickify__input0__all_gather__32-to-32.mlir) | decode | `all_gather` | 32 -> 32 | 5 | bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify | bmm-wtAttnHeadBreak-VirtualReshape_out |
| [C018](C018__decode__Restickify__input0__permutation__32-to-32.mlir) | decode | `permutation` | 32 -> 32 | 5 | bmm_2-wtAttnHeadBreak-VirtualReshape-Output-Restickify | bmm_2-wtAttnHeadBreak-VirtualReshape_out |
| [C019](C019__decode__Scatter__input0__all_gather__16-to-1.mlir) | decode | `all_gather` | 16 -> 1 | 4 | cat_1-kvCacheScatter | view_5-VirtualReshape_out |
| [C020](C020__decode__Stcdp__input0__general_relayout__1-to-32.mlir) | decode | `general_relayout` | 1 -> 32 | 3 | embedding-Output-Stcdp | embedding_out |
| [C021](C021__decode__Sub__input1__replicate_or_owner_remap__8-to-8.mlir) | decode | `replicate_or_owner_remap` | 8 -> 8 | 4 | _safe_softmax-Sub | _safe_softmax-Max_out |
| [C022](C022__decode__Sum__input0__all_gather__32-to-8.mlir) | decode | `all_gather` | 32 -> 8 | 4 | _safe_softmax-Sum | _safe_softmax-Exp_out |
| [C023](C023__prefill__Add__input1__all_gather__16-to-32.mlir) | prefill | `all_gather` | 16 -> 32 | 3 | add_3 | mul_out |
| [C024](C024__prefill__BatchMatMulV2__input0__all_gather__32-to-32.mlir) | prefill | `all_gather` | 32 -> 32 | 5 | bmm_2-BMM_1 | bmm_2-actAttnHeadBreak-VirtualReshape_out |
| [C025](C025__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__16-to-8.mlir) | prefill | `grouped_all_gather_with_replication` | 16 -> 8 | 3 | mm-BMM_1 | mean-LayerNormNorm_out |
| [C026](C026__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir) | prefill | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_280-BMM_1 | slice_161-Stcdp_out |
| [C027](C027__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-8.mlir) | prefill | `grouped_all_gather_with_replication` | 32 -> 8 | 3 | mm_10-BMM_1 | view_41-VirtualReshape_out |
| [C028](C028__prefill__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-1.mlir) | prefill | `grouped_all_gather_with_replication` | 32 -> 1 | 5 | bmm_2-BMM_1 | mul_17_out |
| [C029](C029__prefill__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-1.mlir) | prefill | `grouped_all_gather_with_replication` | 32 -> 1 | 5 | bmm_3-BMM_1 | bmm_3-wtAttnHeadBreak-VirtualReshape_out |
| [C030](C030__prefill__Exx2__input0__all_gather__32-to-8.mlir) | prefill | `all_gather` | 32 -> 8 | 3 | mean_1-Exx2 | add_3_out |
| [C031](C031__prefill__LayerNormNorm__input1__replicate_or_owner_remap__8-to-8.mlir) | prefill | `replicate_or_owner_remap` | 8 -> 8 | 3 | mean_1-LayerNormNorm | mean_1-Exx2_out |
| [C032](C032__prefill__LayerNormNorm__input2__replicate_or_owner_remap__8-to-8.mlir) | prefill | `replicate_or_owner_remap` | 8 -> 8 | 3 | mean_1-LayerNormNorm | mean_1-LayerNormScale_out |
| [C033](C033__prefill__Mul__input1__grouped_all_gather_with_replication__16-to-8.mlir) | prefill | `grouped_all_gather_with_replication` | 16 -> 8 | 6 | mul_14-mul_1 | mul_14-split_2-Ds0_out |
| [C034](C034__prefill__Restickify__input0__permutation__32-to-32.mlir) | prefill | `permutation` | 32 -> 32 | 5 | bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify | bmm-wtAttnHeadBreak-VirtualReshape_out |
| [C035](C035__prefill__Stcdp__input0__permutation__32-to-32.mlir) | prefill | `permutation` | 32 -> 32 | 3 | slice_161-Stcdp | mean_80-LayerNormNorm_out |

## Provenance

Generated from `lx-relayout-sdsc-examples-for-ktir-20260915`, whose canonical
Granite artifacts are pinned to commit

    36804f23ede70325c21ba234a7e102537a8eda95

originally at `AdnanHoque/torch-spyre/experiments/granite_relayout/`. The package
holds 130 relayout records in total; these 35 are its `curated/` set, one
representative per distinct combination of phase, consumer, movement class, piece
counts, extents and owner geometry.

A capture is not a specification: do not read performance, correctness or current
compiler support out of these. They say what one recorded run needed moved, and how
that movement is spelled in KTIR.
