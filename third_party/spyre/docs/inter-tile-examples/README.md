# KTIR examples: inter-tile communication

Thirty-five relayouts a real model performs, each written as KTIR, in **two
spellings of the same movement**:

| | |
|---|---|
| [`dist-store/`](dist-store/) | both sides composed. The destination is a distributed memory view too, so a core stores into **its slot of the whole** rather than into a buffer of its own. |
| [`if-store/`](if-store/) | only the **source** is composed. Each core lands its own piece in its own scratchpad, at local coordinate zero. |

**Neither is decided.** They are here to be read against each other, and each
directory's README states what its spelling buys and what it costs. The question
they are evidence for is open: §4 of
[inter-tile-lowering-to-mem-view.md](../inter-tile-lowering-to-mem-view.md)
composes only the source, and whether a composed destination is the better
spelling -- or admissible at all -- is what these two sets exist to inform.

The one thing both forms agree on, and which an earlier revision got wrong, is
that **the breadth of a relayout belongs in the IR**. That revision wrote each
example from one tile's point of view, with a comment saying the other cores do
the same: C001 lands the whole tensor on all 32 cores, and the only "32" in the
file was prose. `dist-store` puts it in the type system, as one view per
*(piece, owner)* pair; `if-store` leaves it out of the destination side entirely,
because a landing in this core's own scratchpad says nothing about the others.

**Both forms derive a core's box from the tile id**, by the same arithmetic.
That used to be `if-store`'s distinguishing feature and is not: once the composed
destination is the whole of what its views hold, a store has to say which slot of
it this core writes. What is left of the contrast is two lines:

| | `dist-store` | `if-store` |
|---|---|---|
| the store's destination | the composed view, at this core's slot: the piece's position in the box the pieces tile, plus its replica on dimension 0 | a local view, at zero. Which global coordinates it holds is not in the IR |
| participation | **not stated.** The slot carries its holder's `ct_id`, so a core that addresses a slot it does not own is asking for a remote write | an `scf.if` on the tile id, in 10 of the 35. A local landing names no holder, so nothing else could say it

The second line is the sharper difference, and it is the one to argue about: it
says a composed destination makes participation a **consequence** of the two
statements the file already makes, where a local landing makes it a third
statement that could disagree with them. Checked over all 35 and every core: the
cores whose addressed slot is their own are exactly the owner set, so the guard
`dist-store` does not have would have been redundant with its types.

They are **examples, not tests**. Nothing lowers
`ktdp.construct_distributed_memory_view` in tree yet and dbo-opt's legality check
rejects it, so the bar these meet is that they parse and verify. One lit test,
`test/docs/inter-tile-examples-parse.mlir`, keeps both directories at that bar.

The counterpart set is `test/fixtures/inter_tile_*/`, which approaches the same
subject from the other end: hand-written Triton kernels for a movement family,
each with a *Target KTIR* section saying what its lowering should produce. Those
are chosen to isolate one question at a time; these are whatever a real model did.

## How the SDSC becomes KTIR

| Source record | KTIR |
|---|---|
| a piece's `start` and `size` | the `coordinate_set`, as the box `[start, start + size)` in the tensor's global index space |
| each of that piece's `owners` | one view per owner, `memory_space = #ktdp.memory_space<ct_local, ct_id = N>`, repeated in the result memref type |
| a side's pieces and their owners | that side's composed result shape: the box the pieces tile, with dimension 0 scaled by the number of owners each piece has |
| `word_length` | the element type: 2 bytes, so `f16` throughout |

**A composed view is the whole of what its views hold**, which is not the same
thing as the box they cover, and two of the 35 show why each half of that
matters. The rule is one sentence -- every `(piece, owner)` pair is a slot of the
result, and the slots of one piece stack on dimension 0 -- and the consequences
are:

- **Replication multiplies.** C002's destination is the whole 4096-element
  tensor held by 16 cores, so its composed type is `memref<65536x1x1xf16>` and
  not `memref<4096x1x1xf16>`. The 16 copies are real memory and each is
  addressable; a type that named the coordinates once would describe a sixteenth
  of what the views hold.
- **A side need not cover the tensor.** C035's destination pieces all sit at
  `mb = 511`, so the box they tile is `1x4096x1` -- one row of 512 -- and that is
  its composed extent. An earlier revision used the tensor's extents on both
  sides and typed it `512x4096x1`, a view 512 times the memory behind it.

So the two sides of one movement do **not** in general have the same composed
type, even though the source and destination *extents* are identical in all 35.
A load and a store therefore name a box rather than the whole: the box is this
core's piece, and its position is where each spelling differs.

Three conventions the source record does not dictate:

**One `%src` and one `%dst`.** The design requires the views on a side to differ
*only* in `coordinate_set` and `ct_id`, which means one address, at the same place
in each core's scratchpad. The capture carries no LX addresses, so these are
`index` arguments rather than invented constants.

**`access_tile_set` is a range over the block shape**, not the box in global
coordinates. `buildAccessTile`
(`lib/Dialect/KTDP/Transforms/Utility.cpp`) is the authority: it passes
`buildRangeSetND(ctx, blockShape)` and no symbol operands, so the **set** carries
the block's extent and the **base indices** carry its position. Where a piece
starts at the origin the two spellings coincide, which is why a generated name
like `#src0` can appear in both roles -- the sets are textually identical and the
generator emits each one once.

**Core ids are never arbitrary.** Across all 130 relayouts in the package -- not
just these 35 -- every owner set is either a single core or an exact arithmetic
progression: 3109 single, 542 contiguous, and 195 strided by 2, 4, 8 or 16. Zero
irregular sets out of 3846 checked. That is what lets `if-store` write a
participation guard as a stride test rather than a list of 28 comparisons, and it
is what makes a holder derivable from a work-slice coordinate. The format *could*
express an irregular set, so a verifier should not rely on this.

## What the set does and does not cover

Ranks run 3 to 6. Fan-in runs from 1 source piece (C020, a broadcast) to 32.

Every **source** piece has exactly one owner, so a region held by several tiles on
the *source* side does not appear here at all. Destination replication is
everywhere, up to all 32 cores holding the whole tensor.

## The 35

| ID | Phase | Route class | Pieces | Rank | Consumer | Tensor |
|---|---|---|---:|---:|---|---|
| C001 ([dist](dist-store/C001__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__25-to-1.mlir), [if](if-store/C001__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__25-to-1.mlir)) | decode | `grouped_all_gather_with_replication` | 25 -> 1 | 3 | mm_13-BMM_1 | mul_263_out |
| C002 ([dist](dist-store/C002__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir), [if](if-store/C002__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir)) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_1-BMM_1 | mean-LayerNormNorm_out |
| C003 ([dist](dist-store/C003__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir), [if](if-store/C003__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir)) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_11-BMM_1 | mean_3-LayerNormNorm_out |
| C004 ([dist](dist-store/C004__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir), [if](if-store/C004__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir)) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_280-BMM_1 | mean_80-LayerNormNorm_out |
| C005 ([dist](dist-store/C005__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir), [if](if-store/C005__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir)) | decode | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm-BMM_1 | mean-LayerNormNorm_out |
| C006 ([dist](dist-store/C006__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-8.mlir), [if](if-store/C006__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-8.mlir)) | decode | `grouped_all_gather_with_replication` | 32 -> 8 | 5 | bmm_2-BMM_1 | bmm_2-actAttnHeadBreak-VirtualReshape_out |
| C007 ([dist](dist-store/C007__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-16.mlir), [if](if-store/C007__decode__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-16.mlir)) | decode | `grouped_all_gather_with_replication` | 32 -> 16 | 5 | bmm_3-BMM_1 | bmm_3-actAttnHeadBreak-VirtualReshape_out |
| C008 ([dist](dist-store/C008__decode__BatchMatMulV2__input1__all_gather__32-to-32.mlir), [if](if-store/C008__decode__BatchMatMulV2__input1__all_gather__32-to-32.mlir)) | decode | `all_gather` | 32 -> 32 | 5 | bmm-BMM_1 | mul_114_out |
| C009 ([dist](dist-store/C009__decode__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-16.mlir), [if](if-store/C009__decode__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-16.mlir)) | decode | `grouped_all_gather_with_replication` | 32 -> 16 | 5 | bmm_3-BMM_1 | bmm_3-wtAttnHeadBreak-VirtualReshape_out |
| C010 ([dist](dist-store/C010__decode__Exx2__input0__all_gather__32-to-1.mlir), [if](if-store/C010__decode__Exx2__input0__all_gather__32-to-1.mlir)) | decode | `all_gather` | 32 -> 1 | 3 | mean-Exx2 | mul_out |
| C011 ([dist](dist-store/C011__decode__LayerNormNorm__input1__replicate_or_owner_remap__1-to-1.mlir), [if](if-store/C011__decode__LayerNormNorm__input1__replicate_or_owner_remap__1-to-1.mlir)) | decode | `replicate_or_owner_remap` | 1 -> 1 | 3 | mean-LayerNormNorm | mean-Exx2_out |
| C012 ([dist](dist-store/C012__decode__LayerNormNorm__input2__replicate_or_owner_remap__1-to-1.mlir), [if](if-store/C012__decode__LayerNormNorm__input2__replicate_or_owner_remap__1-to-1.mlir)) | decode | `replicate_or_owner_remap` | 1 -> 1 | 3 | mean-LayerNormNorm | mean-LayerNormScale_out |
| C013 ([dist](dist-store/C013__decode__Max__input0__all_gather__32-to-8.mlir), [if](if-store/C013__decode__Max__input0__all_gather__32-to-8.mlir)) | decode | `all_gather` | 32 -> 8 | 4 | _safe_softmax-Max | _safe_softmax-Where3-inp0Masking_out |
| C014 ([dist](dist-store/C014__decode__Mul__input1__grouped_all_gather_with_replication__2-to-1.mlir), [if](if-store/C014__decode__Mul__input1__grouped_all_gather_with_replication__2-to-1.mlir)) | decode | `grouped_all_gather_with_replication` | 2 -> 1 | 6 | mul_138-mul_1 | mul_138-split_2-Ds0_out |
| C015 ([dist](dist-store/C015__decode__Mul__input1__grouped_all_gather_with_replication__2-to-1.mlir), [if](if-store/C015__decode__Mul__input1__grouped_all_gather_with_replication__2-to-1.mlir)) | decode | `grouped_all_gather_with_replication` | 2 -> 1 | 6 | mul_137-mul_1 | mul_137-split_2-Ds0_out |
| C016 ([dist](dist-store/C016__decode__Mul__input1__replicate_or_owner_remap__8-to-8.mlir), [if](if-store/C016__decode__Mul__input1__replicate_or_owner_remap__8-to-8.mlir)) | decode | `replicate_or_owner_remap` | 8 -> 8 | 4 | _safe_softmax_1-Mul | _safe_softmax_1-Reciprocal_out |
| C017 ([dist](dist-store/C017__decode__Restickify__input0__all_gather__32-to-32.mlir), [if](if-store/C017__decode__Restickify__input0__all_gather__32-to-32.mlir)) | decode | `all_gather` | 32 -> 32 | 5 | bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify | bmm-wtAttnHeadBreak-VirtualReshape_out |
| C018 ([dist](dist-store/C018__decode__Restickify__input0__permutation__32-to-32.mlir), [if](if-store/C018__decode__Restickify__input0__permutation__32-to-32.mlir)) | decode | `permutation` | 32 -> 32 | 5 | bmm_2-wtAttnHeadBreak-VirtualReshape-Output-Restickify | bmm_2-wtAttnHeadBreak-VirtualReshape_out |
| C019 ([dist](dist-store/C019__decode__Scatter__input0__all_gather__16-to-1.mlir), [if](if-store/C019__decode__Scatter__input0__all_gather__16-to-1.mlir)) | decode | `all_gather` | 16 -> 1 | 4 | cat_1-kvCacheScatter | view_5-VirtualReshape_out |
| C020 ([dist](dist-store/C020__decode__Stcdp__input0__general_relayout__1-to-32.mlir), [if](if-store/C020__decode__Stcdp__input0__general_relayout__1-to-32.mlir)) | decode | `general_relayout` | 1 -> 32 | 3 | embedding-Output-Stcdp | embedding_out |
| C021 ([dist](dist-store/C021__decode__Sub__input1__replicate_or_owner_remap__8-to-8.mlir), [if](if-store/C021__decode__Sub__input1__replicate_or_owner_remap__8-to-8.mlir)) | decode | `replicate_or_owner_remap` | 8 -> 8 | 4 | _safe_softmax-Sub | _safe_softmax-Max_out |
| C022 ([dist](dist-store/C022__decode__Sum__input0__all_gather__32-to-8.mlir), [if](if-store/C022__decode__Sum__input0__all_gather__32-to-8.mlir)) | decode | `all_gather` | 32 -> 8 | 4 | _safe_softmax-Sum | _safe_softmax-Exp_out |
| C023 ([dist](dist-store/C023__prefill__Add__input1__all_gather__16-to-32.mlir), [if](if-store/C023__prefill__Add__input1__all_gather__16-to-32.mlir)) | prefill | `all_gather` | 16 -> 32 | 3 | add_3 | mul_out |
| C024 ([dist](dist-store/C024__prefill__BatchMatMulV2__input0__all_gather__32-to-32.mlir), [if](if-store/C024__prefill__BatchMatMulV2__input0__all_gather__32-to-32.mlir)) | prefill | `all_gather` | 32 -> 32 | 5 | bmm_2-BMM_1 | bmm_2-actAttnHeadBreak-VirtualReshape_out |
| C025 ([dist](dist-store/C025__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__16-to-8.mlir), [if](if-store/C025__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__16-to-8.mlir)) | prefill | `grouped_all_gather_with_replication` | 16 -> 8 | 3 | mm-BMM_1 | mean-LayerNormNorm_out |
| C026 ([dist](dist-store/C026__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir), [if](if-store/C026__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-1.mlir)) | prefill | `grouped_all_gather_with_replication` | 32 -> 1 | 3 | mm_280-BMM_1 | slice_161-Stcdp_out |
| C027 ([dist](dist-store/C027__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-8.mlir), [if](if-store/C027__prefill__BatchMatMulV2__input0__grouped_all_gather_with_replication__32-to-8.mlir)) | prefill | `grouped_all_gather_with_replication` | 32 -> 8 | 3 | mm_10-BMM_1 | view_41-VirtualReshape_out |
| C028 ([dist](dist-store/C028__prefill__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-1.mlir), [if](if-store/C028__prefill__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-1.mlir)) | prefill | `grouped_all_gather_with_replication` | 32 -> 1 | 5 | bmm_2-BMM_1 | mul_17_out |
| C029 ([dist](dist-store/C029__prefill__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-1.mlir), [if](if-store/C029__prefill__BatchMatMulV2__input1__grouped_all_gather_with_replication__32-to-1.mlir)) | prefill | `grouped_all_gather_with_replication` | 32 -> 1 | 5 | bmm_3-BMM_1 | bmm_3-wtAttnHeadBreak-VirtualReshape_out |
| C030 ([dist](dist-store/C030__prefill__Exx2__input0__all_gather__32-to-8.mlir), [if](if-store/C030__prefill__Exx2__input0__all_gather__32-to-8.mlir)) | prefill | `all_gather` | 32 -> 8 | 3 | mean_1-Exx2 | add_3_out |
| C031 ([dist](dist-store/C031__prefill__LayerNormNorm__input1__replicate_or_owner_remap__8-to-8.mlir), [if](if-store/C031__prefill__LayerNormNorm__input1__replicate_or_owner_remap__8-to-8.mlir)) | prefill | `replicate_or_owner_remap` | 8 -> 8 | 3 | mean_1-LayerNormNorm | mean_1-Exx2_out |
| C032 ([dist](dist-store/C032__prefill__LayerNormNorm__input2__replicate_or_owner_remap__8-to-8.mlir), [if](if-store/C032__prefill__LayerNormNorm__input2__replicate_or_owner_remap__8-to-8.mlir)) | prefill | `replicate_or_owner_remap` | 8 -> 8 | 3 | mean_1-LayerNormNorm | mean_1-LayerNormScale_out |
| C033 ([dist](dist-store/C033__prefill__Mul__input1__grouped_all_gather_with_replication__16-to-8.mlir), [if](if-store/C033__prefill__Mul__input1__grouped_all_gather_with_replication__16-to-8.mlir)) | prefill | `grouped_all_gather_with_replication` | 16 -> 8 | 6 | mul_14-mul_1 | mul_14-split_2-Ds0_out |
| C034 ([dist](dist-store/C034__prefill__Restickify__input0__permutation__32-to-32.mlir), [if](if-store/C034__prefill__Restickify__input0__permutation__32-to-32.mlir)) | prefill | `permutation` | 32 -> 32 | 5 | bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify | bmm-wtAttnHeadBreak-VirtualReshape_out |
| C035 ([dist](dist-store/C035__prefill__Stcdp__input0__permutation__32-to-32.mlir), [if](if-store/C035__prefill__Stcdp__input0__permutation__32-to-32.mlir)) | prefill | `permutation` | 32 -> 32 | 3 | slice_161-Stcdp | mean_80-LayerNormNorm_out |

## Provenance, and regenerating

Generated from `lx-relayout-sdsc-examples-for-ktir-20260915`, whose canonical
Granite artifacts are pinned to commit

    36804f23ede70325c21ba234a7e102537a8eda95

originally at `AdnanHoque/torch-spyre/experiments/granite_relayout/`. The package
holds 130 relayout records in total; these 35 are its `curated/` set, one
representative per distinct combination of phase, consumer, movement class, piece
counts, extents and owner geometry.

`gen_ktir_inter_tile_examples.py` here writes both directories. **The capture is
not in this repo** -- it is a pinned historical package -- so pass its path:

    ./gen_ktir_inter_tile_examples.py PACKAGE_DIR .

It exists so that a KTDP dialect change is answered by regenerating rather than
by hand-editing 70 files and some 1600 memory views.

A capture is not a specification: do not read performance, correctness or current
compiler support out of these. They say what one recorded run needed moved, and
how that movement is spelled in KTIR.
