// C007 -- decode, BatchMatMulV2 input0, consumer bmm_3-BMM_1
// grouped_all_gather_with_replication
// tensor bmm_3-actAttnHeadBreak-VirtualReshape_out, rank 5 (in, mb, x, x1, y), extents {in: 768, mb: 1, x: 8, x1: 1, y: 4}, f16
//
// source      32 view(s) = 32 piece(s) x 1 owner, ct 0..31
// destination 32 view(s) = 16 piece(s) x 2 owner(s), ct 0..31
// composed    source 768x1x8x1x4, destination 1536x1x8x1x4 = 768x1x8x1x4 x 2 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination view is the whole of what its views
// hold -- one slot per (piece, owner) -- so a core says which slot it writes:
//     in = 0
//     mb = 0
//     x = ((tid // 2) % 8) * 1
//     x1 = 0
//     y = ((tid // 1) % 2) * 2
//     replica = (tid // 16) % 2, at 768 per slot on dimension 0
// Which cores take part is not stated: 32 of the 32 address a slot they
// own, and the slot's own ct_id is what says so. See README.md for the mapping
// and the provenance.

#src0 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src2 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 - 2 >= 0, -d2 + 2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src3 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 - 3 >= 0, -d2 + 3 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src4 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 - 4 >= 0, -d2 + 4 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src5 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 - 5 >= 0, -d2 + 5 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src6 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 - 6 >= 0, -d2 + 6 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src7 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 - 7 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src8 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src9 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src10 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 - 2 >= 0, -d2 + 2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src11 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 - 3 >= 0, -d2 + 3 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src12 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 - 2 >= 0, -d2 + 2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src13 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 - 4 >= 0, -d2 + 4 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src14 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 - 5 >= 0, -d2 + 5 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src15 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 - 6 >= 0, -d2 + 6 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src16 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 384 >= 0, -d0 + 575 >= 0, d1 >= 0, -d1 >= 0, d2 - 7 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src17 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src18 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src19 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 2 >= 0, -d2 + 2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src20 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 3 >= 0, -d2 + 3 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src21 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 4 >= 0, -d2 + 4 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src22 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 5 >= 0, -d2 + 5 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src23 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 - 3 >= 0, -d2 + 3 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src24 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 6 >= 0, -d2 + 6 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src25 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 576 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 7 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src26 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 - 4 >= 0, -d2 + 4 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src27 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 - 5 >= 0, -d2 + 5 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src28 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 - 6 >= 0, -d2 + 6 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src29 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 >= 0, d2 - 7 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src30 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#src31 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 192 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 3 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#dst2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 5 >= 0, -d2 + 5 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst3 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 5 >= 0, -d2 + 5 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#dst4 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 6 >= 0, -d2 + 6 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst5 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 6 >= 0, -d2 + 6 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#dst6 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 7 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst7 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 7 >= 0, -d2 + 7 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#dst8 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst9 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#dst10 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 2 >= 0, -d2 + 2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst11 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 2 >= 0, -d2 + 2 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#dst12 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 3 >= 0, -d2 + 3 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst13 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 3 >= 0, -d2 + 3 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#dst14 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 4 >= 0, -d2 + 4 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 1 >= 0)>
#dst15 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 767 >= 0, d1 >= 0, -d1 >= 0, d2 - 4 >= 0, -d2 + 4 >= 0, d3 >= 0, -d3 >= 0, d4 - 2 >= 0, -d4 + 3 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @c007_decode_batchmatmulv2_input0(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c768 = arith.constant 768 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %s19 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s20 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %s21 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s22 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %s23 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s24 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s25 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %s26 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s27 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s28 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s29 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s30 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s31 = ktdp.construct_memory_view %src, sizes: [192, 1, 1, 1, 4], strides: [4, 4, 4, 4, 1]
      {coordinate_set = #src31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24, %s25, %s26, %s27, %s28, %s29, %s30, %s31
      : memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<192x1x1x1x4xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<768x1x8x1x4xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views with the SAME coordinate_set and different ct_id --
  // which is what replication is, stated rather than left to the launch. The
  // composed type counts those views, so the slots are distinct coordinates
  // and no coordinate has two writers.
  %d0 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst8, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst8, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst9, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst9, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst10, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst10, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst11, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst11, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst12, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst12, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst13, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst13, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst14, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst14, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst15, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 9>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [768, 1, 1, 1, 2], strides: [2, 2, 2, 2, 1]
      {coordinate_set = #dst15, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 25>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 9>>, memref<768x1x1x1x2xf16, #ktdp.memory_space<ct_local, ct_id = 25>>) : memref<1536x1x8x1x4xf16>

  // Where this core's box sits: one independent field of the tile id per
  // divided dimension, exactly as in ../if-store/. No branch selects it --
  // the index is computed.
  %q_x = arith.divsi %tid, %c2 : index
  %i_x = arith.remsi %q_x, %c8 : index
  %i_y = arith.remsi %tid, %c2 : index
  %o_y = arith.muli %i_y, %c2 : index

  // Which of the slots is this core's: its position in its piece's owner
  // list, stacked on dimension 0 of the composed destination.
  %q_rep = arith.divsi %tid, %c16 : index
  %i_rep = arith.remsi %q_rep, %c2 : index
  %o_rep = arith.muli %i_rep, %c768 : index

  // The movement, and no control flow: one load of a coordinate REGION --
  // which source partitions that touches is the composed view's to resolve,
  // and a region may span several -- and one store into a named slot of the
  // destination, whose ct_id is what says whether this core owns it.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %i_x, %c0, %o_y]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<768x1x8x1x4xf16> -> !ktdp.access_tile<768x1x1x1x2xindex>
  %val = ktdp.load %rt : <768x1x1x1x2xindex> -> tensor<768x1x1x1x2xf16>
  %wt = ktdp.construct_access_tile %to[%o_rep, %c0, %i_x, %c0, %o_y]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<1536x1x8x1x4xf16> -> !ktdp.access_tile<768x1x1x1x2xindex>
  ktdp.store %val, %wt : tensor<768x1x1x1x2xf16>, <768x1x1x1x2xindex>

  return
}
