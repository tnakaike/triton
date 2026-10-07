// C001 -- decode, BatchMatMulV2 input0, consumer mm_13-BMM_1
// grouped_all_gather_with_replication
// tensor mul_263_out, rank 3 (in, mb, y), extents {in: 12800, mb: 1, y: 1}, f16
//
// source      25 view(s) = 25 piece(s) x 1 owner, ct 0..24
// destination 32 view(s) = 1 piece(s) x 32 owner(s), ct 0..31
// composed    source 12800x1x1, destination 409600x1x1 = 12800x1x1 x 32 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination view is the whole of what its views
// hold -- one slot per (piece, owner) -- so a core says which slot it writes:
//     in = 0
//     mb = 0
//     y = 0
//     replica = (tid // 1) % 32, at 12800 per slot on dimension 0
// Which cores take part is not stated: 32 of the 32 address a slot they
// own, and the slot's own ct_id is what says so. See README.md for the mapping
// and the provenance.

#src0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src1 = affine_set<(d0, d1, d2) : (d0 - 512 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src2 = affine_set<(d0, d1, d2) : (d0 - 5120 >= 0, -d0 + 5631 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src3 = affine_set<(d0, d1, d2) : (d0 - 5632 >= 0, -d0 + 6143 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src4 = affine_set<(d0, d1, d2) : (d0 - 6144 >= 0, -d0 + 6655 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src5 = affine_set<(d0, d1, d2) : (d0 - 6656 >= 0, -d0 + 7167 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src6 = affine_set<(d0, d1, d2) : (d0 - 7168 >= 0, -d0 + 7679 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src7 = affine_set<(d0, d1, d2) : (d0 - 7680 >= 0, -d0 + 8191 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src8 = affine_set<(d0, d1, d2) : (d0 - 8192 >= 0, -d0 + 8703 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src9 = affine_set<(d0, d1, d2) : (d0 - 8704 >= 0, -d0 + 9215 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src10 = affine_set<(d0, d1, d2) : (d0 - 9216 >= 0, -d0 + 9727 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src11 = affine_set<(d0, d1, d2) : (d0 - 9728 >= 0, -d0 + 10239 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src12 = affine_set<(d0, d1, d2) : (d0 - 1024 >= 0, -d0 + 1535 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src13 = affine_set<(d0, d1, d2) : (d0 - 10240 >= 0, -d0 + 10751 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src14 = affine_set<(d0, d1, d2) : (d0 - 10752 >= 0, -d0 + 11263 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src15 = affine_set<(d0, d1, d2) : (d0 - 11264 >= 0, -d0 + 11775 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src16 = affine_set<(d0, d1, d2) : (d0 - 11776 >= 0, -d0 + 12287 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src17 = affine_set<(d0, d1, d2) : (d0 - 12288 >= 0, -d0 + 12799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src18 = affine_set<(d0, d1, d2) : (d0 - 1536 >= 0, -d0 + 2047 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src19 = affine_set<(d0, d1, d2) : (d0 - 2048 >= 0, -d0 + 2559 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src20 = affine_set<(d0, d1, d2) : (d0 - 2560 >= 0, -d0 + 3071 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src21 = affine_set<(d0, d1, d2) : (d0 - 3072 >= 0, -d0 + 3583 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src22 = affine_set<(d0, d1, d2) : (d0 - 3584 >= 0, -d0 + 4095 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src23 = affine_set<(d0, d1, d2) : (d0 - 4096 >= 0, -d0 + 4607 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#src24 = affine_set<(d0, d1, d2) : (d0 - 4608 >= 0, -d0 + 5119 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 12799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c001_decode_batchmatmulv2_input0(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c12800 = arith.constant 12800 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src18, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s19 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src19, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s20 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src20, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s21 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src21, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s22 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src22, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s23 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src23, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s24 = ktdp.construct_memory_view %src, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #src24, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<12800x1x1xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views with the SAME coordinate_set and different ct_id --
  // which is what replication is, stated rather than left to the launch. The
  // composed type counts those views, so the slots are distinct coordinates
  // and no coordinate has two writers.
  %d0 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>) : memref<409600x1x1xf16>

  // Which of the slots is this core's: its position in its piece's owner
  // list, stacked on dimension 0 of the composed destination.
  %i_rep = arith.remsi %tid, %c32 : index
  %o_rep = arith.muli %i_rep, %c12800 : index

  // The movement, and no control flow: one load of a coordinate REGION --
  // which source partitions that touches is the composed view's to resolve,
  // and a region may span several -- and one store into a named slot of the
  // destination, whose ct_id is what says whether this core owns it.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<12800x1x1xf16> -> !ktdp.access_tile<12800x1x1xindex>
  %val = ktdp.load %rt : <12800x1x1xindex> -> tensor<12800x1x1xf16>
  %wt = ktdp.construct_access_tile %to[%o_rep, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<409600x1x1xf16> -> !ktdp.access_tile<12800x1x1xindex>
  ktdp.store %val, %wt : tensor<12800x1x1xf16>, <12800x1x1xindex>

  return
}
