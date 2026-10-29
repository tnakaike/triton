// C001 -- decode, BatchMatMulV2 input0, consumer mm_13-BMM_1
// grouped_all_gather_with_replication
// tensor mul_263_out, rank 3 (in, mb, y), extents {in: 12800, mb: 1, y: 1}, f16
//
// source      25 view(s) = 25 piece(s) x 1 owner, ct 0..24
// destination 32 view(s) = 1 piece(s) x 32 owner(s), ct 0..31
// composed    source 12800x1x1, destination 409600x1x1 = 12800x1x1 x 32 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination is a global view of disjoint
// slots, one per (piece, owner), so a core says which slot it writes:
//     in = 0
//     mb = 0
//     y = 0
//     replica = (tid // 1) % 32, at 12800 per slot on dimension 0
// Every core owns a slot. See README.md for the mapping and the provenance.

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
#dst1 = affine_set<(d0, d1, d2) : (d0 - 12800 >= 0, -d0 + 25599 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst2 = affine_set<(d0, d1, d2) : (d0 - 25600 >= 0, -d0 + 38399 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst3 = affine_set<(d0, d1, d2) : (d0 - 38400 >= 0, -d0 + 51199 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst4 = affine_set<(d0, d1, d2) : (d0 - 51200 >= 0, -d0 + 63999 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst5 = affine_set<(d0, d1, d2) : (d0 - 64000 >= 0, -d0 + 76799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst6 = affine_set<(d0, d1, d2) : (d0 - 76800 >= 0, -d0 + 89599 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst7 = affine_set<(d0, d1, d2) : (d0 - 89600 >= 0, -d0 + 102399 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst8 = affine_set<(d0, d1, d2) : (d0 - 102400 >= 0, -d0 + 115199 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst9 = affine_set<(d0, d1, d2) : (d0 - 115200 >= 0, -d0 + 127999 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst10 = affine_set<(d0, d1, d2) : (d0 - 128000 >= 0, -d0 + 140799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst11 = affine_set<(d0, d1, d2) : (d0 - 140800 >= 0, -d0 + 153599 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst12 = affine_set<(d0, d1, d2) : (d0 - 153600 >= 0, -d0 + 166399 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst13 = affine_set<(d0, d1, d2) : (d0 - 166400 >= 0, -d0 + 179199 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst14 = affine_set<(d0, d1, d2) : (d0 - 179200 >= 0, -d0 + 191999 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst15 = affine_set<(d0, d1, d2) : (d0 - 192000 >= 0, -d0 + 204799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst16 = affine_set<(d0, d1, d2) : (d0 - 204800 >= 0, -d0 + 217599 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst17 = affine_set<(d0, d1, d2) : (d0 - 217600 >= 0, -d0 + 230399 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst18 = affine_set<(d0, d1, d2) : (d0 - 230400 >= 0, -d0 + 243199 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst19 = affine_set<(d0, d1, d2) : (d0 - 243200 >= 0, -d0 + 255999 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst20 = affine_set<(d0, d1, d2) : (d0 - 256000 >= 0, -d0 + 268799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst21 = affine_set<(d0, d1, d2) : (d0 - 268800 >= 0, -d0 + 281599 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst22 = affine_set<(d0, d1, d2) : (d0 - 281600 >= 0, -d0 + 294399 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst23 = affine_set<(d0, d1, d2) : (d0 - 294400 >= 0, -d0 + 307199 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst24 = affine_set<(d0, d1, d2) : (d0 - 307200 >= 0, -d0 + 319999 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst25 = affine_set<(d0, d1, d2) : (d0 - 320000 >= 0, -d0 + 332799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst26 = affine_set<(d0, d1, d2) : (d0 - 332800 >= 0, -d0 + 345599 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst27 = affine_set<(d0, d1, d2) : (d0 - 345600 >= 0, -d0 + 358399 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst28 = affine_set<(d0, d1, d2) : (d0 - 358400 >= 0, -d0 + 371199 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst29 = affine_set<(d0, d1, d2) : (d0 - 371200 >= 0, -d0 + 383999 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst30 = affine_set<(d0, d1, d2) : (d0 - 384000 >= 0, -d0 + 396799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#dst31 = affine_set<(d0, d1, d2) : (d0 - 396800 >= 0, -d0 + 409599 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
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
  // is that many views, one per owner, each a distinct SLOT of the composed
  // view: the slots of one piece stack on dimension 0, so their coordinate
  // sets are disjoint and no coordinate has two writers. Coordinates are the
  // composed view's own, starting at 0.
  %d0 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst8, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst9, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst10, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst11, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst12, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst13, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst14, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst15, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst16, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst17, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst18, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst19, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst20, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst21, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst22, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst23, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst24, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst25, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst26, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst27, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst28, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst29, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst30, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #dst31, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<12800x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>) : memref<409600x1x1xf16>

  // Which of the slots is this core's: its position in its piece's owner
  // list, stacked on dimension 0 of the composed destination.
  %i_rep = arith.remsi %tid, %c32 : index
  %o_rep = arith.muli %i_rep, %c12800 : index

  // The movement: one load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, and a region may span
  // several -- and one store into this core's own slot of the destination.
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
