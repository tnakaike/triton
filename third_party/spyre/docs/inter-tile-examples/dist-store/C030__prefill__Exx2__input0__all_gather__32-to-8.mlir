// C030 -- prefill, Exx2 input0, consumer mean_1-Exx2
// all_gather
// tensor add_3_out, rank 3 (mb, out, y), extents {mb: 512, out: 4096, y: 1}, f16
//
// source      32 view(s) = 32 piece(s) x 1 owner, ct 0..31
// destination  8 view(s) = 8 piece(s) x 1 owner(s), ct 0, 4, ... 28 (stride 4)
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. See README.md for the mapping and the provenance.

#src0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src2 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 191 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src3 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 191 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src4 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src5 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 255 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src6 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 255 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src7 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 255 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src8 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 319 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src9 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 319 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src10 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 319 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src11 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 319 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src12 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src13 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src14 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 383 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src15 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 383 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src16 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 383 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src17 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 447 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src18 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 447 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src19 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 447 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src20 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 447 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src21 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src22 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 511 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src23 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src24 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 511 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src25 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 511 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src26 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src27 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src28 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 - 2048 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src29 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 - 3072 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src30 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src31 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 191 >= 0, d1 - 1024 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#dst0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#dst1 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#dst2 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#dst3 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#dst4 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 319 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#dst5 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#dst6 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 447 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#dst7 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#all0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c030_prefill_exx2_input0(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %s19 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s20 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %s21 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s22 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %s23 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s24 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s25 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %s26 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s27 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s28 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s29 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s30 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s31 = ktdp.construct_memory_view %src, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #src31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24, %s25, %s26, %s27, %s28, %s29, %s30, %s31
      : memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<64x1024x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<512x4096x1xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views with the SAME coordinate_set and different ct_id --
  // which is what replication is, stated rather than left to the launch.
  %d0 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [64, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7
      : memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<64x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>) : memref<512x4096x1xf16>

  // The movement. Whole-tensor on both sides: a relayout is a view-to-view
  // copy, and with both distributions named there is no per-tile share left
  // to anchor. Which side does the transferring is a lowering's choice.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #all0}
      : memref<512x4096x1xf16> -> !ktdp.access_tile<512x4096x1xindex>
  %val = ktdp.load %rt : <512x4096x1xindex> -> tensor<512x4096x1xf16>
  %wt = ktdp.construct_access_tile %to[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #all0}
      : memref<512x4096x1xf16> -> !ktdp.access_tile<512x4096x1xindex>
  ktdp.store %val, %wt : tensor<512x4096x1xf16>, <512x4096x1xindex>

  return
}
