// C033 -- prefill, Mul input1, consumer mul_14-mul_1
// grouped_all_gather_with_replication
// tensor mul_14-split_2-Ds0_out, rank 6 (i, j, mb, out, x, y), extents {i: 1, j: 1, mb: 2, out: 64, x: 1, y: 512}, f16
//
// source      16 view(s) = 16 piece(s) x 1 owner, ct 0, 2, ... 30 (stride 2)
// destination 32 view(s) = 8 piece(s) x 4 owner(s), ct 0..31
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. See README.md for the mapping and the provenance.

#src0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 + 31 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 32 >= 0, -d5 + 63 >= 0)>
#src2 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 320 >= 0, -d5 + 351 >= 0)>
#src3 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 352 >= 0, -d5 + 383 >= 0)>
#src4 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 384 >= 0, -d5 + 415 >= 0)>
#src5 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 416 >= 0, -d5 + 447 >= 0)>
#src6 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 448 >= 0, -d5 + 479 >= 0)>
#src7 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 480 >= 0, -d5 + 511 >= 0)>
#src8 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 64 >= 0, -d5 + 95 >= 0)>
#src9 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 96 >= 0, -d5 + 127 >= 0)>
#src10 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 128 >= 0, -d5 + 159 >= 0)>
#src11 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 160 >= 0, -d5 + 191 >= 0)>
#src12 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 192 >= 0, -d5 + 223 >= 0)>
#src13 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 224 >= 0, -d5 + 255 >= 0)>
#src14 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 256 >= 0, -d5 + 287 >= 0)>
#src15 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 288 >= 0, -d5 + 319 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 + 63 >= 0)>
#dst1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 64 >= 0, -d5 + 127 >= 0)>
#dst2 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 128 >= 0, -d5 + 191 >= 0)>
#dst3 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 192 >= 0, -d5 + 255 >= 0)>
#dst4 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 256 >= 0, -d5 + 319 >= 0)>
#dst5 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 320 >= 0, -d5 + 383 >= 0)>
#dst6 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 384 >= 0, -d5 + 447 >= 0)>
#dst7 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 - 448 >= 0, -d5 + 511 >= 0)>
#all0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 + 511 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

func.func @c033_prefill_mul_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s2 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s3 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s4 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s5 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s6 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s7 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s8 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s9 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s10 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s11 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s12 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s13 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s14 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s15 = ktdp.construct_memory_view %src, sizes: [1, 1, 2, 64, 1, 32], strides: [4096, 4096, 2048, 32, 32, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 18>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15
      : memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x1x2x64x1x32xf16, #ktdp.memory_space<ct_local, ct_id = 18>>) : memref<1x1x2x64x1x512xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views with the SAME coordinate_set and different ct_id --
  // which is what replication is, stated rather than left to the launch.
  %d0 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 64], strides: [8192, 8192, 4096, 64, 64, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 31>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<1x1x2x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 31>>) : memref<1x1x2x64x1x512xf16>

  // The movement. Whole-tensor on both sides: a relayout is a view-to-view
  // copy, and with both distributions named there is no per-tile share left
  // to anchor. Which side does the transferring is a lowering's choice.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #all0}
      : memref<1x1x2x64x1x512xf16> -> !ktdp.access_tile<1x1x2x64x1x512xindex>
  %val = ktdp.load %rt : <1x1x2x64x1x512xindex> -> tensor<1x1x2x64x1x512xf16>
  %wt = ktdp.construct_access_tile %to[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #all0}
      : memref<1x1x2x64x1x512xf16> -> !ktdp.access_tile<1x1x2x64x1x512xindex>
  ktdp.store %val, %wt : tensor<1x1x2x64x1x512xf16>, <1x1x2x64x1x512xindex>

  return
}
