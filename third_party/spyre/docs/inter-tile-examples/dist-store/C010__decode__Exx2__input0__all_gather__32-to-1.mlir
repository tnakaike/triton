// C010 -- decode, Exx2 input0, consumer mean-Exx2
// all_gather
// tensor mul_out, rank 4 (y, out, mb, out%), extents {y: 1, out: 64, mb: 1, out%: 64}, f16
//
// source      32 view(s) = 32 piece(s) x 1 owner, ct 0..31
// destination  1 view(s) = 1 piece(s) x 1 owner(s), ct 0
// composed    source 1x64x1x64, destination 1x64x1x64 = 1x64x1x64 x 1 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination is a global view of disjoint
// slots, one per (piece, owner), so a core says which slot it writes:
//     y = 0
//     out = 0
//     mb = 0
//     out% = 0
// 1 of the 32 cores own a slot, and an scf.if on the tile id keeps the
// others from storing. See README.md for the mapping and the provenance.

#src0 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 2 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 20 >= 0, -d1 + 21 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src3 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 22 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 24 >= 0, -d1 + 25 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src5 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 26 >= 0, -d1 + 27 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src6 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 28 >= 0, -d1 + 29 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src7 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 30 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src8 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 32 >= 0, -d1 + 33 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src9 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 34 >= 0, -d1 + 35 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src10 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 36 >= 0, -d1 + 37 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src11 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 38 >= 0, -d1 + 39 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src12 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 4 >= 0, -d1 + 5 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src13 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 40 >= 0, -d1 + 41 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src14 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 42 >= 0, -d1 + 43 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src15 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 44 >= 0, -d1 + 45 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src16 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 46 >= 0, -d1 + 47 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src17 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 48 >= 0, -d1 + 49 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src18 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 50 >= 0, -d1 + 51 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src19 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 52 >= 0, -d1 + 53 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src20 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 54 >= 0, -d1 + 55 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src21 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 56 >= 0, -d1 + 57 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src22 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 58 >= 0, -d1 + 59 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src23 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 6 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src24 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 60 >= 0, -d1 + 61 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src25 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 62 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src26 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 8 >= 0, -d1 + 9 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src27 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 10 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src28 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 12 >= 0, -d1 + 13 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src29 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 14 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src30 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 16 >= 0, -d1 + 17 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src31 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 18 >= 0, -d1 + 19 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @c010_decode_exx2_input0(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %s19 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s20 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %s21 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s22 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %s23 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s24 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s25 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %s26 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s27 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s28 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s29 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s30 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s31 = ktdp.construct_memory_view %src, sizes: [1, 2, 1, 64], strides: [128, 64, 64, 1]
      {coordinate_set = #src31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24, %s25, %s26, %s27, %s28, %s29, %s30, %s31
      : memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x2x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<1x64x1x64xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views, one per owner, each a distinct SLOT of the composed
  // view: the slots of one piece stack on dimension 0, so their coordinate
  // sets are disjoint and no coordinate has two writers. Coordinates are the
  // composed view's own, starting at 0.
  %d0 = ktdp.construct_memory_view %dst, sizes: [1, 64, 1, 64], strides: [4096, 64, 64, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>

  %to = ktdp.construct_distributed_memory_view (%d0
      : memref<1x64x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>) : memref<1x64x1x64xf16>

  // The movement: one load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, and a region may span
  // several -- and one store into this core's own slot of the destination.
  %part = arith.cmpi eq, %tid, %c0 : index
  // Not every core owns a slot. The function is SPMD, so without the guard a
  // core that owns none would store into another core's slot: a remote write.
  // The load is guarded with the store, since it IS the transfer.
  scf.if %part {
    %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0, %c0]
        {access_tile_order = #ord0, access_tile_set = #dst0}
        : memref<1x64x1x64xf16> -> !ktdp.access_tile<1x64x1x64xindex>
    %val = ktdp.load %rt : <1x64x1x64xindex> -> tensor<1x64x1x64xf16>
    %wt = ktdp.construct_access_tile %to[%c0, %c0, %c0, %c0]
        {access_tile_order = #ord0, access_tile_set = #dst0}
        : memref<1x64x1x64xf16> -> !ktdp.access_tile<1x64x1x64xindex>
    ktdp.store %val, %wt : tensor<1x64x1x64xf16>, <1x64x1x64xindex>
  }

  return
}
