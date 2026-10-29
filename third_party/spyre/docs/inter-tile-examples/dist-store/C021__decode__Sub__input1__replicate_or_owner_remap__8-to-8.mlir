// C021 -- decode, Sub input1, consumer _safe_softmax-Sub
// replicate_or_owner_remap
// tensor _safe_softmax-Max_out, rank 5 (x, mb, i, out, out%), extents {x: 1, mb: 32, i: 1, out: 1, out%: 64}, f16
//
// source       8 view(s) = 8 piece(s) x 1 owner, ct 0, 4, ... 28 (stride 4)
// destination 32 view(s) = 8 piece(s) x 4 owner(s), ct 0..31
// composed    source 1x32x1x1x64, destination 4x32x1x1x64 = 1x32x1x1x64 x 4 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination is a global view of disjoint
// slots, one per (piece, owner), so a core says which slot it writes:
//     x = 0
//     mb = ((tid // 1) % 8) * 4
//     i = 0
//     out = 0
//     out% = 0
//     replica = (tid // 8) % 4, at 1 per slot on dimension 0
// Every core owns a slot. See README.md for the mapping and the provenance.

#src0 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 - 4 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#src2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 - 8 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#src3 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 - 12 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#src4 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 - 16 >= 0, -d1 + 19 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#src5 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 - 20 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#src6 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 - 24 >= 0, -d1 + 27 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#src7 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 >= 0, d1 - 28 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst1 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst2 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst3 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 - 4 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst4 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 - 4 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst5 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 - 4 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst6 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 - 8 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst7 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 - 8 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst8 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 - 8 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst9 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 - 12 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst10 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 - 12 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst11 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 - 12 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst12 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 - 16 >= 0, -d1 + 19 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst13 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 - 16 >= 0, -d1 + 19 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst14 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 - 16 >= 0, -d1 + 19 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst15 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 - 20 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst16 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 - 20 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst17 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 - 20 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst18 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 - 24 >= 0, -d1 + 27 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst19 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 - 24 >= 0, -d1 + 27 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst20 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 - 24 >= 0, -d1 + 27 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst21 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 - 28 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst22 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 - 28 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#dst23 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 - 28 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 63 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @c021_decode_sub_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s2 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s3 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s4 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s5 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s6 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s7 = ktdp.construct_memory_view %src, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>) : memref<1x32x1x1x64xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views, one per owner, each a distinct SLOT of the composed
  // view: the slots of one piece stack on dimension 0, so their coordinate
  // sets are disjoint and no coordinate has two writers. Coordinates are the
  // composed view's own, starting at 0.
  %d0 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst8, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst9, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst10, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst11, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst12, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst14, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst15, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst16, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst17, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst18, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst19, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst20, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst21, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst22, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [1, 4, 1, 1, 64], strides: [256, 64, 64, 64, 1]
      {coordinate_set = #dst23, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 31>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<1x4x1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 31>>) : memref<4x32x1x1x64xf16>

  // Where this core's box sits: one independent field of the tile id per
  // divided dimension, exactly as in ../if-store/. No branch selects it --
  // the index is computed.
  %i_mb = arith.remsi %tid, %c8 : index
  %o_mb = arith.muli %i_mb, %c4 : index

  // Which of the slots is this core's: its position in its piece's owner
  // list, stacked on dimension 0 of the composed destination.
  %q_rep = arith.divsi %tid, %c8 : index
  %i_rep = arith.remsi %q_rep, %c4 : index

  // The movement: one load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, and a region may span
  // several -- and one store into this core's own slot of the destination.
  %rt = ktdp.construct_access_tile %from[%c0, %o_mb, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<1x32x1x1x64xf16> -> !ktdp.access_tile<1x4x1x1x64xindex>
  %val = ktdp.load %rt : <1x4x1x1x64xindex> -> tensor<1x4x1x1x64xf16>
  %wt = ktdp.construct_access_tile %to[%i_rep, %o_mb, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<4x32x1x1x64xf16> -> !ktdp.access_tile<1x4x1x1x64xindex>
  ktdp.store %val, %wt : tensor<1x4x1x1x64xf16>, <1x4x1x1x64xindex>

  return
}
