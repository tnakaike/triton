// C015 -- decode, Mul input1, consumer mul_137-mul_1
// grouped_all_gather_with_replication
// tensor mul_137-split_2-Ds0_out, rank 6 (i, j, mb, out, x, y), extents {i: 1, j: 1, mb: 2, out: 64, x: 1, y: 1}, f16
//
// source       2 view(s) = 2 piece(s) x 1 owner, ct 0, 16, ... 16 (stride 16)
// destination 32 view(s) = 1 piece(s) x 32 owner(s), ct 0..31
// composed    source 1x1x2x64x1x1, destination 32x1x2x64x1x1 = 1x1x2x64x1x1 x 32 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination view is the whole of what its views
// hold -- one slot per (piece, owner) -- so a core says which slot it writes:
//     i = 0
//     j = 0
//     mb = 0
//     out = 0
//     x = 0
//     y = 0
//     replica = (tid // 1) % 32, at 1 per slot on dimension 0
// Which cores take part is not stated: 32 of the 32 address a slot they
// own, and the slot's own ct_id is what says so. See README.md for the mapping
// and the provenance.

#src0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

func.func @c015_decode_mul_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 1, 1, 64, 1, 1], strides: [64, 64, 64, 1, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 1, 1, 64, 1, 1], strides: [64, 64, 64, 1, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>) : memref<1x1x2x64x1x1xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views with the SAME coordinate_set and different ct_id --
  // which is what replication is, stated rather than left to the launch. The
  // composed type counts those views, so the slots are distinct coordinates
  // and no coordinate has two writers.
  %d0 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>) : memref<32x1x2x64x1x1xf16>

  // Which of the slots is this core's: its position in its piece's owner
  // list, stacked on dimension 0 of the composed destination.
  %i_rep = arith.remsi %tid, %c32 : index

  // The movement, and no control flow: one load of a coordinate REGION --
  // which source partitions that touches is the composed view's to resolve,
  // and a region may span several -- and one store into a named slot of the
  // destination, whose ct_id is what says whether this core owns it.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
  %val = ktdp.load %rt : <1x1x2x64x1x1xindex> -> tensor<1x1x2x64x1x1xf16>
  %wt = ktdp.construct_access_tile %to[%i_rep, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<32x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
  ktdp.store %val, %wt : tensor<1x1x2x64x1x1xf16>, <1x1x2x64x1x1xindex>

  return
}
