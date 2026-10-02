// C014 -- decode, Mul input1, consumer mul_138-mul_1
// grouped_all_gather_with_replication
// tensor mul_138-split_2-Ds0_out, rank 6 (i, j, mb, out, x, y), extents {i: 1, j: 1, mb: 2, out: 64, x: 1, y: 1}, f16
//
// source       2 view(s) = 2 piece(s) x 1 owner, ct 0, 16, ... 16 (stride 16)
// destination  8 view(s) = 1 piece(s) x 8 owner(s), ct 0, 4, ... 28 (stride 4)
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. See README.md for the mapping and the provenance.

#src0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

func.func @c014_decode_mul_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index

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
  // which is what replication is, stated rather than left to the launch.
  %d0 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>) : memref<1x1x2x64x1x1xf16>

  // The movement. Whole-tensor on both sides: a relayout is a view-to-view
  // copy, and with both distributions named there is no per-tile share left
  // to anchor. Which side does the transferring is a lowering's choice.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
  %val = ktdp.load %rt : <1x1x2x64x1x1xindex> -> tensor<1x1x2x64x1x1xf16>
  %wt = ktdp.construct_access_tile %to[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
  ktdp.store %val, %wt : tensor<1x1x2x64x1x1xf16>, <1x1x2x64x1x1xindex>

  return
}
