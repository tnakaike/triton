// C014 -- decode, Mul input1, consumer mul_138-mul_1
// grouped_all_gather_with_replication
// tensor mul_138-split_2-Ds0_out, rank 6 (i, j, mb, out, x, y), extents {i: 1, j: 1, mb: 2, out: 64, x: 1, y: 1}, f16
//
// source       2 view(s) = 2 piece(s) x 1 owner, ct 0, 16, ... 16 (stride 16)
// destination  8 view(s) = 1 piece(s) x 8 owner(s), ct 0, 4, ... 28 (stride 4)
// composed    source 1x1x2x64x1x1, destination 8x1x2x64x1x1 = 1x1x2x64x1x1 x 8 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination is a global view of disjoint
// slots, one per (piece, owner), so a core says which slot it writes:
//     i = 0
//     j = 0
//     mb = 0
//     out = 0
//     x = 0
//     y = 0
//     replica = (tid // 4) % 8, at 1 per slot on dimension 0
// 8 of the 32 cores own a slot, and an scf.if on the tile id keeps the
// others from storing. See README.md for the mapping and the provenance.

#src0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 - 1 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst2 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 - 2 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst3 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 - 3 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst4 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 - 4 >= 0, -d0 + 4 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst5 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 - 5 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst6 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 - 6 >= 0, -d0 + 6 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#dst7 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 - 7 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

func.func @c014_decode_mul_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
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
  // is that many views, one per owner, each a distinct SLOT of the composed
  // view: the slots of one piece stack on dimension 0, so their coordinate
  // sets are disjoint and no coordinate has two writers. Coordinates are the
  // composed view's own, starting at 0.
  %d0 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7
      : memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x1x2x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>) : memref<8x1x2x64x1x1xf16>

  // Which of the slots is this core's: its position in its piece's owner
  // list, stacked on dimension 0 of the composed destination.
  %q_rep = arith.divsi %tid, %c4 : index
  %i_rep = arith.remsi %q_rep, %c8 : index

  // The movement: one load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, and a region may span
  // several -- and one store into this core's own slot of the destination.
  %r = arith.remsi %tid, %c4 : index
  %part = arith.cmpi eq, %r, %c0 : index
  // Not every core owns a slot. The function is SPMD, so without the guard a
  // core that owns none would store into another core's slot: a remote write.
  // The load is guarded with the store, since it IS the transfer.
  scf.if %part {
    %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0, %c0, %c0, %c0]
        {access_tile_order = #ord0, access_tile_set = #dst0}
        : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
    %val = ktdp.load %rt : <1x1x2x64x1x1xindex> -> tensor<1x1x2x64x1x1xf16>
    %wt = ktdp.construct_access_tile %to[%i_rep, %c0, %c0, %c0, %c0, %c0]
        {access_tile_order = #ord0, access_tile_set = #dst0}
        : memref<8x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
    ktdp.store %val, %wt : tensor<1x1x2x64x1x1xf16>, <1x1x2x64x1x1xindex>
  }

  return
}
