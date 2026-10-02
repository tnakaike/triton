// C014 -- decode, Mul input1, consumer mul_138-mul_1
// grouped_all_gather_with_replication
// tensor mul_138-split_2-Ds0_out, rank 6 (i, j, mb, out, x, y), extents {i: 1, j: 1, mb: 2, out: 64, x: 1, y: 1}, f16
//
// source       2 view(s) = 2 piece(s) x 1 owner, ct 0, 16, ... 16 (stride 16)
// destination 1 piece(s) x 8 owner(s) -- NOT composed. Each core lands its own.
//
// if-store: the source is composed, the destination is this core's own
// scratchpad, and the box a core lands comes from the tile id:
//     i = 0
//     j = 0
//     mb = 0
//     out = 0
//     x = 0
//     y = 0
// The counterpart spelling is ../dist-store/, which composes both sides.

#src0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#blk0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

func.func @c014_decode_mul_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id. Identical to the dist-store form.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 1, 1, 64, 1, 1], strides: [64, 64, 64, 1, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 1, 1, 64, 1, 1], strides: [64, 64, 64, 1, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>) : memref<1x1x2x64x1x1xf16>

  // The landing: the piece, in THIS core's scratchpad. No ct_id, because a
  // core writes only its own; and the coordinate_set is a range over the
  // piece rather than a box in the tensor, because the view is local. Both
  // make it identical on every core, so it is hoisted with its access tile.
  %land = ktdp.construct_memory_view %dst, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x1x2x64x1x1xf16>
  %wt = ktdp.construct_access_tile %land[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>

  // Where in the tensor this core's box starts: one independent field of the
  // tile id per divided dimension. No branch, because nothing is selected --
  // the index is computed.

  // The movement. One load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, exactly as in dist-store,
  // and a region may span several of them.
  %r = arith.remsi %tid, %c4 : index
  %part = arith.cmpi eq, %r, %c0 : index
  // Not every core takes part, so the load is guarded with the store: the
  // load IS the transfer, and a core that lands nothing must read nothing.
  scf.if %part {
    %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0, %c0, %c0, %c0]
        {access_tile_order = #ord0, access_tile_set = #blk0}
        : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
    %val = ktdp.load %rt : <1x1x2x64x1x1xindex> -> tensor<1x1x2x64x1x1xf16>
    ktdp.store %val, %wt : tensor<1x1x2x64x1x1xf16>, <1x1x2x64x1x1xindex>
  }

  return
}
