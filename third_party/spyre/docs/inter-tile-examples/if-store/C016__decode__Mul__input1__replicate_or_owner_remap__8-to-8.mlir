// C016 -- decode, Mul input1, consumer _safe_softmax_1-Mul
// replicate_or_owner_remap
// tensor _safe_softmax_1-Reciprocal_out, rank 4 (i, mb, out, x), extents {i: 1, mb: 32, out: 64, x: 1}, f16
//
// source       8 view(s) = 8 piece(s) x 1 owner, ct 0, 4, ... 28 (stride 4)
// destination 8 piece(s) x 4 owner(s) -- NOT composed. Each core lands its own.
//
// if-store: the source is composed, the destination is this core's own
// scratchpad, and the box a core lands comes from the tile id:
//     i = 0
//     mb = ((tid // 1) % 8) * 4
//     out = 0
//     x = 0
// The counterpart spelling is ../dist-store/, which composes both sides.

#src0 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 4 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#src2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 8 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#src3 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 12 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#src4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 16 >= 0, -d1 + 19 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#src5 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 20 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#src6 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 24 >= 0, -d1 + 27 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#src7 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 28 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @c016_decode_mul_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id. Identical to the dist-store form.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s2 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s3 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s4 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s5 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s6 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s7 = ktdp.construct_memory_view %src, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7
      : memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x4x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>) : memref<1x32x64x1xf16>

  // The landing: the piece, in THIS core's scratchpad. No ct_id, because a
  // core writes only its own; and the coordinate_set is a range over the
  // piece rather than a box in the tensor, because the view is local. Both
  // make it identical on every core, so it is hoisted with its access tile.
  %land = ktdp.construct_memory_view %dst, sizes: [1, 4, 64, 1], strides: [256, 64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x4x64x1xf16>
  %wt = ktdp.construct_access_tile %land[%c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<1x4x64x1xf16> -> !ktdp.access_tile<1x4x64x1xindex>

  // Where in the tensor this core's box starts: one independent field of the
  // tile id per divided dimension. No branch, because nothing is selected --
  // the index is computed.
  %i_mb = arith.remsi %tid, %c8 : index
  %o_mb = arith.muli %i_mb, %c4 : index

  // The movement. One load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, exactly as in dist-store,
  // and a region may span several of them.
  %rt = ktdp.construct_access_tile %from[%c0, %o_mb, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<1x32x64x1xf16> -> !ktdp.access_tile<1x4x64x1xindex>
  %val = ktdp.load %rt : <1x4x64x1xindex> -> tensor<1x4x64x1xf16>
  ktdp.store %val, %wt : tensor<1x4x64x1xf16>, <1x4x64x1xindex>

  return
}
