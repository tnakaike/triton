// C011 -- decode, LayerNormNorm input1, consumer mean-LayerNormNorm
// replicate_or_owner_remap
// tensor mean-Exx2_out, rank 3 (mb, out, y), extents {mb: 1, out: 64, y: 1}, f16
//
// source       1 view(s) = 1 piece(s) x 1 owner, ct 0
// destination 1 piece(s) x 32 owner(s) -- NOT composed. Each core lands its own.
//
// if-store: the source is composed, the destination is this core's own
// scratchpad, and the box a core lands comes from the tile id:
//     mb = 0
//     out = 0
//     y = 0
// The counterpart spelling is ../dist-store/, which composes both sides.

#src0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c011_decode_layernormnorm_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id. Identical to the dist-store form.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>

  %from = ktdp.construct_distributed_memory_view (%s0
      : memref<1x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>) : memref<1x64x1xf16>

  // The landing: the piece, in THIS core's scratchpad. No ct_id, because a
  // core writes only its own; and the coordinate_set is a range over the
  // piece rather than a box in the tensor, because the view is local. Both
  // make it identical on every core, so it is hoisted with its access tile.
  %land = ktdp.construct_memory_view %dst, sizes: [1, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x64x1xf16>
  %wt = ktdp.construct_access_tile %land[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<1x64x1xf16> -> !ktdp.access_tile<1x64x1xindex>

  // Where in the tensor this core's box starts: one independent field of the
  // tile id per divided dimension. No branch, because nothing is selected --
  // the index is computed.

  // The movement. One load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, exactly as in dist-store,
  // and a region may span several of them.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<1x64x1xf16> -> !ktdp.access_tile<1x64x1xindex>
  %val = ktdp.load %rt : <1x64x1xindex> -> tensor<1x64x1xf16>
  ktdp.store %val, %wt : tensor<1x64x1xf16>, <1x64x1xindex>

  return
}
