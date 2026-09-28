// C011 -- decode, LayerNormNorm input1, consumer mean-LayerNormNorm
// replicate_or_owner_remap: 1 source piece(s) -> 1 destination piece(s)
// tensor mean-Exx2_out, rank 3 (mb, out, y), extents {mb: 1, out: 64, y: 1}, f16
//
// Written for one tile holding destination piece p0 -- 32 cores own it, so every
// one of them runs this same read. That replication IS the movement; ct 0 is shown.
//
// Generated from the curated LX-relayout SDSC package. See README.md for the
// mapping, the conventions and the provenance.

#box0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c011_decode_layernormnorm_input1(%off: index, %land: index) {
  %c0 = arith.constant 0 : index

  // Phase 1 -- one view per source partition. They differ ONLY in
  // coordinate_set and ct_id; offset, sizes and strides are identical.
  %s0 = ktdp.construct_memory_view %off, sizes: [1, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>

  // ... composed into one distributed view over the whole tensor. It moves
  // nothing: it maps a global coordinate to the partition holding it.
  %whole = ktdp.construct_distributed_memory_view (%s0
      : memref<1x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>) : memref<1x64x1xf16>

  // Phase 2 -- name the coordinates this tile consumes. Still nothing moved:
  // both view ops are Pure, so a core may name another core's scratchpad.
  %tile = ktdp.construct_access_tile %whole[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #box0}
      : memref<1x64x1xf16> -> !ktdp.access_tile<1x64x1xindex>

  // Phase 3 -- the load IS the transfer, and the store is the landing: a
  // received tile must be resident before a compute unit can read it.
  %got = ktdp.load %tile : <1x64x1xindex> -> tensor<1x64x1xf16>
  %lv = ktdp.construct_memory_view %land, sizes: [1, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x64x1xf16>
  %lt = ktdp.construct_access_tile %lv[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #box0}
      : memref<1x64x1xf16> -> !ktdp.access_tile<1x64x1xindex>
  ktdp.store %got, %lt : tensor<1x64x1xf16>, <1x64x1xindex>

  return
}
