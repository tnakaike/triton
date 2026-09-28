// C032 -- prefill, LayerNormNorm input2, consumer mean_1-LayerNormNorm
// replicate_or_owner_remap: 8 source piece(s) -> 8 destination piece(s)
// tensor mean_1-LayerNormScale_out, rank 3 (mb, out, y), extents {mb: 512, out: 64, y: 1}, f16
//
// Written for one tile holding destination piece p0 -- 4 cores own it, so every
// one of them runs this same read. That replication IS the movement; ct 0 is shown.
//
// Generated from the curated LX-relayout SDSC package. See README.md for the
// mapping, the conventions and the provenance.

#box0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box1 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box2 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box3 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box4 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 319 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box5 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box6 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 447 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box7 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c032_prefill_layernormnorm_input2(%off: index, %land: index) {
  %c0 = arith.constant 0 : index

  // Phase 1 -- one view per source partition. They differ ONLY in
  // coordinate_set and ct_id; offset, sizes and strides are identical.
  %s0 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s2 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s3 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s4 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s5 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s6 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s7 = ktdp.construct_memory_view %off, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>

  // ... composed into one distributed view over the whole tensor. It moves
  // nothing: it maps a global coordinate to the partition holding it.
  %whole = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>) : memref<512x64x1xf16>

  // Phase 2 -- name the coordinates this tile consumes. Still nothing moved:
  // both view ops are Pure, so a core may name another core's scratchpad.
  %tile = ktdp.construct_access_tile %whole[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #box0}
      : memref<512x64x1xf16> -> !ktdp.access_tile<64x64x1xindex>

  // Phase 3 -- the load IS the transfer, and the store is the landing: a
  // received tile must be resident before a compute unit can read it.
  %got = ktdp.load %tile : <64x64x1xindex> -> tensor<64x64x1xf16>
  %lv = ktdp.construct_memory_view %land, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x64x1xf16>
  %lt = ktdp.construct_access_tile %lv[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #box0}
      : memref<64x64x1xf16> -> !ktdp.access_tile<64x64x1xindex>
  ktdp.store %got, %lt : tensor<64x64x1xf16>, <64x64x1xindex>

  return
}
