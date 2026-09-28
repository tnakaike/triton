// C015 -- decode, Mul input1, consumer mul_137-mul_1
// grouped_all_gather_with_replication: 2 source piece(s) -> 1 destination piece(s)
// tensor mul_137-split_2-Ds0_out, rank 6 (i, j, mb, out, x, y), extents {i: 1, j: 1, mb: 2, out: 64, x: 1, y: 1}, f16
//
// Written for one tile holding destination piece p0 -- 32 cores own it, so every
// one of them runs this same read. That replication IS the movement; ct 0 is shown.
//
// Generated from the curated LX-relayout SDSC package. See README.md for the
// mapping, the conventions and the provenance.

#box0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#box1 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 - 1 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#blk0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 >= 0, d5 >= 0, -d5 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>

func.func @c015_decode_mul_input1(%off: index, %land: index) {
  %c0 = arith.constant 0 : index

  // Phase 1 -- one view per source partition. They differ ONLY in
  // coordinate_set and ct_id; offset, sizes and strides are identical.
  %s0 = ktdp.construct_memory_view %off, sizes: [1, 1, 1, 64, 1, 1], strides: [64, 64, 64, 1, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %off, sizes: [1, 1, 1, 64, 1, 1], strides: [64, 64, 64, 1, 1, 1]
      {coordinate_set = #box1, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>

  // ... composed into one distributed view over the whole tensor. It moves
  // nothing: it maps a global coordinate to the partition holding it.
  %whole = ktdp.construct_distributed_memory_view (%s0, %s1
      : memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x1x64x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>) : memref<1x1x2x64x1x1xf16>

  // Phase 2 -- name the coordinates this tile consumes. Still nothing moved:
  // both view ops are Pure, so a core may name another core's scratchpad.
  %tile = ktdp.construct_access_tile %whole[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>

  // Phase 3 -- the load IS the transfer, and the store is the landing: a
  // received tile must be resident before a compute unit can read it.
  %got = ktdp.load %tile : <1x1x2x64x1x1xindex> -> tensor<1x1x2x64x1x1xf16>
  %lv = ktdp.construct_memory_view %land, sizes: [1, 1, 2, 64, 1, 1], strides: [128, 128, 64, 1, 1, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x1x2x64x1x1xf16>
  %lt = ktdp.construct_access_tile %lv[%c0, %c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x1x2x64x1x1xf16> -> !ktdp.access_tile<1x1x2x64x1x1xindex>
  ktdp.store %got, %lt : tensor<1x1x2x64x1x1xf16>, <1x1x2x64x1x1xindex>

  return
}
