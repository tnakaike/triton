// C025 -- prefill, BatchMatMulV2 input0, consumer mm-BMM_1
// grouped_all_gather_with_replication: 16 source piece(s) -> 8 destination piece(s)
// tensor mean-LayerNormNorm_out, rank 3 (in, mb, y), extents {in: 4096, mb: 512, y: 1}, f16
//
// Written for one tile holding destination piece p0 -- 4 cores own it, so every
// one of them runs this same read. That replication IS the movement; ct 0 is shown.
//
// Generated from the curated LX-relayout SDSC package. See README.md for the
// mapping, the conventions and the provenance.

#box0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 >= 0)>
#box1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 32 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#box2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 320 >= 0, -d1 + 351 >= 0, d2 >= 0, -d2 >= 0)>
#box3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 352 >= 0, -d1 + 383 >= 0, d2 >= 0, -d2 >= 0)>
#box4 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 384 >= 0, -d1 + 415 >= 0, d2 >= 0, -d2 >= 0)>
#box5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 416 >= 0, -d1 + 447 >= 0, d2 >= 0, -d2 >= 0)>
#box6 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 448 >= 0, -d1 + 479 >= 0, d2 >= 0, -d2 >= 0)>
#box7 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 480 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 >= 0)>
#box8 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 64 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 >= 0)>
#box9 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 96 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0)>
#box10 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 128 >= 0, -d1 + 159 >= 0, d2 >= 0, -d2 >= 0)>
#box11 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 160 >= 0, -d1 + 191 >= 0, d2 >= 0, -d2 >= 0)>
#box12 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 192 >= 0, -d1 + 223 >= 0, d2 >= 0, -d2 >= 0)>
#box13 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 224 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 >= 0)>
#box14 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 256 >= 0, -d1 + 287 >= 0, d2 >= 0, -d2 >= 0)>
#box15 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 - 288 >= 0, -d1 + 319 >= 0, d2 >= 0, -d2 >= 0)>
#blk0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 4095 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c025_prefill_batchmatmulv2_input0(%off: index, %land: index) {
  %c0 = arith.constant 0 : index

  // Phase 1 -- one view per source partition. They differ ONLY in
  // coordinate_set and ct_id; offset, sizes and strides are identical.
  %s0 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box1, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s2 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box2, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s3 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box3, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s4 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box4, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s5 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box5, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s6 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box6, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s7 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box7, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s8 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box8, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s9 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box9, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s10 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box10, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s11 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box11, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s12 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box12, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s13 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box13, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s14 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box14, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s15 = ktdp.construct_memory_view %off, sizes: [4096, 32, 1], strides: [32, 1, 1]
      {coordinate_set = #box15, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>

  // ... composed into one distributed view over the whole tensor. It moves
  // nothing: it maps a global coordinate to the partition holding it.
  %whole = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15
      : memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<4096x32x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>) : memref<4096x512x1xf16>

  // Phase 2 -- name the coordinates this tile consumes. Still nothing moved:
  // both view ops are Pure, so a core may name another core's scratchpad.
  %tile = ktdp.construct_access_tile %whole[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<4096x512x1xf16> -> !ktdp.access_tile<4096x64x1xindex>

  // Phase 3 -- the load IS the transfer, and the store is the landing: a
  // received tile must be resident before a compute unit can read it.
  %got = ktdp.load %tile : <4096x64x1xindex> -> tensor<4096x64x1xf16>
  %lv = ktdp.construct_memory_view %land, sizes: [4096, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<4096x64x1xf16>
  %lt = ktdp.construct_access_tile %lv[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<4096x64x1xf16> -> !ktdp.access_tile<4096x64x1xindex>
  ktdp.store %got, %lt : tensor<4096x64x1xf16>, <4096x64x1xindex>

  return
}
