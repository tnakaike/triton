// C001 -- decode, BatchMatMulV2 input0, consumer mm_13-BMM_1
// grouped_all_gather_with_replication: 25 source piece(s) -> 1 destination piece(s)
// tensor mul_263_out, rank 3 (in, mb, y), extents {in: 12800, mb: 1, y: 1}, f16
//
// Written for one tile holding destination piece p0 -- 32 cores own it, so every
// one of them runs this same read. That replication IS the movement; ct 0 is shown.
//
// Generated from the curated LX-relayout SDSC package. See README.md for the
// mapping, the conventions and the provenance.

#box0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box1 = affine_set<(d0, d1, d2) : (d0 - 512 >= 0, -d0 + 1023 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box2 = affine_set<(d0, d1, d2) : (d0 - 5120 >= 0, -d0 + 5631 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box3 = affine_set<(d0, d1, d2) : (d0 - 5632 >= 0, -d0 + 6143 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box4 = affine_set<(d0, d1, d2) : (d0 - 6144 >= 0, -d0 + 6655 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box5 = affine_set<(d0, d1, d2) : (d0 - 6656 >= 0, -d0 + 7167 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box6 = affine_set<(d0, d1, d2) : (d0 - 7168 >= 0, -d0 + 7679 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box7 = affine_set<(d0, d1, d2) : (d0 - 7680 >= 0, -d0 + 8191 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box8 = affine_set<(d0, d1, d2) : (d0 - 8192 >= 0, -d0 + 8703 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box9 = affine_set<(d0, d1, d2) : (d0 - 8704 >= 0, -d0 + 9215 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box10 = affine_set<(d0, d1, d2) : (d0 - 9216 >= 0, -d0 + 9727 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box11 = affine_set<(d0, d1, d2) : (d0 - 9728 >= 0, -d0 + 10239 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box12 = affine_set<(d0, d1, d2) : (d0 - 1024 >= 0, -d0 + 1535 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box13 = affine_set<(d0, d1, d2) : (d0 - 10240 >= 0, -d0 + 10751 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box14 = affine_set<(d0, d1, d2) : (d0 - 10752 >= 0, -d0 + 11263 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box15 = affine_set<(d0, d1, d2) : (d0 - 11264 >= 0, -d0 + 11775 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box16 = affine_set<(d0, d1, d2) : (d0 - 11776 >= 0, -d0 + 12287 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box17 = affine_set<(d0, d1, d2) : (d0 - 12288 >= 0, -d0 + 12799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box18 = affine_set<(d0, d1, d2) : (d0 - 1536 >= 0, -d0 + 2047 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box19 = affine_set<(d0, d1, d2) : (d0 - 2048 >= 0, -d0 + 2559 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box20 = affine_set<(d0, d1, d2) : (d0 - 2560 >= 0, -d0 + 3071 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box21 = affine_set<(d0, d1, d2) : (d0 - 3072 >= 0, -d0 + 3583 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box22 = affine_set<(d0, d1, d2) : (d0 - 3584 >= 0, -d0 + 4095 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box23 = affine_set<(d0, d1, d2) : (d0 - 4096 >= 0, -d0 + 4607 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#box24 = affine_set<(d0, d1, d2) : (d0 - 4608 >= 0, -d0 + 5119 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#blk0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 12799 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c001_decode_batchmatmulv2_input0(%off: index, %land: index) {
  %c0 = arith.constant 0 : index

  // Phase 1 -- one view per source partition. They differ ONLY in
  // coordinate_set and ct_id; offset, sizes and strides are identical.
  %s0 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box18, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s19 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box19, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s20 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box20, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s21 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box21, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s22 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box22, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s23 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box23, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s24 = ktdp.construct_memory_view %off, sizes: [512, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #box24, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  // ... composed into one distributed view over the whole tensor. It moves
  // nothing: it maps a global coordinate to the partition holding it.
  %whole = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24
      : memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<512x1x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<12800x1x1xf16>

  // Phase 2 -- name the coordinates this tile consumes. Still nothing moved:
  // both view ops are Pure, so a core may name another core's scratchpad.
  %tile = ktdp.construct_access_tile %whole[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<12800x1x1xf16> -> !ktdp.access_tile<12800x1x1xindex>

  // Phase 3 -- the load IS the transfer, and the store is the landing: a
  // received tile must be resident before a compute unit can read it.
  %got = ktdp.load %tile : <12800x1x1xindex> -> tensor<12800x1x1xf16>
  %lv = ktdp.construct_memory_view %land, sizes: [12800, 1, 1], strides: [1, 1, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<12800x1x1xf16>
  %lt = ktdp.construct_access_tile %lv[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<12800x1x1xf16> -> !ktdp.access_tile<12800x1x1xindex>
  ktdp.store %got, %lt : tensor<12800x1x1xf16>, <12800x1x1xindex>

  return
}
