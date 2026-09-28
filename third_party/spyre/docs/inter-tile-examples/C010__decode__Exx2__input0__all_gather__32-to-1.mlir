// C010 -- decode, Exx2 input0, consumer mean-Exx2
// all_gather: 32 source piece(s) -> 1 destination piece(s)
// tensor mul_out, rank 3 (mb, out, y), extents {mb: 1, out: 4096, y: 1}, f16
//
// Written for the tile holding destination piece p0, ct 0.
//
// Generated from the curated LX-relayout SDSC package. See README.md for the
// mapping, the conventions and the provenance.

#box0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0)>
#box1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 128 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 >= 0)>
#box2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1280 >= 0, -d1 + 1407 >= 0, d2 >= 0, -d2 >= 0)>
#box3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1408 >= 0, -d1 + 1535 >= 0, d2 >= 0, -d2 >= 0)>
#box4 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1536 >= 0, -d1 + 1663 >= 0, d2 >= 0, -d2 >= 0)>
#box5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1664 >= 0, -d1 + 1791 >= 0, d2 >= 0, -d2 >= 0)>
#box6 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1792 >= 0, -d1 + 1919 >= 0, d2 >= 0, -d2 >= 0)>
#box7 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1920 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#box8 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2048 >= 0, -d1 + 2175 >= 0, d2 >= 0, -d2 >= 0)>
#box9 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2176 >= 0, -d1 + 2303 >= 0, d2 >= 0, -d2 >= 0)>
#box10 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2304 >= 0, -d1 + 2431 >= 0, d2 >= 0, -d2 >= 0)>
#box11 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2432 >= 0, -d1 + 2559 >= 0, d2 >= 0, -d2 >= 0)>
#box12 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 256 >= 0, -d1 + 383 >= 0, d2 >= 0, -d2 >= 0)>
#box13 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2560 >= 0, -d1 + 2687 >= 0, d2 >= 0, -d2 >= 0)>
#box14 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2688 >= 0, -d1 + 2815 >= 0, d2 >= 0, -d2 >= 0)>
#box15 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2816 >= 0, -d1 + 2943 >= 0, d2 >= 0, -d2 >= 0)>
#box16 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2944 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#box17 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3072 >= 0, -d1 + 3199 >= 0, d2 >= 0, -d2 >= 0)>
#box18 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3200 >= 0, -d1 + 3327 >= 0, d2 >= 0, -d2 >= 0)>
#box19 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3328 >= 0, -d1 + 3455 >= 0, d2 >= 0, -d2 >= 0)>
#box20 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3456 >= 0, -d1 + 3583 >= 0, d2 >= 0, -d2 >= 0)>
#box21 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3584 >= 0, -d1 + 3711 >= 0, d2 >= 0, -d2 >= 0)>
#box22 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3712 >= 0, -d1 + 3839 >= 0, d2 >= 0, -d2 >= 0)>
#box23 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 384 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 >= 0)>
#box24 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3840 >= 0, -d1 + 3967 >= 0, d2 >= 0, -d2 >= 0)>
#box25 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3968 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#box26 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 512 >= 0, -d1 + 639 >= 0, d2 >= 0, -d2 >= 0)>
#box27 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 640 >= 0, -d1 + 767 >= 0, d2 >= 0, -d2 >= 0)>
#box28 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 768 >= 0, -d1 + 895 >= 0, d2 >= 0, -d2 >= 0)>
#box29 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 896 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#box30 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1024 >= 0, -d1 + 1151 >= 0, d2 >= 0, -d2 >= 0)>
#box31 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1152 >= 0, -d1 + 1279 >= 0, d2 >= 0, -d2 >= 0)>
#blk0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c010_decode_exx2_input0(%off: index, %land: index) {
  %c0 = arith.constant 0 : index

  // Phase 1 -- one view per source partition. They differ ONLY in
  // coordinate_set and ct_id; offset, sizes and strides are identical.
  %s0 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %s19 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s20 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %s21 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s22 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %s23 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s24 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s25 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %s26 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s27 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s28 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s29 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s30 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s31 = ktdp.construct_memory_view %off, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #box31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  // ... composed into one distributed view over the whole tensor. It moves
  // nothing: it maps a global coordinate to the partition holding it.
  %whole = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24, %s25, %s26, %s27, %s28, %s29, %s30, %s31
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<1x4096x1xf16>

  // Phase 2 -- name the coordinates this tile consumes. Still nothing moved:
  // both view ops are Pure, so a core may name another core's scratchpad.
  %tile = ktdp.construct_access_tile %whole[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x4096x1xf16> -> !ktdp.access_tile<1x4096x1xindex>

  // Phase 3 -- the load IS the transfer, and the store is the landing: a
  // received tile must be resident before a compute unit can read it.
  %got = ktdp.load %tile : <1x4096x1xindex> -> tensor<1x4096x1xf16>
  %lv = ktdp.construct_memory_view %land, sizes: [1, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x4096x1xf16>
  %lt = ktdp.construct_access_tile %lv[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x4096x1xf16> -> !ktdp.access_tile<1x4096x1xindex>
  ktdp.store %got, %lt : tensor<1x4096x1xf16>, <1x4096x1xindex>

  return
}
