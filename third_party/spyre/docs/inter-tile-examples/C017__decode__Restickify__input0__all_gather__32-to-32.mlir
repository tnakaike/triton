// C017 -- decode, Restickify input0, consumer bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify
// all_gather: 32 source piece(s) -> 32 destination piece(s)
// tensor bmm-wtAttnHeadBreak-VirtualReshape_out, rank 5 (j, mb, out, x, y), extents {j: 8, mb: 1, out: 128, x: 1, y: 768}, f16
//
// Written for the tile holding destination piece p0, ct 0.
//
// Generated from the curated LX-relayout SDSC package. See README.md for the
// mapping, the conventions and the provenance.

#box0 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 23 >= 0)>
#box1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 24 >= 0, -d4 + 47 >= 0)>
#box2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 240 >= 0, -d4 + 263 >= 0)>
#box3 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 264 >= 0, -d4 + 287 >= 0)>
#box4 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 288 >= 0, -d4 + 311 >= 0)>
#box5 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 312 >= 0, -d4 + 335 >= 0)>
#box6 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 336 >= 0, -d4 + 359 >= 0)>
#box7 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 360 >= 0, -d4 + 383 >= 0)>
#box8 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 407 >= 0)>
#box9 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 408 >= 0, -d4 + 431 >= 0)>
#box10 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 432 >= 0, -d4 + 455 >= 0)>
#box11 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 456 >= 0, -d4 + 479 >= 0)>
#box12 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 48 >= 0, -d4 + 71 >= 0)>
#box13 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 480 >= 0, -d4 + 503 >= 0)>
#box14 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 504 >= 0, -d4 + 527 >= 0)>
#box15 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 528 >= 0, -d4 + 551 >= 0)>
#box16 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 552 >= 0, -d4 + 575 >= 0)>
#box17 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 599 >= 0)>
#box18 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 600 >= 0, -d4 + 623 >= 0)>
#box19 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 624 >= 0, -d4 + 647 >= 0)>
#box20 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 648 >= 0, -d4 + 671 >= 0)>
#box21 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 672 >= 0, -d4 + 695 >= 0)>
#box22 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 696 >= 0, -d4 + 719 >= 0)>
#box23 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 72 >= 0, -d4 + 95 >= 0)>
#box24 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 720 >= 0, -d4 + 743 >= 0)>
#box25 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 744 >= 0, -d4 + 767 >= 0)>
#box26 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 96 >= 0, -d4 + 119 >= 0)>
#box27 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 120 >= 0, -d4 + 143 >= 0)>
#box28 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 144 >= 0, -d4 + 167 >= 0)>
#box29 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 168 >= 0, -d4 + 191 >= 0)>
#box30 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 215 >= 0)>
#box31 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 216 >= 0, -d4 + 239 >= 0)>
#blk0 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @c017_decode_restickify_input0(%off: index, %land: index) {
  %c0 = arith.constant 0 : index

  // Phase 1 -- one view per source partition. They differ ONLY in
  // coordinate_set and ct_id; offset, sizes and strides are identical.
  %s0 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %s19 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s20 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %s21 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s22 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %s23 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s24 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s25 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %s26 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s27 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s28 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s29 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s30 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s31 = ktdp.construct_memory_view %off, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #box31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  // ... composed into one distributed view over the whole tensor. It moves
  // nothing: it maps a global coordinate to the partition holding it.
  %whole = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24, %s25, %s26, %s27, %s28, %s29, %s30, %s31
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<8x1x128x1x768xf16>

  // Phase 2 -- name the coordinates this tile consumes. Still nothing moved:
  // both view ops are Pure, so a core may name another core's scratchpad.
  %tile = ktdp.construct_access_tile %whole[%c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<8x1x128x1x768xf16> -> !ktdp.access_tile<2x1x64x1x192xindex>

  // Phase 3 -- the load IS the transfer, and the store is the landing: a
  // received tile must be resident before a compute unit can read it.
  %got = ktdp.load %tile : <2x1x64x1x192xindex> -> tensor<2x1x64x1x192xf16>
  %lv = ktdp.construct_memory_view %land, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<2x1x64x1x192xf16>
  %lt = ktdp.construct_access_tile %lv[%c0, %c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<2x1x64x1x192xf16> -> !ktdp.access_tile<2x1x64x1x192xindex>
  ktdp.store %got, %lt : tensor<2x1x64x1x192xf16>, <2x1x64x1x192xindex>

  return
}
