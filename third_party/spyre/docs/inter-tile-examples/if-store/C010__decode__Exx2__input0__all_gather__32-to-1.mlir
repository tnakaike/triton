// C010 -- decode, Exx2 input0, consumer mean-Exx2
// all_gather
// tensor mul_out, rank 3 (mb, out, y), extents {mb: 1, out: 4096, y: 1}, f16
//
// source      32 view(s) = 32 piece(s) x 1 owner, ct 0..31
// destination 1 piece(s) x 1 owner(s) -- NOT composed. Each core lands its own.
//
// if-store: the source is composed, the destination is this core's own
// scratchpad, and the box a core lands comes from the tile id:
//     mb = 0
//     out = 0
//     y = 0
// The counterpart spelling is ../dist-store/, which composes both sides.

#src0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0)>
#src1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 128 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 >= 0)>
#src2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1280 >= 0, -d1 + 1407 >= 0, d2 >= 0, -d2 >= 0)>
#src3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1408 >= 0, -d1 + 1535 >= 0, d2 >= 0, -d2 >= 0)>
#src4 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1536 >= 0, -d1 + 1663 >= 0, d2 >= 0, -d2 >= 0)>
#src5 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1664 >= 0, -d1 + 1791 >= 0, d2 >= 0, -d2 >= 0)>
#src6 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1792 >= 0, -d1 + 1919 >= 0, d2 >= 0, -d2 >= 0)>
#src7 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1920 >= 0, -d1 + 2047 >= 0, d2 >= 0, -d2 >= 0)>
#src8 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2048 >= 0, -d1 + 2175 >= 0, d2 >= 0, -d2 >= 0)>
#src9 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2176 >= 0, -d1 + 2303 >= 0, d2 >= 0, -d2 >= 0)>
#src10 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2304 >= 0, -d1 + 2431 >= 0, d2 >= 0, -d2 >= 0)>
#src11 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2432 >= 0, -d1 + 2559 >= 0, d2 >= 0, -d2 >= 0)>
#src12 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 256 >= 0, -d1 + 383 >= 0, d2 >= 0, -d2 >= 0)>
#src13 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2560 >= 0, -d1 + 2687 >= 0, d2 >= 0, -d2 >= 0)>
#src14 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2688 >= 0, -d1 + 2815 >= 0, d2 >= 0, -d2 >= 0)>
#src15 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2816 >= 0, -d1 + 2943 >= 0, d2 >= 0, -d2 >= 0)>
#src16 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2944 >= 0, -d1 + 3071 >= 0, d2 >= 0, -d2 >= 0)>
#src17 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3072 >= 0, -d1 + 3199 >= 0, d2 >= 0, -d2 >= 0)>
#src18 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3200 >= 0, -d1 + 3327 >= 0, d2 >= 0, -d2 >= 0)>
#src19 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3328 >= 0, -d1 + 3455 >= 0, d2 >= 0, -d2 >= 0)>
#src20 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3456 >= 0, -d1 + 3583 >= 0, d2 >= 0, -d2 >= 0)>
#src21 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3584 >= 0, -d1 + 3711 >= 0, d2 >= 0, -d2 >= 0)>
#src22 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3712 >= 0, -d1 + 3839 >= 0, d2 >= 0, -d2 >= 0)>
#src23 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 384 >= 0, -d1 + 511 >= 0, d2 >= 0, -d2 >= 0)>
#src24 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3840 >= 0, -d1 + 3967 >= 0, d2 >= 0, -d2 >= 0)>
#src25 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3968 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src26 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 512 >= 0, -d1 + 639 >= 0, d2 >= 0, -d2 >= 0)>
#src27 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 640 >= 0, -d1 + 767 >= 0, d2 >= 0, -d2 >= 0)>
#src28 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 768 >= 0, -d1 + 895 >= 0, d2 >= 0, -d2 >= 0)>
#src29 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 896 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#src30 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1024 >= 0, -d1 + 1151 >= 0, d2 >= 0, -d2 >= 0)>
#src31 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1152 >= 0, -d1 + 1279 >= 0, d2 >= 0, -d2 >= 0)>
#blk0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c010_decode_exx2_input0(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id. Identical to the dist-store form.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %s19 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s20 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %s21 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s22 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %s23 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s24 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s25 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %s26 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s27 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s28 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s29 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s30 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s31 = ktdp.construct_memory_view %src, sizes: [1, 128, 1], strides: [128, 1, 1]
      {coordinate_set = #src31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24, %s25, %s26, %s27, %s28, %s29, %s30, %s31
      : memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x128x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<1x4096x1xf16>

  // The landing: the piece, in THIS core's scratchpad. No ct_id, because a
  // core writes only its own; and the coordinate_set is a range over the
  // piece rather than a box in the tensor, because the view is local. Both
  // make it identical on every core, so it is hoisted with its access tile.
  %land = ktdp.construct_memory_view %dst, sizes: [1, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x4096x1xf16>
  %wt = ktdp.construct_access_tile %land[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x4096x1xf16> -> !ktdp.access_tile<1x4096x1xindex>

  // Where in the tensor this core's box starts: one independent field of the
  // tile id per divided dimension. No branch, because nothing is selected --
  // the index is computed.

  // The movement. One load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, exactly as in dist-store,
  // and a region may span several of them.
  %part = arith.cmpi eq, %tid, %c0 : index
  // Not every core takes part, so the load is guarded with the store: the
  // load IS the transfer, and a core that lands nothing must read nothing.
  scf.if %part {
    %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0]
        {access_tile_order = #ord0, access_tile_set = #blk0}
        : memref<1x4096x1xf16> -> !ktdp.access_tile<1x4096x1xindex>
    %val = ktdp.load %rt : <1x4096x1xindex> -> tensor<1x4096x1xf16>
    ktdp.store %val, %wt : tensor<1x4096x1xf16>, <1x4096x1xindex>
  }

  return
}
