// C001 -- decode, BatchMatMulV2 input0, consumer mm_13-BMM_1
// grouped_all_gather_with_replication
// tensor mul_263_out, rank 4 (y, in, mb, in%), extents {y: 1, in: 200, mb: 1, in%: 64}, f16
//
// source      25 view(s) = 25 piece(s) x 1 owner, ct 0..24
// destination 1 piece(s) x 32 owner(s) -- NOT composed. Each core lands its own.
//
// if-store: the source is composed, the destination is this core's own
// scratchpad, and the box a core lands comes from the tile id:
//     y = 0
//     in = 0
//     mb = 0
//     in% = 0
// The counterpart spelling is ../dist-store/, which composes both sides.

#src0 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 8 >= 0, -d1 + 15 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 80 >= 0, -d1 + 87 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src3 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 88 >= 0, -d1 + 95 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 96 >= 0, -d1 + 103 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src5 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 104 >= 0, -d1 + 111 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src6 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 112 >= 0, -d1 + 119 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src7 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 120 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src8 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 128 >= 0, -d1 + 135 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src9 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 136 >= 0, -d1 + 143 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src10 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 144 >= 0, -d1 + 151 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src11 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 152 >= 0, -d1 + 159 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src12 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 16 >= 0, -d1 + 23 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src13 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 160 >= 0, -d1 + 167 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src14 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 168 >= 0, -d1 + 175 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src15 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 176 >= 0, -d1 + 183 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src16 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 184 >= 0, -d1 + 191 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src17 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 192 >= 0, -d1 + 199 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src18 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 24 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src19 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 32 >= 0, -d1 + 39 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src20 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 40 >= 0, -d1 + 47 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src21 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 48 >= 0, -d1 + 55 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src22 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 56 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src23 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 64 >= 0, -d1 + 71 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#src24 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 - 72 >= 0, -d1 + 79 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#blk0 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 199 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @c001_decode_batchmatmulv2_input0(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id. Identical to the dist-store form.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src18, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s19 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src19, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s20 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src20, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s21 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src21, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s22 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src22, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s23 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src23, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s24 = ktdp.construct_memory_view %src, sizes: [1, 8, 1, 64], strides: [512, 64, 64, 1]
      {coordinate_set = #src24, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24
      : memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<1x8x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<1x200x1x64xf16>

  // The landing: the piece, in THIS core's scratchpad. No ct_id, because a
  // core writes only its own; and the coordinate_set is a range over the
  // piece rather than a box in the tensor, because the view is local. Both
  // make it identical on every core, so it is hoisted with its access tile.
  %land = ktdp.construct_memory_view %dst, sizes: [1, 200, 1, 64], strides: [12800, 64, 64, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<1x200x1x64xf16>
  %wt = ktdp.construct_access_tile %land[%c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x200x1x64xf16> -> !ktdp.access_tile<1x200x1x64xindex>

  // Where in the tensor this core's box starts: one independent field of the
  // tile id per divided dimension. No branch, because nothing is selected --
  // the index is computed.

  // The movement. One load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, exactly as in dist-store,
  // and a region may span several of them.
  %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<1x200x1x64xf16> -> !ktdp.access_tile<1x200x1x64xindex>
  %val = ktdp.load %rt : <1x200x1x64xindex> -> tensor<1x200x1x64xf16>
  ktdp.store %val, %wt : tensor<1x200x1x64xf16>, <1x200x1x64xindex>

  return
}
