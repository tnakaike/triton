// C023 -- prefill, Add input1, consumer add_3
// all_gather
// tensor mul_out, rank 3 (mb, out, y), extents {mb: 512, out: 4096, y: 1}, f16
//
// source      16 view(s) = 16 piece(s) x 1 owner, ct 0, 2, ... 30 (stride 2)
// destination 32 piece(s) x 1 owner(s) -- NOT composed. Each core lands its own.
//
// if-store: the source is composed, the destination is this core's own
// scratchpad, and the box a core lands comes from the tile id:
//     mb = ((tid // 4) % 8) * 64
//     out = ((tid // 1) % 4) * 1024
//     y = 0
// The counterpart spelling is ../dist-store/, which composes both sides.

#src0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src1 = affine_set<(d0, d1, d2) : (d0 - 32 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src2 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 351 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src3 = affine_set<(d0, d1, d2) : (d0 - 352 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src4 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 415 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src5 = affine_set<(d0, d1, d2) : (d0 - 416 >= 0, -d0 + 447 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src6 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 479 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src7 = affine_set<(d0, d1, d2) : (d0 - 480 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src8 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src9 = affine_set<(d0, d1, d2) : (d0 - 96 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src10 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 159 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src11 = affine_set<(d0, d1, d2) : (d0 - 160 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src12 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 223 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src13 = affine_set<(d0, d1, d2) : (d0 - 224 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src14 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 287 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#src15 = affine_set<(d0, d1, d2) : (d0 - 288 >= 0, -d0 + 319 >= 0, d1 >= 0, -d1 + 4095 >= 0, d2 >= 0, -d2 >= 0)>
#blk0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1023 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c023_prefill_add_input1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c1024 = arith.constant 1024 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id. Identical to the dist-store form.
  %s0 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s2 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s3 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s4 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s5 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s6 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s7 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s8 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s9 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s10 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s11 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s12 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s13 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s14 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s15 = ktdp.construct_memory_view %src, sizes: [32, 4096, 1], strides: [4096, 1, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15
      : memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<32x4096x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>) : memref<512x4096x1xf16>

  // The landing: the piece, in THIS core's scratchpad. No ct_id, because a
  // core writes only its own; and the coordinate_set is a range over the
  // piece rather than a box in the tensor, because the view is local. Both
  // make it identical on every core, so it is hoisted with its access tile.
  %land = ktdp.construct_memory_view %dst, sizes: [64, 1024, 1], strides: [1024, 1, 1]
      {coordinate_set = #blk0, memory_space = #ktdp.memory_space<ct_local>}
      : memref<64x1024x1xf16>
  %wt = ktdp.construct_access_tile %land[%c0, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<64x1024x1xf16> -> !ktdp.access_tile<64x1024x1xindex>

  // Where in the tensor this core's box starts: one independent field of the
  // tile id per divided dimension. No branch, because nothing is selected --
  // the index is computed.
  %q_mb = arith.divsi %tid, %c4 : index
  %i_mb = arith.remsi %q_mb, %c8 : index
  %o_mb = arith.muli %i_mb, %c64 : index
  %i_out = arith.remsi %tid, %c4 : index
  %o_out = arith.muli %i_out, %c1024 : index

  // The movement. One load of a coordinate REGION -- which source partitions
  // that touches is the composed view's to resolve, exactly as in dist-store,
  // and a region may span several of them.
  %rt = ktdp.construct_access_tile %from[%o_mb, %o_out, %c0]
      {access_tile_order = #ord0, access_tile_set = #blk0}
      : memref<512x4096x1xf16> -> !ktdp.access_tile<64x1024x1xindex>
  %val = ktdp.load %rt : <64x1024x1xindex> -> tensor<64x1024x1xf16>
  ktdp.store %val, %wt : tensor<64x1024x1xf16>, <64x1024x1xindex>

  return
}
