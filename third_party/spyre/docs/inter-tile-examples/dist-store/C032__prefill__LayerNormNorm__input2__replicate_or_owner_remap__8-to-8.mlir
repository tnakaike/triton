// C032 -- prefill, LayerNormNorm input2, consumer mean_1-LayerNormNorm
// replicate_or_owner_remap
// tensor mean_1-LayerNormScale_out, rank 3 (mb, out, y), extents {mb: 512, out: 64, y: 1}, f16
//
// source       8 view(s) = 8 piece(s) x 1 owner, ct 0, 4, ... 28 (stride 4)
// destination 32 view(s) = 8 piece(s) x 4 owner(s), ct 0..31
// composed    source 512x64x1, destination 2048x64x1 = 512x64x1 x 4 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination view is the whole of what its views
// hold -- one slot per (piece, owner) -- so a core says which slot it writes:
//     mb = ((tid // 4) % 8) * 64
//     out = 0
//     y = 0
//     replica = (tid // 1) % 4, at 512 per slot on dimension 0
// Which cores take part is not stated: 32 of the 32 address a slot they
// own, and the slot's own ct_id is what says so. See README.md for the mapping
// and the provenance.

#src0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#src1 = affine_set<(d0, d1, d2) : (d0 - 64 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#src2 = affine_set<(d0, d1, d2) : (d0 - 128 >= 0, -d0 + 191 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#src3 = affine_set<(d0, d1, d2) : (d0 - 192 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#src4 = affine_set<(d0, d1, d2) : (d0 - 256 >= 0, -d0 + 319 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#src5 = affine_set<(d0, d1, d2) : (d0 - 320 >= 0, -d0 + 383 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#src6 = affine_set<(d0, d1, d2) : (d0 - 384 >= 0, -d0 + 447 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#src7 = affine_set<(d0, d1, d2) : (d0 - 448 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @c032_prefill_layernormnorm_input2(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c512 = arith.constant 512 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s2 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s3 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s4 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s5 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s6 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s7 = ktdp.construct_memory_view %src, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>) : memref<512x64x1xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views with the SAME coordinate_set and different ct_id --
  // which is what replication is, stated rather than left to the launch. The
  // composed type counts those views, so the slots are distinct coordinates
  // and no coordinate has two writers.
  %d0 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [64, 64, 1], strides: [64, 1, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 9>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<64x64x1xf16, #ktdp.memory_space<ct_local, ct_id = 31>>) : memref<2048x64x1xf16>

  // Where this core's box sits: one independent field of the tile id per
  // divided dimension, exactly as in ../if-store/. No branch selects it --
  // the index is computed.
  %q_mb = arith.divsi %tid, %c4 : index
  %i_mb = arith.remsi %q_mb, %c8 : index
  %o_mb = arith.muli %i_mb, %c64 : index

  // Which of the slots is this core's: its position in its piece's owner
  // list, stacked on dimension 0 of the composed destination.
  %i_rep = arith.remsi %tid, %c4 : index
  %o_rep = arith.muli %i_rep, %c512 : index
  %base_mb = arith.addi %o_rep, %o_mb : index

  // The movement, and no control flow: one load of a coordinate REGION --
  // which source partitions that touches is the composed view's to resolve,
  // and a region may span several -- and one store into a named slot of the
  // destination, whose ct_id is what says whether this core owns it.
  %rt = ktdp.construct_access_tile %from[%o_mb, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<512x64x1xf16> -> !ktdp.access_tile<64x64x1xindex>
  %val = ktdp.load %rt : <64x64x1xindex> -> tensor<64x64x1xf16>
  %wt = ktdp.construct_access_tile %to[%base_mb, %c0, %c0]
      {access_tile_order = #ord0, access_tile_set = #src0}
      : memref<2048x64x1xf16> -> !ktdp.access_tile<64x64x1xindex>
  ktdp.store %val, %wt : tensor<64x64x1xf16>, <64x64x1xindex>

  return
}
