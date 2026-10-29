// One of two KTIRs for no_unique_ktir.py; the other is ktir_2.mlir. See README.md.
//
// Physical coordinates (out // 64, mb, out % 64). The two copies of the
// destination are laid out along mb (dimension 1): copy r covers mb [4r, 4r + 4),
// so the composed destination is [1, 8, 64] and tile 2r stores at [0, 4r, 0].

#src0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#src1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#src2 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 2 >= 0, -d1 + 2 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#src3 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 3 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#dst0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#dst1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 - 4 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#blk0 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#ord0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @no_unique_ktir_1(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source: region k on tile k. Identical in ktir_2.mlir.
  %s0 = ktdp.construct_memory_view %src, sizes: [1, 1, 64], strides: [64, 64, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [1, 1, 64], strides: [64, 64, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [1, 1, 64], strides: [64, 64, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s3 = ktdp.construct_memory_view %src, sizes: [1, 1, 64], strides: [64, 64, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3
      : memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<1x1x64xf16, #ktdp.memory_space<ct_local, ct_id = 3>>) : memref<1x4x64xf16>

  // The destination: copy r on tile 2r, the copies laid out along mb.
  %d0 = ktdp.construct_memory_view %dst, sizes: [1, 4, 64], strides: [256, 64, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<1x4x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [1, 4, 64], strides: [256, 64, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<1x4x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1
      : memref<1x4x64xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<1x4x64xf16, #ktdp.memory_space<ct_local, ct_id = 2>>) : memref<1x8x64xf16>

  // Copy r = tid // 2, at mb = 4r.
  %q = arith.divsi %tid, %c2 : index
  %o_mb = arith.muli %q, %c4 : index

  // The even tiles receive, as the kernel's `if` says.
  %r = arith.remsi %tid, %c2 : index
  %part = arith.cmpi eq, %r, %c0 : index
  scf.if %part {
    %rt = ktdp.construct_access_tile %from[%c0, %c0, %c0]
        {access_tile_order = #ord0, access_tile_set = #blk0}
        : memref<1x4x64xf16> -> !ktdp.access_tile<1x4x64xindex>
    %val = ktdp.load %rt : <1x4x64xindex> -> tensor<1x4x64xf16>
    %wt = ktdp.construct_access_tile %to[%c0, %o_mb, %c0]
        {access_tile_order = #ord0, access_tile_set = #blk0}
        : memref<1x8x64xf16> -> !ktdp.access_tile<1x4x64xindex>
    ktdp.store %val, %wt : tensor<1x4x64xf16>, <1x4x64xindex>
  }

  return
}
