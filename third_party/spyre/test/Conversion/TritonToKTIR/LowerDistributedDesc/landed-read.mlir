// RUN: spyre-triton-opt %s --analyze-distributed-desc=grid=4 --lower-distributed-desc=grid=4 -split-input-file | FileCheck %s

// A read whose value lands in a pin, lowered from what AnalyzeDistributedDesc
// stated on the landing marker (design section 3 phases 3 and 4). Both passes run,
// as in the pipeline, so the inputs are what a kernel gives: no tts.dist_landing.
//
// What each case checks:
//
//   the dst view   one ktdp.construct_memory_view per dst tile, at the landing
//                  pin's offset, of the landing's shape, with that tile's ct_id
//                  and its place in the dst view as coordinate_set; composed into
//                  dst_shape;
//   the transfer   ahead of the outermost if and unguarded: a load from the src
//                  view at this tile's src coordinates, and a store into the dst
//                  view at its dst coordinates. A quasi-affine coordinate is the
//                  read's own expression, cloned ahead of the if, and the dst
//                  coordinate is it minus the dst view's origin; any other is a
//                  select chain on the tile id;
//   the read       replaced by a load from this tile's own landing, inside the if
//                  where its uses are; the landing marker is gone.
//
// Note there are no rules of dashes anywhere in this file, since -split-input-file
// matches its marker as a substring.

// Tiles 2 and 3 receive rows 2-3 and 4-5 of an [8, 64] source. The dst view is
// [4, 64]: tile 2's landing is its rows 0-1 and tile 3's its rows 2-3.
// CHECK-DAG: #[[DST2:.*]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[DST3:.*]] = affine_set<(d0, d1) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL: tt.func @some_tiles_receive
// CHECK: %[[SRC:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<8x64xf16>
// CHECK: %[[PID:.*]] = tt.get_program_id x : i32
// CHECK: %[[OFF:.*]] = arith.constant 4096 : index
// CHECK: %[[L2:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[DST2]], memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
// CHECK: %[[L3:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[DST3]], memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
// CHECK: %[[DST:.*]] = ktdp.construct_distributed_memory_view(%[[L2]], %[[L3]] : {{.*}}) : memref<4x64xf16>
// The src row is the read's (pid - 1) * 2, cloned out of the if, and the dst row
// is it minus 2, the dst view's origin. On tiles 0 and 1, which do not receive,
// they are -2 and 0, and the transfer does nothing there.
// CHECK: %[[S:.*]] = arith.subi %[[PID]], %c1_i32 : i32
// CHECK: %[[M:.*]] = arith.muli %[[S]], %c2_i32 : i32
// CHECK: %[[SROW:.*]] = arith.index_cast %[[M]] : i32 to index
// CHECK: %[[DROW:.*]] = arith.subi %[[SROW]], %c2{{.*}} : index
// CHECK-NOT: arith.select
// CHECK: %[[ST:.*]] = ktdp.construct_access_tile %[[SRC]][%[[SROW]], %{{.*}}]
// CHECK: %[[MOVED:.*]] = ktdp.load %[[ST]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK: %[[DT:.*]] = ktdp.construct_access_tile %[[DST]][%[[DROW]], %{{.*}}]
// CHECK: ktdp.store %[[MOVED]], %[[DT]]
// CHECK: %[[OWN:.*]] = ktdp.construct_memory_view %[[OFF]], sizes: [2, 64], strides: [64, 1] {{{.*}}memory_space = #ktdp.memory_space<ct_local>} : memref<2x64xf16>
// CHECK: scf.if
// CHECK: %[[OT:.*]] = ktdp.construct_access_tile %[[OWN]]
// CHECK: %[[BACK:.*]] = ktdp.load %[[OT]]
// CHECK: scf.yield %[[BACK]]
// CHECK-NOT: tts.pin {{.*}}offset = 4096
// CHECK-NOT: tt.descriptor_load
tt.func @some_tiles_receive(%x: tensor<2x64xf16>) -> tensor<2x64xf16> {
  %share = math.exp %x : tensor<2x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<2x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 2, 64>}
      : tensor<2x64xf16> -> !tt.tensordesc<2x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %ge = arith.cmpi sge, %pid, %c2 : i32
  %r = scf.if %ge -> tensor<2x64xf16> {
    %s = arith.subi %pid, %c1 : i32
    %row = arith.muli %s, %c2 : i32
    %v = tt.descriptor_load %w[%row, %c0] : !tt.tensordesc<2x64xf16> -> tensor<2x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<2x64xf16>
    scf.yield %v : tensor<2x64xf16>
  } else {
    scf.yield %x : tensor<2x64xf16>
  }
  tt.return %r : tensor<2x64xf16>
}

// -----
// A gather onto one tile: only tile 0 receives, so the dst view has one landing,
// the whole [4, 64], and every coordinate is the constant 0.
// CHECK-LABEL: tt.func @gather_onto_one_tile
// CHECK: %[[SRC:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<4x64xf16>
// CHECK: %[[L0:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [4, 64], strides: [64, 1] {coordinate_set = #{{.*}}, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
// CHECK: %[[DST:.*]] = ktdp.construct_distributed_memory_view(%[[L0]] : {{.*}}) : memref<4x64xf16>
// CHECK-NOT: arith.select
// CHECK: %[[ST:.*]] = ktdp.construct_access_tile %[[SRC]]
// CHECK: %[[MOVED:.*]] = ktdp.load %[[ST]]
// CHECK: %[[DT:.*]] = ktdp.construct_access_tile %[[DST]]
// CHECK: ktdp.store %[[MOVED]], %[[DT]]
// CHECK: scf.if
// CHECK: ktdp.load
tt.func @gather_onto_one_tile(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 4, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<4x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %first = arith.cmpi eq, %pid, %c0 : i32
  scf.if %first {
    %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  }
  tt.return
}

// -----
// Rows to columns, from the lower half: an [8, 64] source held two rows per core,
// and tile t receives the [4, 16] block at [4, 16t]. The src and dst coordinates
// differ on dimension 0 (4 against 4 - 4) and agree on dimension 1, where the
// read's pid * 16, already ahead of the read, serves both accesses as it is.
// CHECK-LABEL: tt.func @lower_half_to_columns
// CHECK: %[[SRC:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<8x64xf16>
// CHECK: %[[PID:.*]] = tt.get_program_id x : i32
// CHECK: %[[MUL:.*]] = arith.muli %[[PID]], %c16_i32 : i32
// CHECK: %[[DST:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<4x64xf16>
// CHECK: %[[ROW:.*]] = arith.index_cast %c4_i32 : i32 to index
// CHECK: %[[DROW:.*]] = arith.subi %[[ROW]], %c4 : index
// CHECK: %[[COL:.*]] = arith.index_cast %[[MUL]] : i32 to index
// CHECK-NOT: arith.select
// CHECK: ktdp.construct_access_tile %[[SRC]][%[[ROW]], %[[COL]]]
// CHECK: ktdp.construct_access_tile %[[DST]][%[[DROW]], %[[COL]]]
tt.func @lower_half_to_columns(%x: tensor<2x64xf16>) {
  %share = math.exp %x : tensor<2x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<2x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 4, 16>}
      : tensor<2x64xf16> -> !tt.tensordesc<4x16xf16>
  %pid = tt.get_program_id x : i32
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %col = arith.muli %pid, %c16 : i32
  %v = tt.descriptor_load %w[%c4, %col] : !tt.tensordesc<4x16xf16> -> tensor<4x16xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x16xf16>
  tt.return
}

// -----
// A coordinate that is not quasi-affine in the program id: pid ^ 1, the
// permutation 1, 0, 3, 2. It is not cloned; the coordinates come from
// tts.dist_landing as a select chain on the tile id, src and dst sharing it since
// the dst view is the whole source. Tile 0's value, 1, is the chain's start.
// CHECK-LABEL: tt.func @not_quasi_affine
// CHECK: %[[SRC:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<4x64xf16>
// CHECK: %[[DST:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<4x64xf16>
// CHECK: %[[TID:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: arith.cmpi eq, %[[TID]], %c1{{.*}} : index
// CHECK: arith.select {{.*}} : index
// CHECK: arith.cmpi eq, %[[TID]], %c2{{.*}} : index
// CHECK: arith.select {{.*}} : index
// CHECK: arith.cmpi eq, %[[TID]], %c3{{.*}} : index
// CHECK: %[[ROW:.*]] = arith.select {{.*}} : index
// CHECK-NOT: arith.select
// CHECK: ktdp.construct_access_tile %[[SRC]][%[[ROW]], %{{.*}}]
// CHECK: ktdp.construct_access_tile %[[DST]][%[[ROW]], %{{.*}}]
tt.func @not_quasi_affine(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %row = arith.xori %pid, %c1 : i32
  %v = tt.descriptor_load %w[%row, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return
}

// -----
// Copies, the design doc's worked example: the even tiles both read the whole
// [4, 64], and the copies are laid along dimension 0, so the dst view is [8, 64]
// with tile 0's landing at rows 0-3 and tile 2's at rows 4-7. The dst row is the
// copy's slab, 4 * (t / 2): no remainder, since t / 2 stays below 2 on all four
// tiles. The src coordinate is the constant 0 on every tile.
// CHECK-DAG: #[[COPY0:.*]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-DAG: #[[COPY1:.*]] = affine_set<(d0, d1) : (d0 - 4 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL: tt.func @gather_onto_the_even_tiles
// CHECK: %[[SRC:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<4x64xf16>
// CHECK: %[[L0:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [4, 64], strides: [64, 1] {coordinate_set = #[[COPY0]], memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
// CHECK: %[[L2:.*]] = ktdp.construct_memory_view %{{.*}}, sizes: [4, 64], strides: [64, 1] {coordinate_set = #[[COPY1]], memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
// CHECK: %[[DST:.*]] = ktdp.construct_distributed_memory_view(%[[L0]], %[[L2]] : {{.*}}) : memref<8x64xf16>
// CHECK: %[[ROW:.*]] = arith.index_cast %c0_i32 : i32 to index
// CHECK: %[[TID:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[Q:.*]] = arith.divsi %[[TID]], %c2{{.*}} : index
// CHECK-NOT: arith.remsi {{.*}} : index
// CHECK: %[[SLAB:.*]] = arith.muli %[[Q]], %c4{{.*}} : index
// CHECK: %[[DROW:.*]] = arith.addi %[[ROW]], %[[SLAB]] : index
// CHECK-NOT: arith.select
// CHECK: ktdp.construct_access_tile %[[SRC]][%[[ROW]], %{{.*}}]
// CHECK: ktdp.construct_access_tile %[[DST]][%[[DROW]], %{{.*}}]
tt.func @gather_onto_the_even_tiles(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 4, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<4x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %m = arith.remsi %pid, %c2 : i32
  %even = arith.cmpi eq, %m, %c0 : i32
  scf.if %even {
    %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  }
  tt.return
}

// -----
// Copies whose index needs the remainder: tile t reads rows 2 * (t / 2) .. + 1,
// so tiles 0 and 1 read one block and tiles 2 and 3 the other, and each block's
// second copy is on an odd tile: the copy index is t % 2, and the dst row is the
// src row plus 4 * (t % 2).
// CHECK-LABEL: tt.func @copy_index_is_a_remainder
// CHECK: %[[DST:.*]] = ktdp.construct_distributed_memory_view({{.*}}) : memref<8x64xf16>
// CHECK: %[[ROW:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[TID:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[C:.*]] = arith.remsi %[[TID]], %c2{{.*}} : index
// CHECK: %[[SLAB:.*]] = arith.muli %[[C]], %c4{{.*}} : index
// CHECK: %[[DROW:.*]] = arith.addi %[[ROW]], %[[SLAB]] : index
// CHECK: ktdp.construct_access_tile %[[DST]][%[[DROW]], %{{.*}}]
tt.func @copy_index_is_a_remainder(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 2, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<2x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %h = arith.divsi %pid, %c2 : i32
  %row = arith.muli %h, %c2 : i32
  %v = tt.descriptor_load %w[%row, %c0] : !tt.tensordesc<2x64xf16> -> tensor<2x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<2x64xf16>
  tt.return
}
