// RUN: spyre-triton-opt %s --analyze-distributed-desc=grid=4 -split-input-file | FileCheck %s

// What AnalyzeDistributedDesc finds about a pinned read through a distributed
// descriptor, stated as tts.dist_landing on the landing marker.
//
// Each case is evaluated once per tile with tt.get_program_id x replaced by the
// tile id 0 .. 3. The tiles whose enclosing if conditions hold are the ones that
// read; src_coordinates is what each passes to .load(), unit for a tile that does
// not; the blocks laid side by side are the destination, dst_shape its extent,
// and dst_coordinates where each block sits in it.
//
// The IR is otherwise unchanged. The share's own pin is not a landing and gets
// no attribute, which the first case checks by matching its whole dictionary.

// No if: every tile reads, and they read a permutation of the four rows. Tile t
// reads row 3 - t, so the destination is the whole source and tile t's block
// sits at row 3 - t of it.
// CHECK-LABEL: tt.func @every_tile_reads
// CHECK: tts.pin %{{.*}} {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
// CHECK: tt.descriptor_load
// CHECK-NEXT: tts.pin %{{.*}} {memory_space = "ct_local", offset = 4096 : i32, tts.dist_landing = {dst_coordinates = [array<i64: 3, 0>, array<i64: 2, 0>, array<i64: 1, 0>, array<i64: 0, 0>], dst_shape = array<i64: 4, 64>, src_coordinates = [array<i64: 3, 0>, array<i64: 2, 0>, array<i64: 1, 0>, array<i64: 0, 0>]}}
tt.func @every_tile_reads(%x: tensor<1x64xf16>) -> tensor<1x64xf16> {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c3 = arith.constant 3 : i32
  %c0 = arith.constant 0 : i32
  %row = arith.subi %c3, %pid : i32
  %v = tt.descriptor_load %w[%row, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return %v : tensor<1x64xf16>
}

// -----
// A gather onto one tile: only tile 0 reads, and it reads the whole source.
// CHECK-LABEL: tt.func @gather_onto_one_tile
// CHECK: tts.dist_landing = {dst_coordinates = [array<i64: 0, 0>, unit, unit, unit], dst_shape = array<i64: 4, 64>, src_coordinates = [array<i64: 0, 0>, unit, unit, unit]}
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
// Half of the tiles read, from an offset: tiles 2 and 3 read rows 2-3 and 4-5.
// The destination starts where the first block does, so dst_coordinates are the
// source's minus 2 on the gather dimension and dst_shape is two blocks, not the
// source's eight rows.
// CHECK-LABEL: tt.func @some_tiles_read
// CHECK: tts.dist_landing = {dst_coordinates = [unit, unit, array<i64: 0, 0>, array<i64: 2, 0>], dst_shape = array<i64: 4, 64>, src_coordinates = [unit, unit, array<i64: 2, 0>, array<i64: 4, 0>]}
tt.func @some_tiles_read(%x: tensor<2x64xf16>) -> tensor<2x64xf16> {
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
// The else side, and nested ifs: a tile reads where the outer condition holds
// (pid < 3) and the inner one does not (pid != 1), so tiles 0 and 2. The
// coordinate is pid // 2 on the column blocks, with index arithmetic.
// CHECK-LABEL: tt.func @nested_else
// CHECK: tts.dist_landing = {dst_coordinates = [array<i64: 0, 0>, unit, array<i64: 0, 32>, unit], dst_shape = array<i64: 1, 64>, src_coordinates = [array<i64: 0, 0>, unit, array<i64: 0, 32>, unit]}
tt.func @nested_else(%x: tensor<1x32xf16>) {
  %share = math.exp %x : tensor<1x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x32xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64, n = 0 : i64}, {mb = 0 : i64, n = 1 : i64},
                      {mb = 1 : i64, n = 0 : i64}, {mb = 1 : i64, n = 1 : i64}],
       axes = ["mb", "n"], block_shape = array<i64: 1, 32>}
      : tensor<1x32xf16> -> !tt.tensordesc<1x32xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c32 = arith.constant 32 : index
  %lt = arith.cmpi slt, %pid, %c3 : i32
  scf.if %lt {
    %is1 = arith.cmpi eq, %pid, %c1 : i32
    scf.if %is1 {
    } else {
      %p = arith.index_cast %pid : i32 to index
      %c2 = arith.constant 2 : index
      %h = arith.divsi %p, %c2 : index
      %col = arith.muli %h, %c32 : index
      %coli = arith.index_cast %col : index to i32
      %v = tt.descriptor_load %w[%c0, %coli] : !tt.tensordesc<1x32xf16> -> tensor<1x32xf16>
      tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x32xf16>
    }
  }
  tt.return
}

// -----
// Two gather dimensions: the four tiles read the four [1, 32] blocks of a
// [2, 64] source, tile t at row t % 2 and column (t // 2) * 32.
// CHECK-LABEL: tt.func @two_gather_dims
// CHECK: tts.dist_landing = {dst_coordinates = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 0, 32>, array<i64: 1, 32>], dst_shape = array<i64: 2, 64>, src_coordinates = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 0, 32>, array<i64: 1, 32>]}
tt.func @two_gather_dims(%x: tensor<1x32xf16>) {
  %share = math.exp %x : tensor<1x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x32xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64, n = 0 : i64}, {mb = 0 : i64, n = 1 : i64},
                      {mb = 1 : i64, n = 0 : i64}, {mb = 1 : i64, n = 1 : i64}],
       axes = ["mb", "n"], block_shape = array<i64: 1, 32>}
      : tensor<1x32xf16> -> !tt.tensordesc<1x32xf16>
  %pid = tt.get_program_id x : i32
  %c2 = arith.constant 2 : i32
  %c32 = arith.constant 32 : i32
  %row = arith.remsi %pid, %c2 : i32
  %h = arith.divsi %pid, %c2 : i32
  %col = arith.muli %h, %c32 : i32
  %v = tt.descriptor_load %w[%row, %col] : !tt.tensordesc<1x32xf16> -> tensor<1x32xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x32xf16>
  tt.return
}

// -----
// tt.get_num_programs x is the grid, so pid's mirror (num - 1 - pid) is a
// constant per tile too.
// CHECK-LABEL: tt.func @num_programs
// CHECK: src_coordinates = [array<i64: 3, 0>, array<i64: 2, 0>, array<i64: 1, 0>, array<i64: 0, 0>]
tt.func @num_programs(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %num = tt.get_num_programs x : i32
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %last = arith.subi %num, %c1 : i32
  %row = arith.subi %last, %pid : i32
  %v = tt.descriptor_load %w[%row, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return
}

// -----
// A read whose value is not pinned is skipped: no attribute, and no requirement
// on its coordinates. That is right for a reduction's read (§7), which needs no
// landing. A copy's would have to be refused (§10), which the pass does not do
// yet (the TODO in AnalyzeDistributedDesc.cpp); when it does, this case, whose
// value feeds no reduction, becomes an error.
// CHECK-LABEL: tt.func @unpinned_read
// CHECK-NOT: tts.dist_landing
tt.func @unpinned_read(%x: tensor<1x64xf16>, %row: i32) -> tensor<1x64xf16> {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %c0 = arith.constant 0 : i32
  %v = tt.descriptor_load %w[%row, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tt.return %v : tensor<1x64xf16>
}

// -----
// Copies: the design doc's worked example. The even tiles both read the whole
// [4, 64], so each receives a copy, and the copies are laid along dimension 0 in
// ascending tile id: tile 0's at rows 0-3 and tile 2's at rows 4-7 of an [8, 64]
// destination.
// CHECK-LABEL: tt.func @gather_onto_the_even_tiles
// CHECK: tts.dist_landing = {dst_coordinates = [array<i64: 0, 0>, unit, array<i64: 4, 0>, unit], dst_shape = array<i64: 8, 64>, src_coordinates = [array<i64: 0, 0>, unit, array<i64: 0, 0>, unit]}
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
// A broadcast: every tile reads the whole source, so there are four copies and a
// [16, 64] destination.
// CHECK-LABEL: tt.func @broadcast
// CHECK: tts.dist_landing = {dst_coordinates = [array<i64: 0, 0>, array<i64: 4, 0>, array<i64: 8, 0>, array<i64: 12, 0>], dst_shape = array<i64: 16, 64>, src_coordinates = [array<i64: 0, 0>, array<i64: 0, 0>, array<i64: 0, 0>, array<i64: 0, 0>]}
tt.func @broadcast(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 4, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// Distinct blocks and copies together: tile t reads rows 2 * (t % 2) .. + 1, so
// two blocks with two copies each. The distinct blocks make a [4, 64] slab, and
// the second copy of each lands one slab down: tiles 0 and 1 in the first slab,
// tiles 2 and 3 in the second.
// CHECK-LABEL: tt.func @blocks_and_copies
// CHECK: tts.dist_landing = {dst_coordinates = [array<i64: 0, 0>, array<i64: 2, 0>, array<i64: 4, 0>, array<i64: 6, 0>], dst_shape = array<i64: 8, 64>, src_coordinates = [array<i64: 0, 0>, array<i64: 2, 0>, array<i64: 0, 0>, array<i64: 2, 0>]}
tt.func @blocks_and_copies(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 2, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<2x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %m = arith.remsi %pid, %c2 : i32
  %row = arith.muli %m, %c2 : i32
  %v = tt.descriptor_load %w[%row, %c0] : !tt.tensordesc<2x64xf16> -> tensor<2x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<2x64xf16>
  tt.return
}
