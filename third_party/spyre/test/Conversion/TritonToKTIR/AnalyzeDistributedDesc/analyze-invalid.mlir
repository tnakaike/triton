// RUN: spyre-triton-opt %s --analyze-distributed-desc=grid=4 -split-input-file -verify-diagnostics

// What AnalyzeDistributedDesc refuses, each at the read: the line a Triton kernel
// points at. These are the design's "checks before the analysis" (every condition
// and coordinate a constant on every tile, every block inside its composed view)
// and the rules of its phase 3: the blocks tile a box, and every block has the
// same number of copies.
//
// Every case uses the same four-tile source, one [1, 64] share per tile composed
// over `mb` into [4, 64], unless it says otherwise.

// A coordinate read from memory rather than computed from the program id. The
// note points at the value that is not a constant.
tt.func @coordinate_from_data(%x: tensor<1x64xf16>, %p: !tt.ptr<i32>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-note @+1 {{this value is not a function of tl.program_id(0) and constants}}
  %r = tt.load %p : !tt.ptr<i32>
  // expected-error @+1 {{the coordinate this read passes on dimension 0 must be a constant on every tile, and on tile 0 it is not}}
  %v = tt.descriptor_load %w[%r, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return
}

// -----
// The same for the if that decides which tiles read: a data-dependent receiver
// set cannot be composed at compile time.
tt.func @condition_from_data(%x: tensor<1x64xf16>, %p: !tt.ptr<i32>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  // expected-note @+1 {{this value is not a function of tl.program_id(0) and constants}}
  %flag = tt.load %p : !tt.ptr<i32>
  %go = arith.cmpi ne, %flag, %c0 : i32
  scf.if %go {
    // expected-error @+1 {{which tiles perform this read is decided by an if whose condition must be a constant on every tile, and on tile 0 it is not}}
    %v = tt.descriptor_load %w[%pid, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  }
  tt.return
}

// -----
// A coordinate that is undefined on a tile, here a division by zero, says so
// rather than calling the value unknown.
tt.func @undefined_on_a_tile(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %z = arith.subi %pid, %pid : i32
  // expected-note @+1 {{this value is undefined on tile 0 (a division by zero, an overflowing division, or a shift by the width or more)}}
  %r = arith.divsi %c3, %z : i32
  // expected-error @+1 {{the coordinate this read passes on dimension 0 must be a constant on every tile, and on tile 0 it is not}}
  %v = tt.descriptor_load %w[%r, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return
}

// -----
// Inside a loop the read would land once per iteration into the one buffer the
// pin names.
tt.func @inside_a_loop(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  // expected-note @+1 {{it is inside this}}
  scf.for %i = %c0 to %c2 step %c1 : i32 {
    // expected-error @+1 {{a read through a distributed descriptor whose value is pinned may be guarded by an if, but not run inside 'scf.for': it would land more than once}}
    %v = tt.descriptor_load %w[%pid, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  }
  tt.return
}

// -----
// Tile 3 reads row 4 of a four-row source: memory no partition holds.
tt.func @outside_the_source(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = arith.addi %pid, %c1 : i32
  // expected-error @+1 {{on tile 3 this read takes 1 elements from 4 on dimension 0, outside the composed source's extent 4}}
  %v = tt.descriptor_load %w[%r, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return
}

// -----
// An if that no tile passes: the pinned value never lands.
tt.func @no_tile_reads(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c8 = arith.constant 8 : i32
  %go = arith.cmpi sge, %pid, %c8 : i32
  scf.if %go {
    // expected-error @+1 {{no tile performs this read: on every tile an if around it skips it, so the value it pins never lands}}
    %v = tt.descriptor_load %w[%pid, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  }
  tt.return
}

// -----
// Blocks read by differing numbers of tiles: tiles 0, 1 and 2 read row 0 and
// tile 3 reads row 1, so row 0 has three copies and row 1 one. Laid along
// dimension 0, the copies could not tile the destination.
tt.func @differing_copies(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %r = arith.divsi %pid, %c3 : i32
  // expected-error @+1 {{the block at [0, 0] is read by 3 tiles and the block at [1, 0] by 1: every block needs the same number of copies for the copies to tile the destination}}
  %v = tt.descriptor_load %w[%r, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return
}

// -----
// Tiles 0 and 1 read rows 0 and 2: a gap between the blocks, so their union is
// not a box.
tt.func @blocks_with_a_gap(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %lt = arith.cmpi slt, %pid, %c2 : i32
  scf.if %lt {
    %r = arith.muli %pid, %c2 : i32
    // expected-error @+1 {{on dimension 0 the tiles that read start at [0, 2]: the blocks must sit side by side, 1 apart, for the destination to be contiguous}}
    %v = tt.descriptor_load %w[%r, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  }
  tt.return
}

// -----
// Three tiles read three of the four [1, 32] blocks of a [2, 64] source. Each
// dimension is side by side on its own, but the corner (1, 32) is not read, so
// the union is not a box.
tt.func @a_missing_corner(%x: tensor<1x32xf16>) {
  %share = math.exp %x : tensor<1x32xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x32xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64, n = 0 : i64}, {mb = 0 : i64, n = 1 : i64},
                      {mb = 1 : i64, n = 0 : i64}, {mb = 1 : i64, n = 1 : i64}],
       axes = ["mb", "n"], block_shape = array<i64: 1, 32>}
      : tensor<1x32xf16> -> !tt.tensordesc<1x32xf16>
  %pid = tt.get_program_id x : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c32 = arith.constant 32 : i32
  %lt = arith.cmpi slt, %pid, %c3 : i32
  scf.if %lt {
    %r = arith.divsi %pid, %c2 : i32
    %m = arith.remsi %pid, %c2 : i32
    %col = arith.muli %m, %c32 : i32
    // expected-error @+1 {{3 distinct blocks are read, and they span 4 places: every place in the box they span must be read for the destination to be whole}}
    %v = tt.descriptor_load %w[%r, %col] : !tt.tensordesc<1x32xf16> -> tensor<1x32xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x32xf16>
  }
  tt.return
}

// -----
// The descriptor made inside the if that guards the read. It is a collective that
// every tile makes, and the transfer goes ahead of the if, where it does not exist.
tt.func @descriptor_inside_the_guard(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %lt = arith.cmpi slt, %pid, %c2 : i32
  // expected-note @+1 {{this if}}
  scf.if %lt {
    %w = tts.make_distributed_descriptor %share
        {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
         axes = ["mb", ""], block_shape = array<i64: 1, 64>}
        : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
    // expected-error @+1 {{the distributed descriptor this reads through is made inside an if that guards the read; every tile makes the descriptor, so make it outside}}
    %v = tt.descriptor_load %w[%pid, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
    tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  }
  tt.return
}

// -----
// The reads below are not pinned, so each has to be a reduction's read (§7): the
// stack moves data between cores only scratchpad to scratchpad, through a pinned
// landing, or straight into a reduction across cores, so every use has to be a
// reduction taking the value directly. The four shares are composed on n,
// dimension 1, in every case.

// A copy with no landing: the gathered block is returned as it is.
tt.func @copy_without_a_landing(%x: tensor<4x16xf16>) -> tensor<4x64xf16> {
  %share = math.exp %x : tensor<4x16xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<4x16xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 4, 64>}
      : tensor<4x16xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{this read through a distributed descriptor is not pinned, and its value is used by 'tt.return', which is not a reduction: a read either lands in the scratchpad, pinned with tl.spyre_pin, or feeds a reduction directly over a dimension the descriptor divides}}
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  // expected-note @+1 {{used here}}
  tt.return %v : tensor<4x64xf16>
}

// -----
// Computing on the value is a use that is not a reduction, whatever follows it.
tt.func @element_wise_then_a_copy(%x: tensor<4x16xf16>) -> tensor<4x64xf16> {
  %share = math.exp %x : tensor<4x16xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<4x16xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 4, 64>}
      : tensor<4x16xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{this read through a distributed descriptor is not pinned, and its value is used by 'math.exp', which is not a reduction: a read either lands in the scratchpad, pinned with tl.spyre_pin, or feeds a reduction directly over a dimension the descriptor divides}}
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  // expected-note @+1 {{used here}}
  %e = math.exp %v : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// A reduction and a copy of one read: the reduction is fine, the copy is not.
tt.func @a_reduction_and_a_copy(%x: tensor<4x16xf16>) -> (tensor<4xf16>, tensor<4x64xf16>) {
  %share = math.exp %x : tensor<4x16xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<4x16xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 4, 64>}
      : tensor<4x16xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{this read through a distributed descriptor is not pinned, and its value is used by 'tt.return', which is not a reduction: a read either lands in the scratchpad, pinned with tl.spyre_pin, or feeds a reduction directly over a dimension the descriptor divides}}
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<4xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<4xf16>) -> tensor<4xf16>
  %r = linalg.reduce ins(%v : tensor<4x64xf16>) outs(%init : tensor<4xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  // expected-note @+1 {{used here}}
  tt.return %r, %v : tensor<4xf16>, tensor<4x64xf16>
}

// -----
// A reduction over dimension 0, which the descriptor does not divide: each core's
// columns are summed down, a gather followed by a local reduction, not a
// reduction across cores. The gathered block has to land.
tt.func @reduction_over_an_undivided_dimension(%x: tensor<4x16xf16>) -> tensor<64xf16> {
  %share = math.exp %x : tensor<4x16xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<4x16xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 4, 64>}
      : tensor<4x16xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{this read through a distributed descriptor is not pinned, and the reduction it feeds folds dimensions [0], which the descriptor does not divide: that is a gather followed by a local reduction, not a reduction across cores, so the gathered block has to land; pin the read with tl.spyre_pin}}
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<64xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<64xf16>) -> tensor<64xf16>
  // expected-note @+1 {{the reduction}}
  %r = linalg.reduce ins(%v : tensor<4x64xf16>) outs(%init : tensor<64xf16>) dimensions = [0]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  tt.return %r : tensor<64xf16>
}

// -----
// An operation that changes the shape: after the transpose, dimension 0 is the
// divided one, so the fold below would cross the cores, but the dimensions are
// not followed through a shape change, so it is refused like a copy.
tt.func @a_transpose_before_the_reduction(%x: tensor<4x16xf16>) -> tensor<4xf16> {
  %share = math.exp %x : tensor<4x16xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<4x16xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 4, 64>}
      : tensor<4x16xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{this read through a distributed descriptor is not pinned, and its value is used by 'tt.trans', which is not a reduction: a read either lands in the scratchpad, pinned with tl.spyre_pin, or feeds a reduction directly over a dimension the descriptor divides}}
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  // expected-note @+1 {{used here}}
  %t = tt.trans %v {order = array<i32: 1, 0>} : tensor<4x64xf16> -> tensor<64x4xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<4xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<4xf16>) -> tensor<4xf16>
  %r = linalg.reduce ins(%t : tensor<64x4xf16>) outs(%init : tensor<4xf16>) dimensions = [0]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  tt.return %r : tensor<4xf16>
}

// -----
// A pin on a value computed from the read is not the read's landing: the
// computation would read the received block before it lands.
tt.func @pin_after_a_computation(%x: tensor<4x16xf16>) {
  %share = math.exp %x : tensor<4x16xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<4x16xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 4, 64>}
      : tensor<4x16xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{this read through a distributed descriptor is not pinned, and its value is used by 'math.exp', which is not a reduction: a read either lands in the scratchpad, pinned with tl.spyre_pin, or feeds a reduction directly over a dimension the descriptor divides}}
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  // expected-note @+1 {{used here}}
  %e = math.exp %v : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// An element-wise operation before a reduction across cores. Each core could
// apply it to its own share before the fold, but whether the backend moves it
// there is not established, so the value has to land first.
tt.func @element_wise_before_the_reduction(%x: tensor<4x16xf16>, %y: tensor<4x64xf16>) -> tensor<4xf16> {
  %share = math.exp %x : tensor<4x16xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<4x16xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{n = 0 : i64}, {n = 1 : i64}, {n = 2 : i64}, {n = 3 : i64}],
       axes = ["", "n"], block_shape = array<i64: 4, 64>}
      : tensor<4x16xf16> -> !tt.tensordesc<4x64xf16>
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{this read through a distributed descriptor is not pinned, and its value is used by 'arith.mulf', which is not a reduction: a read either lands in the scratchpad, pinned with tl.spyre_pin, or feeds a reduction directly over a dimension the descriptor divides}}
  %v = tt.descriptor_load %w[%c0, %c0] : !tt.tensordesc<4x64xf16> -> tensor<4x64xf16>
  // expected-note @+1 {{used here}}
  %m = arith.mulf %v, %y : tensor<4x64xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<4xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<4xf16>) -> tensor<4xf16>
  %r = linalg.reduce ins(%m : tensor<4x64xf16>) outs(%init : tensor<4xf16>) dimensions = [1]
    (%in: f16, %acc: f16) {
      %s = arith.addf %in, %acc : f16
      linalg.yield %s : f16
    }
  tt.return %r : tensor<4xf16>
}
