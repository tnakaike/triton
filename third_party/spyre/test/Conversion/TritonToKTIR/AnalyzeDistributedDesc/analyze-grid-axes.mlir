// RUN: spyre-triton-opt %s --analyze-distributed-desc=grid=2,2 -verify-diagnostics

// A tile is tt.get_program_id x only on a one-axis grid. On more axes the tile id
// is DistributeWork's linearization of all of them, which the evaluation does not
// model, so a pinned read through a distributed descriptor is refused there.
tt.func @two_axis_grid(%x: tensor<1x64xf16>) {
  %share = math.exp %x : tensor<1x64xf16>
  tts.pin %share {memory_space = "ct_local", offset = 0 : i32} : tensor<1x64xf16>
  %w = tts.make_distributed_descriptor %share
      {work_slices = [{mb = 0 : i64}, {mb = 1 : i64}, {mb = 2 : i64}, {mb = 3 : i64}],
       axes = ["mb", ""], block_shape = array<i64: 1, 64>}
      : tensor<1x64xf16> -> !tt.tensordesc<1x64xf16>
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  // expected-error @+1 {{a pinned read through a distributed descriptor needs a one-axis grid, and this one has 2 axes}}
  %v = tt.descriptor_load %w[%pid, %c0] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<1x64xf16>
  tt.return
}
