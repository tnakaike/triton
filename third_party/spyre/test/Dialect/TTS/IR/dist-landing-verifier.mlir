// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// The tts.dist_landing ATTRIBUTE: its shape, as the dialect checks it.
//
// Unlike `tts.pin`, this attribute is not built from fields an op's verifier has
// already passed. AnalyzeDistributedDesc writes it, and hand-written IR can spell
// it, so the dialect reads it back through the same reader its consumer uses. What
// the reader cannot see -- the block shape, the source's extent, the grid -- is the
// analysis's to check, in Conversion/TritonToKTIR/AnalyzeDistributedDesc/.

// Two of four tiles read; the two lists agree on which.
// CHECK-LABEL: tt.func @attribute_verifies
// CHECK: tts.dist_landing = {dst_coordinates = [unit, unit, array<i64: 0, 0>, array<i64: 2, 0>], dst_shape = array<i64: 4, 64>, src_coordinates = [unit, unit, array<i64: 2, 0>, array<i64: 4, 0>]}
tt.func @attribute_verifies(%v: tensor<2x64xf16>) {
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32,
              tts.dist_landing = {
                src_coordinates = [unit, unit, array<i64: 2, 0>, array<i64: 4, 0>],
                dst_shape = array<i64: 4, 64>,
                dst_coordinates = [unit, unit, array<i64: 0, 0>, array<i64: 2, 0>]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
// Only on a landing marker: anywhere else it would be inert.
tt.func @not_on_a_pin(%x: tensor<2x64xf16>) -> tensor<2x64xf16> {
  // expected-error @+1 {{'tts.dist_landing' describes a landing, so it belongs on a tts.pin marker}}
  %e = math.exp %x {tts.dist_landing = {src_coordinates = [array<i64: 0, 0>],
                                        dst_shape = array<i64: 2, 64>,
                                        dst_coordinates = [array<i64: 0, 0>]}}
      : tensor<2x64xf16>
  tt.return %e : tensor<2x64xf16>
}

// -----
tt.func @not_a_dictionary(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: must be a dictionary of src_coordinates, dst_shape and dst_coordinates}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32, tts.dist_landing = unit}
      : tensor<2x64xf16>
  tt.return
}

// -----
tt.func @missing_dst_shape(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: 'dst_shape' must be a dense i64 array}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>],
                                  dst_coordinates = [array<i64: 0, 0>]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
// An unknown entry is refused rather than ignored: a typo would otherwise leave
// the attribute half-stated and still verifying.
tt.func @extra_entry(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: expected exactly the entries src_coordinates, dst_shape and dst_coordinates, got 4 entries}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>],
                                  dst_shape = array<i64: 2, 64>,
                                  dst_coordinates = [array<i64: 0, 0>],
                                  receivers = array<i64: 0>}}
      : tensor<2x64xf16>
  tt.return
}

// -----
tt.func @entry_not_coordinates(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: 'src_coordinates'[1] must be a dense i64 array, or unit for a tile that does not read}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>, 1 : i64],
                                  dst_shape = array<i64: 2, 64>,
                                  dst_coordinates = [array<i64: 0, 0>, unit]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
tt.func @negative_coordinate(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: 'dst_coordinates'[0] has a negative coordinate -2}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>],
                                  dst_shape = array<i64: 2, 64>,
                                  dst_coordinates = [array<i64: -2, 0>]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
tt.func @zero_extent(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: 'dst_shape' has a non-positive extent 0}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>],
                                  dst_shape = array<i64: 0, 64>,
                                  dst_coordinates = [array<i64: 0, 0>]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
// Both lists are indexed by tile, so they have the same length.
tt.func @lengths_differ(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: 'src_coordinates' has 2 entries and 'dst_coordinates' has 1: both have one per tile}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>, unit],
                                  dst_shape = array<i64: 2, 64>,
                                  dst_coordinates = [array<i64: 0, 0>]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
// And they agree on which tiles read: a read lands, and a landing was read.
tt.func @unit_in_one_list_only(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: tile 1 is unit in 'src_coordinates' only: a tile that reads also lands}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>, unit],
                                  dst_shape = array<i64: 4, 64>,
                                  dst_coordinates = [array<i64: 0, 0>, array<i64: 2, 0>]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
tt.func @no_reader(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: no tile reads: every entry is unit}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [unit, unit],
                                  dst_shape = array<i64: 2, 64>,
                                  dst_coordinates = [unit, unit]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
// The read and the landing are one block, so both coordinates have the
// destination's rank.
tt.func @rank_differs(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: tile 0's coordinates are not of the destination's rank 2}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0>],
                                  dst_shape = array<i64: 2, 64>,
                                  dst_coordinates = [array<i64: 0, 0>]}}
      : tensor<2x64xf16>
  tt.return
}

// -----
tt.func @outside_destination(%v: tensor<2x64xf16>) {
  // expected-error @+1 {{tts.dist_landing: tile 0's destination coordinate 64 on dimension 1 is outside the destination's extent 64}}
  tts.pin %v {memory_space = "ct_local", offset = 0 : i32,
              tts.dist_landing = {src_coordinates = [array<i64: 0, 0>],
                                  dst_shape = array<i64: 2, 64>,
                                  dst_coordinates = [array<i64: 0, 64>]}}
      : tensor<2x64xf16>
  tt.return
}
