// RUN: for f in %S/../../docs/inter-tile-examples/*-store/*.mlir; do spyre-triton-opt "$f" -o /dev/null || exit 1; done

// A CHECKER, not an example. The KTIR it runs on lives in
// docs/inter-tile-examples/, because that is documentation rather than test input
// -- nothing in this tree lowers `ktdp.construct_distributed_memory_view` yet and
// dbo-opt's legality check rejects it, so there is no pass for a lit test to drive
// and no device run to compare against.
//
// What is left worth pinning is that those files still parse and verify. The glob
// is `*-store/*.mlir`, so it covers BOTH spellings -- `dist-store/`, which composes
// the destination too, and `if-store/`, which lands each core's piece locally and
// derives the piece from the tile id. 70 files in all.
//
// They are generated from a pinned capture by
// `docs/inter-tile-examples/gen_ktir_inter_tile_examples.py` and hand-editing them
// is not expected, so the failure this catches is the KTDP dialect moving under
// them: an attribute renamed, an assembly format changed, a verifier tightened.
// Without this the examples would go stale silently and be discovered wrong by
// whoever next read them.
//
// This file's own body is never parsed -- the RUN line above does not reference
// %s. It is a `.mlir` file because `config.suffixes` admits only that extension.
