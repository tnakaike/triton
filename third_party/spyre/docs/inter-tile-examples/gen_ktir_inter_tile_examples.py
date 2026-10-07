#!/usr/bin/env python3
"""Generate KTIR inter-tile examples from the curated LX-relayout SDSC package.

Two spellings of the same 35 movements, into two sibling directories:

    dist-store/   both sides composed -- the destination is a distributed memory
                  view too, and the whole movement is one load and one store.
    if-store/     only the SOURCE is composed. Each core lands its own piece in
                  its own scratchpad, and WHICH piece comes from the tile id.

They exist to be read against each other: dist-store states the breadth of a
relayout in the type system, as one view per (piece, owner) pair; if-store states
it in control flow and index arithmetic. Neither is ratified -- see each
directory's README.

**The capture is not in this repo.** It is a pinned historical package, so this
script cannot be rerun from a clean checkout: pass its path. What it is for is
regenerating both forms when the KTDP dialect moves, instead of hand-editing 70
files and ~1600 memory views.

Usage:
    ./gen_ktir_inter_tile_examples.py [PACKAGE_DIR] [OUT_DIR]

OUT_DIR is the parent; the two form directories are created under it.
"""
import csv
import functools
import json
import operator
import pathlib
import re
import sys

DEFAULT_PKG = pathlib.Path(
    "/home/nakaike/dt-inductor/lx-relayout-sdsc-examples-for-ktir-20260915")
DEFAULT_OUT = pathlib.Path(
    "/home/nakaike/dt-inductor/triton/third_party/spyre/docs/inter-tile-examples")

# The capture these examples come from. Stated in the README too.
PROVENANCE = "36804f23ede70325c21ba234a7e102537a8eda95"

ELEM = {2: "f16", 4: "f32"}


def box_set(rank, lo, hi):
    """affine_set for the half-open box [lo, hi) over `rank` dims.

    Spelled the way MLIR prints one, so a generated file round-trips through
    spyre-triton-opt without the sets being restated: a lower bound at 0 is
    `d0 >= 0` rather than `d0 - 0 >= 0`, and an upper bound at 1 is `-d0 >= 0`.
    """
    parts = []
    for i in range(rank):
        d = f"d{i}"
        parts.append(f"{d} >= 0" if lo[i] == 0 else f"{d} - {lo[i]} >= 0")
        top = hi[i] - 1
        parts.append(f"-{d} >= 0" if top == 0 else f"-{d} + {top} >= 0")
    dims = ", ".join(f"d{i}" for i in range(rank))
    return f"affine_set<({dims}) : ({', '.join(parts)})>"


def row_major(size):
    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]
    return strides


def ident(name):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()


def owners_str(ids):
    """A core-id set, summarized. Every owner set in the capture is a single core
    or an exact arithmetic progression -- 0 irregular sets out of 3846 checked
    across all 130 relayouts -- so a stride is always enough to state one."""
    ids = sorted(ids)
    if len(ids) == 1:
        return f"ct {ids[0]}"
    steps = {ids[i + 1] - ids[i] for i in range(len(ids) - 1)}
    if len(steps) == 1:
        s = steps.pop()
        if s == 1:
            return f"ct {ids[0]}..{ids[-1]}"
        return f"ct {ids[0]}, {ids[0] + s}, ... {ids[-1]} (stride {s})"
    return "ct " + ", ".join(map(str, ids))  # not observed; stated exactly anyway


def range_set(rank, size):
    """affine_set for the half-open range [0, size) -- a range over the block shape.

    This is what `access_tile_set` is, and `buildAccessTile`
    (`lib/Dialect/KTDP/Transforms/Utility.cpp`) is the authority: it passes
    `buildRangeSetND(ctx, blockShape)` and no symbol operands, so the SET carries
    the block's extent and the BASE INDICES carry its position. A global box in
    the set would be a second, contradictory spelling of the position.
    """
    return box_set(rank, [0] * rank, list(size))


def dest_derivation(rec, keys, cores):
    """How a core finds the box it lands, dimension by dimension.

    Returns `{dim: ("const", v)}` or `{dim: ("arith", a, b, extent)}`, where the
    arithmetic form is

        start[dim] = ((tid // a) % b) * extent

    with `a` and `b` powers of two. The decomposition is PER DIMENSION and
    independent, which is the whole reason no branch is needed: a destination piece
    index can look like a permutation of 0..31 while each dimension's slice index is
    still a clean field of the tile id. C008 is the case that shows it -- its piece
    order runs 0, 1, 12, 23, 26, ... and yet `out` is `(tid // 8) % 4` and `x` is
    `tid % 8`.

    A dimension that does not vary across the destination pieces is `("const", v)`,
    and `v` is NOT always zero -- C035 lands every piece at `mb = 511`.

    Raises if a dimension admits no such form; every record in the pinned package
    does, checked over all 900 (core, dimension) pairs.
    """
    dps = rec["destination_pieces"]
    varying = [k for k in keys if len({p["start"][k] for p in dps}) > 1]
    core2piece = {o: i for i, p in enumerate(dps) for o in p["owners"]}
    out = {}
    for k in keys:
        if k not in varying:
            out[k] = ("const", dps[0]["start"][k])
            continue
        extent = dps[0]["size"][k]
        want = {c: dps[core2piece[c]]["start"][k] // extent for c in cores}
        for a in (1, 2, 4, 8, 16, 32):
            for b in (2, 4, 8, 16, 32):
                if all(want[c] == (c // a) % b for c in want):
                    out[k] = ("arith", a, b, extent)
                    break
            if k in out:
                break
        else:
            raise AssertionError(
                f"{rec['relayout']}: no (tid // a) % b form for dimension {k}")
    # Re-check the whole thing per core rather than trusting the search.
    for c in cores:
        got = {k: (v[1] if v[0] == "const" else ((c // v[1]) % v[2]) * v[3])
               for k, v in out.items()}
        assert got == {k: dps[core2piece[c]]["start"][k] for k in keys}, (
            rec["relayout"], c, got)
    return out


def dest_geometry(rec, keys):
    """The destination side, as the composed view has to describe it.

    Returns `(pieces, piece, lo, union, R, stride)`: the piece shape every
    destination piece shares, the origin and extent of the box they tile, the
    number of owners each piece has, and the stride of one piece's owner set.

    `union` is NOT always the tensor's extents -- C035's pieces all sit at
    `mb = 511`, so the box they tile is one row of 512 -- and the pieces tile it
    exactly, which is what makes a composed shape well defined.
    """
    dps = rec["destination_pieces"]
    piece = [dps[0]["size"][k] for k in keys]
    assert all([p["size"][k] for k in keys] == piece for p in dps), rec["relayout"]
    R = len(dps[0]["owners"])
    assert all(len(p["owners"]) == R for p in dps), rec["relayout"]
    lo = [min(p["start"][k] for p in dps) for k in keys]
    hi = [max(p["start"][k] + p["size"][k] for p in dps) for k in keys]
    union = [hi[i] - lo[i] for i in range(len(keys))]
    vol = lambda v: functools.reduce(operator.mul, v, 1)
    assert vol(piece) * len(dps) == vol(union), rec["relayout"]
    strides = set()
    for p in dps:
        o = sorted(p["owners"])
        strides |= {o[i + 1] - o[i] for i in range(len(o) - 1)} or {1}
    assert len(strides) == 1, (rec["relayout"], strides)
    return dps, piece, lo, union, R, strides.pop()


def replica_index(dps, cores, R, stride):
    """`None` where nothing is replicated, else `(a, b)` with

        replica = (tid // a) % b

    the position of a core in its piece's owner list -- which is WHICH of the
    `R` slots the composed destination holds for that piece this core writes.

    `a` is the owner set's own stride and `b` is `R`, so the form is read off the
    geometry rather than searched for; it is still checked per core, and `b` is
    not required to be a power of two (C003 has 25 owners, C004 28).
    """
    if R == 1:
        return None
    rep = {o: i for p in dps for i, o in enumerate(sorted(p["owners"]))}
    assert all(rep[c] == (c // stride) % R for c in cores), rep
    return stride, R


def emit_base(A, keys, deriv):
    """Emit the index arithmetic for a core's box, and return one base per dim.

    Shared by both spellings, which derive the same box: `dist-store` uses it for
    the read out of the composed source and, shifted, for the write into the
    composed destination; `if-store` for the read alone, its landing being local.
    """
    base = []
    for k in keys:
        v = deriv[k]
        if v[0] == "const":
            base.append(f"%c{v[1]}")
            continue
        _, a, b, extent = v
        q = "%tid"
        if a != 1:
            A(f"  %q_{k} = arith.divsi %tid, %c{a} : index")
            q = f"%q_{k}"
        A(f"  %i_{k} = arith.remsi {q}, %c{b} : index")
        if extent == 1:
            base.append(f"%i_{k}")
        else:
            A(f"  %o_{k} = arith.muli %i_{k}, %c{extent} : index")
            base.append(f"%o_{k}")
    return base


def participation(rec, grid):
    """`None` when every core participates, else a guard on the tile id.

    Every owner set in the package is an exact arithmetic progression (the README
    says so and the generator re-checks it), so the guard is a stride test, a
    bound, or an equality -- never a list of 28 comparisons.
    """
    cores = sorted({o for p in rec["destination_pieces"] for o in p["owners"]})
    n = len(cores)
    if n == grid:
        return None
    stride = cores[1] - cores[0] if n > 1 else 1
    assert all(cores[i] - cores[i - 1] == stride for i in range(1, n)), cores
    assert cores[0] == 0, cores
    if n == 1:
        return [("eq", 0)]
    if stride == 1:
        return [("lt", n)]
    # offset 0, so the stride test is `tid % stride == 0`; the upper bound is
    # implied whenever stride * n covers the grid, which it does in every record.
    assert stride * n == grid, (stride, n, grid)
    return [("mod0", stride)]


def emit_dist(row, rec):
    """One curated record -> one self-contained KTIR example.

    BOTH sides are composed. The source pieces become one distributed view and the
    destination (piece, owner) pairs become another, so the function is the whole
    movement rather than one tile's share of it -- see the README on what that buys
    and what it leaves to a lowering.

    **The composed destination is the WHOLE of what its views hold**, not the box
    they cover: `R` owners of one piece are `R` slots, so the result shape is the
    union of the pieces with dimension 0 multiplied by `R`. C002's destination is
    the whole 4096-element tensor held by 16 cores, so its composed type is
    `memref<65536x1x1xf16>`.

    That is what puts the tile id in this spelling: with one slot per (piece,
    owner) there is no coordinate two cores write, and so a store has to say WHICH
    slot -- the piece from the same fields of the tile id `if-store` uses, and the
    replica from one more.

    **Participation is not stated, because it follows.** The slot a core addresses
    carries its holder's `ct_id`, so a core that addresses a slot it does not own
    is asking for a remote write; nothing has to say so in control flow. Checked
    over all 35 and every core: the set of cores whose addressed slot is their own
    is exactly the owner set, so a guard would only repeat the two composed types.
    Leaving it out is also what keeps this spelling neutral on transport, which an
    `scf.if` on the owner would have decided for pull.
    """
    keys = list(rec["source_extents"])
    rank = len(keys)
    elem = ELEM[rec["word_length"]]
    extents = [rec["source_extents"][k] for k in keys]

    named, order = {}, []

    def name_for(text, stem):
        if text not in named:
            n = sum(1 for v in named.values() if v.startswith("#" + stem))
            named[text] = f"#{stem}{n}"
            order.append((named[text], text))
        return named[text]

    def box_name(piece, stem):
        lo = [piece["start"][k] for k in keys]
        hi = [lo[i] + piece["size"][k] for i, k in enumerate(keys)]
        return name_for(box_set(rank, lo, hi), stem), [piece["size"][k] for k in keys]

    # One view per source piece; one per destination (piece, owner) PAIR, because a
    # piece with several owners is that many views differing only in ct_id.
    src = [(p, p["owners"][0], *box_name(p, "src")) for p in rec["source_pieces"]]
    dst = [(p, o, *box_name(p, "dst"))
           for p in rec["destination_pieces"] for o in p["owners"]]

    dps, piece, dlo, union, R, ostride = dest_geometry(rec, keys)
    cores = sorted({o for p in dps for o in p["owners"]})
    deriv = dest_derivation(rec, keys, cores)
    rep = replica_index(dps, cores, R, ostride)
    # The slots of one piece stack on dimension 0, so the composed destination is
    # the union with that one dimension scaled by the replication factor.
    agg = [union[0] * R] + union[1:]

    blk = name_for(range_set(rank, piece), "blk")
    dims = ", ".join(f"d{i}" for i in range(rank))
    omap = name_for(f"affine_map<({dims}) -> ({dims})>", "ord")

    src_ty = f"memref<{'x'.join(map(str, extents))}x{elem}>"
    agg_ty = f"memref<{'x'.join(map(str, agg))}x{elem}>"
    pc = "x".join(map(str, piece))
    tile_ty = f"!ktdp.access_tile<{pc}xindex>"
    idx_ty = f"<{pc}xindex>"
    tensor_ty = f"tensor<{pc}x{elem}>"

    consts = {0}

    def c(v):
        consts.add(v)
        return f"%c{v}"

    for v in deriv.values():
        if v[0] == "const":
            c(v[1])
        else:
            c(v[1]); c(v[2]); c(v[3])
    if rep:
        c(rep[0]); c(rep[1]); c(union[0])

    def view_lines(tag, entries, offset):
        out = []
        for i, (p, owner, boxnm, size) in enumerate(entries):
            sp = f"#ktdp.memory_space<ct_local, ct_id = {owner}>"
            out.append(f"  %{tag}{i} = ktdp.construct_memory_view {offset}, "
                       f"sizes: [{', '.join(map(str, size))}], "
                       f"strides: [{', '.join(map(str, row_major(size)))}]")
            out.append(f"      {{coordinate_set = {boxnm}, memory_space = {sp}}}")
            out.append(f"      : memref<{'x'.join(map(str, size))}x{elem}, {sp}>")
        return out

    def compose_lines(tag, entries, result, result_ty):
        ops = ", ".join(f"%{tag}{i}" for i in range(len(entries)))
        tys = ", ".join(
            f"memref<{'x'.join(map(str, e[3]))}x{elem}, "
            f"#ktdp.memory_space<ct_local, ct_id = {e[1]}>>" for e in entries)
        return [f"  %{result} = ktdp.construct_distributed_memory_view ({ops}",
                f"      : {tys}) : {result_ty}"]

    ext_str = ", ".join(f"{k}: {rec['source_extents'][k]}" for k in keys)
    L = []
    A = L.append
    A(f"// {row['curated_id']} -- {rec['phase']}, {rec['consumer_family']} "
      f"input{rec['consumer_input_lds']}, consumer {rec['consumer']}")
    A(f"// {rec['route_class']}")
    A(f"// tensor {rec['tensor']}, rank {rank} ({', '.join(keys)}), "
      f"extents {{{ext_str}}}, {elem}")
    A("//")
    A(f"// source      {len(src):>2} view(s) = {len(rec['source_pieces'])} piece(s) "
      f"x 1 owner, {owners_str([e[1] for e in src])}")
    A(f"// destination {len(dst):>2} view(s) = "
      f"{len(rec['destination_pieces'])} piece(s) x "
      f"{len(rec['destination_pieces'][0]['owners'])} owner(s), "
      f"{owners_str([e[1] for e in dst])}")
    A(f"// composed    source {'x'.join(map(str, extents))}, "
      f"destination {'x'.join(map(str, agg))} = "
      f"{'x'.join(map(str, union))} x {R} slot(s) per piece")
    A("//")
    A("// Both sides composed, so the function states the whole movement and needs no")
    A("// launch table to be read. The destination view is the whole of what its views")
    A("// hold -- one slot per (piece, owner) -- so a core says which slot it writes:")
    for k in keys:
        v = deriv[k]
        A(f"//     {k} = {v[1]}" if v[0] == "const"
          else f"//     {k} = ((tid // {v[1]}) % {v[2]}) * {v[3]}")
    if rep:
        A(f"//     replica = (tid // {rep[0]}) % {rep[1]}, "
          f"at {union[0]} per slot on dimension 0")
    A(f"// Which cores take part is not stated: {len(cores)} of the 32 address a slot they")
    A("// own, and the slot's own ct_id is what says so. See README.md for the mapping")
    A("// and the provenance.")
    A("")
    for name, text in order:
        A(f"{name} = {text}")
    A("")

    fn = ident(f"{row['curated_id']}_{rec['phase']}_{rec['consumer_family']}"
               f"_input{rec['consumer_input_lds']}")
    A(f"func.func @{fn}(%src: index, %dst: index) {{")
    for v in sorted(consts):
        A(f"  %c{v} = arith.constant {v} : index")
    # One destination piece held by one core needs no tile id at all: there is one
    # slot, every core addresses it, and its ct_id says whose it is.
    if rep or any(v[0] == "arith" for v in deriv.values()):
        A("  %tid = ktdp.get_compute_tile_id : index")
    A("")
    A("  // The source distribution: one view per partition, differing ONLY in")
    A("  // coordinate_set and ct_id.")
    L.extend(view_lines("s", src, "%src"))
    A("")
    L.extend(compose_lines("s", src, "from", src_ty))
    A("")
    A("  // The destination distribution, the same way. A piece with several owners")
    A("  // is that many views with the SAME coordinate_set and different ct_id --")
    A("  // which is what replication is, stated rather than left to the launch. The")
    A("  // composed type counts those views, so the slots are distinct coordinates")
    A("  // and no coordinate has two writers.")
    L.extend(view_lines("d", dst, "%dst"))
    A("")
    L.extend(compose_lines("d", dst, "to", agg_ty))
    A("")
    mark = len(L)
    if any(v[0] == "arith" for v in deriv.values()):
        A("  // Where this core's box sits: one independent field of the tile id per")
        A("  // divided dimension, exactly as in ../if-store/. No branch selects it --")
        A("  // the index is computed.")
    base = emit_base(A, keys, deriv)

    # The destination side is indexed in the COMPOSED view's own space: a constant
    # dimension is the union's origin, so it is 0 there whatever its global value,
    # and dimension 0 carries the replica.
    store = ["%c0" if deriv[k][0] == "const" else base[i]
             for i, k in enumerate(keys)]
    if rep:
        a, b = rep
        if len(L) > mark:
            A("")
        A("  // Which of the slots is this core's: its position in its piece's owner")
        A("  // list, stacked on dimension 0 of the composed destination.")
        q = "%tid"
        if a != 1:
            A(f"  %q_rep = arith.divsi %tid, %c{a} : index")
            q = "%q_rep"
        A(f"  %i_rep = arith.remsi {q}, %c{b} : index")
        slot = "%i_rep"
        if union[0] != 1:
            A(f"  %o_rep = arith.muli %i_rep, %c{union[0]} : index")
            slot = "%o_rep"
        if store[0] == "%c0":
            store[0] = slot
        else:
            A(f"  %base_{keys[0]} = arith.addi {slot}, {store[0]} : index")
            store[0] = f"%base_{keys[0]}"
    if len(L) > mark:
        A("")

    def movement(indent):
        pad = " " * indent
        A(f"{pad}%rt = ktdp.construct_access_tile %from[{', '.join(base)}]")
        A(f"{pad}    {{access_tile_order = {omap}, access_tile_set = {blk}}}")
        A(f"{pad}    : {src_ty} -> {tile_ty}")
        A(f"{pad}%val = ktdp.load %rt : {idx_ty} -> {tensor_ty}")
        A(f"{pad}%wt = ktdp.construct_access_tile %to[{', '.join(store)}]")
        A(f"{pad}    {{access_tile_order = {omap}, access_tile_set = {blk}}}")
        A(f"{pad}    : {agg_ty} -> {tile_ty}")
        A(f"{pad}ktdp.store %val, %wt : {tensor_ty}, {idx_ty}")

    A("  // The movement, and no control flow: one load of a coordinate REGION --")
    A("  // which source partitions that touches is the composed view's to resolve,")
    A("  // and a region may span several -- and one store into a named slot of the")
    A("  // destination, whose ct_id is what says whether this core owns it.")
    movement(2)
    A("")
    A("  return")
    A("}")
    return "\n".join(L) + "\n"



def emit_if(row, rec, grid=32):
    """One curated record -> the if-store spelling of the same movement.

    Only the SOURCE is composed. Each core lands its own piece in its own
    scratchpad, so the destination needs no distributed view -- which is what §4
    of the design asks for. What replaces it is the tile id: by arithmetic where
    the owner map is a quotient or a remainder, by one `scf.if` per piece where it
    is a permutation.

    The landing view and its access tile are hoisted above any guard, because they
    are identical on every core: the landing is LOCAL, so its `coordinate_set` is a
    range over the piece and not a box in the tensor's global space. Only global
    coordinates vary per core, and they live in the load's base indices.

    `ktdp.load` stays INSIDE the guard with the store. The load is the transfer, so
    hoisting it -- which `Pure` would permit -- would have every core read all P
    pieces and keep one.
    """
    keys = list(rec["source_extents"])
    rank = len(keys)
    elem = ELEM[rec["word_length"]]
    extents = [rec["source_extents"][k] for k in keys]

    named, order = {}, []

    def name_for(text, stem):
        if text not in named:
            n = sum(1 for v in named.values() if v.startswith("#" + stem))
            named[text] = f"#{stem}{n}"
            order.append((named[text], text))
        return named[text]

    def box_name(piece, stem):
        lo = [piece["start"][k] for k in keys]
        hi = [lo[i] + piece["size"][k] for i, k in enumerate(keys)]
        return name_for(box_set(rank, lo, hi), stem), [piece["size"][k] for k in keys]

    src = [(p, p["owners"][0], *box_name(p, "src")) for p in rec["source_pieces"]]
    dps = rec["destination_pieces"]
    piece = [dps[0]["size"][k] for k in keys]
    assert all([p["size"][k] for k in keys] == piece for p in dps), row["curated_id"]

    blk = name_for(range_set(rank, piece), "blk")
    dims = ", ".join(f"d{i}" for i in range(rank))
    omap = name_for(f"affine_map<({dims}) -> ({dims})>", "ord")

    whole_ty = f"memref<{'x'.join(map(str, extents))}x{elem}>"
    pc = "x".join(map(str, piece))
    piece_ty = f"memref<{pc}x{elem}>"
    tile_ty = f"!ktdp.access_tile<{pc}xindex>"
    idx_ty = f"<{pc}xindex>"
    tensor_ty = f"tensor<{pc}x{elem}>"

    cores = sorted({o for dp in dps for o in dp["owners"]})
    deriv = dest_derivation(rec, keys, cores)
    guard = participation(rec, grid)

    consts = {0}
    def c(v):
        consts.add(v)
        return f"%c{v}"

    # Every constant the body will reference, collected before the block is rendered.
    for k, v in deriv.items():
        if v[0] == "const":
            c(v[1])
        else:
            c(v[1]); c(v[2]); c(v[3])
    for g in guard or []:
        c(g[1])

    L = []
    A = L.append
    A(f"// {row['curated_id']} -- {rec['phase']}, {rec['consumer_family']} "
      f"input{rec['consumer_input_lds']}, consumer {rec['consumer']}")
    A(f"// {rec['route_class']}")
    A(f"// tensor {rec['tensor']}, rank {rank} ({', '.join(keys)}), "
      f"extents {{{', '.join(f'{k}: {rec['source_extents'][k]}' for k in keys)}}}, {elem}")
    A("//")
    A(f"// source      {len(src):>2} view(s) = {len(rec['source_pieces'])} piece(s) "
      f"x 1 owner, {owners_str([e[1] for e in src])}")
    A(f"// destination {len(dps)} piece(s) x {len(dps[0]['owners'])} owner(s) "
      f"-- NOT composed. Each core lands its own.")
    A("//")
    A("// if-store: the source is composed, the destination is this core's own")
    A("// scratchpad, and the box a core lands comes from the tile id:")
    for k in keys:
        v = deriv[k]
        A(f"//     {k} = {v[1]}" if v[0] == "const"
          else f"//     {k} = ((tid // {v[1]}) % {v[2]}) * {v[3]}")
    A("// The counterpart spelling is ../dist-store/, which composes both sides.")
    A("")
    for name, text in order:
        A(f"{name} = {text}")
    A("")

    fn = ident(f"{row['curated_id']}_{rec['phase']}_{rec['consumer_family']}"
               f"_input{rec['consumer_input_lds']}")
    A(f"func.func @{fn}(%src: index, %dst: index) {{")
    for v in sorted(consts):
        A(f"  %c{v} = arith.constant {v} : index")
    A("  %tid = ktdp.get_compute_tile_id : index")
    A("")
    A("  // The source distribution: one view per partition, differing ONLY in")
    A("  // coordinate_set and ct_id. Identical to the dist-store form.")
    for i, (pp, owner, boxnm, size) in enumerate(src):
        sp = f"#ktdp.memory_space<ct_local, ct_id = {owner}>"
        A(f"  %s{i} = ktdp.construct_memory_view %src, "
          f"sizes: [{', '.join(map(str, size))}], "
          f"strides: [{', '.join(map(str, row_major(size)))}]")
        A(f"      {{coordinate_set = {boxnm}, memory_space = {sp}}}")
        A(f"      : memref<{'x'.join(map(str, size))}x{elem}, {sp}>")
    A("")
    ops = ", ".join(f"%s{i}" for i in range(len(src)))
    tys = ", ".join(f"memref<{'x'.join(map(str, e[3]))}x{elem}, "
                    f"#ktdp.memory_space<ct_local, ct_id = {e[1]}>>" for e in src)
    A(f"  %from = ktdp.construct_distributed_memory_view ({ops}")
    A(f"      : {tys}) : {whole_ty}")
    A("")
    A("  // The landing: the piece, in THIS core's scratchpad. No ct_id, because a")
    A("  // core writes only its own; and the coordinate_set is a range over the")
    A("  // piece rather than a box in the tensor, because the view is local. Both")
    A("  // make it identical on every core, so it is hoisted with its access tile.")
    A(f"  %land = ktdp.construct_memory_view %dst, "
      f"sizes: [{', '.join(map(str, piece))}], "
      f"strides: [{', '.join(map(str, row_major(piece)))}]")
    A(f"      {{coordinate_set = {blk}, memory_space = #ktdp.memory_space<ct_local>}}")
    A(f"      : {piece_ty}")
    A(f"  %wt = ktdp.construct_access_tile %land[{', '.join('%c0' for _ in extents)}]")
    A(f"      {{access_tile_order = {omap}, access_tile_set = {blk}}}")
    A(f"      : {piece_ty} -> {tile_ty}")
    A("")

    # The base index per dimension. No branch selects it: each dimension is an
    # independent field of the tile id.
    A("  // Where in the tensor this core's box starts: one independent field of the")
    A("  // tile id per divided dimension. No branch, because nothing is selected --")
    A("  // the index is computed.")
    base = emit_base(A, keys, deriv)
    A("")

    def movement(indent):
        pad = " " * indent
        A(f"{pad}%rt = ktdp.construct_access_tile %from[{', '.join(base)}]")
        A(f"{pad}    {{access_tile_order = {omap}, access_tile_set = {blk}}}")
        A(f"{pad}    : {whole_ty} -> {tile_ty}")
        A(f"{pad}%val = ktdp.load %rt : {idx_ty} -> {tensor_ty}")
        A(f"{pad}ktdp.store %val, %wt : {tensor_ty}, {idx_ty}")

    A("  // The movement. One load of a coordinate REGION -- which source partitions")
    A("  // that touches is the composed view's to resolve, exactly as in dist-store,")
    A("  // and a region may span several of them.")
    if guard:
        kindg, val = guard[0]
        if kindg == "eq":
            A(f"  %part = arith.cmpi eq, %tid, %c{val} : index")
        elif kindg == "lt":
            A(f"  %part = arith.cmpi slt, %tid, %c{val} : index")
        else:
            A(f"  %r = arith.remsi %tid, %c{val} : index")
            A(f"  %part = arith.cmpi eq, %r, %c0 : index")
        A("  // Not every core takes part, so the load is guarded with the store: the")
        A("  // load IS the transfer, and a core that lands nothing must read nothing.")
        A("  scf.if %part {")
        movement(4)
        A("  }")
    else:
        movement(2)
    A("")
    A("  return")
    A("}")
    return "\n".join(L) + "\n"




def _geom(records):
    """Per record: the derivation, the participation guard, and the source spanning."""
    out = []
    for row, rec in records:
        keys = list(rec["source_extents"])
        dps, sps = rec["destination_pieces"], rec["source_pieces"]
        cores = sorted({o for p in dps for o in p["owners"]})
        deriv = dest_derivation(rec, keys, cores)
        guard = participation(rec, 32)

        def overlap(a, b):
            return all(max(a[0][k], b[0][k]) < min(a[1][k], b[1][k]) for k in keys)

        spans = []
        for dp in dps:
            d = ({k: dp["start"][k] for k in keys},
                 {k: dp["start"][k] + dp["size"][k] for k in keys})
            spans.append(sum(1 for sp in sps if overlap(
                d, ({k: sp["start"][k] for k in keys},
                    {k: sp["start"][k] + sp["size"][k] for k in keys}))))
        out.append((row["curated_id"], deriv, guard, len(cores), max(spans)))
    return out


def _spanning(records):
    return sum(1 for _, _, _, _, m in _geom(records) if m > 1)


def _max_span(records):
    return max(m for _, _, _, _, m in _geom(records))


def _dim_hist(records):
    """How many dimensions a file derives from the tile id."""
    import collections
    c = collections.Counter(
        sum(1 for v in deriv.values() if v[0] == "arith")
        for _, deriv, _, _, _ in _geom(records))
    label = {0: "no dimension divided -- one destination piece",
             1: "one dimension from the tile id",
             2: "two dimensions, independently",
             3: "three dimensions, independently"}
    return [(label.get(k, f"{k} dimensions"), c[k]) for k in sorted(c)]


def _guarded(records):
    out = []
    for cid, _deriv, guard, n, _m in _geom(records):
        if not guard:
            continue
        kind, val = guard[0]
        g = {"eq": f"tid == {val}", "lt": f"tid < {val}"}.get(
            kind, f"tid % {val} == 0")
        out.append((cid, n, g))
    return out


def readme(records, form):
    """The index for one directory, written from the same records the examples were.

    `form` is "index" for the parent, "dist" or "if" for a spelling. The shared
    facts -- provenance, the mapping, what the set covers, the table of 35 -- live in
    the parent so the two spellings are not two copies of them.
    """
    L = []
    A = L.append

    if form == "index":
        A("# KTIR examples: inter-tile communication")
        A("")
        A("Thirty-five relayouts a real model performs, each written as KTIR, in **two")
        A("spellings of the same movement**:")
        A("")
        A("| | |")
        A("|---|---|")
        A("| [`dist-store/`](dist-store/) | both sides composed. The destination is a distributed memory view too, so a core stores into **its slot of the whole** rather than into a buffer of its own. |")
        A("| [`if-store/`](if-store/) | only the **source** is composed. Each core lands its own piece in its own scratchpad, at local coordinate zero. |")
        A("")
        A("**Neither is decided.** They are here to be read against each other, and each")
        A("directory\'s README states what its spelling buys and what it costs. The question")
        A("they are evidence for is open: §4 of")
        A("[inter-tile-lowering-to-mem-view.md](../inter-tile-lowering-to-mem-view.md)")
        A("composes only the source, and whether a composed destination is the better")
        A("spelling -- or admissible at all -- is what these two sets exist to inform.")
        A("")
        A("The one thing both forms agree on, and which an earlier revision got wrong, is")
        A("that **the breadth of a relayout belongs in the IR**. That revision wrote each")
        A("example from one tile\'s point of view, with a comment saying the other cores do")
        A("the same: C001 lands the whole tensor on all 32 cores, and the only \"32\" in the")
        A("file was prose. `dist-store` puts it in the type system, as one view per")
        A("*(piece, owner)* pair; `if-store` leaves it out of the destination side entirely,")
        A("because a landing in this core\'s own scratchpad says nothing about the others.")
        A("")
        A("**Both forms derive a core\'s box from the tile id**, by the same arithmetic.")
        A("That used to be `if-store`\'s distinguishing feature and is not: once the composed")
        A("destination is the whole of what its views hold, a store has to say which slot of")
        A("it this core writes. What is left of the contrast is two lines:")
        A("")
        A("| | `dist-store` | `if-store` |")
        A("|---|---|---|")
        A("| the store\'s destination | the composed view, at this core\'s slot: the piece\'s position in the box the pieces tile, plus its replica on dimension 0 | a local view, at zero. Which global coordinates it holds is not in the IR |")
        A("| participation | **not stated.** The slot carries its holder\'s `ct_id`, so a core that addresses a slot it does not own is asking for a remote write | an `scf.if` on the tile id, in %d of the 35. A local landing names no holder, so nothing else could say it" % len(_guarded(records)))
        A("")
        A("The second line is the sharper difference, and it is the one to argue about: it")
        A("says a composed destination makes participation a **consequence** of the two")
        A("statements the file already makes, where a local landing makes it a third")
        A("statement that could disagree with them. Checked over all 35 and every core: the")
        A("cores whose addressed slot is their own are exactly the owner set, so the guard")
        A("`dist-store` does not have would have been redundant with its types.")
        A("")
        A("They are **examples, not tests**. Nothing lowers")
        A("`ktdp.construct_distributed_memory_view` in tree yet and dbo-opt\'s legality check")
        A("rejects it, so the bar these meet is that they parse and verify. One lit test,")
        A("`test/docs/inter-tile-examples-parse.mlir`, keeps both directories at that bar.")
        A("")
        A("The counterpart set is `test/fixtures/inter_tile_*/`, which approaches the same")
        A("subject from the other end: hand-written Triton kernels for a movement family,")
        A("each with a *Target KTIR* section saying what its lowering should produce. Those")
        A("are chosen to isolate one question at a time; these are whatever a real model did.")
        A("")
        A("## How the SDSC becomes KTIR")
        A("")
        A("| Source record | KTIR |")
        A("|---|---|")
        A("| a piece\'s `start` and `size` | the `coordinate_set`, as the box `[start, start + size)` in the tensor\'s global index space |")
        A("| each of that piece\'s `owners` | one view per owner, `memory_space = #ktdp.memory_space<ct_local, ct_id = N>`, repeated in the result memref type |")
        A("| a side\'s pieces and their owners | that side\'s composed result shape: the box the pieces tile, with dimension 0 scaled by the number of owners each piece has |")
        A("| `word_length` | the element type: 2 bytes, so `f16` throughout |")
        A("")
        A("**A composed view is the whole of what its views hold**, which is not the same")
        A("thing as the box they cover, and two of the 35 show why each half of that")
        A("matters. The rule is one sentence -- every `(piece, owner)` pair is a slot of the")
        A("result, and the slots of one piece stack on dimension 0 -- and the consequences")
        A("are:")
        A("")
        A("- **Replication multiplies.** C002\'s destination is the whole 4096-element")
        A("  tensor held by 16 cores, so its composed type is `memref<65536x1x1xf16>` and")
        A("  not `memref<4096x1x1xf16>`. The 16 copies are real memory and each is")
        A("  addressable; a type that named the coordinates once would describe a sixteenth")
        A("  of what the views hold.")
        A("- **A side need not cover the tensor.** C035\'s destination pieces all sit at")
        A("  `mb = 511`, so the box they tile is `1x4096x1` -- one row of 512 -- and that is")
        A("  its composed extent. An earlier revision used the tensor\'s extents on both")
        A("  sides and typed it `512x4096x1`, a view 512 times the memory behind it.")
        A("")
        A("So the two sides of one movement do **not** in general have the same composed")
        A("type, even though the source and destination *extents* are identical in all 35.")
        A("A load and a store therefore name a box rather than the whole: the box is this")
        A("core\'s piece, and its position is where each spelling differs.")
        A("")
        A("Three conventions the source record does not dictate:")
        A("")
        A("**One `%src` and one `%dst`.** The design requires the views on a side to differ")
        A("*only* in `coordinate_set` and `ct_id`, which means one address, at the same place")
        A("in each core\'s scratchpad. The capture carries no LX addresses, so these are")
        A("`index` arguments rather than invented constants.")
        A("")
        A("**`access_tile_set` is a range over the block shape**, not the box in global")
        A("coordinates. `buildAccessTile`")
        A("(`lib/Dialect/KTDP/Transforms/Utility.cpp`) is the authority: it passes")
        A("`buildRangeSetND(ctx, blockShape)` and no symbol operands, so the **set** carries")
        A("the block\'s extent and the **base indices** carry its position. Where a piece")
        A("starts at the origin the two spellings coincide, which is why a generated name")
        A("like `#src0` can appear in both roles -- the sets are textually identical and the")
        A("generator emits each one once.")
        A("")
        A("**Core ids are never arbitrary.** Across all 130 relayouts in the package -- not")
        A("just these 35 -- every owner set is either a single core or an exact arithmetic")
        A("progression: 3109 single, 542 contiguous, and 195 strided by 2, 4, 8 or 16. Zero")
        A("irregular sets out of 3846 checked. That is what lets `if-store` write a")
        A("participation guard as a stride test rather than a list of 28 comparisons, and it")
        A("is what makes a holder derivable from a work-slice coordinate. The format *could*")
        A("express an irregular set, so a verifier should not rely on this.")
        A("")
        A("## What the set does and does not cover")
        A("")
        A("Ranks run 3 to 6. Fan-in runs from 1 source piece (C020, a broadcast) to 32.")
        A("")
        A("Every **source** piece has exactly one owner, so a region held by several tiles on")
        A("the *source* side does not appear here at all. Destination replication is")
        A("everywhere, up to all 32 cores holding the whole tensor.")
        A("")
        A("## The 35")
        A("")
        A("| ID | Phase | Route class | Pieces | Rank | Consumer | Tensor |")
        A("|---|---|---|---:|---:|---|---|")
        for row, rec in records:
            stem = row["curated_file"][:-5]
            A(f"| {row['curated_id']} "
              f"([dist](dist-store/{stem}.mlir), [if](if-store/{stem}.mlir)) "
              f"| {rec['phase']} | `{rec['route_class']}` "
              f"| {len(rec['source_pieces'])} -> {len(rec['destination_pieces'])} "
              f"| {len(rec['source_extents'])} | {rec['consumer']} | {rec['tensor']} |")
        A("")
        A("## Provenance, and regenerating")
        A("")
        A("Generated from `lx-relayout-sdsc-examples-for-ktir-20260915`, whose canonical")
        A("Granite artifacts are pinned to commit")
        A("")
        A(f"    {PROVENANCE}")
        A("")
        A("originally at `AdnanHoque/torch-spyre/experiments/granite_relayout/`. The package")
        A("holds 130 relayout records in total; these 35 are its `curated/` set, one")
        A("representative per distinct combination of phase, consumer, movement class, piece")
        A("counts, extents and owner geometry.")
        A("")
        A("`gen_ktir_inter_tile_examples.py` here writes both directories. **The capture is")
        A("not in this repo** -- it is a pinned historical package -- so pass its path:")
        A("")
        A("    ./gen_ktir_inter_tile_examples.py PACKAGE_DIR .")
        A("")
        A("It exists so that a KTDP dialect change is answered by regenerating rather than")
        A("by hand-editing 70 files and some 1600 memory views.")
        A("")
        A("A capture is not a specification: do not read performance, correctness or current")
        A("compiler support out of these. They say what one recorded run needed moved, and")
        A("how that movement is spelled in KTIR.")
        return "\n".join(L) + "\n"

    if form == "dist":
        A("# dist-store: both sides composed")
        A("")
        A("One of two spellings of the same 35 movements. The other is")
        A("[`../if-store/`](../if-store/), and [the parent README](../README.md) carries the")
        A("provenance, the SDSC mapping and the table of 35. **Neither spelling is decided.**")
        A("")
        A("## What one file contains")
        A("")
        A("| | |")
        A("|---|---|")
        A("| the source distribution | one `ktdp.construct_memory_view` per source piece, composed by `ktdp.construct_distributed_memory_view` |")
        A("| the destination distribution | the same, one view per destination *(piece, owner)* pair, composed the same way |")
        A("| the movement | `ktdp.get_compute_tile_id` and the index arithmetic for this core\'s box, then one `ktdp.construct_access_tile` + `ktdp.load` on the source and one tile + `ktdp.store` into this core\'s slot of the destination |")
        A("")
        A("**There is no control flow in these files.** Every core runs the same two")
        A("transfers, and which of them is the slot\'s holder is carried by the `ct_id` in")
        A("the composed destination\'s type rather than by a guard -- see the parent README\'s")
        A("participation row. Two files have no tile id either (C010, C019: one piece, one")
        A("owner, so there is one slot and nothing to select).")
        A("")
        A("**The composed destination is the whole of what its views hold.** Every")
        A("*(piece, owner)* pair is a slot of the result and the slots of one piece stack on")
        A("dimension 0, so C002 -- the whole 4096-element tensor on 16 cores -- composes to")
        A("`memref<65536x1x1xf16>`. [The parent README](../README.md) states the rule and")
        A("the other case it decides, C035, whose destination covers one row of 512.")
        A("")
        A("## What composing the destination buys")
        A("")
        A("**The file states its own movement.** A piece with several owners is that many")
        A("views with the **same** `coordinate_set` and a different `ct_id`, which is exactly")
        A("what replication is. C001\'s destination is one piece held by all 32 cores, and")
        A("here that is 32 views -- the breadth is in the IR rather than in the launch.")
        A("")
        A("**It answers §4\'s objection rather than setting it aside.**")
        A("[§4](../../inter-tile-lowering-to-mem-view.md) calls a destination view")
        A("\"meaningless where destinations are replicated\", because it would have \"two")
        A("writers for one coordinate with nothing saying which wins\". Under the rule above")
        A("there are no two writers: the 16 copies of C002 are 16 slots, each written by its")
        A("own owner, and a reader of the IR can say which core wrote which bytes. The")
        A("objection holds against a composed destination typed as the coordinates once,")
        A("which is what an earlier revision of these files emitted.")
        A("")
        A("**Ownership stays readable at the destination.** A store names the global box it")
        A("lands, so the IR says which coordinates a core holds; `if-store` moves that into")
        A("the load\'s base indices and leaves the landing anonymous.")
        A("")
        A("**It takes no side on transport.** With both distributions named and no guard on")
        A("the store, *which* side does the transferring is a lowering\'s choice: pull, where")
        A("the slot\'s holder is the one that moves it, or push. A guard on \"am I the holder\"")
        A("would have chosen pull in the IR, which is `if-store`\'s shape and is the reason")
        A("there is no `scf.if` here.")
        A("")
        A("## What it costs")
        A("")
        A("**A written distributed view implies remote writes.** §4\'s other objection")
        A("stands: cross-core *reads* are what the interconnect is described as supporting,")
        A("and storing through a composed destination asks for the unverified direction.")
        A("Nothing in this tree can yet say whether a lowering would -- though with the")
        A("slots distinct, a lowering is free to realize each store locally, which is")
        A("`if-store`\'s shape arrived at by analysis rather than by spelling.")
        A("")
        A("**It asks more of the backend.** Participation is derivable rather than stated,")
        A("so a lowering has to fold the index expression against the tile id and compare")
        A("the result with the slot\'s `ct_id`. `if-store` asks nothing: the guard is there")
        A("to read. These files are evidence for the first reading being enough, not proof")
        A("-- nothing in tree lowers either form yet.")
        A("")
        A("**The index arithmetic is not saved.** An earlier revision of these files had no")
        A("tile id at all, because every core stored the whole tensor through a view typed")
        A("as the coordinates once. That was the thing the rule above rejects, so the saving")
        A("went with it: both spellings now derive the box from the tile id.")
        A("")
        A("**A written distributed view implies remote writes.** §4\'s other objection is")
        A("that cross-core *reads* are what the interconnect is described as supporting while")
        A("remote writes are unverified. Storing through a composed destination asks for the")
        A("unverified direction, and nothing in this tree can yet say whether a lowering")
        A("would.")
        A("")
        A("**It is verbose where replication is wide.** The destination side is one view per")
        A("*(piece, owner)* pair, so C008 has 32 source views and 32 destination views, and a")
        A("reader checks a 32-operand compose rather than an arithmetic expression.")
        return "\n".join(L) + "\n"

    A("# if-store: the source composed, the destination local")
    A("")
    A("One of two spellings of the same 35 movements. The other is")
    A("[`../dist-store/`](../dist-store/), and [the parent README](../README.md) carries the")
    A("provenance, the SDSC mapping and the table of 35. **Neither spelling is decided.**")
    A("")
    A("## What one file contains")
    A("")
    A("| | |")
    A("|---|---|")
    A("| the source distribution | one `ktdp.construct_memory_view` per source piece, composed by `ktdp.construct_distributed_memory_view` -- identical to `dist-store` |")
    A("| the landing | **one** `ktdp.construct_memory_view` in `ct_local` with **no `ct_id`**, the piece\'s shape, plus its access tile |")
    A("| the movement | `ktdp.get_compute_tile_id`, then a `ktdp.load` from the composed source and a `ktdp.store` into the landing |")
    A("")
    A("This is [§4](../../inter-tile-lowering-to-mem-view.md)\'s pull model: a destination")
    A("holder writes only into its own scratchpad, so the destination side needs no")
    A("distributed view and no remote write.")
    A("")
    A("## Three conventions, and why")
    A("")
    A("**The landing is identical on every core, so it is hoisted above any guard.** Its")
    A("`coordinate_set` is a range over the piece rather than a box in the tensor, because")
    A("the view is *local*: it describes a buffer in this core\'s scratchpad, indexed from")
    A("zero, and it carries no `ct_id` because a core writes only its own. Global")
    A("coordinates appear in the load alone, which is where the per-core difference is.")
    A("")
    A("**A load names a coordinate REGION, not a source tile.** Which source partitions a")
    A("region touches is the composed view\'s to resolve, and a region may span many of")
    A("them: in %d of the 35 a destination box overlaps more than one source piece, up to"
      % _spanning(records))
    A("all %d of them (C001, whose single destination piece is the whole tensor)."
      % _max_span(records))
    A("This is exactly `dist-store`\'s property too -- the two spellings share both the box")
    A("and the arithmetic that places it -- but it is worth saying here, because \"each core")
    A("reads its own piece\" invites the reading that one load is one source tile. It is not.")
    A("")
    A("**`ktdp.load` stays inside the guard, with the store**, in the files that have one.")
    A("The load *is* the transfer. `construct_access_tile` is `Pure` and may be hoisted;")
    A("the load may not, or a core that lands nothing would still read.")
    A("")
    A("## Where the breadth went: one field of the tile id per dimension")
    A("")
    A("`ktdp.get_compute_tile_id`, then for each divided dimension")
    A("")
    A("    start[dim] = ((tid // a) % b) * extent")
    A("")
    A("with `a` and `b` powers of two. The decomposition is **per dimension and")
    A("independent**, and that is what removes the branches: a destination piece *index*")
    A("can look like a permutation of 0..31 while each dimension\'s slice index is still a")
    A("clean field of the tile id. C008 is the case that shows it -- its piece order runs")
    A("0, 1, 12, 23, 26, ... and yet `out` is `(tid // 8) % 4` and `x` is `tid % 8`.")
    A("")
    A("All 35 admit this form, checked over every (core, dimension) pair, so **no file")
    A("branches to select a piece**:")
    A("")
    A("| | files |")
    A("|---|---:|")
    for label, n in _dim_hist(records):
        A(f"| {label} | {n} |")
    A("")
    A("A dimension that does not vary is a constant, and **not always zero** -- C035 lands")
    A("every piece at `mb = 511`. A dimension whose per-piece extent is 1 needs no")
    A("multiply, so its slice index is the base index.")
    A("")
    A("## The only branch: participation")
    A("")
    A("%d of the 35 are launched on fewer than all 32 cores, and there the movement is"
      % len(_guarded(records)))
    A("guarded. Because every owner set is an arithmetic progression from zero (see the")
    A("parent README), the guard is one comparison, never a list:")
    A("")
    A("| file | cores | guard |")
    A("|---|---:|---|")
    for cid, n, g in _guarded(records):
        A(f"| {cid} | {n} | `{g}` |")
    A("")
    A("The other %d need none. Three of the guarded ones -- C013, C022, C030 -- carry both"
      % (len(records) - len(_guarded(records))))
    A("a guard and the index arithmetic, which is the general shape.")
    A("")
    A("## What it costs")
    A("")
    A("**The IR no longer says which global coordinates a core\'s landing holds.** That")
    A("moved into the load\'s base indices, so a reader recovers the ownership that")
    A("`dist-store` states in a type by executing the arithmetic instead. The header")
    A("comment of each file writes the derivation out for exactly that reason.")
    A("")
    A("**Pull is baked in.** A lowering cannot choose to push from these files, because")
    A("there is no destination view to push into.")
    A("")
    A("**The arithmetic is only as good as the owner map.** Every record here decomposes,")
    A("but the format could express an owner map that does not, and such a record would")
    A("need a branch per piece -- which is what this directory looked like before the")
    A("decomposition was found to be per dimension.")
    return "\n".join(L) + "\n"


def main(argv):
    pkg = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_PKG
    out = pathlib.Path(argv[2]) if len(argv) > 2 else DEFAULT_OUT

    man = json.load(open(pkg / "source/sendnn_sdsc_lx_replay_manifest.json"))
    by = {(r["phase"], r["relayout"]): r for r in man["relayouts"]}
    rows = list(csv.DictReader(open(pkg / "CURATED-INDEX.tsv"), delimiter="\t"))
    dist_dir, if_dir = out / "dist-store", out / "if-store"
    for d in (dist_dir, if_dir):
        d.mkdir(parents=True, exist_ok=True)

    records = []
    for row in rows:
        key = (row["phase"], row["original_file"][:-5])
        rec = by[key]

        # The premises the emitter relies on, re-checked per record rather than
        # trusted: a generator that silently emits a wrong extent is worse than
        # one that stops.
        keys = list(rec["source_extents"])
        assert rec["word_length"] in ELEM, (row["curated_id"], rec["word_length"])
        n = functools.reduce(operator.mul, rec["source_extents"].values(), 1)
        assert n * rec["word_length"] == rec["logical_tensor_bytes"], row["curated_id"]
        assert {k: max(p["start"][k] + p["size"][k] for p in rec["source_pieces"])
                for k in keys} == rec["source_extents"], row["curated_id"]
        for p in rec["source_pieces"] + rec["destination_pieces"]:
            assert list(p["start"]) == keys and list(p["size"]) == keys, row["curated_id"]
        for p in rec["source_pieces"]:
            assert len(p["owners"]) == 1, (row["curated_id"], p["key"])

        name = row["curated_file"][:-5] + ".mlir"
        (dist_dir / name).write_text(emit_dist(row, rec))
        (if_dir / name).write_text(emit_if(row, rec))
        records.append((row, rec))

    (out / "README.md").write_text(readme(records, "index"))
    (dist_dir / "README.md").write_text(readme(records, "dist"))
    (if_dir / "README.md").write_text(readme(records, "if"))
    print(f"wrote {len(records)} examples + README.md to each of "
          f"{dist_dir.name}/ and {if_dir.name}/, plus the index README.md, under {out}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
