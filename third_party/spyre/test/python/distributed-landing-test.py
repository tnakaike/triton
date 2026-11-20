# RUN: %python -m pytest %s -q
"""A pinned read through a distributed descriptor, from Python to KTIR.

`distributed-relayout-test.py` beside this file runs a relayout with no `if`, in which
every tile reads the region it already holds. What the design doc (§3 phases 3 and 4)
adds on top of that is checked here, on kernels an author writes rather than on the
hand-written TTIR of the lit tests:

  * the `if` that chooses the receivers is read at compile time and does not reach
    the transfer: the transfer is emitted ahead of the `scf.if`, unguarded, and the
    read inside it becomes a load of the tile's own landing;
  * the destination is composed too, from the receivers' landings, in the shape of
    the blocks they read -- copies of one block laid along dimension 0;
  * a quasi-affine coordinate stays the kernel's own expression, cloned out of the
    `if` when it was computed inside it, so the KTIR has no select;
  * what the design rejects is reported at the kernel's own line.

Like the relayout test, this cannot reach a device: dbo's legality check rejects
`construct_distributed_memory_view`, so the IR is read instead.

Every kernel runs on a grid of 4 over a [4, 64]-row-shaped source held as one or two
rows per tile.
"""

import inspect
import os
import tempfile

import pytest

GRID = [4]

SIGNATURE = {
    "x_ptr": "*fp16",
    "out_ptr": "*fp16",
    "WORK_SLICES": "constexpr",
    "AXES": "constexpr",
}

#: One region per tile, divided on dimension 0.
CONSTANTS = {
    "WORK_SLICES": [{"mb": i} for i in range(4)],
    "AXES": ["mb", None],
}


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_onto_the_even_tiles(x_ptr, out_ptr, WORK_SLICES: tl.constexpr,
                                   AXES: tl.constexpr):
        # The design doc's worked example: one row per tile, gathered whole onto
        # tiles 0 and 2, which each receive a copy.
        pid = tl.program_id(0)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[4, 64], strides=[64, 1], block_shape=[1, 64])
        out_desc = tl.make_tensor_descriptor(
            out_ptr, shape=[4, 64], strides=[64, 1], block_shape=[4, 64])
        share = tl.exp(x_desc.load([pid, 0]))
        tl.spyre_pin(share, "ct_local", offset=0)
        whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES, [4, 64])
        if pid % 2 == 0:
            mine = whole.load([0, 0])
            tl.spyre_pin(mine, "ct_local", offset=4096)
            out_desc.store([0, 0], mine)

    @triton.jit
    def some_tiles_receive(x_ptr, out_ptr, WORK_SLICES: tl.constexpr,
                           AXES: tl.constexpr):
        # Tiles 2 and 3 receive rows 2-3 and 4-5 of an [8, 64] source, at a
        # coordinate computed inside the `if`.
        pid = tl.program_id(0)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[8, 64], strides=[64, 1], block_shape=[2, 64])
        out_desc = tl.make_tensor_descriptor(
            out_ptr, shape=[8, 64], strides=[64, 1], block_shape=[2, 64])
        share = tl.exp(x_desc.load([pid * 2, 0]))
        tl.spyre_pin(share, "ct_local", offset=0)
        whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES, [2, 64])
        if pid >= 2:
            mine = whole.load([(pid - 1) * 2, 0])
            tl.spyre_pin(mine, "ct_local", offset=4096)
            out_desc.store([pid * 2, 0], mine)

    @triton.jit
    def rows_to_columns(x_ptr, out_ptr, WORK_SLICES: tl.constexpr,
                        AXES: tl.constexpr):
        # Held by rows, received by columns: the block is not the share's shape.
        pid = tl.program_id(0)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[4, 64], strides=[64, 1], block_shape=[1, 64])
        out_desc = tl.make_tensor_descriptor(
            out_ptr, shape=[4, 64], strides=[64, 1], block_shape=[4, 16])
        share = tl.exp(x_desc.load([pid, 0]))
        tl.spyre_pin(share, "ct_local", offset=0)
        whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES, [4, 16])
        mine = whole.load([0, pid * 16])
        tl.spyre_pin(mine, "ct_local", offset=4096)
        out_desc.store([0, pid * 16], mine)

    return {k.fn.__name__: k for k in
            (gather_onto_the_even_tiles, some_tiles_receive, rows_to_columns)}


def _invalid_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def coordinate_from_data(x_ptr, out_ptr, WORK_SLICES: tl.constexpr,
                             AXES: tl.constexpr):
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[4, 64], strides=[64, 1], block_shape=[1, 64])
        share = tl.exp(x_desc.load([0, 0]))
        tl.spyre_pin(share, "ct_local", offset=0)
        whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES, [1, 64])
        row = tl.load(out_ptr.to(tl.pointer_type(tl.int32)))
        mine = whole.load([row, 0])
        tl.spyre_pin(mine, "ct_local", offset=4096)
        x_desc.store([0, 0], mine)

    @triton.jit
    def read_in_a_loop(x_ptr, out_ptr, WORK_SLICES: tl.constexpr,
                       AXES: tl.constexpr):
        pid = tl.program_id(0)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[4, 64], strides=[64, 1], block_shape=[1, 64])
        share = tl.exp(x_desc.load([pid, 0]))
        tl.spyre_pin(share, "ct_local", offset=0)
        whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES, [1, 64])
        for i in range(2):
            mine = whole.load([pid, 0])
            tl.spyre_pin(mine, "ct_local", offset=4096)
            x_desc.store([pid, 0], mine)

    @triton.jit
    def descriptor_inside_the_guard(x_ptr, out_ptr, WORK_SLICES: tl.constexpr,
                                    AXES: tl.constexpr):
        pid = tl.program_id(0)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[4, 64], strides=[64, 1], block_shape=[1, 64])
        share = tl.exp(x_desc.load([pid, 0]))
        tl.spyre_pin(share, "ct_local", offset=0)
        if pid < 2:
            whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES,
                                                   [1, 64])
            mine = whole.load([pid, 0])
            tl.spyre_pin(mine, "ct_local", offset=4096)
            x_desc.store([pid, 0], mine)

    return {k.fn.__name__: k for k in
            (coordinate_from_data, read_in_a_loop, descriptor_inside_the_guard)}


def _module(kernel):
    from utils import compile_to_ttir, make_ktir_mod

    ttir = compile_to_ttir(kernel, SIGNATURE, CONSTANTS)
    fd, path = tempfile.mkstemp(suffix=".ttir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(ttir)
        return make_ktir_mod(path, grid=GRID)
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def ktir():
    """Each kernel through the whole TTIR -> KTIR pipeline, as printed IR."""
    return {name: str(_module(k)) for name, k in _kernels().items()}


@pytest.fixture(scope="module")
def spyrecode():
    """Each kernel through both stages, as printed IR."""
    from triton._C.libtriton import ir, spyre

    out = {}
    for name, k in _kernels().items():
        mod = _module(k)
        pm = ir.pass_manager(mod.context)
        spyre.passes.ttir_to_ktdp.add_spyrecode_pipeline(pm)
        pm.run(mod, f"distributed_landing_{name}")
        out[name] = str(mod)
    return out


class _IR:
    """Just enough reading of printed IR to follow values to their definitions."""

    def __init__(self, text):
        self.lines = [line.strip() for line in text.splitlines()]

    def defining(self, value):
        """The line that defines `value`, and its position."""
        for i, line in enumerate(self.lines):
            if line.startswith(f"{value} = "):
                return i, line
        raise AssertionError(f"{value} is not defined")

    def constant(self, value):
        _, line = self.defining(value)
        assert "arith.constant" in line, line
        return int(line.split("arith.constant ")[1].split(" ")[0])

    def operands(self, value):
        _, line = self.defining(value)
        args = line.split(" = ", 1)[1].split(" ", 1)[1].split(":")[0]
        return [a.strip() for a in args.split(",")]

    def op(self, value):
        _, line = self.defining(value)
        return line.split(" = ", 1)[1].split(" ")[0]

    def composes(self):
        """(name, line) of each construct_distributed_memory_view: src, then dst."""
        return [(line.split(" =")[0], line) for line in self.lines
                if "construct_distributed_memory_view(" in line]

    def partitions(self, which):
        _, line = self.composes()[which]
        names = [o.strip() for o in line.split("(", 1)[1].split(":")[0].split(",")]
        return [self.defining(n)[1] for n in names]

    def access(self, view):
        """The coordinates of the one access tile on `view`, and its position."""
        for i, line in enumerate(self.lines):
            if "construct_access_tile" in line and f" {view}[" in line:
                return i, [c.strip() for c in
                           line.split(f"{view}[")[1].split("]")[0].split(",")]
        raise AssertionError(f"no access tile on {view}")

    def position(self, needle):
        return next(i for i, line in enumerate(self.lines) if needle in line)


class TestGatherOntoTheEvenTiles:
    """The design doc's worked example, with copies."""

    def test_the_dst_view_composes_the_two_landings(self, ktir):
        ir = _IR(ktir["gather_onto_the_even_tiles"])
        (_, src), (_, dst) = ir.composes()
        assert ": memref<4x64xf16>" in src
        # Two copies of the [4, 64] whole, laid along dimension 0.
        assert ": memref<8x64xf16>" in dst
        landings = ir.partitions(1)
        assert ["ct_id = 0" in v for v in landings] == [True, False]
        assert ["ct_id = 2" in v for v in landings] == [False, True]

    def test_the_transfer_is_ahead_of_the_if(self, ktir):
        ir = _IR(ktir["gather_onto_the_even_tiles"])
        dst, _ = ir.composes()[1]
        tile_at, _ = ir.access(dst)
        assert tile_at < ir.position("scf.if")

    def test_the_dst_row_is_the_copys_slab(self, ktir):
        # 4 * (t / 2), as the design doc writes it: tile 0 at row 0, tile 2 at
        # row 4. No select, and no remainder.
        ir = _IR(ktir["gather_onto_the_even_tiles"])
        dst, _ = ir.composes()[1]
        _, (row, _) = ir.access(dst)
        assert ir.op(row) == "arith.muli"
        q, four = ir.operands(row)
        assert ir.constant(four) == 4
        assert ir.op(q) == "arith.divsi"
        tid, two = ir.operands(q)
        assert ir.constant(two) == 2
        assert ir.op(tid) == "ktdp.get_compute_tile_id"
        assert not any("arith.select" in line for line in ir.lines)

    def test_the_read_is_a_load_of_the_own_landing(self, ktir):
        # Inside the if, where the kernel's own store still is, the value is read
        # back from this tile's landing: ct_local with no ct_id, at the pin's
        # offset.
        ir = _IR(ktir["gather_onto_the_even_tiles"])
        inside = ir.lines[ir.position("scf.if"):]
        load = next(line for line in inside if "ktdp.load" in line)
        tile = load.split("ktdp.load ")[1].split(" ")[0]
        view = ir.operands(tile)[0].split("[")[0]
        _, own = ir.defining(view)
        assert "memory_space = #ktdp.memory_space<ct_local>}" in own, own
        assert ir.constant(own.split("construct_memory_view ")[1].split(",")[0]) \
            == 4096

    def test_no_pin_survives_the_second_stage(self, spyrecode):
        assert "tts.pin" not in spyrecode["gather_onto_the_even_tiles"]

    def test_the_share_is_stored_before_it_is_composed(self, spyrecode):
        ir = _IR(spyrecode["gather_onto_the_even_tiles"])
        assert ir.position("ktdp.store") < \
            ir.position("construct_distributed_memory_view")


class TestSomeTilesReceive:
    """A coordinate computed inside the `if`, cloned out of it."""

    def test_the_dst_view_is_the_received_rows(self, ktir):
        ir = _IR(ktir["some_tiles_receive"])
        (_, src), (_, dst) = ir.composes()
        assert ": memref<8x64xf16>" in src
        assert ": memref<4x64xf16>" in dst
        landings = ir.partitions(1)
        assert "ct_id = 2" in landings[0] and "ct_id = 3" in landings[1]

    def test_the_src_row_is_the_kernels_expression_ahead_of_the_if(self, ktir):
        # (pid - 1) * 2, defined before the scf.if rather than inside it.
        ir = _IR(ktir["some_tiles_receive"])
        src, _ = ir.composes()[0]
        _, (row, _) = ir.access(src)
        if ir.op(row) == "arith.index_cast":
            row = ir.operands(row)[0]
        assert ir.op(row) == "arith.muli"
        at, _ = ir.defining(row)
        assert at < ir.position("scf.if")
        sub, two = ir.operands(row)
        assert ir.constant(two) == 2
        assert ir.op(sub) == "arith.subi"

    def test_the_dst_row_is_the_src_row_less_the_origin(self, ktir):
        ir = _IR(ktir["some_tiles_receive"])
        src, _ = ir.composes()[0]
        dst, _ = ir.composes()[1]
        _, (src_row, _) = ir.access(src)
        _, (dst_row, _) = ir.access(dst)
        assert ir.op(dst_row) == "arith.subi"
        base, origin = ir.operands(dst_row)
        assert base == src_row
        assert ir.constant(origin) == 2
        assert not any("arith.select" in line for line in ir.lines)


class TestRowsToColumns:
    """Held by rows, received by columns."""

    def test_the_dst_view_is_divided_by_columns(self, ktir):
        ir = _IR(ktir["rows_to_columns"])
        (_, src), (_, dst) = ir.composes()
        assert ": memref<4x64xf16>" in src and ": memref<4x64xf16>" in dst
        landings = ir.partitions(1)
        assert all("sizes: [4, 16]" in v for v in landings), landings
        assert [f"ct_id = {t}" in v for t, v in enumerate(landings)] == [True] * 4

    def test_both_accesses_share_the_kernels_column(self, ktir):
        # pid * 16, once, for the src and the dst view alike.
        ir = _IR(ktir["rows_to_columns"])
        src, _ = ir.composes()[0]
        dst, _ = ir.composes()[1]
        _, (_, src_col) = ir.access(src)
        _, (_, dst_col) = ir.access(dst)
        assert src_col == dst_col
        col = src_col
        if ir.op(col) == "arith.index_cast":
            col = ir.operands(col)[0]
        assert ir.op(col) == "arith.muli"
        assert not any("arith.select" in line for line in ir.lines)


def _line_of(kernel, needle):
    """The source line of `kernel` that contains `needle`."""
    lines, start = inspect.getsourcelines(kernel.fn)
    return start + next(i for i, line in enumerate(lines) if needle in line)


class TestRefusedAtTheKernelsLine:
    """What the design rejects, reported where the author wrote it."""

    @pytest.mark.parametrize("name, needle, message", [
        ("coordinate_from_data", "mine = whole.load",
         "the coordinate this read passes on dimension 0 must be a constant on "
         "every tile"),
        ("read_in_a_loop", "mine = whole.load",
         "may be guarded by an if, but not run inside 'scf.for'"),
        ("descriptor_inside_the_guard", "mine = whole.load",
         "is made inside an if that guards the read"),
    ])
    def test_refused(self, capfd, name, needle, message):
        kernel = _invalid_kernels()[name]
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            _module(kernel)
        err = capfd.readouterr().err
        line = _line_of(kernel, needle)
        errors = [l for l in err.splitlines() if ": error: " in l and message in l]
        assert errors, err
        assert f":{line}:" in errors[0], (line, errors[0])

    def test_the_note_points_at_the_value(self, capfd):
        # The coordinate that is not a constant is traced to the load it came from.
        kernel = _invalid_kernels()["coordinate_from_data"]
        with pytest.raises(RuntimeError):
            _module(kernel)
        err = capfd.readouterr().err
        notes = [l for l in err.splitlines() if ": note: this value is not a "
                 "function of tl.program_id(0) and constants" in l]
        assert notes, err
        assert f":{_line_of(kernel, 'row = tl.load')}:" in notes[0]
