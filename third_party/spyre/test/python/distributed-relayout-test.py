# RUN: %python -m pytest %s -q
"""A relayout, from Python to KTIR: `tl.spyre_pin` and `tl.make_distributed_descriptor`.

The two ops only mean something together, and this file is where that is checked.
The lit tests beside it drive each pass on hand-written IR, which is what keeps a
diagnostic honest; none of them shows that a kernel an author could actually write
reaches the composed form, because each starts from the shape the previous pass was
supposed to produce.

The kernel is the smallest thing that is a real redistribution:

    share = exp(x)                                  produced on-chip
    pin(share, ct_local, 0)                         where my share lives
    whole = make_distributed_descriptor(share, ...)  every core's share, composed
    mine  = whole.load([0, pid * N])                 my region under the new division
    pin(mine, ct_local, 8192)                        where the received tile lands

Both pins are load-bearing and for different reasons, which is the point of testing
them here rather than apart. The first is the only thing that gives a partition an
offset — the compose refuses a share nothing pinned — and it is honoured by
`MaterializePinnedBuffers` in the `spyrecode` stage. The second is the landing the
design requires before a compute unit may read a received tile. `AnalyzeDistributedDesc`
reads which tiles receive and where each block sits in the destination, and
`LowerDistributedDesc` consumes the pin: it composes the landings into a dst view and
emits the transfer as a load from the src view and a store into the dst view. So the
landing is in the `ktir` stage already, and the share's store a stage later, and
nothing but an end-to-end run shows the two agreeing.

What is NOT checked here is numerical: this cannot reach a device. dbo's V1 legality
check rejects `construct_distributed_memory_view` outright, and a hand-written
`ct_local` memory view makes the scheduler abort — not diagnose — on a missing
`applicable_units` attribute. So the artifact is read instead, at the `ktir` stage,
which is also where `PLAN-DistDesc.md` puts this layer's acceptance.

Four claims, and they are separable.

**The partitions differ only in coordinate_set and ct_id.** Design section 3 says
those two attributes are "the entire content of the distribution", so the offsets
being EQUAL across partitions is as much the claim as the sets being different: every
core runs the same program text, so a share sits at the same offset in its own core's
scratchpad as every other share does in its. A lowering that made the offsets differ
would look plausible and address the wrong memory on 31 cores out of 32.

**The composed extent is the union, not a share.** The share is [64, 32] and the
composition of two of them is [64, 64]. That number appears nowhere in the kernel —
the slice count is derived as one more than the largest index in the table — so it is
the derivation being checked and not a transcription.

**The read is per-instance.** The access tile's coordinate is `pid * N`, which is
what makes this a redistribution rather than copies of the same read. It is
quasi-affine in the program id, so the lowering keeps the read's own expression,
an SSA value, because `construct_access_tile` takes runtime base indices.

**The received tile lands in a composed dst view.** Each tile receives, so the dst
view composes two landings at the landing's offset, one per core, and the store of the
transfer goes into it. Here every tile receives the region it already holds, so the
dst view has the src view's extent and the same coordinate sets.
"""

import pytest

SIGNATURE = {
    "x_ptr": "*fp16",
    "out_ptr": "*fp16",
    "M": "constexpr",
    "N": "constexpr",
    "WORK_SLICES": "constexpr",
    "AXES": "constexpr",
}

#: Two cores, the work divided on the last dimension. The table is the PARTITION
#: table -- one entry per region -- and here it has one entry per tile, which is the
#: only length the lowering admits so far: the holder is then the entry's own index.
CONSTANTS = {
    "M": 64,
    "N": 32,
    "WORK_SLICES": [{"n": 0}, {"n": 1}],
    "AXES": [None, "n"],
}

GRID = [2]

#: Element indices, not byte addresses: what `construct_memory_view`'s offset feeds.
SHARE_ADDRESS = 0
LANDING_ADDRESS = 8192


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def relayout(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                 WORK_SLICES: tl.constexpr, AXES: tl.constexpr):
        pid = tl.program_id(0)
        x_desc = tl.make_tensor_descriptor(
            x_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])
        out_desc = tl.make_tensor_descriptor(
            out_ptr, shape=[M, N], strides=[N, 1], block_shape=[M, N])

        share = tl.exp(x_desc.load([0, 0]))
        tl.spyre_pin(share, "ct_local", offset=0)

        whole = tl.make_distributed_descriptor(share, WORK_SLICES, AXES, [M, N])
        mine = whole.load([0, pid * N])
        tl.spyre_pin(mine, "ct_local", offset=8192)

        out_desc.store([0, 0], mine)

    return relayout


@pytest.fixture(scope="module")
def ktir():
    """The kernel through the whole TTIR -> KTIR pipeline, as printed IR."""
    import tempfile
    import os
    from utils import compile_to_ttir, make_ktir_mod

    ttir = compile_to_ttir(_kernel(), SIGNATURE, CONSTANTS)
    fd, path = tempfile.mkstemp(suffix=".ttir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(ttir)
        return str(make_ktir_mod(path, grid=GRID))
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def spyrecode():
    """The same kernel through BOTH stages, as printed IR.

    The `ktir` fixture above stops where that stage does, and the share's pin is
    still an attribute there: `LowerTTSMarkers` moved it onto its producer and
    `MaterializePinnedBuffers` -- which builds the buffer, the store and the loads --
    runs at the head of `spyrecode`. So anything about the share's BUFFER has to be
    asked of this fixture. The landing's pin is consumed a stage earlier, by
    `LowerDistributedDesc`, so its store and its read-back are in both.

    That split is the whole reason this file exists. The compose is emitted by one
    pass in one stage and the buffers it names are built by another pass in the next,
    and neither pass's own lit tests can show them agreeing -- one hand-writes the
    buffers it wants to see, the other never sees a compose.

    dbo-opt is not invoked: it rejects `construct_distributed_memory_view` outright,
    so the pipeline is run for its IR and stopped before the tool.
    """
    import tempfile
    import os
    from utils import compile_to_ttir, make_ktir_mod
    from triton._C.libtriton import ir, spyre

    ttir = compile_to_ttir(_kernel(), SIGNATURE, CONSTANTS)
    fd, path = tempfile.mkstemp(suffix=".ttir")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(ttir)
        mod = make_ktir_mod(path, grid=GRID)
    finally:
        os.unlink(path)

    pm = ir.pass_manager(mod.context)
    spyre.passes.ttir_to_ktdp.add_spyrecode_pipeline(pm)
    pm.run(mod, "distributed_relayout_spyrecode")
    return str(mod)


def _views(ktir, needle):
    return [line.strip() for line in ktir.splitlines()
            if "ktdp.construct_memory_view" in line and needle in line]


def _composes(ktir):
    """Each `construct_distributed_memory_view` as (result name, operand names,
    line), in program order: the src view first, made where the descriptor was, then
    the dst view, made where the read was."""
    out = []
    for line in ktir.splitlines():
        if "construct_distributed_memory_view(" not in line:
            continue
        name = line.split(" =")[0].strip()
        operands = line.split("(", 1)[1].split(":")[0]
        out.append((name, [o.strip() for o in operands.split(",")], line))
    return out


def _partitions(ktir, which):
    """The memory views composed by the `which`-th compose (0 = src, 1 = dst)."""
    _, operands, _ = _composes(ktir)[which]
    return [line.strip() for line in ktir.splitlines()
            if "ktdp.construct_memory_view" in line
            and line.split(" =")[0].strip() in operands]


class TestTheComposedView:

    def test_one_partition_per_tile(self, ktir):
        # Two cores, two partitions, each naming a different core's scratchpad.
        src = _partitions(ktir, 0)
        assert len([v for v in src if "ct_id = 0" in v]) == 1, src
        assert len([v for v in src if "ct_id = 1" in v]) == 1, src

    def test_the_partitions_share_one_offset(self, ktir):
        # The claim that matters most, and the one a wrong lowering would pass every
        # other check with. Compared as the offset TOKEN rather than a number,
        # because the value is an SSA name.
        offsets = [line.split("construct_memory_view ")[1].split(",")[0]
                   for line in _partitions(ktir, 0)]
        assert len(set(offsets)) == 1, offsets

    def test_the_partitions_differ_only_in_set_and_ct_id(self, ktir):
        # Everything except the two distinguishing attributes must be identical.
        # Normalised by naming exactly what may differ -- the coordinate-set
        # reference and the ct_id -- rather than by a pattern over the line, so a
        # future difference shows up as a failure instead of being absorbed.
        def strip(line, ct_id):
            body = line.split(" = ", 1)[1]        # drop the result's SSA name
            body = body.replace(f"ct_id = {ct_id}", "ct_id = _")
            head, _, rest = body.partition("coordinate_set = ")
            _, _, tail = rest.partition(",")
            return head + "coordinate_set = _," + tail

        src = _partitions(ktir, 0)
        a = next(v for v in src if "ct_id = 0" in v)
        b = next(v for v in src if "ct_id = 1" in v)
        assert strip(a, 0) == strip(b, 1)
        # And they really do differ before normalising, or the assertion above
        # proves nothing.
        assert a != b

    def test_the_two_regions_tile_the_whole(self, ktir):
        # d1 in [0, 32) and d1 in [32, 64): the slice width is the share's extent and
        # the coordinate picks which slice. The `- 32` is what says the second
        # partition owns the upper half.
        sets = {}
        for line in ktir.splitlines():
            if line.strip().startswith("#set"):
                name, body = line.split(" = ", 1)
                sets[name.strip()] = body
        bodies = " ".join(sets.values())
        assert "d1 >= 0" in bodies and "-d1 + 31 >= 0" in bodies
        assert "d1 - 32 >= 0" in bodies and "-d1 + 63 >= 0" in bodies

    def test_the_composed_extent_is_the_union(self, ktir):
        # [64, 32] composed twice is [64, 64], and 64 appears in no kernel line: the
        # slice count is derived from the table.
        assert "ktdp.construct_distributed_memory_view" in ktir
        _, _, compose = _composes(ktir)[0]
        assert ": memref<64x64xf16>" in compose

    def test_the_marker_is_gone(self, ktir):
        assert "tts.make_distributed_descriptor" not in ktir
        assert "tt.descriptor_load" not in ktir


class TestTheTransfer:

    def test_the_read_is_on_the_composed_view(self, ktir):
        name, _, _ = _composes(ktir)[0]
        tiles = [line for line in ktir.splitlines()
                 if "construct_access_tile" in line and f"{name}[" in line]
        assert len(tiles) == 1, tiles
        # memref<64x64xf16> in, a 64x32 tile out: the instance takes its own block
        # from the whole.
        assert "memref<64x64xf16> -> !ktdp.access_tile<64x32xindex>" in tiles[0]

    def test_the_coordinate_is_per_instance(self, ktir):
        # pid * N is quasi-affine, so the lowering uses the read's own expression:
        # arithmetic on the compute-tile id, and the two instances read two
        # different regions of one view. A folded constant here would mean every
        # core read the same thing.
        assert "ktdp.get_compute_tile_id" in ktir
        name, _, _ = _composes(ktir)[0]
        tile = next(line for line in ktir.splitlines()
                    if "construct_access_tile" in line and f"{name}[" in line)
        coord = tile.split(f"{name}[")[1].split("]")[0].split(",")[1].strip()

        def definition(value):
            return next(line for line in ktir.splitlines()
                        if line.strip().startswith(f"{value} = "))
        line = definition(coord)
        if "arith.index_cast" in line:
            line = definition(line.split("arith.index_cast ")[1].split(" ")[0])
        assert "arith.muli" in line, line
        assert "arith.select" not in ktir

    def test_the_transfer_stores_into_the_dst_view(self, ktir):
        # The landing of design section 3 phase 4: the transfer's load is followed by
        # a store through the composed dst view, at this tile's dst coordinate.
        loads = [line for line in ktir.splitlines() if "ktdp.load" in line]
        dst, _, _ = _composes(ktir)[1]
        tile = next(line for line in ktir.splitlines()
                    if "construct_access_tile" in line and f"{dst}[" in line)
        tile_name = tile.split(" =")[0].strip()
        store = next(line for line in ktir.splitlines()
                     if "ktdp.store" in line and tile_name in line)
        moved = store.split("ktdp.store ")[1].split(",")[0]
        assert any(line.strip().startswith(f"{moved} = ") for line in loads), store

    def test_the_transfer_is_not_guarded(self, ktir):
        # The `if` that chooses the receivers does not reach KTIR (design section 3
        # phase 4). This kernel has none, so there is no scf.if at all.
        assert "scf.if" not in ktir


class TestTheDstView:

    def test_one_landing_per_receiver(self, ktir):
        # Both tiles receive, so the dst view composes two landings, one per core.
        dst = _partitions(ktir, 1)
        assert len([v for v in dst if "ct_id = 0" in v]) == 1, dst
        assert len([v for v in dst if "ct_id = 1" in v]) == 1, dst

    def test_the_landings_are_at_the_landing_offset(self, ktir):
        # Every landing sits at the landing pin's offset, and that is not the share's
        # offset, or the relayout would overwrite its own source.
        def offset(line):
            return line.split("construct_memory_view ")[1].split(",")[0]
        dst = {offset(v) for v in _partitions(ktir, 1)}
        src = {offset(v) for v in _partitions(ktir, 0)}
        assert len(dst) == 1, dst
        assert dst != src, (dst, src)

    def test_the_dst_extent_is_the_union(self, ktir):
        # Each tile receives a [64, 32] block at [0, 32 * pid], so the dst view is
        # their union, [64, 64].
        _, _, compose = _composes(ktir)[1]
        assert ": memref<64x64xf16>" in compose


class TestTheTwoPins:
    """The pins' BUFFERS, so the tests here read the `spyrecode` fixture.

    In the `ktir` artifact the share's pin is still an attribute on its producer --
    that is the two-stage design -- so asking the `ktir` stage about the share's
    store would be asking the wrong stage. (The landing's store is in the `ktir`
    stage already; TestTheTransfer checks it there.) An earlier revision of this file
    asked the `ktir` stage about both, and it is worth recording what
    that cost: one test failed honestly, and one PASSED for the wrong reason, matching
    the kernel's own `out_desc.store` rather than any landing.
    """

    def test_the_share_pin_is_an_attribute_in_the_ktir_artifact(self, ktir):
        # The premise of the share's tests below: the `ktir` stage carries the
        # share's request, not its buffer. If this ever fails, materialization has
        # moved back and those tests are asking the wrong stage again. The landing's
        # pin is not here: LowerDistributedDesc consumed it.
        assert ktir.count("tts.pin = {") == 1
        assert "ktdp.store" not in ktir.split("construct_distributed_memory_view")[0]

    def test_the_share_is_stored_before_it_is_composed(self, spyrecode):
        # Phase 1 needs the share resident: the compose names each core's buffer, so
        # something must have written it. The store comes from MaterializePinnedBuffers
        # in the NEXT stage, inserted immediately after the producer -- which is what
        # puts it ahead of a compose emitted a stage earlier and further down.
        lines = spyrecode.splitlines()
        store = next(i for i, l in enumerate(lines) if "ktdp.store" in l)
        compose = next(i for i, l in enumerate(lines)
                       if "construct_distributed_memory_view" in l)
        assert store < compose, "the share is composed before anything wrote it"

    def test_the_received_tile_lands(self, spyrecode):
        # The mandatory landing of design section 3 phase 4 survives the second
        # stage: after the transfer's load there is a store.
        lines = spyrecode.splitlines()
        compose = next(i for i, l in enumerate(lines)
                       if "construct_distributed_memory_view" in l)
        transfer = next(i for i, l in enumerate(lines)
                        if i > compose and "ktdp.load" in l)
        assert any("ktdp.store" in l for l in lines[transfer:]), \
            "the received tile was never stored"

    def test_the_landing_is_its_own_buffer(self, spyrecode):
        # Two ct_local views with no ct_id: the share's, built by
        # MaterializePinnedBuffers, and the landing's, which the read is read back
        # from. They must be distinct buffers, or the relayout would overwrite its own
        # source.
        own = _views(spyrecode, "memory_space = #ktdp.memory_space<ct_local>}")
        assert len(own) == 2, own
        offsets = [line.split("construct_memory_view ")[1].split(",")[0] for line in own]
        assert len(set(offsets)) == 2, offsets

    def test_no_pin_request_survives_the_second_stage(self, spyrecode):
        # MaterializePinnedBuffers removes the attribute it honours, so a request that
        # reached the binary would mean one was missed.
        assert "tts.pin" not in spyrecode
