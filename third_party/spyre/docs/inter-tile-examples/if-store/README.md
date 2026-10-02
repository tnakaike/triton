# if-store: the source composed, the destination local

One of two spellings of the same 35 movements. The other is
[`../dist-store/`](../dist-store/), and [the parent README](../README.md) carries the
provenance, the SDSC mapping and the table of 35. **Neither spelling is decided.**

## What one file contains

| | |
|---|---|
| the source distribution | one `ktdp.construct_memory_view` per source piece, composed by `ktdp.construct_distributed_memory_view` -- identical to `dist-store` |
| the landing | **one** `ktdp.construct_memory_view` in `ct_local` with **no `ct_id`**, the piece's shape, plus its access tile |
| the movement | `ktdp.get_compute_tile_id`, then a `ktdp.load` from the composed source and a `ktdp.store` into the landing |

This is [§4](../../inter-tile-lowering-to-mem-view.md)'s pull model: a destination
holder writes only into its own scratchpad, so the destination side needs no
distributed view and no remote write.

## Three conventions, and why

**The landing is identical on every core, so it is hoisted above any guard.** Its
`coordinate_set` is a range over the piece rather than a box in the tensor, because
the view is *local*: it describes a buffer in this core's scratchpad, indexed from
zero, and it carries no `ct_id` because a core writes only its own. Global
coordinates appear in the load alone, which is where the per-core difference is.

**A load names a coordinate REGION, not a source tile.** Which source partitions a
region touches is the composed view's to resolve, and a region may span many of
them: in 25 of the 35 a destination box overlaps more than one source piece, up to
all 32 of them (C001, whose single destination piece is the whole tensor).
This is exactly `dist-store`'s property too -- its one whole-tensor load spans every
source piece -- but it is worth saying here, because "each core reads its own
piece" invites the reading that one load is one source tile. It is not.

**`ktdp.load` stays inside the guard, with the store**, in the files that have one.
The load *is* the transfer. `construct_access_tile` is `Pure` and may be hoisted;
the load may not, or a core that lands nothing would still read.

## Where the breadth went: one field of the tile id per dimension

`ktdp.get_compute_tile_id`, then for each divided dimension

    start[dim] = ((tid // a) % b) * extent

with `a` and `b` powers of two. The decomposition is **per dimension and
independent**, and that is what removes the branches: a destination piece *index*
can look like a permutation of 0..31 while each dimension's slice index is still a
clean field of the tile id. C008 is the case that shows it -- its piece order runs
0, 1, 12, 23, 26, ... and yet `out` is `(tid // 8) % 4` and `x` is `tid % 8`.

All 35 admit this form, checked over every (core, dimension) pair, so **no file
branches to select a piece**:

| | files |
|---|---:|
| no dimension divided -- one destination piece | 14 |
| one dimension from the tile id | 14 |
| two dimensions, independently | 6 |
| three dimensions, independently | 1 |

A dimension that does not vary is a constant, and **not always zero** -- C035 lands
every piece at `mb = 511`. A dimension whose per-piece extent is 1 needs no
multiply, so its slice index is the base index.

## The only branch: participation

10 of the 35 are launched on fewer than all 32 cores, and there the movement is
guarded. Because every owner set is an arithmetic progression from zero (see the
parent README), the guard is one comparison, never a list:

| file | cores | guard |
|---|---:|---|
| C002 | 16 | `tid % 2 == 0` |
| C003 | 25 | `tid < 25` |
| C004 | 28 | `tid < 28` |
| C010 | 1 | `tid == 0` |
| C013 | 8 | `tid % 4 == 0` |
| C014 | 8 | `tid % 4 == 0` |
| C019 | 1 | `tid == 0` |
| C022 | 8 | `tid % 4 == 0` |
| C026 | 28 | `tid < 28` |
| C030 | 8 | `tid % 4 == 0` |

The other 25 need none. Three of the guarded ones -- C013, C022, C030 -- carry both
a guard and the index arithmetic, which is the general shape.

## What it costs

**The IR no longer says which global coordinates a core's landing holds.** That
moved into the load's base indices, so a reader recovers the ownership that
`dist-store` states in a type by executing the arithmetic instead. The header
comment of each file writes the derivation out for exactly that reason.

**Pull is baked in.** A lowering cannot choose to push from these files, because
there is no destination view to push into.

**The arithmetic is only as good as the owner map.** Every record here decomposes,
but the format could express an owner map that does not, and such a record would
need a branch per piece -- which is what this directory looked like before the
decomposition was found to be per dimension.
