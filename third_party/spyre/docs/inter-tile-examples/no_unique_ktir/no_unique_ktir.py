"""A gather onto the even tiles, whose destination the kernel does not determine.

  tensor     [mb, out] = [4, 64], fp16, stick on out
  source     4 regions of [1, 64], region k on tile k
  receivers  tiles 0 and 2, the whole tensor each
  grid       [4]

ktir_1.mlir and ktir_2.mlir are two correct KTIRs for it; README.md says why both
are, and what that implies.
"""

import triton
import triton.language as tl

# Entry k is the region tile k holds: mb is divided 4 ways.
SRC = [{"mb": 0}, {"mb": 1}, {"mb": 2}, {"mb": 3}]

SRC_OFF = 0x0000
DST_OFF = 0x0100


@triton.jit
def no_unique_ktir(x_ptr):
    pid = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[4, 64], strides=[64, 1],
                                       block_shape=[1, 64])
    tl.spyre_tensor_layout(x_desc, [(1, "floordiv", 64), 0, (1, "mod", 64)])
    share = x_desc.load([pid, 0])
    tl.spyre_pin(share, "ct_local", offset=SRC_OFF)

    whole = tl.make_distributed_descriptor(share, work_slices=SRC, axes=["mb", None],
                                           block_shape=[4, 64])        # composed: [4, 64]

    # The even tiles each receive the whole tensor.
    if pid % 2 == 0:
        mine = whole.load([0, 0])
        tl.spyre_pin(mine, "ct_local", offset=DST_OFF)
