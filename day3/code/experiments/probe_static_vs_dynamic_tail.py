"""Does the runtime ``tail`` (dynamic slice extent) cost bandwidth?

The tail-safe abs kernel copies ``A[offset:offset+tail]`` where
``tail = min(block_N, shape-offset)`` is a runtime value.  A DMA engine needs
static extents to plan a transfer; a dynamic extent may force the compiler to
emit a per-element path.

Compare, at the same N=1048576 (divisible, so the tail path is never actually
short here):

* dynamic: the shipped kernel, extent = min(block_N, shape-offset)
* static : extent = block_N, no min, no runtime shape

Both produce correct results; only the addressing differs.
"""

import sys
import time

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode, compile_abs, make_tensors  # noqa: E402

N, BLOCK = 1048576, 4096


def static_kernel(N, block_N, dtype="float32"):
    num_blocks = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            a_buf = T.alloc_ub((block_N,), dtype)
            b_buf = T.alloc_ub((block_N,), dtype)
            offset = cid * block_N
            T.copy(A[offset : offset + block_N], a_buf)
            T.vabs(a_buf, b_buf)
            T.copy(b_buf, B[offset : offset + block_N])

    return main


def bench(kernel, a, b, iters=50, rounds=9):
    for _ in range(5):
        kernel(a, b, N)
    torch.npu.synchronize()
    out = []
    for _ in range(rounds):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            kernel(a, b, N)
        e.record()
        torch.npu.synchronize()
        out.append(s.elapsed_time(e) / iters * 1000.0)
    out.sort()
    return out[len(out) // 2]


a, b = make_tensors(N, "float32")

dyn = compile_abs(N, BLOCK, memory="ub", dtype="float32")
with ascend_mode("Expert"):
    sta = tilelang.compile(static_kernel(N, BLOCK), target="npuir")

dyn(a, b, N)
torch.npu.synchronize()
ok_dyn = torch.equal(b, torch.abs(a))
b.zero_()
sta(a, b, N)
torch.npu.synchronize()
ok_sta = torch.equal(b, torch.abs(a))

gbs = N * 4 * 2 / 1e9
print(f"N={N}, block_N={BLOCK}, fp32, 50 calls/window, median of 9")
print(f"dynamic tail : {bench(dyn, a, b):>8.2f} us  correct={ok_dyn}")
print(f"static block : {bench(sta, a, b):>8.2f} us  correct={ok_sta}")
print(f"8 MiB traffic => GB/s: dynamic {gbs/ (bench(dyn,a,b)*1e-6):>7.1f}"
      f"  static {gbs/(bench(sta,a,b)*1e-6):>7.1f}")
