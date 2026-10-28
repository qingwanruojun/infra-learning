"""Host cost of one launch: taskqueue wrapper on vs off, repeated 5x.

probe_npugraph.py showed two host clusters (~250 and ~440 us/call) for the same
kernel, and TILELANG_ENABLE_TASKQUEUE changes whether the generated launcher
wraps rtKernelLaunch in at_npu::native::OpCommand::RunOpApi(name, call, true)
(which synchronizes; that is also why graph capture aborts with it on).
Measure the same kernel 5 times per setting, in separate processes, to see
whether the wrapper or the run-to-run state explains the difference.

Run twice:
  python probe_launch_taskqueue.py                        # taskqueue on (default)
  TILELANG_ENABLE_TASKQUEUE=0 python probe_launch_taskqueue.py
Use a fresh TILELANG_CACHE_DIR for the second one: the launcher source is baked
into the cached .so, so the env var only takes effect on a cache miss.
"""

import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode  # noqa: E402

N, BLOCK, NCORE = 1 << 25, 8192, 40


def fixed_grid_kernel(n, block_N, ncore, dtype="float32"):
    tiles = T.ceildiv(n, block_N)
    per_core = T.ceildiv(tiles, ncore)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), shape: T.int32):
        with T.Kernel(ncore, is_npu=True) as (cid, _):
            a_buf = T.alloc_ub((block_N,), dtype)
            b_buf = T.alloc_ub((block_N,), dtype)
            for i in T.serial(per_core):
                t = cid + i * ncore
                if t * block_N < shape:
                    off = t * block_N
                    tail = T.min(block_N, shape - off)
                    T.copy(A[off : off + tail], a_buf[0:tail])
                    T.vabs(a_buf, b_buf)
                    T.copy(b_buf[0:tail], B[off : off + tail])

    return main


def host_wall(fn, iters=200):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    us = (time.perf_counter() - t0) * 1e6 / iters
    torch.npu.synchronize()
    return us


a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)
with ascend_mode("Expert"):
    kern = tilelang.compile(fixed_grid_kernel(N, BLOCK, NCORE), target="npuir")
kern(a, b, N)
torch.npu.synchronize()
assert torch.equal(b, torch.abs(a))

tq = os.getenv("TILELANG_ENABLE_TASKQUEUE", "true")
ours = [host_wall(lambda: kern(a, b, N)) for _ in range(5)]
their = [host_wall(lambda: torch.abs(a, out=b)) for _ in range(5)]
print(f"TILELANG_ENABLE_TASKQUEUE={tq}")
print("ours  host us/call:", " ".join(f"{v:.1f}" for v in ours),
      f" median {statistics.median(ours):.1f}")
print("torch host us/call:", " ".join(f"{v:.1f}" for v in their),
      f" median {statistics.median(their):.1f}")
