"""Close the host-side budget: does the per-call cost add up across frames?

Open question from review: cProfile gave launch 939.5 + load_kernel_binary 65 +
eval 40 + wrapper 27 at 4096 blocks (and launch 138.2 at 1024).  That does not
obviously explain 277 us at 1 block or 239 us at 256 blocks, and the implied
per-block slope differs between the two big configs (0.229 vs 0.135 us/block).
So measure both numbers for the same kernel across block counts and check that
the frames sum to the measured wall time:

  wall    : time.perf_counter over 50 calls, no sync inside   (the real budget)
  profile : cProfile stats for the same call path              (the breakdown)

Block counts come from N/block_N on one plain-grid kernel shape, plus the
16M/4096 config used before.
"""

import cProfile
import io
import pstats
import sys
import time

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode  # noqa: E402


def plain_grid_kernel(N, block_N, dtype="float32"):
    num_blocks = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            a_buf = T.alloc_ub((block_N,), dtype)
            b_buf = T.alloc_ub((block_N,), dtype)
            off = cid * block_N
            if off < shape:
                tail = T.min(block_N, shape - off)
                T.copy(A[off : off + tail], a_buf[0:tail])
                T.vabs(a_buf, b_buf)
                T.copy(b_buf[0:tail], B[off : off + tail])

    return main


CONFIGS = [
    (1 << 12, 1 << 12),        # 1 block
    (1 << 20, 1 << 14),        # 64 blocks
    (1 << 20, 1 << 12),        # 256 blocks
    (1 << 24, 1 << 14),        # 1024 blocks
    (1 << 24, 1 << 12),        # 4096 blocks
]

print(f"{'blocks':>7}{'N':>10}{'block_N':>9}{'wall us/call':>14}{'prof total':>12}"
      f"{'launch':>10}{'load_bin':>10}{'eval':>8}{'wrapper':>9}")
for N, block_N in CONFIGS:
    with ascend_mode("Expert"):
        kern = tilelang.compile(plain_grid_kernel(N, block_N), target="npuir")
    a = torch.randn(N, device="npu", dtype=torch.float32)
    b = torch.zeros_like(a)
    for _ in range(5):
        kern(a, b, N)
    torch.npu.synchronize()

    iters = 50
    torch.npu.synchronize()
    w0 = time.perf_counter()
    for _ in range(iters):
        kern(a, b, N)
    wall_us = (time.perf_counter() - w0) * 1e6 / iters
    torch.npu.synchronize()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(iters):
        kern(a, b, N)
    pr.disable()
    torch.npu.synchronize()
    st = pstats.Stats(pr, stream=io.StringIO())

    def per_call(fragment):
        for (fn, lineno, func), (_cc, nc, tt, _ct, _call) in st.stats.items():
            if fragment in func:
                return tt / iters * 1e6
        return 0.0

    total = sum(v[2] for v in st.stats.values()) / iters * 1e6
    print(
        f"{N // block_N:>7}{N:>10}{block_N:>9}{wall_us:>14.1f}{total:>12.1f}"
        f"{per_call('__tilelang_launcher.launch'):>10.1f}"
        f"{per_call('load_kernel_binary'):>10.1f}"
        f"{per_call('builtins.eval'):>8.1f}"
        f"{per_call('JitKernel_NPU.__call__'):>9.1f}"
    )
    del a, b
    torch.npu.empty_cache()
