"""Which host function scales with block count?

probe_hostcost_4096blocks.py: host submission alone is ~1092 us/call at 4096
blocks vs ~269 us/call at 1024 blocks (same N).  The generated launcher has no
per-block host loop and no syncBlockLock / rtMalloc in this kernel's wrapper, so
cProfile the Python call path and compare the two grid sizes: whichever frame
grows by ~4x is the culprit (or the cost is inside the compiled stub, which
cProfile will show as one opaque frame).
"""

import cProfile
import io
import pstats
import sys

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode  # noqa: E402

N = 1 << 24


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


for block_N in (4096, 16384):
    with ascend_mode("Expert"):
        kern = tilelang.compile(plain_grid_kernel(N, block_N), target="npuir")
    a = torch.randn(N, device="npu", dtype=torch.float32)
    b = torch.zeros_like(a)
    for _ in range(5):
        kern(a, b, N)
    torch.npu.synchronize()

    iters = 40
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(iters):
        kern(a, b, N)
    pr.disable()
    torch.npu.synchronize()

    st = pstats.Stats(pr, stream=io.StringIO())
    total = sum(v[2] for v in st.stats.values() if v[0] not in (0, None)) or 1.0
    print(f"\n=== block_N={block_N}, blocks={N // block_N}, {iters} calls ===")
    rows = []
    for (fn, lineno, func), (_cc, nc, tt, _ct, _callers) in st.stats.items():
        if tt <= 0:
            continue
        rows.append((tt / iters * 1e6, nc / iters, func))
    for us, calls_per_iter, func in sorted(rows, reverse=True)[:8]:
        print(f"  {func:<40}{us:>10.1f} us/call  ({calls_per_iter:.1f} calls/call)")
    del a, b
    torch.npu.empty_cache()
