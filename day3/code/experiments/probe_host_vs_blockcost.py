"""Two decisive checks for the Day-2 measurement post-mortem.

A) Is the ~178-240 us per-call cost host submission or device time?
   Time 200 ``kernel(a,b,N)`` calls WITHOUT synchronizing: if the host needs
   ~240 us per call, the loop cannot outrun the queue and the measured latency
   is host-bound.

B) Is the 16M block_N effect "per-block cost" or "chunk size"?
   Compare the normal grid (one block per tile, 4096 blocks) against a
   fixed 40-block grid where each core serially loops over its tiles.  Same
   chunk size (block_N=4096), same total DMA count, different block count.
   If block count were the driver, the fixed grid would be ~4x faster; if
   chunk size is the driver, both stay ~1120 us.
"""

import time

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

import sys

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode, compile_abs, make_tensors  # noqa: E402

N, BLOCK, NCORE = 1 << 24, 4096, 40


def fixed_grid_kernel(N, block_N, ncore, dtype="float32"):
    tiles = T.ceildiv(N, block_N)
    per_core = T.ceildiv(tiles, ncore)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
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


def device_bench(kernel, a, b, n, iters=20, rounds=7):
    for _ in range(5):
        kernel(a, b, n)
    torch.npu.synchronize()
    out = []
    for _ in range(rounds):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            kernel(a, b, n)
        e.record()
        torch.npu.synchronize()
        out.append(s.elapsed_time(e) / iters * 1000.0)
    out.sort()
    return out[len(out) // 2]


a, b = make_tensors(N, "float32")

# --- A: host submission, no sync inside the loop -------------------------
print("A) host submission cost (200 calls, no sync until the end)")
for name, k in (("standard 4096", compile_abs(N, BLOCK, memory="ub", dtype="float32")),):
    k(a, b, N)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        k(a, b, N)
    host_us = (time.perf_counter() - t0) / 200 * 1e6
    torch.npu.synchronize()
    total_us = (time.perf_counter() - t0) / 200 * 1e6
    print(f"   {name}: host enqueue {host_us:>7.1f} us/call, "
          f"end-to-end {total_us:>7.1f} us/call")
    std = k
    std_us = device_bench(k, a, b, N)

# --- B: block count vs chunk size at N=16M ------------------------------
print()
print("B) block count vs chunk size (N=16M, block_N=4096, 128 MiB traffic)")
b.zero_()
with ascend_mode("Expert"):
    fixed = tilelang.compile(fixed_grid_kernel(N, BLOCK, NCORE), target="npuir")
ok = False
try:
    fixed(a, b, N)
    torch.npu.synchronize()
    ok = torch.equal(b, torch.abs(a))
    fx_us = device_bench(fixed, a, b, N)
    print(f"   normal grid  ({N // BLOCK} blocks){std_us:>9.1f} us")
    print(f"   fixed grid   ({NCORE} blocks, serial){fx_us:>9.1f} us  correct={ok}")
except Exception as exc:  # noqa: BLE001
    print(f"   fixed grid FAILED: {type(exc).__name__}: {exc}")
