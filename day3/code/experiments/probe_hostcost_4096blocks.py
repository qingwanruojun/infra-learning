"""What is the extra ~900 us in the 4096-block Event number?

probe_blockcount_16M.py measured the plain 4096-block kernel at N=16M as 156-159 us
on device, but the same configuration reports 1095 us through the Event-wrapped
loop.  158 + ~240 us host does not reach 1095, so either the host cost per call
grows with the number of blocks, or the Event window is measuring something else.
Decompose it three ways on the same kernel:

  enqueue-only : 200 calls back-to-back, no sync inside  -> host submission cost
  sync-per-call: sync after every call                   -> host + device, serial
  event-window : 1 Event pair around 10 calls (abs.py)   -> reproduces the table

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_hostcost_4096blocks.py

Archived log(s): logs/hostcost_4096blocks.log;
                   logs/hostcost_1M.log (earlier run, the script now pins N=16M)
Status: primary evidence (splits the 1095 us Event number three ways).
"""

import sys

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


def report(label, per_call_us):
    print(f"{label:<44}{per_call_us:>10.1f} us/call")


N = 1 << 24
for BLOCK in (4096, 16384):
    with ascend_mode("Expert"):
        kern = tilelang.compile(plain_grid_kernel(N, BLOCK), target="npuir")
    a = torch.randn(N, device="npu", dtype=torch.float32)
    b = torch.zeros_like(a)
    kern(a, b, N)
    torch.npu.synchronize()
    assert torch.equal(b, torch.abs(a)), "kernel wrong"

    print(f"\nN={N}, block_N={BLOCK}, blocks={N // BLOCK}")
    for _ in range(5):
        kern(a, b, N)
    torch.npu.synchronize()

    # 1. enqueue only: no sync inside the loop
    iters = 200
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    import time

    torch.npu.synchronize()
    w0 = time.perf_counter()
    t0.record()
    for _ in range(iters):
        kern(a, b, N)
    t1.record()
    torch.npu.synchronize()
    wall = time.perf_counter() - w0
    report(f"enqueue-only, {iters} calls (host wall)", wall * 1e6 / iters)
    report(f"  same, Event window", t0.elapsed_time(t1) * 1000 / iters)

    # 2. sync per call
    iters2 = 30
    torch.npu.synchronize()
    w0 = time.perf_counter()
    for _ in range(iters2):
        kern(a, b, N)
        torch.npu.synchronize()
    wall = time.perf_counter() - w0
    report(f"sync-per-call, {iters2} calls (host wall)", wall * 1e6 / iters2)

    # 3. abs.py style: one Event pair around 10 calls
    for _ in range(3):
        t0.record()
        for _ in range(10):
            kern(a, b, N)
        t1.record()
        torch.npu.synchronize()
        report("  event-window x10 (abs.py harness)", t0.elapsed_time(t1) * 1000 / 10)
    del a, b
    torch.npu.empty_cache()
