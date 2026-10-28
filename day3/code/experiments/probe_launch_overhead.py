"""Is the abs benchmark host-bound?

``benchmark()`` in own_ops/abs.py wraps ``iters`` calls in one event pair and
divides by ``iters``.  If the Python/tilelang launcher costs more per call than
the kernel itself, the device idles between launches and the reported "us per
call" is really host submission latency -- a constant that swamps real kernel
differences (which is exactly what the knob table shows: 240-335 us, all
within 25% except anomalies).

This probe separates the two:

* single-call event  -> device time of one launch (no host gap inside)
* 10-call event      -> what benchmark() reports
* host wall clock    -> end-to-end including host

Run: python notes/day2/experiments/probe_launch_overhead.py
"""

import time

import torch
import torch_npu  # noqa: F401

import sys

sys.path.insert(0, ".")
from own_ops.abs import compile_abs, make_tensors  # noqa: E402

N, BLOCK = 1048576, 4096
kernel = compile_abs(N, BLOCK, memory="ub", dtype="float32")
a, b = make_tensors(N, "float32")


def warm():
    for _ in range(5):
        kernel(a, b, N)
    torch.npu.synchronize()


def event_batch(iters, rounds=9):
    warm()
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


def wall_batch(iters, rounds=9):
    warm()
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            kernel(a, b, N)
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) / iters * 1e6)
    out.sort()
    return out[len(out) // 2]


warm()
print(f"N={N}, block_N={BLOCK}, ub, fp32 (8 MiB traffic)")
print(f"{'measurement':<34}{'us/call':>10}")
print("-" * 44)
for iters in (1, 2, 5, 10, 50):
    print(f"{'event, %d call(s) per window' % iters:<34}{event_batch(iters):>10.2f}")
for iters in (1, 10, 50):
    print(f"{'wall clock, %d call(s)' % iters:<34}{wall_batch(iters):>10.2f}")
print()
print("If '1 call' is much smaller than '10 calls', the window is host-bound:")
print("device work per launch is the 1-call number, the rest is Python/launcher")
print("latency that the 10-call window cannot hide.")
