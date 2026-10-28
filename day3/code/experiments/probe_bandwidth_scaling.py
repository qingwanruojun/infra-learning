"""Separate per-launch cost from bandwidth in the hand-written abs kernel.

If per-call time is flat in N, the kernel is launch-bound and the knob table is
comparing a constant.  If it scales, the slope gives the real bandwidth and the
intercept gives the per-launch cost.

Same block_N=4096 for every N, so the grid grows with N exactly as it does in
the knob table.

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_bandwidth_scaling.py

Archived log(s): logs/bandwidth_scaling.log
Status: superseded by probe_host_vs_blockcost.py + probe_hostbudget.py:
    the flat-in-N answer turned out to be host submission, not bandwidth.
"""

import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, ".")
from own_ops.abs import compile_abs, make_tensors  # noqa: E402

BLOCK = 4096


def bench(kernel, a, b, n, iters=50, rounds=7):
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


print(f"{'N':>10}{'MiB traffic':>13}{'abs_us':>9}{'GB/s':>9}{'torch_us':>10}{'torch GB/s':>12}")
print("-" * 68)
rows = []
for N in (1 << 20, 1 << 22, 1 << 24):
    kernel = compile_abs(N, BLOCK, memory="ub", dtype="float32")
    a, b = make_tensors(N, "float32")
    us = bench(kernel, a, b, N)
    gbps = N * 4 * 2 / (us * 1e-6) / 1e9
    torch.npu.synchronize()
    torch_us = bench(lambda x, y, n: torch.abs(x, out=y), a, b, N)
    tgbps = N * 4 * 2 / (torch_us * 1e-6) / 1e9
    print(f"{N:>10}{N*4*2/2**20:>13.0f}{us:>9.2f}{gbps:>9.1f}{torch_us:>10.2f}{tgbps:>12.1f}")
    rows.append((N, us))
    del a, b
    torch.npu.empty_cache()

print()
(n1, t1), (n2, t2) = rows[0], rows[-1]
slope = (t2 - t1) / ((n2 - n1) * 4 * 2)  # us per byte
intercept = t1 - slope * n1 * 4 * 2
print(f"linear fit from N={n1} to N={n2}:")
print(f"  per-launch intercept ~ {intercept:.1f} us")
print(f"  streaming bandwidth  ~ {1/(slope*1e-6)/1e9:.1f} GB/s")
