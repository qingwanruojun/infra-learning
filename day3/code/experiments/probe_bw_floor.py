"""Measure the launch-overhead floor and achievable bandwidth on this NPU.

Motivation: every abs configuration in ``notes/day2/optimization-knobs.md``
lands at 240-335 us for N=1048576 fp32 (8 MiB of traffic), i.e. 25-35 GB/s.
That is 20-30x below HBM peak on a 910B3, which suggests the absolute number
is dominated by per-iteration overhead (launch + queue) rather than by
bandwidth.  This probe establishes:

1. per-iteration floor: a tiny op (1024 elements) timed with the exact same
   loop shape as ``benchmark()`` in own_ops/abs.py;
2. scaling: torch.abs at 1 MiB / 8 MiB / 64 MiB, so we can see the slope
   (bandwidth) separate from the intercept (overhead).

If the 8 MiB point is mostly intercept, then relative differences measured at
240 us are differences between two mostly-constant quantities.
"""

import torch
import torch_npu  # noqa: F401


def bench(fn, iters=10, rounds=5):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    samples.sort()
    return samples[len(samples) // 2], samples[0], samples[-1]


print(f"{'case':<28}{'median_us':>10}{'min_us':>9}{'max_us':>9}{'GB/s':>8}")
print("-" * 64)

# 1. floor: tiny tensor, same loop shape.  Traffic is 8 KiB, so GB/s is
#    meaningless and the time is almost pure launch/queue overhead.
tiny = torch.randn(1024, device="npu", dtype=torch.float32)
tiny_out = torch.empty_like(tiny)
med, lo, hi = bench(lambda: torch.abs(tiny, out=tiny_out))
print(f"{'tiny 1024 elems (floor)':<28}{med:>10.2f}{lo:>9.2f}{hi:>9.2f}{'-':>8}")

# 2. scaling sweep: 1 / 8 / 64 MiB of useful reads (fp32, 1 read + 1 write).
for nbytes in (1 << 20, 8 << 20, 64 << 20):
    n = nbytes // 4
    a = torch.randn(n, device="npu", dtype=torch.float32)
    out = torch.empty_like(a)
    med, lo, hi = bench(lambda: torch.abs(a, out=out))
    gbps = n * 4 * 2 / (med * 1e-6) / 1e9
    label = f"torch.abs {nbytes >> 20} MiB read"
    print(f"{label:<28}{med:>10.2f}{lo:>9.2f}{hi:>9.2f}{gbps:>8.1f}")

print()
print("Interpretation:")
print("  - If the 1 MiB point is close to the tiny floor, the loop is")
print("    overhead-bound and the abs table compares mostly-constant times.")
print("  - Slope between 8 and 64 MiB gives the sustainable bandwidth;")
print("    intercept at 0 MiB estimates per-iteration overhead.")
