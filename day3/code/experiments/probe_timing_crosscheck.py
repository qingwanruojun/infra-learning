"""Cross-check event timing against host wall clock, and verify correctness
of the torch reference used in probe_bw_floor.py.

The previous probe reported ~41 us for 1 MiB / 8 MiB / 64 MiB *and* for a
1024-element tensor, which would imply 3.2 TB/s at 64 MiB -- impossible, so
either the event timing is not seeing device work, or the `out=` call is not
doing the work.  This checks both.
"""

import time

import torch
import torch_npu  # noqa: F401


def event_bench(fn, iters=10, rounds=5):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    out = []
    for _ in range(rounds):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.npu.synchronize()
        out.append(s.elapsed_time(e) / iters * 1000.0)
    out.sort()
    return out[len(out) // 2]


def wall_bench(fn, iters=10, rounds=5):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        out.append((time.perf_counter() - t0) / iters * 1e6)
    out.sort()
    return out[len(out) // 2]


print(f"{'case':<26}{'event_us':>10}{'wall_us':>10}{'GB/s(wall)':>12}  check")
print("-" * 70)

for nbytes in (1 << 20, 8 << 20, 64 << 20):
    n = nbytes // 4
    a = torch.randn(n, device="npu", dtype=torch.float32)
    out = torch.empty_like(a)

    def fn(a=a, out=out):
        torch.abs(a, out=out)

    ev = event_bench(fn)
    wl = wall_bench(fn)
    fn()
    torch.npu.synchronize()
    ref = a.abs()
    ok = torch.equal(out, ref)
    gbps = n * 4 * 2 / (wl * 1e-6) / 1e9
    print(f"abs {nbytes >> 20:>3} MiB read       {ev:>10.2f}{wl:>10.2f}{gbps:>12.1f}  {ok}")
    del a, out, ref

# Same thing without out=, to rule out the out= path being special.
for nbytes in (8 << 20, 64 << 20):
    n = nbytes // 4
    a = torch.randn(n, device="npu", dtype=torch.float32)
    ev = event_bench(lambda a=a: a.abs())
    wl = wall_bench(lambda a=a: a.abs())
    gbps = n * 4 * 3 / (wl * 1e-6) / 1e9
    print(f"a.abs() {nbytes >> 20:>3} MiB (new out){ev:>10.2f}{wl:>10.2f}{gbps:>12.1f}")
    del a
