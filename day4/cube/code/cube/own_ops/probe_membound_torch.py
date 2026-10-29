"""Why does torch's 8192x8192x64 matmul look faster than HBM allows?

At M=N=8192, K=64 the output C alone is 8192*8192*2 = 128 MiB, so the
measured 69.58 us implies ~1.9 TB/s of writes -- above the machine's 1.27 TB/s
steady HBM rate.  Two candidate explanations:

  a) op_summary only captured part of the work (several kernels per call, or a
     duration that excludes the store), or
  b) C (128 MiB) fits in the 192 MiB L2 (Ascend910B3.ini l2_size), so the write
     is absorbed by cache and the kernel ends before the write-back.

Sweeping C past the L2 size separates the two: under (a) the implied bandwidth
stays impossible at every size; under (b) it drops to the HBM steady rate once
C no longer fits.

For each shape this prints EVERY op CANN reports (name, count, median duration),
so a multi-kernel call shows up as several names instead of being averaged away.

Run:  python day4/code/own_ops/probe_membound_torch.py
"""

import collections
import csv
import glob
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOG_DIR = os.path.abspath(os.path.join(_HERE, "..", "logs"))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
from torch_npu.profiler import ProfilerActivity, profile  # noqa: E402

SHAPES = [(8192, 8192, 64), (12288, 12288, 64), (16384, 16384, 64)]
ITERS = 5
L2_BYTES = 201326592  # Ascend910B3.ini l2_size


def profile_once(fn, iters, tag):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    before = set(glob.glob("export_only_prof_dir/*"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(f"/tmp/trace_mb_{abs(hash(tag)) % (1 << 32):08x}.json")
    new = sorted(set(glob.glob("export_only_prof_dir/*")) - before)
    if not new:
        raise RuntimeError(f"[{tag}] profiler produced no run directory")
    csvs = glob.glob(os.path.join(
        new[-1], "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv"))
    if not csvs:
        raise RuntimeError(f"[{tag}] no op_summary csv under {new[-1]}")
    agg = collections.defaultdict(list)
    for path in csvs:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                agg[row["Op Name"]].append(float(row["Task Duration(us)"].strip()))
    return agg


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"L2 = {L2_BYTES / 2**20:.0f} MiB; HBM steady ~1.27 TB/s (Day 2)")
    for M, N, K in SHAPES:
        a = torch.randn(M, K, dtype=torch.float16).npu()
        b = torch.randn(K, N, dtype=torch.float16).npu()
        c = torch.zeros(M, N, dtype=torch.float16).npu()
        c_bytes = M * N * 2
        print(f"\n=== {M}x{N}x{K}  C={c_bytes / 2**20:.0f} MiB "
              f"({'fits L2' if c_bytes <= L2_BYTES else 'exceeds L2'}) ===")

        t0 = time.perf_counter()
        agg = profile_once(lambda: torch.matmul(a, b, out=c), ITERS, f"{M}_{N}_{K}")
        t1 = time.perf_counter()

        total_med = 0.0
        for name, durs in sorted(agg.items(), key=lambda kv: -len(kv[1])):
            med = statistics.median(durs)
            print(f"  {len(durs):>3d}x  {med:9.2f} us  median  {name[:70]}")
            if len(durs) == ITERS:
                total_med += med
        print(f"  implied write bandwidth (C only, median): "
              f"{c_bytes / (total_med * 1e-6) / 1e12:.2f} TB/s  "
              f"[sum of per-call medians = {total_med:.2f} us]")
        print(f"  wall time for {ITERS} calls + profiler parse: {t1 - t0:.1f} s")
        del a, b, c
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
