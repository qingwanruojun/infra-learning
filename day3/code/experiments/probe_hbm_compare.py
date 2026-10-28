"""Fair HBM-scale comparison against torch: 128 MiB tensor (256 MiB traffic).

At N=1M/16M both our kernel and torch.abs hit on-chip cache, so the comparison
is not bandwidth-limited.  N=32M (128 MiB fp32 input + 128 MiB output) is past
the cache, so both should be HBM-bound.  Compares device durations from CANN's
op_summary (see probe_profiler_kernel_time.py for why not chrome trace).

Kernel: 40 blocks (one per vector core) x block_N=8192, serial over tiles.

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_hbm_compare.py

Archived log(s): logs/hbm_compare.log
Status: superseded by probe_hbm_interleaved.py, then probe_abba_hbm.py.
    Its 166.93 us torch 'peak' was an outlier; probe_hbm_peak.py re-measured 214 us.
"""

import collections
import csv
import glob
import os
import statistics
import sys

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode  # noqa: E402

from torch_npu.profiler import ProfilerActivity, profile  # noqa: E402


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


def device_median(label, fn, iters=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    before = set(glob.glob("export_only_prof_dir/*"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(f"/tmp/hbm_{abs(hash(label))}.json")
    new = sorted(set(glob.glob("export_only_prof_dir/*")) - before)
    csvs = glob.glob(
        os.path.join(new[-1], "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    )
    agg = collections.defaultdict(list)
    for path in csvs:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                agg[row["Op Name"]].append(float(row["Task Duration(us)"].strip()))
    name, durs = max(agg.items(), key=lambda kv: len(kv[1]))
    med = statistics.median(durs)
    print(f"{label:<46}{len(durs):>6}{med:>11.2f}")
    return med


# block_N=16384 with the serial loop fails to compile: the loop makes the
# backend enable multi-buffering, so UB needs 2 * 2 * 16384 * 4 = 256 KiB >
# 192 KiB ("ub overflow, requires 2097152 bits while 1572864 bits available").
# 8192 fits: 2 * 2 * 8192 * 4 = 128 KiB.
N, BLOCK, NCORE = 1 << 25, 8192, 40
with ascend_mode("Expert"):
    kern = tilelang.compile(fixed_grid_kernel(N, BLOCK, NCORE), target="npuir")
a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)
kern(a, b, N)
torch.npu.synchronize()
assert torch.equal(b, torch.abs(a)), "kernel wrong"

print(f"N={N} (128 MiB tensor, 256 MiB traffic), block_N={BLOCK}, {NCORE} blocks")
print(f"{'case':<46}{'calls':>6}{'device_us':>11}")
ours = device_median(f"tilelang 40 blocks, block_N={BLOCK}", lambda: kern(a, b, N))
ref = device_median("torch.abs", lambda: torch.abs(a, out=b))
print()
for label, us in (("tilelang", ours), ("torch", ref)):
    print(f"  {label:<8} {N * 4 * 2 / (us * 1e-6) / 1e9:>7.1f} GB/s")
print(f"  ratio: {ours / ref:.2f}x")
