"""Bidirectional HBM ladder: is the 7% gap a size effect, a clock effect, or both?

Open question (review point 2): the 128 MiB point of probe_hbm_peak.py (torch
214.4 us = 1.25 TB/s, measured first in a fresh process) disagrees with the
same call inside probe_abba_hbm.py (185.7 us = 1.45 TB/s) by 15%.  If that is
DVFS, then (a) absolute TB/s numbers cannot be compared across processes, and
(b) the ladder's monotonic 1.25 -> 1.27 -> 1.28 may be a warm-up artefact
rather than a hardware curve.

Protocol:
  1. one cold measurement at 128 MiB (torch only) -> reproduces the ladder point
  2. 60 s of sustained load at 512 MiB                        -> force steady clocks
  3. ascending pass   128 / 256 / 512 MiB, A-B-B-A each       -> post-warmup ladder
  4. descending pass  512 / 256 / 128 MiB, A-B-B-A each       -> order check
Each size gets its own compiled kernel (same 40 blocks x block_N=8192, but the
typed extent and the per-core loop trip count are that size's), and torch runs
on the matching slice a[:n] / b[:n].  v1 of this probe got both wrong.

Device duration from CANN op_summary, as in the other probes.
"""

import collections
import csv
import glob
import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode  # noqa: E402

from torch_npu.profiler import ProfilerActivity, profile  # noqa: E402

MIB = 1 << 20
SIZES_MIB = [128, 256, 512]
NCORE, BLOCK = 40, 8192
WARMUP_SECONDS = 60


def fixed_grid_kernel(n_typed, block_N, ncore, dtype="float32"):
    tiles = T.ceildiv(n_typed, block_N)
    per_core = T.ceildiv(tiles, ncore)

    @T.prim_func
    def main(A: T.Tensor((n_typed,), dtype), B: T.Tensor((n_typed,), dtype), shape: T.int32):
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


def device_median(fn, iters=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    before = set(glob.glob("export_only_prof_dir/*"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    prof.export_chrome_trace("/tmp/ladder_trace.json")  # triggers CANN parse
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
    return statistics.median(durs)


tensors, kerns = {}, {}
for mib in SIZES_MIB:
    n = mib * MIB // 4
    a = torch.randn(n, device="npu", dtype=torch.float32)
    b = torch.zeros_like(a)
    tensors[mib] = (n, a, b)
    with ascend_mode("Expert"):
        kerns[mib] = tilelang.compile(fixed_grid_kernel(n, BLOCK, NCORE), target="npuir")
    kerns[mib](a, b, n)
    torch.npu.synchronize()
    assert torch.equal(b, torch.abs(a)), f"wrong at {mib} MiB"
    print(f"compiled + verified {mib} MiB", flush=True)

n128, a128, b128 = tensors[128]
kern = kerns[128]

# 1) cold point, same shape as the ladder's first row
cold = device_median(lambda: torch.abs(a128, out=b128))
print(f"cold 128 MiB torch {cold:8.2f} us  {2 * 128 * MIB / cold / 1e6:.3f} TB/s", flush=True)

# 2) sustained load to force steady clocks
n512, a512, b512 = tensors[512]
t0 = time.perf_counter()
calls = 0
while time.perf_counter() - t0 < WARMUP_SECONDS:
    torch.abs(a512, out=b512)
    calls += 1
    if calls % 64 == 0:
        torch.npu.synchronize()
torch.npu.synchronize()
print(f"warmup: {calls} torch.abs calls at 512 MiB over {time.perf_counter() - t0:.1f} s",
      flush=True)


def abba(mib):
    n, a, b = tensors[mib]
    kern = kerns[mib]
    t1 = device_median(lambda: torch.abs(a, out=b))
    o1 = device_median(lambda: kern(a, b, n))
    o2 = device_median(lambda: kern(a, b, n))
    t2 = device_median(lambda: torch.abs(a, out=b))
    t, o = (t1 + t2) / 2, (o1 + o2) / 2
    traffic = 2 * mib * MIB
    print(
        f"  {mib:4d} MiB  torch {t1:7.2f}/{t2:7.2f} (avg {t:7.2f}, {traffic / t / 1e6:5.3f} TB/s)"
        f"  ours {o1:7.2f}/{o2:7.2f} (avg {o:7.2f}, {traffic / o / 1e6:5.3f} TB/s)"
        f"  ratio {o / t:.3f}x",
        flush=True,
    )
    return mib, t, o


print("ascending pass", flush=True)

asc = [abba(m) for m in SIZES_MIB]
print("descending pass", flush=True)
desc = [abba(m) for m in reversed(SIZES_MIB)]

print()
print("size   torch asc/desc      ours asc/desc       ratio asc/desc")
for (m, t_a, o_a) in asc:
    t_d, o_d = [(t, o) for (mm, t, o) in desc if mm == m][0]
    print(f"{m:4d}   {t_a:7.2f} / {t_d:7.2f}   {o_a:7.2f} / {o_d:7.2f}   "
          f"{o_a / t_a:.3f} / {o_d / t_d:.3f}")

t_a = dict((m, (t, o)) for m, t, o in asc)
t_d = dict((m, (t, o)) for m, t, o in desc)
print()
for m in SIZES_MIB:
    print(f"{m:4d} MiB order effect: torch {t_a[m][0] / t_d[m][0]:.3f}x  "
          f"ours {t_a[m][1] / t_d[m][1]:.3f}x")
print(f"cold-vs-warm 128 MiB torch: {cold:.2f} -> {t_d[128][0]:.2f} us "
      f"({cold / t_d[128][0]:.2f}x)")
