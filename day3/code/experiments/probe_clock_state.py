"""Is the 128 MiB measurement clock-state dependent?  Cold vs post-load, both impls.

The same kernel and config measured 217.7 us in probe_config_sweep.py (cold
process) and 198.1 us in probe_ladder_bidir.py (after 60 s of sustained load),
a 10% swing that dwarfs every config knob.  Separate the two candidates:

  clock state : sustained load boosts the vector clock; 128 MiB is partly
                issue-bound so it shows up
  L2 state    : 128 MiB sits near the cache boundary; residency depends on
                what ran before

Protocol: measure ours cold, measure torch cold, then 60 s of load with OUR
kernel only, then measure both again.  If both implementations speed up by the
same factor, it is the clock; if only the one that ran last speeds up, it is L2.
Then repeat the same protocol at 512 MiB (pure HBM) as a control.

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_clock_state.py

Archived log(s): logs/clock_state.log
Status: primary evidence -- the confounder that makes 128 MiB unusable as a
    comparison point (cold vs post-load).
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
BLOCK, NCORE = 8192, 40


def fixed_grid_kernel(N, block_N, ncore, dtype="float32"):
    tiles = T.ceildiv(N, block_N)
    per_core = T.ceildiv(tiles, ncore)

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), shape: T.int32):
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
    prof.export_chrome_trace("/tmp/clock_trace.json")  # triggers CANN parse
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


for mib in (128, 512):
    n = mib * MIB // 4
    a = torch.randn(n, device="npu", dtype=torch.float32)
    b = torch.zeros_like(a)
    traffic = 2 * n * 4
    with ascend_mode("Expert"):
        kern = tilelang.compile(fixed_grid_kernel(n, BLOCK, NCORE), target="npuir")
    kern(a, b, n)
    torch.npu.synchronize()
    assert torch.equal(b, torch.abs(a)), "wrong"

    cold_o = [device_median(lambda: kern(a, b, n)) for _ in range(2)]
    cold_t = [device_median(lambda: torch.abs(a, out=b)) for _ in range(2)]

    t0 = time.perf_counter()
    calls = 0
    while time.perf_counter() - t0 < 60:
        kern(a, b, n)
        calls += 1
        if calls % 64 == 0:
            torch.npu.synchronize()
    torch.npu.synchronize()
    load_s = time.perf_counter() - t0

    warm_o = [device_median(lambda: kern(a, b, n)) for _ in range(2)]
    warm_t = [device_median(lambda: torch.abs(a, out=b)) for _ in range(2)]

    print(f"{mib} MiB ({traffic // MIB} MiB traffic), {calls} calls / {load_s:.0f} s load")
    print(f"  ours  cold {cold_o[0]:7.2f} {cold_o[1]:7.2f} -> warm {warm_o[0]:7.2f} {warm_o[1]:7.2f}"
          f"   speedup {statistics.mean(cold_o) / statistics.mean(warm_o):.3f}x")
    print(f"  torch cold {cold_t[0]:7.2f} {cold_t[1]:7.2f} -> warm {warm_t[0]:7.2f} {warm_t[1]:7.2f}"
          f"   speedup {statistics.mean(cold_t) / statistics.mean(warm_t):.3f}x", flush=True)
    del a, b
    torch.npu.empty_cache()
