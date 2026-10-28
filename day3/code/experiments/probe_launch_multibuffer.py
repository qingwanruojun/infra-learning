"""Does the per-call host cost depend on auto multi-buffering?

The fixed-grid serial kernel costs ~444 us of host per call (taskqueue wrapper
on) while the plain-grid kernel at 1-256 blocks costs 216-232 us.  The serial
tile loop is also what triggers auto multi-buffering, and the launcher template
allocates a sync-lock block per call when lock_num > 0.  Compare the same
kernel with NPUIR_ENABLE_AUTO_MULTI_BUFFER on and off: host wall (5 x 200
enqueues) and device median in one process.

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_launch_multibuffer.py

Archived log(s): logs/launch_multibuffer.log
Status: primary evidence (does auto multi-buffering change the host cost?).
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

N, BLOCK, NCORE = 1 << 25, 8192, 40


def fixed_grid_kernel(n, block_N, ncore, dtype="float32"):
    tiles = T.ceildiv(n, block_N)
    per_core = T.ceildiv(tiles, ncore)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), shape: T.int32):
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
    prof.export_chrome_trace("/tmp/mb_trace.json")  # triggers CANN parse
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


def host_wall(fn, iters=200):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    us = (time.perf_counter() - t0) * 1e6 / iters
    torch.npu.synchronize()
    return us


a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)

print(f"{'auto multi-buffer':>18}{'host us/call (x5)':>34}{'device us':>11}")
for auto in (True, False):
    cfg = {} if auto else {tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False}
    with ascend_mode("Expert"):
        kern = tilelang.compile(fixed_grid_kernel(N, BLOCK, NCORE), target="npuir",
                                pass_configs=cfg)
    kern(a, b, N)
    torch.npu.synchronize()
    assert torch.equal(b, torch.abs(a)), "wrong"
    walls = [host_wall(lambda: kern(a, b, N)) for _ in range(5)]
    dev = device_median(lambda: kern(a, b, N))
    print(f"{str(auto):>18}"
          f"{' '.join(f'{w:.0f}' for w in walls):>34}{dev:>11.2f}"
          f"   (median host {statistics.median(walls):.0f})", flush=True)
