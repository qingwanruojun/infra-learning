"""Pin down host vs device time with the NPU profiler.

Event timing showed a ~240 us/call floor even for N=1024 (16 KiB), i.e. the
harness is host-bound below 240 us.  The profiler reports the kernel's own
device duration, which settles:

* standard abs, N=1M  -- event says 239 us/call.  Device duration?
* fixed 40-block abs, N=16M -- event says 299 us/call.  Device duration?
* torch.abs, N=1M and a 128 MiB tensor -- reference points.

Run: python notes/day2/experiments/probe_profiler_kernel_time.py
"""

import sys

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

sys.path.insert(0, ".")
from own_ops.abs import ascend_mode, compile_abs, make_tensors  # noqa: E402

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


def profile_calls(label, fn, iters=20):
    """Profile, then read device durations from CANN's op_summary CSV.

    Neither ``profile.key_averages()`` (absent in this torch_npu build) nor the
    chrome trace (host events only: async_npu / enqueue / dequeue) exposes the
    kernel duration.  The CANN profiler writes it to
    ``export_only_prof_dir/<run>/PROF_*/mindstudio_profiler_output/op_summary_*.csv``
    under "Task Duration(us)".
    """
    import collections
    import csv
    import glob
    import os
    import statistics

    for _ in range(3):
        fn()
    torch.npu.synchronize()
    before = set(glob.glob("export_only_prof_dir/*"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    # export_chrome_trace is what triggers CANN's own parse pass; without it the
    # run directory contains no op_summary CSV at all.
    prof.export_chrome_trace(f"/tmp/trace_{abs(hash(label))}.json")
    new = sorted(set(glob.glob("export_only_prof_dir/*")) - before)
    if not new:
        print(f"\n===== {label}: no new prof dir =====")
        return None
    csvs = glob.glob(
        os.path.join(new[-1], "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    )
    if not csvs:
        print(f"\n===== {label}: no op_summary.csv under {new[-1]} =====")
        return None

    agg = collections.defaultdict(list)
    with open(csvs[0]) as fh:
        for row in csv.DictReader(fh):
            agg[row["Op Name"]].append(float(row["Task Duration(us)"].strip()))

    print(f"\n===== {label} ({iters} calls) =====")
    print(f"{'op':<46}{'calls':>7}{'median_us':>11}{'mean_us':>10}")
    for name, durs in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print(f"{name[:44]:<46}{len(durs):>7}{statistics.median(durs):>11.2f}"
              f"{sum(durs) / len(durs):>10.2f}")
    return agg


N1 = 1 << 20
std = compile_abs(N1, 4096, memory="ub", dtype="float32")
a1, b1 = make_tensors(N1, "float32")
profile_calls("tilelang abs, 4096 blocks, N=1M", lambda: std(a1, b1, N1))

N2 = 1 << 24
with ascend_mode("Expert"):
    fx = tilelang.compile(fixed_grid_kernel(N2, 4096, 40), target="npuir")
a2, b2 = make_tensors(N2, "float32")
fx(a2, b2, N2)
torch.npu.synchronize()
assert torch.equal(b2, torch.abs(a2)), "fixed-grid kernel wrong"
profile_calls("tilelang abs, 40 blocks serial, N=16M", lambda: fx(a2, b2, N2))

# torch reference, same tensors
profile_calls("torch.abs, N=1M", lambda: torch.abs(a1, out=b1))
profile_calls("torch.abs, N=16M (64 MiB)", lambda: torch.abs(a2, out=b2))
