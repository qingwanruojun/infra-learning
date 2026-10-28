"""Interleaved tilelang-vs-torch at HBM scale (N=32M, 128 MiB tensor).

probe_hbm_compare.py measured torch.abs at 166.93 us (1608 GB/s) but a fresh
process (probe_hbm_peak.py) measured 214.37 us (1252 GB/s) for the same call --
a 28% swing that looks like clock/thermal state rather than kernel differences.
So measure both in ONE process, alternating: torch, ours, torch, ours.  If the
two torch measurements agree, the ratio is trustworthy; if they don't, the gap
is smaller than the run-to-run noise and should not be quoted as a speedup.
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
    # trigger CANN's parse pass; the CSV does not exist without it
    prof.export_chrome_trace(f"/tmp/il_{abs(hash(label))}.json")
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
    print(f"{label:<34}{len(durs):>6}{med:>11.2f}  {name}")
    return med


N, BLOCK, NCORE = 1 << 25, 8192, 40
with ascend_mode("Expert"):
    kern = tilelang.compile(fixed_grid_kernel(N, BLOCK, NCORE), target="npuir")
a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)
kern(a, b, N)
torch.npu.synchronize()
assert torch.equal(b, torch.abs(a)), "kernel wrong"

print(f"N={N} (128 MiB tensor, 256 MiB traffic), block_N={BLOCK}, {NCORE} blocks")
print(f"{'case':<34}{'calls':>6}{'device_us':>11}  op")
torch_meds = []
our_meds = []
for rnd in range(2):
    torch_meds.append(device_median(f"round{rnd} torch.abs", lambda: torch.abs(a, out=b)))
    our_meds.append(device_median(f"round{rnd} tilelang", lambda: kern(a, b, N)))

traffic = N * 4 * 2
print()
print(f"torch     {min(torch_meds):.2f} / {max(torch_meds):.2f} us  "
      f"-> {traffic / (max(torch_meds) * 1e-6) / 1e9:.0f}-{traffic / (min(torch_meds) * 1e-6) / 1e9:.0f} GB/s")
print(f"tilelang  {min(our_meds):.2f} / {max(our_meds):.2f} us  "
      f"-> {traffic / (max(our_meds) * 1e-6) / 1e9:.0f}-{traffic / (min(our_meds) * 1e-6) / 1e9:.0f} GB/s")
print(f"ratio (ours/torch, per-round): "
      f"{our_meds[0] / torch_meds[0]:.2f}x, {our_meds[1] / torch_meds[1]:.2f}x")
