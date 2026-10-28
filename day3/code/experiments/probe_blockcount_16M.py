"""Measure the last remaining estimate: plain 4096-block grid vs 40-block serial
at N=16M, device-side.  Docs so far said "4095-block version is ~800 us by
subtracting 270 us of host from the 1095 us Event number" -- that was inferred,
not measured.  Same block_N=4096 in both variants, so the only difference is
the grid size.

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_blockcount_16M.py

Archived log(s): logs/blockcount_16M.log
Status: primary evidence (4096-block grid vs 40-block serial, §4.4b).
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


def plain_grid_kernel(N, block_N, dtype="float32"):
    num_blocks = T.ceildiv(N, block_N)

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), shape: T.int32):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            a_buf = T.alloc_ub((block_N,), dtype)
            b_buf = T.alloc_ub((block_N,), dtype)
            off = cid * block_N
            if off < shape:
                tail = T.min(block_N, shape - off)
                T.copy(A[off : off + tail], a_buf[0:tail])
                T.vabs(a_buf, b_buf)
                T.copy(b_buf[0:tail], B[off : off + tail])

    return main


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


def device_median(label, fn, iters=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    before = set(glob.glob("export_only_prof_dir/*"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(f"/tmp/bc_{abs(hash(label))}.json")  # triggers CANN parse
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


N, BLOCK = 1 << 24, 4096
with ascend_mode("Expert"):
    plain = tilelang.compile(plain_grid_kernel(N, BLOCK), target="npuir")
    fixed = tilelang.compile(fixed_grid_kernel(N, BLOCK, 40), target="npuir")
a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)
for k in (plain, fixed):
    k(a, b, N)
    torch.npu.synchronize()
    assert torch.equal(b, torch.abs(a)), "kernel wrong"
    b.zero_()

print(f"N={N} (64 MiB), block_N={BLOCK} in both variants")
print(f"{'case':<34}{'calls':>6}{'device_us':>11}  op")
p0 = device_median("round0 plain 4096 blocks", lambda: plain(a, b, N))
f0 = device_median("round0 fixed 40 blocks serial", lambda: fixed(a, b, N))
p1 = device_median("round1 plain 4096 blocks", lambda: plain(a, b, N))
f1 = device_median("round1 fixed 40 blocks serial", lambda: fixed(a, b, N))
print()
print(f"plain {p0:.2f} / {p1:.2f} us     fixed {f0:.2f} / {f1:.2f} us")
print(f"ratio plain/fixed: {p0 / f0:.1f}x, {p1 / f1:.1f}x")
