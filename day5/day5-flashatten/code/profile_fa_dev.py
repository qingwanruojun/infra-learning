#!/usr/bin/env python
"""CANN op_summary device time for the Developer-mode FA kernel.

Event timing is ambiguous here: a start event is recorded on the stream, then
the host spends time enqueueing the kernel, so the measured interval can be
host-enqueue + kernel. This script asks CANN's profiler for the kernel's own
Task Duration and prints per-op medians, so the Cube-side and Vector-side
functions are visible separately.

Usage:
    python day5-flashattn/code/profile_fa_dev.py --seq-lens 512 4096
"""

import argparse
import collections
import csv
import glob
import os
import statistics
import sys

import torch

import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_fa_dev import build_kernel  # noqa: E402


def new_run_dir(log_dir, before):
    runs = glob.glob(os.path.join(log_dir, "export_only_prof_dir", "*"))
    # Sort by mtime: the run-dir name embeds PID before the timestamp, so a
    # lexicographic sort can put the newest PID before older ones.
    new = sorted(set(runs) - before, key=os.path.getmtime)
    if not new:
        raise RuntimeError("profiler produced no run directory")
    return new[-1]


def summary_csv(run_dir):
    csvs = glob.glob(os.path.join(run_dir, "PROF_*", "mindstudio_profiler_output",
                                  "op_summary_*.csv"))
    if not csvs:
        raise RuntimeError(f"no op_summary csv under {run_dir}")
    return csvs


def profile_callable(fn, iters, log_dir, tag):
    from torch_npu.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.npu.synchronize()

    os.makedirs(log_dir, exist_ok=True)
    old = os.getcwd()
    os.chdir(log_dir)
    try:
        before = set(glob.glob(os.path.join(".", "export_only_prof_dir", "*")))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
        trace = f"/tmp/trace_fa_{tag}.json"
        prof.export_chrome_trace(trace)
        run_dir = new_run_dir(".", before)
        agg = collections.defaultdict(list)
        for path in summary_csv(run_dir):
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    name = row.get("Op Name", "").strip()
                    dur = row.get("Task Duration(us)", "").strip()
                    if name and dur:
                        agg[name].append(float(dur))
        return agg
    finally:
        os.chdir(old)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--log-dir", type=str,
                        default="day5-flashattn/logs/profiler")
    args = parser.parse_args()

    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    torch.npu.set_device(0)

    for seq_len in args.seq_lens:
        print(f"\n=== seq_len={seq_len} ours ===", flush=True)
        kernel = build_kernel(seq_len, args.dim, args.block_m, args.block_n,
                              args.block_k, "float16", "float32")
        shape = [seq_len, args.dim]
        q = torch.randn(shape, dtype=torch.float16).npu()
        k = torch.randn(shape, dtype=torch.float16).npu()
        v = torch.randn(shape, dtype=torch.float16).npu()

        agg = profile_callable(lambda: kernel(q, k, v), args.iters,
                               args.log_dir, f"ours_{seq_len}")
        for name, durs in sorted(agg.items(), key=lambda kv: -statistics.median(kv[1])):
            if statistics.median(durs) >= 1.0:
                print(f"  {name:40s} n={len(durs):3d} median={statistics.median(durs):10.2f} us",
                      flush=True)

        print(f"=== seq_len={seq_len} torch manual chain ===", flush=True)
        scale = (1.0 / args.dim) ** 0.5

        def torch_call():
            torch.nn.functional.softmax(
                (q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v

        agg = profile_callable(torch_call, args.iters, args.log_dir,
                               f"torch_{seq_len}")
        for name, durs in sorted(agg.items(), key=lambda kv: -statistics.median(kv[1])):
            if statistics.median(durs) >= 1.0:
                print(f"  {name:40s} n={len(durs):3d} median={statistics.median(durs):10.2f} us",
                      flush=True)


if __name__ == "__main__":
    main()
