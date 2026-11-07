"""Common timing harness for Day 8.

Three clocks, reported separately:
  * host_us   : wall time to enqueue one call on the CPU (no device sync inside)
  * e2e_us    : NPU-event elapsed time per call around a loop (includes gaps)
  * device_us : per-kernel duration from torch_npu profiler kernel_details.csv

`e2e_us` is the same口径 as Day 3's `benchmark()`, so Day 3 vs Day 8 e2e is
comparable. `device_us` is only used for a same-condition kernel comparison
when the profiler produced a usable kernel row.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import tempfile
import time

import torch
import torch_npu  # noqa: F401


def e2e_bench(fn, iters=20, rounds=7, warmup=5) -> dict:
    fn()
    torch.npu.synchronize()
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    samples.sort()
    return {
        "median_us": samples[len(samples) // 2],
        "min_us": samples[0],
        "max_us": samples[-1],
        "samples_us": samples,
        "iters": iters,
        "rounds": rounds,
    }


def host_enqueue_us(fn, iters=50, warmup=5) -> dict:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)
    torch.npu.synchronize()
    samples.sort()
    return {
        "median_us": statistics.median(samples),
        "min_us": samples[0],
        "max_us": samples[-1],
        "iters": iters,
    }


def _read_kernel_details(root: str) -> list[dict]:
    rows = []
    for path in glob.glob(os.path.join(root, "**", "kernel_details.csv"), recursive=True):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def device_bench(fn, kernel_name_substr: str, iters=20, warmup=3) -> dict:
    """Profile one kernel; return per-call duration of matching kernel rows."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    with tempfile.TemporaryDirectory(prefix="day8_prof_") as td:
        try:
            exp = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1
            )
        except Exception:
            exp = None
        kwargs = dict(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(td),
        )
        if exp is not None:
            kwargs["experimental_config"] = exp
        with torch_npu.profiler.profile(**kwargs):
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
        rows = _read_kernel_details(td)
        if not rows:
            return {"available": False, "reason": "no kernel_details.csv rows", "dir": td}

        # Column names vary slightly across torch_npu versions.
        def pick(row, *names):
            for n in names:
                if n in row and row[n] not in ("", None):
                    return row[n]
            return None

        matched = []
        all_names = []
        for row in rows:
            name = pick(row, "Kernel Name", "kernel_name", "Name") or ""
            all_names.append(name)
            if kernel_name_substr.lower() in name.lower():
                dur = pick(row, "Duration(us)", "duration", "Duration")
                if dur is not None:
                    try:
                        matched.append(float(dur))
                    except ValueError:
                        pass
        if not matched:
            return {
                "available": False,
                "reason": f"no row matching {kernel_name_substr!r}",
                "kernel_names": sorted(set(all_names))[:20],
            }
        matched.sort()
        return {
            "available": True,
            "median_us": statistics.median(matched),
            "min_us": matched[0],
            "max_us": matched[-1],
            "n_rows": len(matched),
            "iters": iters,
            "path": "torch_npu.profiler kernel_details.csv",
        }
