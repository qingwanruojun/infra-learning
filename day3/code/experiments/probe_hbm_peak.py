"""Device-side HBM bandwidth ladder for torch.abs (128 / 256 / 512 MiB tensors).

The knobs table had 128 MiB = 1.2 TB/s and 512 MiB = 1.25 TB/s from Event-timed
runs, but the profiler says the 128 MiB kernel is 166.9 us (1.6 TB/s).  Re-measure
all three sizes with the op_summary device duration so the ladder has one clock.
"""

import collections
import csv
import glob
import os
import statistics
import sys

import torch
import torch_npu  # noqa: F401

from torch_npu.profiler import ProfilerActivity, profile  # noqa: E402

sys.path.insert(0, ".")

MIB = 1 << 20


def device_median(label, fn, iters=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    before = set(glob.glob("export_only_prof_dir/*"))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    prof.export_chrome_trace(f"/tmp/peak_{abs(hash(label))}.json")
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
    return statistics.median(durs), len(durs), name


print(f"{'tensor':>10}{'traffic':>10}{'calls':>6}{'device_us':>11}{'GB/s':>9}  op")
for nbytes in (128 * MIB, 256 * MIB, 512 * MIB):
    n = nbytes // 4
    a = torch.randn(n, device="npu", dtype=torch.float32)
    b = torch.zeros_like(a)
    us, calls, name = device_median(f"torch abs {nbytes}", lambda: torch.abs(a, out=b))
    traffic = nbytes * 2
    gbs = traffic / (us * 1e-6) / 1e9
    print(f"{nbytes // MIB:>8} MiB{traffic // MIB:>8} MiB{calls:>6}{us:>11.2f}{gbs:>9.1f}  {name}")
    del a, b
    torch.npu.empty_cache()
