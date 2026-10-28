"""10-round ABBA interleaved comparison at HBM scale (N=32M).

The earlier probe_hbm_interleaved.py ran 2 rounds (torch, ours, torch, ours) and
got 0.93x / 0.94x.  Two rounds only show "these two agreed"; report median and
spread over 10 ABBA rounds instead (A=torch, B=ours, order A B B A per round so
slow drift cancels within a round).

Device duration from CANN op_summary, as in the other probes.

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_abba_hbm.py 67108864    # 256 MiB tensor
    python experiments/probe_abba_hbm.py 134217728   # 512 MiB tensor
    python experiments/probe_abba_hbm.py 33554432    # 128 MiB -- retired point

Archived log(s): logs/abba_hbm_256MiB.log, logs/abba_hbm_512MiB.log,
                   logs/abba_hbm.log (128 MiB, retired)
Status: primary evidence for the final numbers (§3.2, §4.4).
    Device duration comes from op_summary; 10 rounds, round 0 discarded.
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
    prof.export_chrome_trace("/tmp/abba_trace.json")  # triggers CANN parse
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


# size in elements: 1<<25 = 128 MiB, 1<<26 = 256 MiB, 1<<27 = 512 MiB (fp32)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 25
BLOCK, NCORE, ROUNDS = 8192, 40, 10
print(f"N={N} ({N * 4 // (1 << 20)} MiB tensor, {N * 8 // (1 << 20)} MiB traffic)", flush=True)
with ascend_mode("Expert"):
    kern = tilelang.compile(fixed_grid_kernel(N, BLOCK, NCORE), target="npuir")
a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)
kern(a, b, N)
torch.npu.synchronize()
assert torch.equal(b, torch.abs(a)), "kernel wrong"

torch_calls, our_calls = [], []
for rnd in range(ROUNDS):
    ta = device_median(lambda: torch.abs(a, out=b))
    tb = device_median(lambda: kern(a, b, N))
    tb2 = device_median(lambda: kern(a, b, N))
    ta2 = device_median(lambda: torch.abs(a, out=b))
    torch_us = (ta + ta2) / 2
    our_us = (tb + tb2) / 2
    torch_calls.append(torch_us)
    our_calls.append(our_us)
    print(
        f"round {rnd:>2}  torch {ta:7.2f}/{ta2:7.2f} (avg {torch_us:7.2f})  "
        f"ours {tb:7.2f}/{tb2:7.2f} (avg {our_us:7.2f})  ratio {our_us / torch_us:.3f}x"
    )

m_t, m_o = statistics.median(torch_calls), statistics.median(our_calls)
ratios = [o / t for o, t in zip(our_calls, torch_calls)]
print()
print(f"torch  median {m_t:7.2f} us  range {min(torch_calls):.2f}-{max(torch_calls):.2f}"
      f"  stdev {statistics.stdev(torch_calls):.2f}")
print(f"ours   median {m_o:7.2f} us  range {min(our_calls):.2f}-{max(our_calls):.2f}"
      f"  stdev {statistics.stdev(our_calls):.2f}")
print(f"ratio  median {statistics.median(ratios):.3f}x  range {min(ratios):.3f}-{max(ratios):.3f}")
