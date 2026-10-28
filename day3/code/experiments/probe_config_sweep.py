"""block_N x NCORE sweep in the device-time domain, at the 128 MiB working set.

Question from review point 5: is the 128 MiB gap a parallelism problem (too few
cores) or a per-tile transfer-granularity problem (tile too small/large)?
Auto multi-buffering is switched off for the 16384 cells because 2 buffers x
2 stages x 64 KiB = 256 KiB > 192 KiB UB; one 8192 cell is also run with it off
to isolate the effect of that flag.

Device duration from CANN op_summary, as in the other probes.
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

N = 1 << 25          # 128 MiB tensor, 256 MiB traffic
MIB = 1 << 20

# (block_N, ncore, auto_multi_buffer)
CONFIGS = [
    (4096, 40, True),
    (8192, 40, True),
    (16384, 40, False),
    (4096, 20, True),
    (8192, 20, True),
    (16384, 20, False),
    (8192, 40, False),
]


def fixed_grid_kernel(block_N, ncore, dtype="float32"):
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
    prof.export_chrome_trace("/tmp/sweep_trace.json")  # triggers CANN parse
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


a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)
traffic = 2 * N * 4

print(f"N={N} ({N * 4 // MIB} MiB tensor, {traffic // MIB} MiB traffic, {4 * N // MIB} MiB/block at block_N=4096)")
print(f"{'block_N':>8}{'NCORE':>7}{'autoMB':>8}{'tiles/core':>12}{'run1 us':>10}{'run2 us':>10}{'TB/s':>8}")
for block_N, ncore, auto in CONFIGS:
    cfg = {} if auto else {
        tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False
    }
    try:
        with ascend_mode("Expert"):
            kern = tilelang.compile(
                fixed_grid_kernel(block_N, ncore), target="npuir", pass_configs=cfg
            )
        kern(a, b, N)
        torch.npu.synchronize()
        assert torch.equal(b, torch.abs(a)), "wrong"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()
        print(f"{block_N:>8}{ncore:>7}{str(auto):>8}   FAILED: {msg[-1][:70] if msg else exc}")
        continue
    r1 = device_median(lambda: kern(a, b, N))
    r2 = device_median(lambda: kern(a, b, N))
    tiles = -(-N // block_N)
    per_core = -(-tiles // ncore)
    print(f"{block_N:>8}{ncore:>7}{str(auto):>8}{per_core:>12}{r1:>10.2f}{r2:>10.2f}"
          f"{traffic / ((r1 + r2) / 2) / 1e6:>8.3f}", flush=True)
