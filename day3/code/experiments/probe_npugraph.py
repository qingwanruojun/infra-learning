"""NPUGraph ceiling test: how much of the ~240 us host cost can a graph remove?

Review point 4: graph capture can eat the Python/ACL submission path, but the
device time (~198-207 us at 128 MiB) is a hard floor, and replay itself has a
fixed host cost.  Measure all three:

  host   : time.perf_counter over 200 enqueues, no sync inside  (direct vs replay)
  device : CANN op_summary median over 20 calls               (direct vs replay)
  e2e    : Event window / N                                    (direct vs replay)

Also checks that replay produces the right output (the kernel's tail handling
depends on a runtime `shape` scalar, which the graph freezes).

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_npugraph.py

Archived log(s): logs/npugraph.log, logs/npugraph_capture.log,
                   logs/npugraph_capture_noqueue.log, logs/npugraph_direct.log
Status: primary evidence -- the negative result: capture aborts, or replay is a
    silent no-op (§4.3 kernel layer 3). CEILING = device time, not 50 us.
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

N, BLOCK, NCORE = 1 << 25, 8192, 40   # 128 MiB, the same config as the ABBA probe
TASKQUEUE = os.getenv("TILELANG_ENABLE_TASKQUEUE", "true").lower() in ("true", "1")
print(f"TILELANG_ENABLE_TASKQUEUE={TASKQUEUE}  capture={os.getenv('TILELANG_PROBE_CAPTURE')}")


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
    prof.export_chrome_trace("/tmp/graph_trace.json")  # triggers CANN parse
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


def e2e_event(fn, iters=20):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


a = torch.randn(N, device="npu", dtype=torch.float32)
b = torch.zeros_like(a)
with ascend_mode("Expert"):
    kern = tilelang.compile(fixed_grid_kernel(N, BLOCK, NCORE), target="npuir")
kern(a, b, N)
torch.npu.synchronize()
assert torch.equal(b, torch.abs(a)), "direct wrong"

rows = []
rows.append(("direct", host_wall(lambda: kern(a, b, N)),
             device_median(lambda: kern(a, b, N)),
             e2e_event(lambda: kern(a, b, N))))

# torch reference for the same numbers
rows.append(("torch.abs", host_wall(lambda: torch.abs(a, out=b)),
             device_median(lambda: torch.abs(a, out=b)),
             e2e_event(lambda: torch.abs(a, out=b))))

if os.getenv("TILELANG_PROBE_CAPTURE", "0") != "1":
    print()
    print(f"{'path':>10}{'host us/call':>15}{'device us':>11}{'e2e us':>9}")
    for name, h, d, e in rows:
        print(f"{name:>10}{h:>15.1f}{d:>11.2f}{e:>9.1f}")
    print("capture skipped (TILELANG_PROBE_CAPTURE != 1); "
          "the launcher syncs the stream unless TILELANG_ENABLE_TASKQUEUE=0")
    sys.exit(0)

g = torch.npu.NPUGraph()
try:
    side = torch.npu.Stream()
    side.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side):
        for _ in range(5):
            kern(a, b, N)
    torch.npu.current_stream().wait_stream(side)
    torch.npu.synchronize()

    with torch.npu.graph(g):
        kern(a, b, N)
    torch.npu.synchronize()

    def replay_check(graph, label):
        ok = True
        for trial in range(2):
            b.zero_()
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            nz = int(torch.count_nonzero(b))
            diff = float((b - torch.abs(a)).abs().max())
            print(f"  {label} replay trial {trial}: nonzero={nz}/{N} max_abs_diff={diff:.3e}")
            ok = ok and nz == N and torch.equal(b, torch.abs(a))
        return ok

    # control first: does a torch op replay correctly in this same process?
    g2 = torch.npu.NPUGraph()
    with torch.npu.graph(g2):
        torch.abs(a, out=b)
    torch.npu.synchronize()
    replay_check(g2, "torch-abs graph")

    if not replay_check(g, "tilelang graph"):
        raise AssertionError("tilelang graph replay did not reproduce the output")

    rows.append(("graph", host_wall(g.replay),
                 device_median(g.replay),
                 e2e_event(g.replay)))
    graph_ok = True
except Exception as exc:  # noqa: BLE001
    graph_ok = False
    print("GRAPH FAILED:", type(exc).__name__, str(exc).splitlines()[0][:200])

print(f"{'path':>10}{'host us/call':>15}{'device us':>11}{'e2e us':>9}")
for name, h, d, e in rows:
    print(f"{name:>10}{h:>15.1f}{d:>11.2f}{e:>9.1f}")
if not graph_ok:
    print("graph capture unavailable; see message above")
