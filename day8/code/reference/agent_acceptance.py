"""Independent acceptance for the Agent-generated RMSNorm kernel.

Run from /mnt/workspace/learn/tilelang-ascend with:
  PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:<agent_dir>:<this_dir> \
  TILELANG_ASCEND_MODE=Developer python agent_acceptance.py

It uses reference/independent_check.py (float64 CPU golden) and
reference/bench_common.py (host / e2e / device clocks). It never imports the
generated test file's golden.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch_npu  # noqa: F401

AGENT_DIR = "/mnt/workspace/learn/infra-learning/day8/code/agent_output/custom/rmsnorm"
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_common import device_bench, e2e_bench, host_enqueue_us  # noqa: E402
from independent_check import compare, negative_control, torch_ref_fp64  # noqa: E402
from rmsnorm import rmsnorm  # noqa: E402


CASES = [
    dict(M=1024, N=1024, dtype="float32", atol=1e-4, rtol=1e-4, seed=0),
    dict(M=1023, N=1024, dtype="float32", atol=1e-4, rtol=1e-4, seed=1),
    dict(M=1024, N=1024, dtype="float16", atol=1e-2, rtol=1e-2, seed=2),
]


def make_inputs(M, N, dtype, seed):
    torch.manual_seed(seed)
    x_cpu = torch.randn(M, N, dtype=torch.float32).to(getattr(torch, dtype))
    w_cpu = torch.randn(N, dtype=torch.float32).to(getattr(torch, dtype))
    return x_cpu.npu(), w_cpu.npu()


def run_case(case, do_bench=True):
    M, N, dtype = case["M"], case["N"], case["dtype"]
    kernel = rmsnorm(M, N, dtype=dtype, eps=1e-5)
    x, w = make_inputs(M, N, dtype, case["seed"])
    y = kernel(x, w)
    torch.npu.synchronize()

    ref = torch_ref_fp64(x, w, eps=1e-5)
    metrics = compare(y, ref, case["atol"], case["rtol"])
    neg = negative_control(y, ref, case["atol"], case["rtol"])

    out = {
        "shape": [M, N],
        "dtype": dtype,
        "eps": 1e-5,
        "metrics": metrics,
        "negative_control": {k: neg[k] for k in ("pass", "n_mismatch", "max_abs")},
    }
    if do_bench:
        fn = lambda: kernel(x, w)  # noqa: E731
        out["host"] = host_enqueue_us(fn)
        out["e2e"] = e2e_bench(fn)
        out["device"] = device_bench(fn, "main", iters=20)
    return out


def all_ok(results):
    """Exit gate: every case passes AND every negative control is caught."""
    return all(r["metrics"]["pass"] for r in results) and all(
        not r["negative_control"]["pass"] for r in results
    )


def exit_code(results):
    """0 only when all_ok holds, 1 otherwise."""
    return 0 if all_ok(results) else 1


def main():
    torch.npu.set_device(0)
    results = []
    for case in CASES:
        print(f"running {case}", flush=True)
        results.append(run_case(case))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_acceptance.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        m = r["metrics"]
        n = r["negative_control"]
        print(
            f"{r['dtype']:7s} M={r['shape'][0]:4d} pass={m['pass']} "
            f"max_abs={m.get('max_abs'):.3e} max_rel={m.get('max_rel'):.3e} "
            f"mismatch={m.get('n_mismatch')} "
            f"neg_control_pass={n['pass']} neg_mismatch={n['n_mismatch']} "
            f"device={r.get('device', {}).get('median_us')} us "
            f"e2e={r.get('e2e', {}).get('median_us')} us"
        )
    ok = all_ok(results)
    n_pass = sum(1 for r in results if r["metrics"]["pass"])
    n_neg_caught = sum(1 for r in results if not r["negative_control"]["pass"])
    print(
        f"ACCEPTANCE_RESULT: {'PASS' if ok else 'FAIL'} "
        f"({n_pass}/{len(results)} cases pass, "
        f"{n_neg_caught}/{len(results)} negative controls caught)"
    )
    print("WROTE", path)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
