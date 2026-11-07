"""Day 8 baseline: measure the unchanged Day 3 hand-written RMSNorm kernel.

This script does not modify Day 3 code. It imports `own_ops.rmsnorm`, compiles
the same shapes that Day 8 tests, and records accuracy plus device kernel
latency using Day 3's own `benchmark()` (NPU events, per-call microseconds).

Run from /mnt/workspace/learn/tilelang-ascend:
    python /mnt/workspace/learn/infra-learning/day8/code/reference/day3_baseline.py
"""

import json
import os
import sys

import torch
import torch_npu  # noqa: F401

REPO = "/mnt/workspace/learn/tilelang-ascend"
sys.path.insert(0, REPO)

from own_ops.rmsnorm import (  # noqa: E402
    benchmark,
    compile_rmsnorm,
    make_tensors,
    torch_ref,
)


def one_case(M, N, dtype, use_weight=True, rstd_mode="sqrt_rec", block_M=8):
    kernel = compile_rmsnorm(
        M, N, block_M, dtype=dtype, use_weight=use_weight, rstd_mode=rstd_mode
    )
    x, w, y = make_tensors(M, N, dtype=dtype, use_weight=use_weight)
    kernel(x, w, y)
    torch.npu.synchronize()

    ref = torch_ref(x, w if use_weight else None)
    y_cpu = y.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    diff = (y_cpu - ref_cpu).abs()
    denom = ref_cpu.abs().clamp_min(1e-12)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    denom_mask = ref_cpu.abs() > 1e-6
    max_rel_masked = (
        (diff[denom_mask] / ref_cpu[denom_mask].abs()).max().item()
        if denom_mask.any()
        else 0.0
    )
    st = benchmark(kernel, x, w, y)
    return {
        "M": M,
        "N": N,
        "dtype": dtype,
        "use_weight": use_weight,
        "rstd_mode": rstd_mode,
        "block_M": block_M,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_rel_abs_gt_1e-6": max_rel_masked,
        "latency_us": st,
    }


def main():
    torch.npu.set_device(0)
    results = []
    for spec in [
        dict(M=1024, N=1024, dtype="float32", rstd_mode="sqrt_rec"),
        dict(M=1023, N=1024, dtype="float32", rstd_mode="sqrt_rec"),
        dict(M=1024, N=1024, dtype="float16", rstd_mode="sqrt_rec"),
    ]:
        print(f"running {spec}", flush=True)
        results.append(one_case(**spec))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day3_baseline.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        print(
            f"{r['dtype']:7s} M={r['M']:4d} median={r['latency_us']['median_us']:.2f} us "
            f"max_abs={r['max_abs']:.3e} max_rel={r['max_rel']:.3e}"
        )
    print("WROTE", out)


if __name__ == "__main__":
    main()
