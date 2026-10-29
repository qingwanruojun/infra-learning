"""Standalone vs paired timing: how much does the ABBA pairing itself bias?

Found on Day 4: at 16384x16384x64 (C = 512 MiB) torch measures 416.13 us in its
own session but 571.77 us inside the paired ABBA session, and ours moves
1886.56 -> 2114.43 us.  At 4096^3 the same comparison is stable
(499.63 vs 501.54 us), so the effect is size-dependent, not a protocol bug.

This script measures each implementation ALONE, in its own profiler session,
so the paired numbers in logs/bench_membound_16384.log have a reference.

Run:  python day4/code/own_ops/probe_pair_bias.py
"""

import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_CODE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
LOG_DIR = os.path.abspath(os.path.join(_HERE, "..", "logs"))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _CODE_DIR)
sys.path.append(_HERE)
os.chdir(LOG_DIR)  # profile_once() writes export_only_prof_dir/ relative to cwd

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from own_ops.gemm import get_kernel  # noqa: E402
from probe_membound_torch import profile_once  # noqa: E402

SHAPES = [(4096, 4096, 4096), (16384, 16384, 64)]
ITERS = 5


def median_for(agg):
    name, durs = max(agg.items(), key=lambda kv: len(kv[1]))
    return statistics.median(durs), name


def main():
    for M, N, K in SHAPES:
        a = torch.randn(M, K, dtype=torch.float16).npu()
        b = torch.randn(K, N, dtype=torch.float16).npu()
        c = torch.zeros(M, N, dtype=torch.float16).npu()
        kern = get_kernel("blocked", M=M, N=N, K=K,
                          block_M=128, block_N=128, block_K=64)

        t_us, t_name = median_for(
            profile_once(lambda: torch.matmul(a, b, out=c), ITERS, f"t_{M}"))
        o_us, o_name = median_for(
            profile_once(lambda: kern(a, b, c), ITERS, f"o_{M}"))

        flops = 2.0 * M * N * K
        traffic = (M * K + K * N + M * N) * 2  # no-reuse lower bound
        print(f"\n=== {M}x{N}x{K}  C={M * N * 2 / 2**20:.0f} MiB ===")
        print(f"  torch alone {t_us:9.2f} us  {flops / t_us / 1e6:7.2f} TFLOPs  "
              f"equiv HBM {traffic / t_us / 1e3:7.1f} GB/s")
        print(f"  ours  alone {o_us:9.2f} us  {flops / o_us / 1e6:7.2f} TFLOPs  "
              f"equiv HBM {traffic / o_us / 1e3:7.1f} GB/s")
        print(f"  ratio {o_us / t_us:.3f}x   (paired ABBA numbers are in logs/)")
        del a, b, c
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
