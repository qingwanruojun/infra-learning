"""Three-clock measurement of the unchanged Day 3 hand-written RMSNorm.

Run from /mnt/workspace/learn/tilelang-ascend:
  PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:<day8>/code/reference \
  python /mnt/workspace/learn/infra-learning/day8/code/reference/day3_three_clocks.py

It writes day3_three_clocks.json next to this file. The Day 3 source is only
imported, never modified.
"""

import json
import os
import sys

import torch
import torch_npu  # noqa: F401

REPO = "/mnt/workspace/learn/tilelang-ascend"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from bench_common import device_bench, e2e_bench, host_enqueue_us  # noqa: E402
from own_ops.rmsnorm import compile_rmsnorm, make_tensors  # noqa: E402

CASES = [
    (1024, 1024, "float32"),
    (1023, 1024, "float32"),
    (1024, 1024, "float16"),
]


def main():
    torch.npu.set_device(0)
    out = {}
    for M, N, dtype in CASES:
        kernel = compile_rmsnorm(M, N, 8, dtype=dtype, rstd_mode="sqrt_rec")
        x, w, y = make_tensors(M, N, dtype=dtype, use_weight=True)
        fn = lambda: kernel(x, w, y)  # noqa: E731
        key = f"{dtype}_{M}"
        out[key] = {
            "shape": [M, N],
            "dtype": dtype,
            "host": host_enqueue_us(fn),
            "e2e": e2e_bench(fn),
            "device": device_bench(fn, "main", iters=20),
        }
        print(key, out[key]["host"]["median_us"], out[key]["e2e"]["median_us"],
              out[key]["device"], flush=True)

    path = os.path.join(HERE, "day3_three_clocks.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
