"""Edge-shape probes for Day 4 §6 Q1/Q2 (fractal / alignment constraints).

Q1 asks for the Cube basic block size and what happens when a shape violates
it; Q2 asks for hard shape/alignment requirements on the L1->L0A/L0B path.
Both are answered empirically here rather than by guessing:

  * tile 100x100  -- neither dimension is a multiple of the 16x16 fractal
  * tile 16x32    -- minimal fractal-sized tiles, many blocks
  * tile 128x128, N=129 -- one-column N tail
  * tile 128x128, K=271  -- 15-element K tail (already covered by pytest)

Each case reports compile status plus the error metrics; a compile error is
reported as such rather than being swallowed.

Run:  python day4/code/own_ops/probe_shapes.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_CODE_DIR = os.path.abspath(os.path.join(_HERE, ".."))  # day4/code, holds own_ops/
for _p in (_CODE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

from own_ops.gemm import check, fmt_stats, get_kernel  # noqa: E402

CASES = [
    # (M, N, K, block_M, block_N, block_K, note)
    (257, 263, 271, 100, 100, 64, "tile 100x100: neither dim multiple of 16"),
    (257, 263, 271, 16, 32, 32, "tile 16x32: minimal fractal tiles"),
    (129, 128, 64, 128, 128, 64, "N=129: single-column tail"),
    (64, 64, 15, 64, 64, 64, "K=15 < one fractal row"),
]


def main():
    fails = 0
    for M, N, K, bm, bn, bk, note in CASES:
        print(f"\n=== {M}x{N}x{K} tile={bm}x{bn}x{bk} -- {note} ===")
        t0 = time.time()
        try:
            kernel = get_kernel("dynamic", block_M=bm, block_N=bn, block_K=bk)
        except Exception as exc:  # compile error is a result, not a crash
            print(f"  COMPILE FAILED after {time.time() - t0:.1f}s: "
                  f"{type(exc).__name__}: {exc}")
            fails += 1
            continue
        print(f"  compiled in {time.time() - t0:.1f}s")
        try:
            stats = check(kernel, M, N, K)
        except Exception as exc:
            print(f"  RUN/CHECK FAILED: {type(exc).__name__}: {exc}")
            fails += 1
            continue
        print(f"  OK  {fmt_stats(stats)}")
    print(f"\n{fails} failing case(s) out of {len(CASES)}")


if __name__ == "__main__":
    main()
