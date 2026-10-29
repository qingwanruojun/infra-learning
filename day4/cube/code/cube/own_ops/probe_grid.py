"""Minimal repro: a 2D T.Kernel grid is silently serialized on the NPU backend.

``T.Kernel(4, 2)`` compiles and produces correct results, but the second grid
dimension does not become extra parallel blocks: ``blockIdx.y`` maps to
``hivm.hir.get_sub_block_idx`` (the AIV sub-block id) and every block runs the
``by`` loop as an ``scf.for``.  Consequence: only ``gx`` blocks are launched.

Evidence for Day 4 section 3 (tier B).  Run:
    python day4/code/own_ops/probe_grid.py
"""

import contextlib
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOG_DIR = os.path.abspath(os.path.join(_HERE, "..", "logs"))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


@contextlib.contextmanager
def ascend_mode(mode="Expert"):
    prev = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = prev


@T.prim_func
def main(A: T.Tensor((64,), "float16"), B: T.Tensor((64,), "float16")):
    with T.Kernel(4, 2, is_npu=True) as (bx, by, _):
        with T.Scope("Cube"):
            a = T.alloc_L1((16, 16), "float16")
            T.load_nd2nz(A[(bx + by) * 8], a, [16, 16])
            T.store_fixpipe(a, B[(bx + by) * 8], size=[16, 16])


def run():
    with ascend_mode("Expert"):
        kk = tilelang.compile(main, target="npuir")
    irt = kk.mlir_content
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "two_d_grid_serialized.mlir")
    with open(path, "w") as fh:
        fh.write(irt)
    print(f"wrote {path}")
    print("lines mentioning block idx / serial loops:")
    for i, line in enumerate(irt.splitlines(), 1):
        if re.search(r"block_idx|sub_block|scf\.for|get_block", line):
            print(f"  {i:4d}  {line.strip()[:140]}")
    n_for = len(re.findall(r"scf\.for", irt))
    n_idx = len(re.findall(r"get_block_idx", irt))
    print(f"\nget_block_idx occurrences: {n_idx} (1D grid would also be {n_idx})")
    print(f"scf.for occurrences: {n_for}  <- the by-dimension became this loop")


if __name__ == "__main__":
    run()
