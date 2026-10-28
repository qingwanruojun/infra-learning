"""Dump NPUIR lines that expose the compiler-chosen vector width for abs.

The DSL has no explicit vector-width knob; this probe is the IR evidence for
the block-start alignment experiment (block_N=1023/1024/1025).  Run from the
repository root::

    python notes/day2/experiments/probe_vector_width.py

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_vector_width.py

Archived log(s): logs/vector_width_ir.log
Status: evidence -- IR side-note for the block-start alignment experiment.
"""

import contextlib
import os

import torch  # noqa: F401
import torch_npu  # noqa: F401

import tilelang

from own_ops.abs import abs_kernel


@contextlib.contextmanager
def expert_mode():
    prev = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = "Expert"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = prev


def dump(block_N):
    with expert_mode():
        k = tilelang.compile(abs_kernel(4096, block_N), target="npuir")
    print(f"===== block_N={block_N} =====")
    for line in k.mlir_content.splitlines():
        s = line.strip()
        if any(
            tok in s
            for tok in (
                "vabs",
                "vector<",
                "memref<",
                "hivm.hir.load",
                "hivm.hir.store",
            )
        ):
            print(s)


for block_N in (1023, 1024, 1025):
    tilelang.cache.clear_cache()
    dump(block_N)
