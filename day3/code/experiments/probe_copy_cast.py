"""Minimal repro for the two dtype traps documented in the Day-3 notes (§2.3).

Three cases, one compile each:

  float16 -> float32   : ``T.copy`` across dtypes -> Check failed:
                         npuirop.src->dtype == npuirop.dst->dtype
  float32 -> float16   : same failure (T.copy never casts)
  float32 -> float32   : compiles and copies correctly (the only legal T.copy)

The script deliberately has no ``__main__`` guard: it is a one-shot probe that
runs at import time, exactly as it was run when the error was captured.

Run (from ``day3/code/`` on the NPU machine)::

    python experiments/probe_copy_cast.py

Archived log: none -- the output is a compiler error message quoted in §2.3.
Status: evidence for the report; not part of the measurement chain.
"""

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


def kern(M, N, idtype, odtype):
    @T.prim_func
    def main(A: T.Tensor((M, N), idtype), B: T.Tensor((M, N), odtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((M, N), idtype)
            b = T.alloc_ub((M, N), odtype)
            T.copy(A, a)
            T.copy(a, b)
            T.copy(b, B)

    return main


for idt, odt in [("float16", "float32"), ("float32", "float16"), ("float32", "float32")]:
    tilelang.cache.clear_cache()
    k = tilelang.compile(kern(64, 64, idt, odt), target="npuir")
    a = torch.randn(64, 64, dtype=getattr(torch, idt), device="npu")
    b = torch.zeros(64, 64, dtype=getattr(torch, odt), device="npu")
    k(a, b)
    d = (b.cpu().float() - a.cpu().float()).abs().max().item()
    print(idt, "->", odt, "max diff", d)
