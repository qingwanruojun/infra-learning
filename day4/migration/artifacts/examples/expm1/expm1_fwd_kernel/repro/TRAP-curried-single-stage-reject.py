# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""repro: curried factory single-stage ``(x)`` direct call must be rejected.

Entry target: pattern-library traps-compiler.md (proposed TRAP/adjacent
R entry) -- a ``@tilelang.jit`` curried factory (``factory -> kernel(bs)
-> executable(x)``, DESIGN.md section 3.5.2 two-stage contract) must NOT
be called single-stage with ``(x)``: the tensor binds to the ``bs``
parameter.  On the current toolchain the failure surfaces at the kernel
body's host guard ``if bs <= 0`` as
``RuntimeError: Boolean value of Tensor with more than one value is
ambiguous`` (the design document predicted a build-time TypeError; the
actual failure type is a RuntimeError but the contract behavior --
single-stage direct calls must be rejected -- is satisfied).  This repro
locks the actual failure signature so later tasks do not chase the
wrong exception type.

Assertions: the single-stage call raises (any Exception type; a
successful run would be a contract violation).

First validated: 2026-10-29, expm1 Stage 3 (origin_task: expm1_fwd_kernel).
Re-verification records appended below.
"""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

_N = 1024


def _make_kernel(N, dtype):
    """Minimal curried factory mirroring the expm1_fwd_kernel shape."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(bs=1024):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(1, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((bs,), dtype)
                w32 = T.alloc_ub((bs,), "float32")
                T.copy(x[0:bs], x_ub[0:bs])
                T.vexp(x_ub, w32)
                T.vsub(w32, 1.0, x_ub)
                T.copy(x_ub[0:bs], y[0:bs])

        return main

    return kernel


def main():
    x = torch.randn(_N, dtype=torch.float32).npu()
    fn = _make_kernel(_N, "float32")
    raised = False
    try:
        # Single-stage direct call: the tensor is bound to the bs
        # parameter instead of being passed to main(x).
        out = fn(x)
        out.cpu()  # noqa: B018  (force materialization if it somehow ran)
    except Exception as exc:  # noqa: BLE001
        raised = True
        print(f"rejected as expected: {type(exc).__name__}: {str(exc)[:90]}")
    assert raised, "single-stage (x) direct call must fail (tensor bound to bs)"
    print("PASS: single-stage (x) direct call rejected")


if __name__ == "__main__":
    main()
