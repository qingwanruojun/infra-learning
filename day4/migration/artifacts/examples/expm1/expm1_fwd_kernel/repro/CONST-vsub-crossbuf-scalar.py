# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""repro: vsub cross-buffer scalar-literal form (dst = load buffer).

Entry target: pattern-library (proposed CONST/TRAP-adjacent D entry) --
``T.vsub(w32, 1.0, x_ub)``: src0 = fp32 UB buffer, src1 = Python float
literal 1.0 (fp32-exact), dst = a DIFFERENT UB buffer (the load buffer).
This three-buffer form (T.vsub.md section 2.4 example 1) is the fp32
route's "result lands in the load buffer" write-back of the expm1
kernel (DESIGN.md section 1.5/9.2).  DESIGN.md section 9.1#8 flagged the
vsub+scalar+cross-buffer combination as having no verbatim production
precedent on the current toolchain (fallback: one32 buffer via vbrc);
this repro proves the combination compiles, runs, and produces the
correct (non-identity) output on Ascend 910B3 / tilelang
0.1.2+ef4078a697 / CANN 9.0.0, removing the fallback need.

Assertions: (1) the kernel compiles and runs; (2) the output is NOT an
identity of the input (a wrong in-place ``vsub(w32, 1.0, w32)`` plus a
store from x_ub would yield the input unchanged -- the恒等/identity
mapping fingerprint, DESIGN.md section 9.2); (3) the output matches
``torch.expm1`` within the fp32 tolerance envelope.

First validated: 2026-10-29, expm1 Stage 3 (origin_task: expm1_fwd_kernel).
Re-verification records appended below.
"""

import os

# Developer mode (same module-level contract as the expm1 kernel).
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

_N = 4096


@tilelang.jit(out_idx=[1], target="npuir")
def _crossbuf_vsub_kernel(bs):
    """Minimal expm1-shaped kernel: persistent 1-D split + copy -> vexp
    -> cross-buffer vsub (dst = load buffer x_ub) -> copy."""

    @T.prim_func
    def main(x: T.Tensor((_N,), "float32"), y: T.Tensor((_N,), "float32")):
        with T.Kernel(T.ceildiv(_N, bs), is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((bs,), "float32")
            w32 = T.alloc_ub((bs,), "float32")
            t0 = cid * bs
            tail = T.min(bs, _N - t0)
            T.copy(x[t0 : t0 + tail], x_ub[0:tail])
            T.vexp(x_ub, w32)
            # Cross-buffer: dst = load buffer x_ub != src0 = w32.
            T.vsub(w32, 1.0, x_ub)
            T.copy(x_ub[0:tail], y[t0 : t0 + tail])

    return main


def main():
    bs = 1024
    x = torch.randn(_N, dtype=torch.float32).npu()
    out = _crossbuf_vsub_kernel(bs)(x).cpu()
    x_cpu = x.cpu()
    ref = torch.expm1(x_cpu)

    # (2) non-identity: the load buffer must have been updated by the
    # cross-buffer vsub (an in-place mistake would leave it untouched).
    assert not torch.equal(out, x_cpu), "output equals input: cross-buffer vsub did not write x_ub"

    # (3) numeric correctness against the golden (fp32 envelope).
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    max_diff = (out - ref).abs().max().item()
    print(f"PASS: cross-buffer vsub(w32, 1.0, x_ub) compiles/runs, max_diff={max_diff:.3e} (fp32 1e-5 envelope)")


if __name__ == "__main__":
    main()
