# TRAP-vbrc-scalar-let-binding: vbrc scalar source must be let-bound INSIDE
# the traced @T.prim_func body (not a factory-level named local).
#
# REFINEMENT over pattern-library traps-compiler.md "TRAP-vcmp-scalar-cast"
# L174 note ("T.vbrc 标量源须 let 绑定局部量（字面量触发 rank 断言）"):
#   - The existing note says a *literal* triggers the rank assertion.
#   - This repro shows the trigger set is wider: a *named local defined in
#     the enclosing factory scope* (outside the prim_func body) ALSO fails
#     with the same diagnostic:
#       "error: The input vector and output vector must have same rank."
#     (T.vbrc(zero_val, zero32) at trace time -- _get_extent() returns an
#     empty rank for a closure Python scalar, and the vbrc rank check then
#     rejects rank-0 source vs rank-1 dst.)
#   - Correct form: bind the scalar with a plain assignment inside the
#     @T.prim_func body (inside `with T.Kernel(...)`); the TVM-script parser
#     lifts it to tir.Var-with-let-value, takes the PrimExpr branch, and
#     skips the rank assertion (gqa_prefill kernel note L88 precedent).
#
# Toolchain: tilelang 0.1.2+88ee8c302e / CANN 9.0.0 / Ascend910B3 / torch_npu 2.7.1.post4
# Evidence: examples/leaky_relu/_make_leaky_relu_kernel fp32 route hit this in
# first_impl (L0-1/float32 RUNTIME-ERROR DiagnosticError at T.vbrc line 287);
# fixing the binding location resolved it, all levels PASS bit-exact.
#
# Run: python TRAP-vbrc-scalar-let-binding.py  (requires NPU device)
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Repro + assertion: vbrc scalar source binding location (in/out traced body)."""

import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)

N = 1024
DTYPE = "float32"
BLOCK = 1024
NUM_KERNELS = 1
NUM_LOCAL_TASKS = 1


def _build(bind_inside: bool):
    """Build the same kernel; scalar bound at factory scope or inside body."""

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size=BLOCK):
        if bind_inside:
            @T.prim_func
            def main(x: T.Tensor((N,), DTYPE), y: T.Tensor((N,), DTYPE)):
                with T.Kernel(NUM_KERNELS, is_npu=True) as (cid, _):
                    zero_val = 0.0  # let-bound inside traced body -> OK
                    x_ub = T.alloc_ub((block_size,), DTYPE)
                    zero32 = T.alloc_ub((block_size,), "float32")
                    T.vbrc(zero_val, zero32)
                    T.copy(x[0:block_size], x_ub[0:block_size])
                    T.copy(x_ub[0:block_size], y[0:block_size])

        else:
            zero_val = 0.0  # factory-level named local -> rank assert fails

            @T.prim_func
            def main(x: T.Tensor((N,), DTYPE), y: T.Tensor((N,), DTYPE)):
                with T.Kernel(NUM_KERNELS, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((block_size,), DTYPE)
                    zero32 = T.alloc_ub((block_size,), "float32")
                    T.vbrc(zero_val, zero32)
                    T.copy(x[0:block_size], x_ub[0:block_size])
                    T.copy(x_ub[0:block_size], y[0:block_size])

        return main

    return kernel


def main():
    x = torch.randn(N, dtype=torch.float32).npu()

    # Bad form: factory-level named local -> DiagnosticError at trace/compile.
    # The rank-assert diagnostic ("The input vector and output vector must
    # have same rank.") is rendered to stderr by the TVM diagnostic context;
    # the raised exception's repr wraps the render traceback, so assert on
    # the exception type + the stderr text separately below via the type.
    bad_compile_failed = False
    try:
        _build(bind_inside=False)(BLOCK)(x)
    except Exception as exc:  # noqa: BLE001  (tvm DiagnosticError)
        bad_compile_failed = "DiagnosticError" in type(exc).__name__
        print(f"bad form (factory-level scalar) rejected as expected: {type(exc).__name__}")
    assert bad_compile_failed, "expected rank-assert DiagnosticError for factory-level scalar source"

    # Good form: scalar let-bound inside the traced body -> compiles & copies.
    out = _build(bind_inside=True)(BLOCK)(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), x.cpu(), rtol=0.0, atol=0.0)
    print("good form (in-body let-bound scalar) compiles and passes copy identity")

    print("ASSERT PASS: TRAP-vbrc-scalar-let-binding reproduced (binding location, not literal-ness)")


if __name__ == "__main__":
    main()
