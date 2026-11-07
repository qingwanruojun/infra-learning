# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Amin tiled row-reduction kernel (NPU, Developer mode) -- Stage 4 round-3b.

Round-3 branch r3b_unroll2 (single optimization point: straight-line
two-tile interleave for num_full==2 workloads). When the factory-static
tile split has exactly two full tiles and no tail, the per-task body is
generated as straight-line code over TWO staging buffers (x_ub_a/x_ub_b)
and ONE compute fragment: copy(tile0->a) and copy(tile1->b) are issued
back-to-back, then cast/reduce consume a and b in order. copy(tile1)
has no data dependence on cast/reduce(tile0), so MTE2 and the vector
pipe can overlap -- without the smaller-burst / more-instructions cost
that made the tn-downshift form (r3a_pipe) slower. Straight-line code
also leaves no loop for auto-multi-buffer to inflate. All other tile
splits keep the baseline serial-tile loop unchanged.


Migrated from the GPU TileOPs ``_simple_reduce_kernel_tiled`` (amin branch,
``examples/TileOPs/tileops/kernels/reduction/amin_fwd/_amin_fwd_kernels.py``
L131-223) to ``target="npuir"``.

Semantics (DESIGN.md section 0.1, frozen):
    out[m] = min_{n in [0, N)} x[m, n]
    Input ``(M, N)``; output ``(M,)``; dtype in {float16, bfloat16, float32}.
    This kernel covers the N-tile streaming path (N_padded > 32512). Smaller N
    is dispatched by the wrapper to the single-tile sibling function.

NPU redesign (DESIGN.md sections 0.6 / 1.4, frozen):
    R-D1 persistent one-dimensional core split:
        num_kernels = min(ceil(M / block_m), vector_cores)
        num_tasks   = ceil(num_logical / num_kernels)   (factory static bound)
        block_id = s * num_kernels + cid; dynamic guard block_id < num_logical.
    R-D2 exact-width load + static tail tile:
        num_full = N // tn_eff full tiles (exact column slices) and one static
        tail tile of width ``N - num_full * tn_eff`` (trace-time branch).
        No N padding, no ite mask, no kernel-side serial-var condition.
    R-D3 direct ``T.reduce_min`` in the natural dtype domain:
        fp16 -> fp16 domain; fp32 -> fp32 domain (same-dtype copy, no vcast);
        bf16  -> ``T.vcast(..., round_mode="rint")`` into an fp32 fragment
        (``T.reduce.md`` section 2.2.1: bf16 reduce is unsupported).
        Cross-tile merge uses a scalar read-modify-write ``T.min`` (never an
        in-place ``T.vselect``: TRAP-vselect-inplace-carried-state).
    R-D4 O(tile) UB-resident buffer set: shared staging -> fragment compute
        -> (block_m, 1) running state -> shared store.
    R-D5 factory-period ``tn_eff`` budget adaptation (``_fit_tile_n``): the
        full-tile set and the static tail set coexist, so the wrapper tile_n
        upper bound is searched downward in 256-column steps until
        ``((2b + w) * tn + (b + w) * max(tail, 1)) * 1.12 <= 172KB``.

Known toolchain constraints enforced here (DESIGN.md section 9.2):
    - reduce src/dst dtype MUST match (mixed dtype silently corrupts,
      VP-2026-0119).
    - bf16 reduce is unsupported -> ``vcast(rint)`` to fp32 first.
    - ``T.vcast`` f32->f32 is non-identity (CONST-vcast-f32-to-f32-corrupt)
      -> fp32 uses a same-dtype ``T.copy``.
    - ``T.copy`` src/dst use paired explicit slices.
    - ``T.Kernel`` is 1-D only; ``T.serial`` bounds are factory-static; the
      dynamic out-of-range check is an ``if`` guard only.
    - multi-tile + block_m > 1 triggers a compiler SIGSEGV (CG-2026-0014)
      -> block_m is clamped to 1 at factory time.
    - tail buffers are allocated unconditionally and the processing body is
      guarded by a trace-time ``if tail > 0`` (TVM tracer does not track
      allocations inside conditional blocks).
"""

import os

# Developer mode: compiler manages UB (alloc_shared) + fragment + auto sync.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import argparse

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

# Programming mode actually validated by this Stage 3 implementation.
# Must stay consistent with the TILELANG_ASCEND_MODE env above.
ASCEND_MODE = "Developer"

_SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")

# Tolerances from examples/TileOPs/tests/ops/test_amin_fwd.py::_tol
# (fp32 1e-4/1e-4; fp16/bf16 1e-2/1e-2). Results are expected bit-exact
# (DESIGN.md section 8.2); the tolerance is only the gate.
_DTYPE_MAP = {
    "float16": (torch.float16, 1e-2, 1e-2),
    "float32": (torch.float32, 1e-4, 1e-4),
    "bfloat16": (torch.bfloat16, 1e-2, 1e-2),
}

# R-D5 budget predicate constants (DESIGN.md sections 0.6 / 4.5).
# 172KB = 176128B = 192KB UB minus a 20KB safety margin; 1.12 is the upper
# bound of the BishengIR demand inflation (CONST-capacity-910B2C).
_UB_WORK_BUDGET_BYTES = 176128
_UB_INFLATION = 1.12
_TILE_N_STEP = 256

# Input staging element bytes.
_INPUT_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
# Compute-domain element bytes: fp16 keeps the native domain (2B); bf16 is
# upcast to fp32 (4B); fp32 is native (4B).
_COMPUTE_DTYPE_BYTES = {"float16": 2, "bfloat16": 4, "float32": 4}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_amin(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.amin`` along the last dim (fp32 opmath).

    CPU only. Ported from the source-repo test baseline
    (``test_amin_fwd.py`` ``ref_program``: ``x.float().amin(dim=-1).to(x.dtype)``);
    does NOT reproduce the NPU algorithm (no tile loop / reduce_min chain /
    persistent split), guaranteeing validation independence (DESIGN.md 8.1).
    """
    return torch.amin(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel ----------
def _ceildiv(a, b):
    return (a + b - 1) // b


def _fit_tile_n(N, dtype, tile_n):
    """R-D5 factory-period tile budget adaptation (DESIGN.md sections 0.6/4.5).

    The full-tile buffer set ``(block_m, tn)`` and the static tail set
    ``(block_m, tail)`` coexist in UB, while the wrapper budget heuristic only
    accounts for a single set. Search ``tile_n`` downward in 256-column steps
    and return the first width satisfying the double-buffer budget predicate::

        ((2b + w) * tn + (b + w) * max(tail, 1)) * 1.12 <= 172KB

    where ``b`` is the input element size, ``w`` the compute-domain element
    size, ``num_full = N // tn`` and ``tail = N - num_full * tn``.
    ``tn_eff <= tile_n`` always holds (the wrapper value is an upper bound).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    b = _INPUT_DTYPE_BYTES[dtype]
    w = _COMPUTE_DTYPE_BYTES[dtype]
    tn = tile_n
    while tn > 0:
        num_full = N // tn
        tail = N - num_full * tn
        lhs = (2 * b + w) * tn + (b + w) * max(tail, 1)
        if lhs * _UB_INFLATION <= _UB_WORK_BUDGET_BYTES:
            return tn
        tn -= _TILE_N_STEP
    raise ValueError(
        f"no tn_eff satisfies the UB budget for N={N} dtype={dtype!r} "
        f"(searched down from tile_n={tile_n} in steps of {_TILE_N_STEP})"
    )


def _simple_reduce_kernel_tiled(M, N, op_kind, dtype, tile_n):
    """Build the amin tiled NPU kernel factory.

    Mirrors the GPU source structure ``_simple_reduce_kernel_tiled(M, N,
    op_kind, dtype, tile_n) -> _func(block_m) -> main`` (name / signature and
    prim_func parameter names+order unchanged). K3/K9: the ``threads`` backend
    parameter is removed (NPU has no CUDA threads concept); backend config is
    the single ``block_m`` argument plus the factory-period vector-core query
    (DESIGN.md section 5.5).
    """
    if op_kind != "amin":
        raise ValueError(
            f"unsupported op_kind {op_kind!r}; this kernel serves the 'amin' "
            f"dispatch path only (wrapper hard-checks op_kind, amin_fwd.py)"
        )
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_SUPPORTED_DTYPES)}"
        )
    if M <= 0:
        raise ValueError(f"M must be positive, got M={M}")
    if N <= 0:
        raise ValueError(f"reduction dim N must be positive, got N={N}")
    if tile_n <= 0:
        raise ValueError(f"tile_n must be positive, got tile_n={tile_n}")

    # Factory-period physical core count (never a hard-coded assumption;
    # 910B3 verified 20 AI cores / 40 vector cores on 2026-11-05).
    vector_cores = NPUUtils.get().get_aivector_core_num()
    is_bf16 = dtype == "bfloat16"

    # R-D5 budget adaptation: tn_eff <= tile_n, factory-static tile split.
    tn_eff = _fit_tile_n(N, dtype, tile_n)
    num_full = N // tn_eff
    tail = N - num_full * tn_eff
    # Tail buffers are always allocated; tail == 0 keeps a (block_m, 1)
    # placeholder that the trace-time ``if tail > 0`` never touches.
    tail_w = tail if tail > 0 else 1

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        # CG-2026-0014: multi-tile + block_m > 1 triggers a compiler SIGSEGV,
        # so the tiled path clamps block_m to 1. The wrapper default_config
        # already derives block_m=1 for the long-seq workload, and a caller
        # block_m > 1 is safely absorbed (tile_n is its budget upper bound).
        block_m = 1

        # Persistent split (DESIGN.md section 5.5, element 3).
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_tasks = _ceildiv(num_logical, num_kernels)

        # r3b_unroll2: factory-period shape split. Straight-line two-tile
        # interleave applies only when the tile split has exactly 2 full
        # tiles and no tail (the long-seq workload form).
        unroll2 = num_full == 2 and tail == 0

        if is_bf16:
            # bf16 domain: shared staging -> vcast(rint) fp32 fragment ->
            # fp32-domain reduce (bf16 reduce unsupported, T.reduce.md 2.2.1).
            if unroll2:
                @T.prim_func
                def main(
                    x: T.Tensor((M, N), dtype),
                    out: T.Tensor((M,), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        x_ub_a = T.alloc_shared((block_m, tn_eff), dtype)
                        x_ub_b = T.alloc_shared((block_m, tn_eff), dtype)
                        x_work = T.alloc_fragment((block_m, tn_eff), "float32")
                        tile_min = T.alloc_fragment((block_m, 1), "float32")
                        acc = T.alloc_fragment((block_m, 1), "float32")
                        out_ub = T.alloc_shared((block_m,), dtype)
                        value_inf = T.infinity("float32")

                        for s in T.serial(num_tasks):
                            block_id = s * num_kernels + cid
                            if block_id < num_logical:
                                off_m = block_id * block_m
                                real_m = T.min(block_m, M - off_m)
                                T.vbrc(value_inf, acc)
                                # Both tile loads issued back-to-back: the
                                # second load does not depend on the vector
                                # chain consuming the first.
                                T.copy(
                                    x[off_m : off_m + real_m, 0:tn_eff],
                                    x_ub_a[0:real_m, 0:tn_eff],
                                )
                                T.copy(
                                    x[off_m : off_m + real_m, tn_eff : N],
                                    x_ub_b[0:real_m, 0:tn_eff],
                                )
                                T.vcast(x_ub_a, x_work, round_mode="rint")
                                T.reduce_min(x_work, tile_min, dim=1)
                                for i in T.Parallel(block_m):
                                    acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                T.vcast(x_ub_b, x_work, round_mode="rint")
                                T.reduce_min(x_work, tile_min, dim=1)
                                for i in T.Parallel(block_m):
                                    acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                for i in T.Parallel(block_m):
                                    out_ub[i] = T.cast(acc[i, 0], dtype)
                                T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

            else:
                @T.prim_func
                def main(
                        x: T.Tensor((M, N), dtype),
                        out: T.Tensor((M,), dtype),
                    ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        x_ub = T.alloc_shared((block_m, tn_eff), dtype)
                        x_work = T.alloc_fragment((block_m, tn_eff), "float32")
                        tile_min = T.alloc_fragment((block_m, 1), "float32")
                        acc = T.alloc_fragment((block_m, 1), "float32")
                        out_ub = T.alloc_shared((block_m,), dtype)
                        x_tail = T.alloc_shared((block_m, tail_w), dtype)
                        xw_tail = T.alloc_fragment((block_m, tail_w), "float32")
                        value_inf = T.infinity("float32")

                        for s in T.serial(num_tasks):
                            block_id = s * num_kernels + cid
                            if block_id < num_logical:
                                off_m = block_id * block_m
                                real_m = T.min(block_m, M - off_m)
                                # Running-min state re-initialized per task (loop
                                # carried state must not leak across tasks).
                                T.vbrc(value_inf, acc)
                                for t in T.serial(num_full):
                                    T.copy(
                                        x[
                                            off_m : off_m + real_m,
                                            t * tn_eff : (t + 1) * tn_eff,
                                        ],
                                        x_ub[0:real_m, 0:tn_eff],
                                    )
                                    T.vcast(x_ub, x_work, round_mode="rint")
                                    T.reduce_min(x_work, tile_min, dim=1)
                                    for i in T.Parallel(block_m):
                                        acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                # Static tail tile (trace-time branch, exact width).
                                if tail > 0:
                                    T.copy(
                                        x[off_m : off_m + real_m, num_full * tn_eff : N],
                                        x_tail[0:real_m, 0:tail],
                                    )
                                    T.vcast(x_tail, xw_tail, round_mode="rint")
                                    T.reduce_min(xw_tail, tile_min, dim=1)
                                    for i in T.Parallel(block_m):
                                        acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                for i in T.Parallel(block_m):
                                    out_ub[i] = T.cast(acc[i, 0], dtype)
                                T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:
            # fp16/fp32 domain: shared staging -> same-dtype fragment copy
            # (fp32 must NOT use T.vcast f32->f32) -> natural-domain reduce.
            if unroll2:
                @T.prim_func
                def main(
                    x: T.Tensor((M, N), dtype),
                    out: T.Tensor((M,), dtype),
                ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        x_ub_a = T.alloc_shared((block_m, tn_eff), dtype)
                        x_ub_b = T.alloc_shared((block_m, tn_eff), dtype)
                        x_frag = T.alloc_fragment((block_m, tn_eff), dtype)
                        tile_min = T.alloc_fragment((block_m, 1), dtype)
                        acc = T.alloc_fragment((block_m, 1), dtype)
                        out_ub = T.alloc_shared((block_m,), dtype)
                        value_inf = T.infinity(dtype)

                        for s in T.serial(num_tasks):
                            block_id = s * num_kernels + cid
                            if block_id < num_logical:
                                off_m = block_id * block_m
                                real_m = T.min(block_m, M - off_m)
                                T.vbrc(value_inf, acc)
                                T.copy(
                                    x[off_m : off_m + real_m, 0:tn_eff],
                                    x_ub_a[0:real_m, 0:tn_eff],
                                )
                                T.copy(
                                    x[off_m : off_m + real_m, tn_eff : N],
                                    x_ub_b[0:real_m, 0:tn_eff],
                                )
                                T.copy(x_ub_a, x_frag)
                                T.reduce_min(x_frag, tile_min, dim=1)
                                for i in T.Parallel(block_m):
                                    acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                T.copy(x_ub_b, x_frag)
                                T.reduce_min(x_frag, tile_min, dim=1)
                                for i in T.Parallel(block_m):
                                    acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                for i in T.Parallel(block_m):
                                    out_ub[i] = acc[i, 0]
                                T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

            else:
                @T.prim_func
                def main(
                        x: T.Tensor((M, N), dtype),
                        out: T.Tensor((M,), dtype),
                    ):
                    with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                        x_ub = T.alloc_shared((block_m, tn_eff), dtype)
                        x_frag = T.alloc_fragment((block_m, tn_eff), dtype)
                        tile_min = T.alloc_fragment((block_m, 1), dtype)
                        acc = T.alloc_fragment((block_m, 1), dtype)
                        out_ub = T.alloc_shared((block_m,), dtype)
                        x_tail = T.alloc_shared((block_m, tail_w), dtype)
                        x_tail_frag = T.alloc_fragment((block_m, tail_w), dtype)
                        value_inf = T.infinity(dtype)

                        for s in T.serial(num_tasks):
                            block_id = s * num_kernels + cid
                            if block_id < num_logical:
                                off_m = block_id * block_m
                                real_m = T.min(block_m, M - off_m)
                                # Running-min state re-initialized per task.
                                T.vbrc(value_inf, acc)
                                for t in T.serial(num_full):
                                    T.copy(
                                        x[
                                            off_m : off_m + real_m,
                                            t * tn_eff : (t + 1) * tn_eff,
                                        ],
                                        x_ub[0:real_m, 0:tn_eff],
                                    )
                                    T.copy(x_ub, x_frag)
                                    T.reduce_min(x_frag, tile_min, dim=1)
                                    for i in T.Parallel(block_m):
                                        acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                # Static tail tile (trace-time branch, exact width).
                                if tail > 0:
                                    T.copy(
                                        x[off_m : off_m + real_m, num_full * tn_eff : N],
                                        x_tail[0:real_m, 0:tail],
                                    )
                                    T.copy(x_tail, x_tail_frag)
                                    T.reduce_min(x_tail_frag, tile_min, dim=1)
                                    for i in T.Parallel(block_m):
                                        acc[i, 0] = T.min(acc[i, 0], tile_min[i, 0])
                                for i in T.Parallel(block_m):
                                    out_ub[i] = acc[i, 0]
                                T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(M, N, dtype_str, tile_n, block_m, tag):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    tn_eff = _fit_tile_n(N, dtype_str, tile_n)
    num_full = N // tn_eff
    tail = N - num_full * tn_eff

    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _simple_reduce_kernel_tiled(M, N, "amin", dtype_str, tile_n)(block_m)
    y = kernel(x)
    ref = golden_amin(x.cpu())

    assert tuple(y.shape) == tuple(ref.shape), (
        f"[{tag}] output shape {tuple(y.shape)} != expected {tuple(ref.shape)}"
    )
    assert y.dtype == ref.dtype, f"[{tag}] output dtype {y.dtype} != {ref.dtype}"
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} tile_n={tile_n} "
        f"tn_eff={tn_eff} full/tail={num_full}/{tail} max_diff={max_diff:.2e}"
    )
    return max_diff


# ---------- L0: gate tests (must pass) -- DESIGN.md section 8.3 ----------
def run_L0():
    # (M, N, dtype, tile_n, block_m)
    cases = [
        (64, 32768, "bfloat16", 16384, 1),  # L0-1 main workload (2 full tiles)
        (64, 32768, "float16", 16384, 1),  # L0-2 fp16 native domain
        (64, 32768, "float32", 16384, 1),  # L0-3 fp32 (fit downshift 3/512)
        (8, 33000, "bfloat16", 31232, 1),  # L0-4 tail tile (fit -> 2/232)
        (8, 33000, "float16", 16384, 1),  # L0-5 tail x fp16 (2/232)
        (128, 40000, "bfloat16", 8192, 1),  # L0-6 multi tile (4/7232) + guard
        (65, 32768, "bfloat16", 16384, 1),  # L0-7 persistent tail wave (2 tasks)
        (1, 102400, "float16", 25600, 1),  # L0-8 single row big N (4/0)
        (33, 65536, "bfloat16", 16384, 1),  # L0-9 M < cores degenerate (4/0)
        (4, 32513, "float16", 16384, 1),  # L0-10 tiled lower bound (2/257)
    ]
    max_diff = 0.0
    for M, N, dtype_str, tile_n, block_m in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, tile_n, block_m, "L0"))
    print(f"[L0] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    cases = [
        (16, 32768, "float16", 16384, 1),  # small M, 2 full tiles
        (32, 65536, "float32", 16384, 1),  # fp32 multi tile (fit 6/1408)
        (64, 33000, "float16", 16384, 1),  # tail + persistent (2 tasks)
        (128, 32768, "bfloat16", 16384, 1),  # large M persistent (4 tasks)
        (3, 32768, "float16", 16384, 1),  # M < cores degenerate
        (2, 65536, "bfloat16", 16384, 1),  # small M big N (4 full tiles)
        (17, 32768, "float32", 16384, 1),  # non-multiple M + fp32 fit
        (1, 32768, "float32", 16384, 1),  # single row (num_kernels=1)
        (33, 32769, "bfloat16", 16384, 1),  # N unaligned, minimal tail (=1)
        (256, 40000, "float16", 8192, 1),  # 4 full tiles + 7232 tail + guard
    ]
    max_diff = 0.0
    for M, N, dtype_str, tile_n, block_m in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, tile_n, block_m, "L1"))
    print(f"[L1] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (4, 49151, "bfloat16", 24576, 1),  # extreme tail (fit 3/767)
        (1, 65536, "float32", 16384, 1),  # fp32 single row (fit 6/1024)
        (2, 1, "float16", 16384, 1),  # tail-only path (num_full=0, tail=1)
        (1, 40000, "bfloat16", 16384, 1),  # single row + tail (3/64)
        (16, 65536, "float16", 16384, 1),  # 4 full tiles, fp16
    ]
    for M, N, dtype_str, tile_n, block_m in cases:
        try:
            _run_case(M, N, dtype_str, tile_n, block_m, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def _boundary_case(M, N, dtype_str, tile_n, block_m, x, tag, equal_nan=False):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    kernel = _simple_reduce_kernel_tiled(M, N, "amin", dtype_str, tile_n)(block_m)
    y = kernel(x)
    ref = golden_amin(x.cpu())
    torch.testing.assert_close(
        y.cpu(), ref, rtol=0, atol=0, equal_nan=equal_nan
    )
    print(f"[Boundary] PASS: {tag}")


def run_boundary():
    # All +inf / all -inf rows: min semantics preserve the extremes. The
    # -inf case with a tail also validates the +inf initializer convergence.
    extreme_cases = [
        (8, 32768, "float16", 16384, float("inf"), "all +inf (fp16, tail=0)"),
        (8, 32768, "float16", 16384, float("-inf"), "all -inf (fp16, tail=0)"),
        (4, 33000, "float16", 16384, float("-inf"), "all -inf (fp16, tail=232)"),
        (4, 33000, "bfloat16", 31232, float("inf"), "all +inf (bf16, tail=232)"),
    ]
    for M, N, dtype_str, tile_n, val, tag in extreme_cases:
        try:
            torch_dtype, _, _ = _DTYPE_MAP[dtype_str]
            x = torch.full((M, N), val, dtype=torch_dtype, device="npu")
            _boundary_case(M, N, dtype_str, tile_n, 1, x, tag)
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): {tag}: {e}")

    # Mixed rows: one -inf row, one +inf row, the rest random.
    try:
        x = torch.randn(8, 32768, dtype=torch.float16, device="npu")
        x[0, :] = float("-inf")
        x[1, :] = float("inf")
        _boundary_case(8, 32768, "float16", 16384, 1, x, "mixed +-inf rows")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): mixed +-inf rows: {e}")

    # +-0 tie rows: IEEE-equal values compare equal under assert_close.
    try:
        x = torch.zeros(8, 33000, dtype=torch.float32, device="npu")
        x[0, 3] = -0.0
        x[1, 32999] = -0.0  # inside the static tail tile
        _boundary_case(8, 33000, "float32", 16384, 1, x, "+-0 mix (fp32, tail=232)")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): +-0 mix: {e}")

    # NaN row: record the NPU propagation behavior vs torch (DESIGN 9.1-R3).
    try:
        x = torch.randn(4, 33000, dtype=torch.float16, device="npu")
        x[0, 100] = float("nan")
        x[1, 32999] = float("nan")  # NaN inside the static tail tile
        kernel = _simple_reduce_kernel_tiled(4, 33000, "amin", "float16", 16384)(1)
        y = kernel(x)
        ref = golden_amin(x.cpu())
        print(f"[Boundary] NaN record: kernel={y.cpu().tolist()} torch={ref.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): NaN: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
