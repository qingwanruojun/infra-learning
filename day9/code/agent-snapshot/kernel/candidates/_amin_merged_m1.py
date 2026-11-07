# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Amin single-tile row-reduction kernel (NPU, Developer mode) -- Stage 4 merged-m1 (r2a eff_bm uplift + r4a bf16 vcast epilogue).

Round-1 branch merged_m1 (single optimization point: effective block_m uplift).

Change vs baseline: the factory uplifts the wrapper-selected ``block_m`` to
an UB-budget-allowed larger effective row block (``eff_bm``) so each task
processes more rows per launch -- fewer tasks per core (less serial-task
launch/sync overhead), fewer and wider MTE3 stores (the 2-byte-per-task
tiny stores of the bm=1 3d workload cost ~1us each of MTE3 pipe occupancy),
and wider MTE2 copies. The wrapper default_config heuristics are NOT
modifiable; ``eff_bm`` is a tuned factory-internal default exposed as the
module constant ``_EFF_BM_*`` below (wrapper passes block_m; eff_bm >=
block_m always, and the uplift saturates at 8 rows).

UB budget predicate (single set, PlanMemory inflation 1.12x per
CONST-capacity-910B2C; 160KB working budget leaves room for the small
row_min/out_ub fragments)::

    eff_bm <= min(8, 163840 // (N * resident_bytes_per_elem))

where resident_bytes_per_elem covers the shared staging + fragment pair:
fp16/fp32 = 2x/2x dtype bytes (4B fp16, 8B fp32); bf16 = 2B + 4B = 6B.

Semantics (DESIGN.md section 0.1, frozen):
    out[m] = min_{n in [0, N)} x[m, n]
    Input ``(M, N)``; output ``(M,)``; dtype in {float16, bfloat16, float32}.
    This kernel covers the single-tile resident path (N <= 32512); larger N is
    dispatched by the wrapper to the tiled sibling function.

Known toolchain constraints enforced here (DESIGN.md section 9.2):
    - reduce src/dst dtype MUST match (mixed dtype silently corrupts,
      VP-2026-0119).
    - bf16 reduce is unsupported (T.reduce.md section 2.2.1) -> vcast(rint)
      to fp32 first.
    - ``T.vcast`` f32->f32 is non-identity (CONST-vcast-f32-to-f32-corrupt)
      -> fp32 uses a same-dtype ``T.copy``.
    - ``T.copy`` src/dst use paired explicit slices for the M tail (T.copy
      shape-consistency rule).
    - ``T.Kernel`` is 1-D only; ``T.serial`` bounds are factory-static; the
      dynamic out-of-range check is an ``if`` guard only.
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

# merged_m1: tuned factory-internal effective block_m policy (module-level
# constants so the wrapper toggle block can reference them if adopted).
# Working UB budget for the (eff_bm, N) staging+fragment pair, leaving room
# for the small (eff_bm,1)/(eff_bm,) fragments and PlanMemory alignment.
_EFF_BM_UB_BUDGET_BYTES = 160 * 1024
_EFF_BM_MAX = 8
# resident bytes per element for the staging+fragment pair, per dtype
# r2a: resident bytes MUST assume the auto-multi-buffer double-buffered
# staging (merged-m1 (r2a eff_bm uplift + r4a bf16 vcast epilogue) lesson: bf16 eff=6 overflowed by 32B because x_ub was
# double-buffered): fp16/fp32 = 2 x dtype-bytes, bf16 = 2B staged x2 + 4B
# fp32 work = 8B. With this predicate bf16 stays at the wrapper value.
_EFF_BM_RESIDENT_BYTES = {"float16": 4, "bfloat16": 8, "float32": 8}


def _uplift_block_m(block_m, N, dtype):
    """r2a: uplift the wrapper-selected block_m to the UB-budget maximum.

    eff_bm >= block_m always (the wrapper value is the floor); the uplift
    saturates at _EFF_BM_MAX rows and the UB budget predicate. For the
    hidden-state workloads this turns 13 tasks/core (fp16, bm=4) into 7
    (eff=8) and 13 (bf16, bm=4) into 3 (eff=6); for the 3d workload
    (wrapper bm=1, N=16384) it turns 4 tasks/core into 2 (eff=2, the
    budget bound for the 4 B/elem staging+fragment pair at N=16384).
    """
    resident = _EFF_BM_RESIDENT_BYTES[dtype]
    max_by_ub = _EFF_BM_UB_BUDGET_BYTES // (N * resident)
    return max(block_m, min(_EFF_BM_MAX, max_by_ub))

# Tolerances from examples/TileOPs/tests/ops/test_amin_fwd.py::_tol
# (fp32 1e-4/1e-4; fp16/bf16 1e-2/1e-2). Expected results are bit-exact
# (DESIGN.md section 8.2) for all dtypes.
_DTYPE_MAP = {
    "float16": (torch.float16, 1e-2, 1e-2),
    "float32": (torch.float32, 1e-4, 1e-4),
    "bfloat16": (torch.bfloat16, 1e-2, 1e-2),
}


# ---------- Golden (PyTorch CPU reference implementation) ----------
def golden_amin(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: ``torch.amin`` along the last dim (fp32 opmath).

    CPU only. Ported from the source-repo test baseline
    (``test_amin_fwd.py`` ``ref_program``: ``x.float().amin(dim=-1).to(x.dtype)``);
    does NOT reproduce the NPU algorithm (no reduce_min / persistent split),
    guaranteeing validation independence (DESIGN.md section 8.1).
    """
    return torch.amin(x.float(), dim=-1).to(x.dtype)


# ---------- Kernel ----------
def _ceildiv(a, b):
    return (a + b - 1) // b


def _simple_reduce_kernel(M, N, op_kind, dtype):
    """Build the amin single-tile NPU kernel factory.

    Mirrors the GPU source structure ``_simple_reduce_kernel(M, N, op_kind,
    dtype) -> _func(block_m) -> main`` (name and prim_func parameter names /
    order unchanged). K9: the ``threads`` backend parameter is removed (NPU
    has no CUDA threads concept); backend config is the single ``block_m``
    argument plus the factory-period vector-core query (DESIGN.md section 5.5).
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

    # Factory-period physical core count (never a hard-coded assumption;
    # 910B3 verified 20 AI cores / 40 vector cores on 2026-11-05).
    vector_cores = NPUUtils.get().get_aivector_core_num()
    is_bf16 = dtype == "bfloat16"

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):
        # merged_m1: uplift to the UB-budget-allowed effective row block
        # (wrapper block_m is the floor; see _uplift_block_m docstring).
        block_m = _uplift_block_m(block_m, N, dtype)

        # Persistent split (DESIGN.md section 5.5, element 3): num_tasks is a
        # factory-static serial-loop bound; out-of-range tasks are guarded by
        # the dynamic ``if`` below.
        num_logical = _ceildiv(M, block_m)
        num_kernels = min(num_logical, vector_cores)
        num_tasks = _ceildiv(num_logical, num_kernels)

        if is_bf16:
            # bf16 domain: shared staging -> vcast(rint) fp32 fragment ->
            # fp32-domain reduce (bf16 reduce unsupported, T.reduce.md 2.2.1).
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_work = T.alloc_fragment((block_m, N), "float32")
                    row_min = T.alloc_fragment((block_m, 1), "float32")
                    row_min_bf = T.alloc_fragment((block_m, 1), dtype)
                    out_ub = T.alloc_shared((block_m,), dtype)

                    for s in T.serial(num_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(
                                x[off_m : off_m + real_m, 0:N],
                                x_ub[0:real_m, 0:N],
                            )
                            T.vcast(x_ub, x_work, round_mode="rint")
                            T.reduce_min(x_work, row_min, dim=1)
                            # merged r4a: vectorized fp32->bf16 downcast
                            # (full-buffer T.vcast, rint; replaces the
                            # per-row scalar T.cast in T.Parallel).
                            T.vcast(row_min, row_min_bf, round_mode="rint")
                            for i in T.Parallel(block_m):
                                out_ub[i] = row_min_bf[i, 0]
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        else:
            # fp16/fp32 domain: shared staging -> same-dtype fragment copy
            # (fp32 must NOT use T.vcast f32->f32) -> natural-domain reduce.
            @T.prim_func
            def main(
                x: T.Tensor((M, N), dtype),
                out: T.Tensor((M,), dtype),
            ):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_shared((block_m, N), dtype)
                    x_frag = T.alloc_fragment((block_m, N), dtype)
                    row_min = T.alloc_fragment((block_m, 1), dtype)
                    out_ub = T.alloc_shared((block_m,), dtype)

                    for s in T.serial(num_tasks):
                        block_id = s * num_kernels + cid
                        if block_id < num_logical:
                            off_m = block_id * block_m
                            real_m = T.min(block_m, M - off_m)
                            T.copy(
                                x[off_m : off_m + real_m, 0:N],
                                x_ub[0:real_m, 0:N],
                            )
                            T.copy(x_ub, x_frag)
                            T.reduce_min(x_frag, row_min, dim=1)
                            for i in T.Parallel(block_m):
                                out_ub[i] = row_min[i, 0]
                            T.copy(out_ub[0:real_m], out[off_m : off_m + real_m])

        return main

    return _func


# ---------- precision comparison ----------
def _run_case(M, N, dtype_str, block_m, tag):
    torch_dtype, atol, rtol = _DTYPE_MAP[dtype_str]
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    kernel = _simple_reduce_kernel(M, N, "amin", dtype_str)(block_m)
    y = kernel(x)
    ref = golden_amin(x.cpu())

    assert tuple(y.shape) == tuple(ref.shape), (
        f"[{tag}] output shape {tuple(y.shape)} != expected {tuple(ref.shape)}"
    )
    assert y.dtype == ref.dtype, f"[{tag}] output dtype {y.dtype} != {ref.dtype}"
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol, atol=atol)
    max_diff = (y.cpu().float() - ref.float()).abs().max().item()
    print(
        f"[{tag}] PASS: shape=({M},{N}) dtype={dtype_str} block_m={block_m} "
        f"max_diff={max_diff:.2e}"
    )
    return max_diff


# ---------- L0: gate tests (must pass) -- DESIGN.md section 8.3 ----------
def run_L0():
    cases = [
        (32, 256, "float16", 8),  # L0-1 smoke fp16
        (32, 256, "bfloat16", 8),  # L0-2 bf16 vcast(rint) domain prep
        (32, 256, "float32", 8),  # L0-3 fp32 native domain (same-dtype copy)
        (2048, 4096, "float16", 4),  # L0-4 hidden-state fp16 (persistent 13 tasks)
        (2048, 4096, "bfloat16", 4),  # L0-5 hidden-state bf16 (UB stress)
        (128, 16384, "float16", 1),  # L0-6 3d-multidim reshape entry (bm=1)
        (33, 256, "float16", 8),  # L0-7 M tail (33 % 8 = 1)
        (128, 300, "float16", 8),  # L0-8 N unaligned (exact slice path)
        (128, 300, "bfloat16", 8),  # L0-9 N unaligned x bf16
        (129, 512, "float16", 8),  # L0-10 tail fixture shape (129 % 8 = 1)
        (1, 256, "float16", 1),  # L0-11 single row (num_kernels=1 degenerate)
        (2, 1, "float16", 1),  # L0-12 minimal N=1 (narrow-row boundary)
    ]
    max_diff = 0.0
    for M, N, dtype_str, block_m in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, block_m, "L0"))
    print(f"[L0] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L1: functional coverage (must pass) ----------
def run_L1():
    cases = [
        (128, 512, "float16", 8),  # smoke fixture shape
        (128, 512, "float32", 8),
        (128, 512, "bfloat16", 8),
        (256, 4096, "float16", 4),  # full fixture shape
        (256, 4096, "bfloat16", 4),
        (130, 256, "float16", 8),  # M tail 130 % 8 = 2
        (17, 128, "float16", 16),  # M tail 17 % 16 = 1
        (64, 1024, "bfloat16", 4),
        (8, 512, "float16", 8),  # exact divide M = block_m
        (256, 1024, "float32", 8),
        (3, 300, "bfloat16", 2),  # small M + unaligned N
        (4096, 256, "bfloat16", 8),  # large M persistent (512 logical blocks)
    ]
    max_diff = 0.0
    for M, N, dtype_str, block_m in cases:
        max_diff = max(max_diff, _run_case(M, N, dtype_str, block_m, "L1"))
    print(f"[L1] ALL PASS: {len(cases)} cases, max_diff={max_diff:.2e}")


# ---------- L2: boundary (warn only, non-blocking) ----------
def run_L2():
    cases = [
        (1, 1, "float16", 1),  # minimal
        (9, 256, "float16", 8),  # M = block_m + 1
        (32, 1, "float16", 4),  # N=1 single-element reduce
        (256, 4096, "float32", 4),  # UB stress (fp32 not in fixture dtype set)
        (4, 32512, "float16", 1),  # N_padded full bound (vectorizer limit)
    ]
    for M, N, dtype_str, block_m in cases:
        try:
            _run_case(M, N, dtype_str, block_m, "L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (record only): shape=({M},{N}) {dtype_str}: {e}")


# ---------- Boundary: extreme values (warn only, non-blocking) ----------
def run_boundary():
    # All +inf / all -inf rows: min semantics preserve the extremes.
    for val, name in [(float("inf"), "all +inf"), (float("-inf"), "all -inf")]:
        try:
            x = torch.full((8, 256), val, dtype=torch.float16, device="npu")
            kernel = _simple_reduce_kernel(8, 256, "amin", "float16")(8)
            y = kernel(x)
            ref = golden_amin(x.cpu())
            torch.testing.assert_close(y.cpu(), ref, rtol=0, atol=0, equal_nan=True)
            print(f"[Boundary] PASS: {name}")
        except Exception as e:  # noqa: BLE001
            print(f"[Boundary] WARN (record only): {name}: {e}")

    # +-0 tie rows: IEEE-equal values compare equal under assert_close.
    try:
        x = torch.zeros(8, 256, dtype=torch.float32, device="npu")
        x[0, 3] = -0.0
        kernel = _simple_reduce_kernel(8, 256, "amin", "float32")(8)
        y = kernel(x)
        ref = golden_amin(x.cpu())
        torch.testing.assert_close(y.cpu(), ref, rtol=0, atol=0)
        print("[Boundary] PASS: +-0 mix")
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (record only): +-0 mix: {e}")

    # NaN row: record NPU propagation vs torch (DESIGN.md section 9.1-R3).
    try:
        x = torch.randn(4, 256, dtype=torch.float16, device="npu")
        x[0, 100] = float("nan")
        kernel = _simple_reduce_kernel(4, 256, "amin", "float16")(4)
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
