# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Expm1 forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``Expm1FwdKernel`` / ``expm1_fwd_kernel``
(TileOPs, ``FloatUnaryKernel`` family, Pattern-A verbatim extract
``examples/TileOPs/tileops/kernels/elementwise/expm1/_expm1_fwd_kernels.py``)
to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = expm1(x[i]) = exp(x[i]) - 1        (unconditional, elementwise)
on a flattened, contiguous ``(N,)`` tensor of dtype
float16 / bfloat16 / float32; output ``(N,)`` same dtype (manifest
same_as(input)).  The source computes entirely in the fp32 domain with
a single final rounding to the output dtype (``T.exp(T.cast(x, "float32"))
- T.cast(1.0, "float32")``); this kernel structurally carries that path
over (a "structural carry-over" transit, same family as elu):
    - float32: native-domain chain (no transit):  copy -> vexp -> vsub
      (scalar literal 1.0, cross-buffer dst=x_ub) -> copy.
    - float16/bfloat16: fp32 transit chain (single final rounding):
      copy -> vcast(rint) up -> vexp -> vsub(1.0, in-place) ->
      vcast(rint) down (reuses the load buffer) -> copy.
      bfloat16 has no v-prefix arithmetic/transcendental (T.vexp.md /
      T.vsub.md dtype matrices: bf16 x); fp16 transits to align with the
      golden's fp32-opmath rounding path.
Overflow/underflow semantics: x >= 88.73 -> exp(x) = +Inf -> +Inf - 1 =
+Inf (the expected output for the L0 x=88.8 corner, IEEE-equal to the
golden); x <= -103.97 -> exp(x) = 0 -> -1.0; the subnormal transition
band is FTZ-insensitive (|exp(x)| < 1.7e-38 << ulp(1.0)/2).  At x = -0.0
the golden keeps the ATen sign (-0.0) while the NPU subtraction yields
+0.0 -- IEEE value-equal, zero-sign difference exempted (DESIGN.md
section 8.3 L0-5).  NaN propagates (exp(NaN) - 1 = NaN).

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` persistent core split
        (``num_kernels = min(num_logical, vector_cores)``) with an
        in-core ``T.serial`` grid-stride loop and an if-guard for
        out-of-range tiles (static bounds) + a v-prefix vector chain
        (vexp -> vsub with a scalar literal operand) replaces the CUDA
        threads x npt two-level SIMT expansion.  No vcmp/vselect/vbrc:
        the formula is unconditional and the only constant (1.0) rides
        as the vsub scalar-literal operand (zero constant buffers).
    R3: tail handling = tile-level ``tail = T.min(block_size, N - t0)``
        with zero-origin UB slices on store (official vec_add_1d.py
        pattern; no per-element guard, no padding).  Stale [tail:] lanes
        never land in GM.

Factory contract (E2/K11, two-stage; DESIGN.md sections 3.3/3.5.2):
    expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=None,
                     threads=256, num_per_thread=8)
        -> kernel(bs=default_block) -> main   (main(x, y), out_idx=[1])
    where ``main(x)`` returns ``y``.  The factory return value is ALWAYS
    a curried builder (``return kernel`` unconditionally); a factory-level
    ``block_size=`` kwarg only presets ``default_block`` and never changes
    the return shape.  The wrapper (Stage 5) must call it two-stage:
        expm1_fwd_kernel(N, dtype_str, block_size=block_size)(block_size)(x)
    Wrapper adaptation required (DESIGN.md section 3.5.2, one mandatory
    spot): examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py
    L81 currently calls ``expm1_fwd_kernel(N, dtype_str,
    block_size=block_size)(x, N)`` -- the trailing ``, N`` cannot bind in
    the jit callable contract (out_idx=[1], main(x, y) -> callable(x))
    and a single-stage ``(x)`` would bind the tensor to the ``bs``
    parameter (build-time TypeError).  Adaptation = drop the trailing
    ``, N`` and add the curried ``(block_size)`` application (2 edits,
    family-standard two-stage: leaky_relu.py L98 / elu_fwd.py L99).
    ``threads``/``num_per_thread`` stay in the signature for source
    fidelity + lru_cache key space but are not consumed on NPU (K11
    folds threads*npt into the single ``block_size``; default fp32 1024 /
    fp16.bf16 2048 = wrapper ``_BLOCK_SIZE_FP32``/``_BLOCK_SIZE_NON_FP32``).

Run the embedded hierarchical tests:
    python expm1_fwd_kernel.py --level L0
    python expm1_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no
# manual sync points (DESIGN.md section 7); the module-level setdefault
# keeps Expert/Developer mixed-process runs isolated.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

ASCEND_MODE = "Developer"

import argparse
import functools
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["expm1_fwd_kernel", "golden_expm1"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

# Per-dtype block_size cap (DESIGN.md section 4.5, platform-value
# accounting after REVIEW issue 3): fp32 direct chain ~26.00 B/elem ->
# 4096 elements ~= 106.5 KB; fp16/bf16 transit chain ~20.01 B/elem ->
# 8192 elements ~= 163.9 KB (floating upper bound 22.00 B/elem x 8192
# ~= 180.2 KB still safe).  fp32 8192 would sit at ~213 KB > 192 KB and
# is excluded by the tighter fp32 cap.
_UB_MAX_BLOCK = {
    "float32": 4096,
    "float16": 8192,
    "bfloat16": 8192,
}

# Default tile sizes (DESIGN.md section 5.2 / K11: GPU threads(256) *
# npt folded into a single block_size; fp32 256*4=1024,
# fp16/bf16 256*8=2048 -- equals the NPU wrapper
# ``_BLOCK_SIZE_FP32``/``_BLOCK_SIZE_NON_FP32``).
_DEFAULT_BLOCK = {
    "float32": 1024,
    "float16": 2048,
    "bfloat16": 2048,
}

# Precision tolerances (DESIGN.md section 8.2, identical to the source
# repo test_unary_math.py ``_get_tolerances`` L24-L29).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
    "float32": (torch.float32, 1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_expm1(x: torch.Tensor) -> torch.Tensor:
    """Expm1 reference: ``torch.expm1(x)``.

    Ported from the source-repo test anchor (test_unary_math.py
    ``ref_fn = torch.expm1``): the golden IS the fp32-opmath + single
    round-back path (ATen internally computes expm1 for sub-fp32 dtypes
    in fp32 opmath; GOLD-DIAG machine-verified bit-identical to the
    fp32-domain computation, DESIGN.md section 0.1).  Independent of the
    NPU algorithm (no vcast transit / vexp+vsub chain / persistent
    structure knowledge).  Runs on CPU and does not require an NPU
    device.  The explicit ``.to(x.dtype)`` keeps the sub-fp32 golden in
    the input dtype (same_as(input) contract).
    """
    return torch.expm1(x.detach().cpu()).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def expm1_fwd_kernel(
    N,
    dtype,
    output_dtype=None,
    block_size=None,
    threads=256,
    num_per_thread=8,
):
    """Build the Expm1 kernel factory (source signature kept + K11).

    Returns a JIT callable taking a single ``block_size`` (E2/K11),
    which compiles and returns ``main``; ``main(x)`` returns ``y``
    (``out_idx=[1]``, source-operator contract).  The factory return
    value is ALWAYS a curried builder -- the two-stage call
    ``expm1_fwd_kernel(...)(block_size)(x)`` is the only valid form
    (DESIGN.md section 3.5.2; single-stage ``(x)`` binds the tensor to
    the ``bs`` parameter and fails at build time).  Host-side contract
    validation (ValueError, survives ``python -O``):

      - N <= 0            -> ValueError (design adaptation: the source
        Op-layer check ``numel != N_total`` lets N=0 through; NPU
        rejects defensively, DESIGN.md section 9.3);
      - output_dtype not None and != dtype -> ValueError (manifest
        same_as(input));
      - dtype not in {float16, bfloat16, float32} -> ValueError.

    The dtype argument accepts both ``torch.dtype`` objects and str
    (the wrapper passes dtype_str directly, L81/L155); it is normalized
    to ``dtype_key`` before any branch/guard/default selection
    (comparing a torch.dtype object against a str tuple is always
    False and would mis-route fp16/bf16 to the fp32 path).  NOTE:
    torch.dtype has no ``.name`` attribute in torch 2.7.1 --
    ``str(torch.float16) == "torch.float16"``, so the key is derived
    by stripping the ``"torch."`` prefix.
    """
    if N <= 0:
        raise ValueError(f"N must be a positive int, got N={N}")
    dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]
    if dtype_key not in _DEFAULT_BLOCK:
        raise ValueError(
            f"unsupported dtype {dtype!r}; expected one of {sorted(_DEFAULT_BLOCK)}"
        )
    if output_dtype is not None:
        out_key = output_dtype if isinstance(output_dtype, str) else str(output_dtype).split(".")[-1]
        if out_key != dtype_key:
            raise ValueError(
                f"output_dtype {output_dtype!r} != input dtype {dtype_key!r}; "
                "manifest requires same_as(input)"
            )

    # fp32 transit for both half-precision dtypes (DESIGN.md section
    # 0.6 R2, A1: structural carry-over of the source fp32-opmath path;
    # bf16 has no v-prefix arithmetic, fp16 transits to align with the
    # golden rounding path).  fp32 computes in the native domain.
    use_transit = dtype_key in ("float16", "bfloat16")
    # Factory-level block_size kwarg only presets default_block; the
    # return value stays a curried builder (two-stage contract).
    default_block = block_size if block_size is not None else _DEFAULT_BLOCK[dtype_key]
    ub_cap = _UB_MAX_BLOCK[dtype_key]

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; on this device 20 -> 40).  Re-queried per factory
    # call so deployments on other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(bs=default_block):
        # Host-side UB budget guard (DESIGN.md sections 4.5 / 9.2),
        # per-dtype cap.  ValueError is used instead of a bare assert so
        # the guard survives `python -O`.
        if bs <= 0:
            raise ValueError(f"block_size must be positive, got {bs}")
        if bs > ub_cap:
            raise ValueError(
                f"block_size={bs} exceeds the UB budget for {dtype_key} "
                f"(max {ub_cap} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # Persistent core split (DESIGN.md section 5.5).  All quantities
        # are compile-time constants: N and bs are Python ints and
        # T.ceildiv/min fold to IntImm at trace time.
        num_logical = T.ceildiv(N, bs)  # logical tile count
        num_kernels = min(num_logical, vector_cores)  # <= physical cores
        num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

        # Compile-time dtype dispatch (DESIGN.md section 3.3 note): the
        # transit branch is selected inside the factory.  Keeping the
        # branch outside the traced body also avoids the TVM-script
        # parser's if-block variable-table scoping limitation (C11).
        if use_transit:
            # fp16/bf16 transit chain (5 steps).  No vbrc / vcmp /
            # vselect / constant buffers: the formula is unconditional
            # and the only scalar (1.0) rides as the vsub scalar-literal
            # operand (mish L78 same-shaped precedent).  vsub is
            # in-place (Out == src0 = w32); the result is consumed by
            # step 5 within the same T.serial iteration (no
            # loop-carried state, DESIGN.md section 9.2).
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    # UB buffers (DESIGN.md section 4.3).  x_ub is the
                    # load buffer and the round-back / store source.
                    x_ub = T.alloc_ub((bs,), dtype_key)
                    x32 = T.alloc_ub((bs,), "float32")
                    w32 = T.alloc_ub((bs,), "float32")

                    # Persistent in-core serial task loop (static bound,
                    # grid-stride mapping, if-guard masks out-of-range
                    # tiles).
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * bs
                            # Tail handling (DESIGN.md section 6.4):
                            # last tile may be shorter; dst slices are
                            # always zero-origin (UB alignment pattern).
                            tail = T.min(bs, N - t0)

                            # 1) Load GM -> UB (same-dtype copy).
                            T.copy(x[t0 : t0 + tail], x_ub[0:tail])
                            # 2) Upcast to fp32 (rint; lossless since
                            #    fp16/bf16 are subsets of fp32).
                            T.vcast(x_ub, x32, round_mode="rint")
                            # 3) exp(x32) -- full-buffer, fp32 domain.
                            T.vexp(x32, w32)
                            # 4) w32 = w32 - 1.0 (scalar literal
                            #    operand, in-place).
                            T.vsub(w32, 1.0, w32)
                            # 5) Round back to the input dtype with a
                            #    single rint, reusing the load buffer.
                            T.vcast(w32, x_ub, round_mode="rint")
                            # 6) Store UB -> GM ([0:tail] only; stale
                            #    [tail:] lanes never land in GM).  Store
                            #    source is x_ub on both routes.
                            T.copy(x_ub[0:tail], y[t0 : t0 + tail])

        else:
            # fp32 native domain (3 steps): no transit, no independent
            # x32 -- x_ub is the domain input and the vsub destination.
            # The vsub is CROSS-BUFFER (dst = x_ub != src0 = w32, the
            # standard three-buffer form of T.vsub.md section 2.4
            # example 1): it must NOT be written in-place
            # (``T.vsub(w32, 1.0, w32)``) or the store below would copy
            # the untouched load buffer and the fp32 route would output
            # an identity of its input (DESIGN.md section 9.2 guard).
            # x_ub's original value dies at the vexp read (RAW/WAR by
            # Developer auto-sync).  Store source = x_ub, unified with
            # the transit route.
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((bs,), dtype_key)
                    w32 = T.alloc_ub((bs,), "float32")

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * bs
                            tail = T.min(bs, N - t0)

                            # 1) Load GM -> UB (same-dtype copy).
                            T.copy(x[t0 : t0 + tail], x_ub[0:tail])
                            # 3) exp(x_ub) -- full-buffer, fp32 domain
                            #    (domain input = x_ub, no upcast).
                            T.vexp(x_ub, w32)
                            # 4) x_ub = w32 - 1.0 (scalar literal,
                            #    CROSS-BUFFER dst = load buffer).
                            T.vsub(w32, 1.0, x_ub)
                            # 6) Store UB -> GM ([0:tail] only).
                            T.copy(x_ub[0:tail], y[t0 : t0 + tail])

        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------

# Per-dtype worst max_diff across the whole run (for the final summary).
_STATS_MAX_DIFF = {}


def _compare(out, ref, atol, rtol, equal_nan=False):
    """Return (max_diff, violation_count) between out and ref.

    NaN/Inf aware with an ``out == ref`` exact-equality branch FIRST:
    equal infinities (e.g. x=88.8 -> both sides +Inf) and IEEE-equal
    +-0 pairs count as equal here, so inf-inf never degenerates into a
    NaN false violation and the +-0 sign difference (golden -0.0 vs NPU
    +0.0 at x=-0.0) is naturally exempted (DESIGN.md sections 8.2/8.3).
    """
    eq = out == ref
    if equal_nan:
        eq = eq | (torch.isnan(out) & torch.isnan(ref))
    diff = torch.where(
        eq,
        torch.zeros((), dtype=torch.float32),
        (out.float() - ref.float()).abs(),
    )
    max_diff = diff.max().item()
    violations = int((diff > (atol + rtol * ref.float().abs())).sum().item())
    return max_diff, violations


def _assert_special_bits(out, ref):
    """NaN/Inf positions must propagate correctly (Inf sign included).

    Finite values are left to the assert_close tolerance envelope (the
    NPU vexp hardware may differ from the golden's exp by <=1 ulp in
    the fp32 domain, DESIGN.md section 9.1#1).  NaN positions are
    accepted when both sides are NaN regardless of payload (the exp
    chain may alter the payload).  +-0 is NOT checked here -- the
    zero-sign difference is IEEE value-equal and exempted (DESIGN.md
    section 8.3 L0-5; do NOT copy leaky_relu's sign-bit strictness).
    """
    out_c = out.contiguous()
    ref_c = ref.contiguous()
    if out_c.dtype == torch.float16 or out_c.dtype == torch.bfloat16:
        out_i = out_c.view(torch.int16)
        ref_i = ref_c.view(torch.int16)
    else:
        out_i = out_c.view(torch.int32)
        ref_i = ref_c.view(torch.int32)
    special = torch.isnan(out_c) | torch.isinf(out_c) | torch.isnan(ref_c) | torch.isinf(ref_c)
    if special.any():
        both_nan = torch.isnan(out_c) & torch.isnan(ref_c)
        ok = both_nan | (out_i == ref_i)
        n_mismatch = int((~ok[special]).sum().item())
        if n_mismatch:
            raise AssertionError(
                f"special-value (NaN/Inf) bit mismatch: {n_mismatch}/{int(special.sum().item())} positions"
            )
    return True


def _run_case(
    dtype_str,
    N=None,
    block_size=None,
    tag="L0",
    x=None,
    equal_nan=False,
    special_bits=False,
):
    """Run one kernel case and compare against the golden.

    Tensor x may be pre-built (endpoint / special-value cases); by
    default ``randn`` per DESIGN.md section 8.3 L0-1.  The kernel is
    invoked two-stage (``expm1_fwd_kernel(N, dtype_str)(block_size)``,
    the E2/K11 family contract).  Raises AssertionError on tolerance
    violation.
    """
    torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
    if block_size is None:
        block_size = _DEFAULT_BLOCK[dtype_str]
    if x is None:
        # CPU-generate then cast+move (lerp/mish pattern; avoids NPU
        # RNG dtype quirks and keeps data generation device-independent).
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
    else:
        N = x.shape[0]
        assert x.dtype == torch_dtype, f"input dtype {x.dtype} != expected {torch_dtype}"

    kernel = expm1_fwd_kernel(N, dtype_str)(block_size)
    out = kernel(x)
    ref = golden_expm1(x.cpu())
    out_cpu = out.cpu()

    if special_bits:
        _assert_special_bits(out_cpu, ref)
    max_diff, violations = _compare(out_cpu, ref, atol, rtol, equal_nan=equal_nan)
    if max_diff == max_diff:  # NaN guard (all-equal cases give 0.0)
        best = _STATS_MAX_DIFF.get(dtype_str, 0.0)
        if max_diff > best:
            _STATS_MAX_DIFF[dtype_str] = max_diff

    try:
        torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except AssertionError:
        print(
            f"[{tag}] FAIL: N={N} dtype={dtype_str} block_size={block_size} "
            f"max_diff={max_diff:.3e} violations={violations}/{N}"
        )
        raise
    print(
        f"[{tag}] PASS: N={N} dtype={dtype_str} block_size={block_size} "
        f"max_diff={max_diff:.3e}"
    )
    return max_diff, violations


def _try_case(failures, case_id, **kwargs):
    """Run _run_case, collecting (not re-raising) per-case failures.

    AssertionError is treated as a precision failure; any other
    exception is recorded as a runtime error (both block the gate).
    """
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def _run_L0_corners(failures):
    """L0-5: corner & special-value propagation (per dtype).

    Corner set (DESIGN.md section 8.3): +-Inf / NaN / +-0 / +-subnormal
    / +-max-finite (from torch.finfo -- never hand-written decimal
    literals) / +-1.0 / x=88.8 (expected output +Inf, the expm1
    overflow semantics itself) / x=-88.8 (expected -1.0) / exp
    underflow band (-104) / overflow boundary band (+-89) / tiny-x
    cancellation zone (+-1e-8, +-1e-7, +-2^-24).  Assertion rule:
    assert_close value envelope (equal_nan=True) + NaN/Inf bit check,
    +-0 zero-sign difference exempted (DESIGN.md section 8.3 L0-5).
    """
    # Subnormal minima and max-finite per dtype (finfo only).
    corners = {
        "float16": (2 ** -24, float(torch.finfo(torch.float16).max)),
        "bfloat16": (2 ** -133, float(torch.finfo(torch.bfloat16).max)),
        "float32": (2 ** -149, float(torch.finfo(torch.float32).max)),
    }
    N = 8192
    base = torch.randn(N, dtype=torch.float32)
    for dtype_str, (subnorm, max_fin) in corners.items():
        vals = base.clone()
        vals[0] = float("inf")
        vals[1] = float("-inf")
        vals[2] = float("nan")
        vals[3] = 0.0
        vals[4] = -0.0
        vals[5] = subnorm
        vals[6] = -subnorm
        vals[7] = max_fin
        vals[8] = -max_fin
        vals[9] = 1.0
        vals[10] = -1.0
        vals[11] = 88.8
        vals[12] = -88.8
        vals[13] = -104.0
        vals[14] = 89.0
        vals[15] = -89.0
        vals[16] = 1e-8
        vals[17] = -1e-8
        vals[18] = 1e-7
        vals[19] = -1e-7
        vals[20] = 2 ** -24
        vals[21] = -(2 ** -24)
        x = vals.to(_TOLERANCE[dtype_str][0]).npu()
        _try_case(
            failures,
            f"L0-5/{dtype_str}/corners",
            dtype_str=dtype_str,
            tag="L0-5",
            x=x,
            special_bits=True,
            equal_nan=True,
        )


def _run_L0_persistent(failures):
    """L0-6: persistent core-count switch boundary.

    num_logical in {vc-1, vc, vc+1} (fp16 block 2048, vector_cores vc
    queried at runtime; on this device vc=40 -> 39/40/41 per DESIGN.md
    section 8.3): fewer kernels than cores / exact fit / masking +
    uneven tasks active.
    """
    vc = NPUUtils.get().get_aicore_num() * 2
    bs = 2048
    for num_logical, N in ((vc - 1, (vc - 1) * bs), (vc, vc * bs), (vc + 1, vc * bs + 1)):
        _try_case(
            failures,
            f"L0-6/fp16/num_logical={num_logical}",
            dtype_str="float16",
            N=N,
            block_size=bs,
            tag="L0-6",
        )
    # bf16 at the masking boundary (vc+1) to cover the transit vcast
    # chain under the if-guard + serial loop.
    _try_case(
        failures,
        f"L0-6/bf16/num_logical={vc + 1}",
        dtype_str="bfloat16",
        N=vc * bs + 1,
        block_size=bs,
        tag="L0-6",
    )


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, six items + extras).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L0-1: smoke, all dtypes, N=2^20, randn.
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(
            failures,
            f"L0-1/{dtype_str}",
            dtype_str=dtype_str,
            N=2**20,
            tag="L0-1",
        )

    # L0-2: large-scale spot check, N=2^24, fp16 + bf16 (205 in-core
    # serial tasks per core on this device -- multi-iteration coverage
    # for the in-place (transit) / cross-buffer (fp32) vsub "no
    # loop-carried state" discipline).
    for dtype_str in ("float16", "bfloat16"):
        _try_case(
            failures,
            f"L0-2/{dtype_str}",
            dtype_str=dtype_str,
            N=2**24,
            tag="L0-2",
        )

    # L0-3: non-divisible tails (single partial tile / tail==1 /
    # non-32B-aligned tail), fp16 + fp32.
    for dtype_str in ("float16", "float32"):
        for N in (1000, 4097, 5000):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                N=N,
                tag="L0-3",
            )

    # L0-4: tiny shapes (single non-full tile / num_logical=1
    # degenerate).
    _try_case(failures, "L0-4/fp16/N=1", dtype_str="float16", N=1, tag="L0-4")
    _try_case(failures, "L0-4/fp16/N=7", dtype_str="float16", N=7, tag="L0-4")
    _try_case(failures, "L0-4/fp16/N=32", dtype_str="float16", N=32, tag="L0-4")
    _try_case(failures, "L0-4/bf16/N=7", dtype_str="bfloat16", N=7, tag="L0-4")
    _try_case(failures, "L0-4/fp32/N=32", dtype_str="float32", N=32, tag="L0-4")

    # L0-5: corner & special-value propagation.
    _run_L0_corners(failures)

    # L0-6: persistent core-count switch boundary.
    _run_L0_persistent(failures)

    # Extra 1: E2 dual-config reuse -- one factory callable serving two
    # block_sizes (the jit wrapper caches the compiled main per
    # block_size).
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        fn = expm1_fwd_kernel(N, dtype_str)
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        ref = golden_expm1(x.cpu())
        for bs in (2048, 1024):
            out = fn(bs)(x).cpu()
            torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print("[L0-extra] PASS: E2 factory(block_size) dual-config reuse (2048/1024)")
    except AssertionError as exc:
        failures.append(("L0-extra/E2-reuse", exc))
        print(f"[collected] L0-extra/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-extra/E2-reuse", exc))
        print(f"[collected] L0-extra/E2-reuse RUNTIME-ERROR: {exc!r}")

    # Extra 2: two-stage integration form (wrapper L81 adapted call
    # surface, DESIGN.md section 3.5.2 / section 8.3): factory-level
    # block_size kwarg presets default_block + curried application.
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        ref = golden_expm1(x.cpu())
        out = expm1_fwd_kernel(N, dtype_str, block_size=2048)(2048)(x).cpu()
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print("[L0-extra] PASS: two-stage integration form (block_size kwarg + curried)")
    except AssertionError as exc:
        failures.append(("L0-extra/two-stage", exc))
        print(f"[collected] L0-extra/two-stage FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-extra/two-stage", exc))
        print(f"[collected] L0-extra/two-stage RUNTIME-ERROR: {exc!r}")

    # Extra 3: single-stage guard (DESIGN.md section 3.5.2 contract
    # analysis 2): directly calling the factory return value with
    # ``(x)`` binds the tensor to the bs parameter and must fail at
    # build time (TypeError / ValueError / any build-stage exception --
    # a successful run would be a contract violation).
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype = _TOLERANCE[dtype_str][0]
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        out = expm1_fwd_kernel(N, dtype_str, block_size=2048)(x)
        out.cpu()  # noqa: B018  (force materialization if it somehow ran)
        failures.append(
            ("L0-extra/single-stage", AssertionError("single-stage (x) call unexpectedly succeeded"))
        )
        print("[collected] L0-extra/single-stage FAILED: single-stage (x) call unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        print(f"[L0-extra] PASS: single-stage (x) direct call rejected ({type(exc).__name__}: {exc})")

    # Extra 4: UB budget guard, per-dtype cap (DESIGN.md section 4.5,
    # REVIEW issue 3): fp16/bf16 8193 -> ValueError; fp32 4097 AND 8193
    # -> ValueError (the fp32 8192 tier is excluded on the platform
    # value ~213 KB > 192 KB).
    for dtype_str, bad_bs in (("float16", 8193), ("bfloat16", 8193), ("float32", 4097), ("float32", 8193)):
        try:
            expm1_fwd_kernel(4096, dtype_str)(bad_bs)
        except ValueError as exc:
            print(f"[L0-extra] PASS: UB guard rejects {dtype_str} block_size={bad_bs} ({exc})")
        else:
            failures.append(
                (f"L0-extra/UB-guard-{dtype_str}-{bad_bs}", AssertionError(f"{dtype_str} {bad_bs} not rejected"))
            )

    # Extra 5: N defense (N=0 / N<0 -> ValueError at the builder).
    for bad_N in (0, -5):
        try:
            expm1_fwd_kernel(bad_N, "float16")
        except ValueError as exc:
            print(f"[L0-extra] PASS: N defense rejects N={bad_N} ({exc})")
        else:
            failures.append(
                (f"L0-extra/N-defense/{bad_N}", AssertionError(f"N={bad_N} not rejected"))
            )

    # Extra 6: output_dtype defense (manifest same_as(input)).
    try:
        expm1_fwd_kernel(16, "float16", output_dtype="float32")
    except ValueError as exc:
        print(f"[L0-extra] PASS: output_dtype mismatch rejected ({exc})")
    else:
        failures.append(("L0-extra/output_dtype", AssertionError("output_dtype mismatch not rejected")))

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (six items + extras)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size).

    Includes the manifest workloads (DESIGN.md section 5.5:
    elementwise-16m [4096,4096] = 2^24, fp16+bf16), the large
    multi-iteration persistent shapes, and the large block_size sweep
    (the transit 8192 cases sit exactly at the UB cap and double as the
    Stage-3 re-verification of the stale TRAP-UB-multibuffer platform
    value on the current toolchain; fp32 4096 sits exactly at its
    per-dtype cap).  Also re-exercises the factory-level block_size
    compatibility surface (two-stage integration form + single-stage
    TypeError guard).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # dtype x {exact multiple, single partial tile, tail==1-ish}.
    for dtype_str in ("float16", "bfloat16", "float32"):
        for N in (8192, 1000, 4097):
            _try_case(
                failures,
                f"L1/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                N=N,
                tag="L1",
            )

    # Large multi-iteration shapes (persistent in-core serial loop,
    # covers the in-place / cross-buffer vsub no-loop-carried
    # discipline at scale; fp32 2^24 exercises the cross-buffer route
    # with 16384 logical tiles).
    for dtype_str in ("float16", "bfloat16", "float32"):
        _try_case(
            failures,
            f"L1/large/N=2^24/{dtype_str}",
            dtype_str=dtype_str,
            N=2**24,
            tag="L1",
        )

    # Manifest workloads (flattened product of the 2-D bench shapes).
    for dtype_str in ("float16", "bfloat16"):
        for N in (2**23, 2**24):
            _try_case(
                failures,
                f"L1/manifest/N={N}/{dtype_str}",
                dtype_str=dtype_str,
                N=N,
                tag="L1",
            )

    # Non-default block_size sweep (wrapper config space + per-dtype UB
    # cap re-verification).
    _try_case(
        failures,
        "L1/fp16/N=2^20/bs=1024",
        dtype_str="float16",
        N=2**20,
        block_size=1024,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp32/N=2^20/bs=2048",
        dtype_str="float32",
        N=2**20,
        block_size=2048,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp16/N=2^20/bs=4096",
        dtype_str="float16",
        N=2**20,
        block_size=4096,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp16/N=2^20/bs=8192",
        dtype_str="float16",
        N=2**20,
        block_size=8192,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/bf16/N=2^20/bs=8192",
        dtype_str="bfloat16",
        N=2**20,
        block_size=8192,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/fp32/N=2^20/bs=4096",
        dtype_str="float32",
        N=2**20,
        block_size=4096,
        tag="L1",
    )

    # Factory-level block_size compatibility surface: preset 2048 via
    # kwarg but apply 1024 at the curried stage (different values prove
    # the kwarg only presets default_block and never changes the return
    # shape -- DESIGN.md section 3.5.2).
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        ref = golden_expm1(x.cpu())
        out = expm1_fwd_kernel(N, dtype_str, block_size=2048)(1024)(x).cpu()
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print("[L1] PASS: factory block_size kwarg presets default only (preset=2048, applied=1024)")
    except AssertionError as exc:
        failures.append(("L1/factory-kwarg-preset", exc))
        print(f"[collected] L1/factory-kwarg-preset FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L1/factory-kwarg-preset", exc))
        print(f"[collected] L1/factory-kwarg-preset RUNTIME-ERROR: {exc!r}")

    # Single-stage guard re-check (fp32 route this time -- the transit
    # route was covered in L0; both dtype routes must reject (x) direct
    # calls).
    try:
        dtype_str = "float32"
        N = 8192
        torch_dtype = _TOLERANCE[dtype_str][0]
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        out = expm1_fwd_kernel(N, dtype_str, block_size=1024)(x)
        out.cpu()  # noqa: B018
        failures.append(
            ("L1/single-stage-fp32", AssertionError("single-stage (x) call unexpectedly succeeded"))
        )
        print("[collected] L1/single-stage-fp32 FAILED: single-stage (x) call unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        print(f"[L1] PASS: single-stage (x) direct call rejected on fp32 ({type(exc).__name__}: {exc})")

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme / non-power-of-two edge sizes (warn only, non-blocking)."""
    for N, dtype_str in (
        (2, "float16"),
        (3, "float32"),
        (2**22, "bfloat16"),
        (2**25, "float16"),
    ):
        try:
            _run_case(dtype_str=dtype_str, N=N, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking).

    DESIGN.md section 8.3 boundary list: all zeros (expm1(0)=+-0),
    exp underflow band (all-negative large -> -1.0), tiny-x cancellation
    scale (x ~1e-7), high-magnitude finite scale (x ~x10, near the
    overflow boundary but finite), single element, and a NaN/Inf/+-0
    mixed vector.
    """
    N = 4096

    # All zeros (expm1(0) = +-0; zero-sign difference exempted).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-zeros",
            x=torch.zeros(N, dtype=torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # Exp underflow band (all-negative large x -> expm1 ~ -1.0).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-allneg",
            x=torch.full((N,), -10.0, dtype=torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): allneg: {e}")

    # Tiny-x cancellation zone (atol dominates).
    try:
        _run_case(
            dtype_str="float32",
            tag="Boundary-tiny",
            x=(torch.randn(N, dtype=torch.float32) * 1e-7).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): tiny: {e}")

    # High-magnitude finite scale (x * 10; stays inside the overflow
    # boundary band and documents the expm1 output on large finite
    # inputs within the rtol envelope).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-large",
            x=(torch.randn(N, dtype=torch.float32) * 10.0).to(torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): large: {e}")

    # Single element (N=1, num_logical=1 degenerate).
    try:
        _run_case(
            dtype_str="float32",
            tag="Boundary-single",
            N=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): single: {e}")

    # Mixed special-value vector (NaN/Inf/+-0/subnormal/max-finite
    # through the tolerance envelope; duplicate of L0-5 on a different
    # random base).
    try:
        dtype_str = "float32"
        vals = torch.randn(N, dtype=torch.float32)
        vals[0] = float("inf")
        vals[1] = float("-inf")
        vals[2] = float("nan")
        vals[3] = 0.0
        vals[4] = -0.0
        vals[5] = float(torch.finfo(torch.float32).max)
        vals[6] = -float(torch.finfo(torch.float32).max)
        _run_case(
            dtype_str=dtype_str,
            tag="Boundary-mixed",
            x=vals.npu(),
            equal_nan=True,
            special_bits=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): mixed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}={v:.3e}" for k, v in sorted(_STATS_MAX_DIFF.items()))
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
