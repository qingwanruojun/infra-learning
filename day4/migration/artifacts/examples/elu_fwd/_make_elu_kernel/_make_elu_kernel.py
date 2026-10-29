# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""ELU forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``_make_elu_kernel`` (TileOPs, Pattern-B
factory closure) to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = x[i]                     if x[i] >  0   (strictly greater)
    y[i] = alpha * (exp(x[i]) - 1)  if x[i] <= 0
on a flattened, contiguous ``(N,)`` tensor of dtype
float16 / bfloat16 / float32; output ``(N,)`` same dtype.
NaN/Inf propagation follows IEEE (NaN > 0 is false -> negative branch
alpha*(exp(NaN)-1) = NaN; +Inf -> +Inf; -Inf -> -alpha).  The
comparison lives in the fp32 domain; at x = -0.0 the source negative
branch yields +0.0 while the ATen golden identity branch yields -0.0
(IEEE-equal, zero-sign difference exempted, DESIGN.md section 8.3).

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` persistent core split
        (``num_kernels = min(num_logical, vector_cores)``) with an
        in-core ``T.serial`` grid-stride loop and an if-guard for
        out-of-range tiles (static bounds) + a v-prefix vector chain
        (vcmp -> vexp -> vsub -> vmul -> vselect) replaces the CUDA
        threads x npt two-level SIMT expansion.
    R2: fp16/bf16 go through a fp32 transit chain
        (``T.vcast(rint)`` up -> fp32-domain vcmp/vexp/vsub/vmul/vselect
        -> ``T.vcast(rint)`` back, single final rounding).  The source
        itself computes the negative branch in fp32 with a single
        round-back (structural carry-over, not a fix): bfloat16 has no
        v-prefix arithmetic/transcendental (T.vexp/vmul/vsub bf16 x)
        and fp16 transits to match the golden's fp32-opmath path
        bit-exactly (DESIGN.md section 1.6.1 A1, machine-verified
        0 ulp).  fp32 computes in the native dtype domain (no transit,
        vselect Out=A in-place write-back to the load buffer).
    R3: tail handling = tile-level ``tail = T.min(block_size, N - t0)``
        with zero-origin UB slices on store (official vec_add_1d.py
        pattern; no per-element guard, no padding).

Factory contract (wrapper E2/K11, see
examples/TileOPs/tileops/kernels/elementwise/elu_fwd/elu_fwd.py):
    _make_elu_kernel(N, dtype, alpha) -> kernel(block_size) -> main
    where ``main(x)`` returns ``y`` (``out_idx=[1]``); the outer
    signature keeps the source 7-arg form (``output_dtype``/``is_fp8``/
    ``threads``/``npt`` retained for signature compatibility but not
    consumed on NPU; K11 collapses ``threads * npt`` into the single
    ``block_size`` parameter, default fp32 1024 / fp16.bf16 2048).
    ``dtype`` accepts both torch.dtype objects (the wrapper ``__init__``
    passes ``self.dtype`` and the custom_op chain resolves
    ``getattr(torch, dtype_str)``) and str keys; it is normalized to
    ``dtype_key`` before any branch/guard/default selection.

Run the embedded hierarchical tests:
    python _make_elu_kernel.py --level L0
    python _make_elu_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no
# manual sync points (DESIGN.md section 7); the module-level setdefault
# keeps Expert/Developer mixed-process runs isolated.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

ASCEND_MODE = "Developer"

import argparse
import functools
import math
import time

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["_make_elu_kernel", "golden_elu"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

# Conservative block_size cap for all dtypes (DESIGN.md section 4.5:
# fp32 17 B/elem -> 11565, fp16/bf16 19 B/elem -> 10347; resident
# footprint equals the leaky_relu sibling (17/19 B/elem) whose L1 suite
# passes bs=8192 on this toolchain, so the guard stays at the shared
# conservative 8192 -- default 1024/2048 is safe on every accounting
# model, including the naive x2 double-buffer one).
_UB_MAX_BLOCK = 8192

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
# repo test_elu.py ``_get_tolerances``).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
    "float32": (torch.float32, 1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU reference: ``F.elu(x.float(), alpha).to(x.dtype)``.

    Ported from the source-repo test anchor (test_elu.py ``ref_program``
    L40): fp32 opmath + single round-back to the input dtype, exactly
    the golden path DESIGN.md section 8.1 freezes.  Independent of the
    NPU algorithm (no vcast transit / vcmp/vexp/vselect / persistent
    structure).  Runs on CPU and does not require an NPU device.
    """
    return F.elu(x.detach().float(), alpha).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _make_elu_kernel(
    N,
    dtype,
    alpha,
    output_dtype=None,
    is_fp8=False,
    threads=256,
    npt=8,
):
    """Build the ELU kernel factory (source 7-arg signature kept).

    Returns a JIT callable taking a single ``block_size`` (E2/K11),
    which compiles and returns ``main``; ``main(x)`` returns ``y``
    (``out_idx=[1]``, source-operator contract).  Host-side contract
    validation (ValueError, survives ``python -O``):

      - N <= 0            -> ValueError (design adaptation: the source
        Op-layer check ``numel != N_total`` lets N=0 through; NPU
        rejects defensively, DESIGN.md section 9.3);
      - is_fp8=True       -> ValueError (dead code on NPU: the source
        base class rejects fp8 before the builder, DESIGN.md section
        0.5 #5);
      - alpha non-finite  -> ValueError (carries the source host check
        semantics, test_elu_rejects_nonfinite_alpha);
      - output_dtype not None and != dtype -> ValueError (manifest
        same_as(input));
      - dtype not in {float16, bfloat16, float32} -> ValueError.

    The dtype argument accepts both ``torch.dtype`` objects (the
    wrapper ``__init__`` passes ``self.dtype`` and the custom_op chain
    resolves ``getattr(torch, dtype_str)``) and str; it is normalized
    to ``dtype_key`` before any branch/guard/default selection
    (comparing a torch.dtype object against a str tuple is always
    False and would mis-route fp16/bf16 to the fp32 path). NOTE:
    torch.dtype has no ``.name`` attribute in torch 2.7.1 --
    ``str(torch.float16) == "torch.float16"``, so the key is derived
    by stripping the ``"torch."`` prefix.
    """
    if N <= 0:
        raise ValueError(f"N must be a positive int, got N={N}")
    if is_fp8:
        raise ValueError(
            "is_fp8=True is not supported on NPU (the source base class "
            "rejects fp8 dtypes before the builder; manifest contract is "
            "float16 | bfloat16 | float32)"
        )
    if not math.isfinite(float(alpha)):
        raise ValueError(f"alpha must be finite, got alpha={alpha!r}")
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
    # 0.6 R2, PL-1.17): bf16 has no Vector arithmetic/transcendental;
    # fp16 transits to match the golden's fp32-opmath rounding path
    # (bit-exact per DESIGN.md section 1.6.1 A1).
    use_transit = dtype_key in ("float16", "bfloat16")
    default_block = _DEFAULT_BLOCK[dtype_key]

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op). Re-queried per factory call so deployments on
    # other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size=default_block):
        # Host-side UB budget guard (DESIGN.md section 4.5 / 9.2).
        # ValueError is used instead of a bare assert so the guard
        # survives `python -O`.
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if block_size > _UB_MAX_BLOCK:
            raise ValueError(
                f"block_size={block_size} exceeds the UB budget for "
                f"{dtype_key} (max {_UB_MAX_BLOCK} elements, "
                f"{_UB_CAPACITY_BYTES} B UB capacity)"
            )

        # Persistent core split (DESIGN.md section 5.5). All quantities
        # are compile-time constants: N and block_size are Python ints
        # and T.ceildiv/min fold to IntImm at trace time.
        num_logical = T.ceildiv(N, block_size)  # logical tile count
        num_kernels = min(num_logical, vector_cores)  # <= physical cores
        num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

        # Compile-time dtype dispatch (DESIGN.md section 3.3 note): the
        # fp32-transit branch is selected inside the factory. Keeping
        # the branch outside the traced body also avoids the TVM-script
        # parser's if-block variable-table scoping limitation (C11).
        if use_transit:
            # Constant broadcast sources must be let-bound locals inside
            # the traced script, not closure/raw literals: _get_extent()
            # returns an empty rank for a plain Python scalar and the
            # vbrc rank check then fails (TRAP-vbrc-scalar-let-binding;
            # factory-scope named locals trigger the same diagnostic --
            # leaky_relu first-run lesson, both branches audited
            # separately).
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    zero_val = 0.0
                    alpha_val = float(alpha)
                    # UB buffers (DESIGN.md section 4.3). x_ub is the
                    # load buffer and the round-back / store source.
                    x_ub = T.alloc_ub((block_size,), dtype_key)
                    x32 = T.alloc_ub((block_size,), "float32")
                    w32 = T.alloc_ub((block_size,), "float32")
                    cond = T.alloc_ub((block_size,), "bool")
                    zero32 = T.alloc_ub((block_size,), "float32")
                    alpha32 = T.alloc_ub((block_size,), "float32")
                    T.vbrc(zero_val, zero32)  # per-kernel once
                    T.vbrc(alpha_val, alpha32)  # alpha in fp32 domain

                    # Persistent in-core serial task loop (static bound,
                    # grid-stride mapping, if-guard masks out-of-range
                    # tiles).
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * block_size
                            # Tail handling (DESIGN.md section 6.4):
                            # last tile may be shorter; dst slices are
                            # always zero-origin (UB alignment pattern).
                            tail = T.min(block_size, N - t0)

                            # 1) Load GM -> UB (same-dtype copy).
                            T.copy(x[t0 : t0 + tail], x_ub[0:tail])
                            # 2) Upcast to fp32 (rint; lossless since
                            #    fp16/bf16 are subsets of fp32).
                            T.vcast(x_ub, x32, round_mode="rint")
                            # 3) Compare strictly greater than zero in
                            #    the fp32 domain.
                            T.vcmp(x32, zero32, cond, "gt")
                            # 4) exp(x32) -- full-buffer; overflow Inf
                            #    on the discarded positive lanes is
                            #    harmless (lane-independent, masked by
                            #    the select).
                            T.vexp(x32, w32)
                            # 5) w32 = w32 - 1.0 (scalar literal
                            #    operand, in-place; mish L78 precedent).
                            T.vsub(w32, 1.0, w32)
                            # 6) w32 = alpha32 * w32 (in-place).
                            T.vmul(alpha32, w32, w32)
                            # 7) Select: Out=B in-place (result stays
                            #    in w32, consumed within this iteration
                            #    by 8); no loop-carried state).
                            T.vselect(cond, x32, w32, w32)
                            # 8) Round back to the input dtype with a
                            #    single rint, reusing the load buffer.
                            T.vcast(w32, x_ub, round_mode="rint")
                            # 9) Store UB -> GM ([0:tail] only; stale
                            #    [tail:] lanes never land in GM).
                            T.copy(x_ub[0:tail], y[t0 : t0 + tail])

        else:
            # fp32 native domain: no transit, no independent x32 --
            # x_ub is the domain input and the select output (Out=A
            # in-place write-back to the load buffer, gqa_prefill
            # L875 precedent). Store source is x_ub on both routes.
            # Constant broadcast sources are let-bound locals inside
            # the traced script (same rule as the transit branch above;
            # factory-scope scalars fail the vbrc rank check).
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    zero_val = 0.0
                    alpha_val = float(alpha)
                    x_ub = T.alloc_ub((block_size,), dtype_key)
                    w32 = T.alloc_ub((block_size,), "float32")
                    cond = T.alloc_ub((block_size,), "bool")
                    zero32 = T.alloc_ub((block_size,), "float32")
                    alpha32 = T.alloc_ub((block_size,), "float32")
                    T.vbrc(zero_val, zero32)
                    T.vbrc(alpha_val, alpha32)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * block_size
                            tail = T.min(block_size, N - t0)

                            T.copy(x[t0 : t0 + tail], x_ub[0:tail])
                            T.vcmp(x_ub, zero32, cond, "gt")
                            T.vexp(x_ub, w32)
                            T.vsub(w32, 1.0, w32)
                            T.vmul(alpha32, w32, w32)
                            # Out=A in-place: result written back into
                            # the load buffer (gqa_prefill L875).
                            T.vselect(cond, x_ub, w32, x_ub)
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

    NaN/Inf aware: positions where both are NaN count as equal when
    equal_nan is set; equal infinities count as equal (IEEE sign bit of
    +-0 is not distinguished by the numeric comparison -- DESIGN.md
    section 8.3 L0-5 asserts +-0 by value equivalence).
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
    """NaN/Inf positions must propagate bit-exactly (Inf sign included).

    Finite values are left to the assert_close tolerance envelope (the
    NPU vexp hardware may differ from the golden's exp by <=1 ulp in the
    fp32 domain, DESIGN.md section 9.1#1); NaN positions are accepted
    when both sides are NaN regardless of payload (the exp chain may
    alter the payload); +-0 zero-sign difference is exempt (DESIGN.md
    section 8.3 L0-5 -- the source x>0 condition and ATen x<0 land
    -0.0 on different branches, IEEE-equal).
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
    alpha=1.0,
    equal_nan=False,
    special_bits=False,
):
    """Run one kernel case and compare against the golden.

    Tensor x may be pre-built (endpoint / special-value cases); by
    default ``randn`` per DESIGN.md section 8.3 L0-1.
    Raises AssertionError on tolerance violation.
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

    kernel = _make_elu_kernel(N, dtype_str, alpha)(block_size)
    out = kernel(x)
    ref = golden_elu(x.cpu(), alpha)
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
            f"alpha={alpha} max_diff={max_diff:.3e} violations={violations}/{N}"
        )
        raise
    print(
        f"[{tag}] PASS: N={N} dtype={dtype_str} block_size={block_size} "
        f"alpha={alpha} max_diff={max_diff:.3e}"
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
    literals) / +-1.0 / exp overflow lanes (+-89, positive-axis Inf is
    discarded by the select) / exp underflow band (-104, -144) /
    tiny-x cancellation zone (+-1e-8, +-1e-7, +-2^-24); run with the
    default alpha (1.0) and a non-default alpha (0.5).  Assertion rule:
    assert_close value envelope (equal_nan) + NaN/Inf bit check, +-0
    zero-sign difference exempted (DESIGN.md section 8.3 L0-5).
    """
    # Subnormal minima per dtype (fp16 2^-24, bf16 2^-133, fp32 2^-149).
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
        vals[11] = 89.0
        vals[12] = -89.0
        vals[13] = -104.0
        vals[14] = -144.0
        vals[15] = 1e-8
        vals[16] = -1e-8
        vals[17] = 1e-7
        vals[18] = -1e-7
        vals[19] = 2 ** -24
        vals[20] = -(2 ** -24)
        x = vals.to(_TOLERANCE[dtype_str][0]).npu()
        for alpha in (1.0, 0.5):
            _try_case(
                failures,
                f"L0-5/{dtype_str}/corners-alpha{alpha}",
                dtype_str=dtype_str,
                tag="L0-5",
                x=x,
                alpha=alpha,
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
    # bf16 at the masking boundary (vc+1) to cover the vcast path under
    # the if-guard + serial loop.
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
    # for the in-place vselect/vsub/vmul "no loop-carried state"
    # discipline).
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

    # Extra: E2 dual-config reuse -- one factory callable serving two
    # block_sizes (the jit wrapper caches the compiled main per
    # block_size).
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        fn = _make_elu_kernel(N, dtype_str, 1.0)
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        ref = golden_elu(x.cpu(), 1.0)
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

    # Extra: UB budget guard (block_size=8193 rejected on all dtypes).
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            _make_elu_kernel(4096, dtype_str, 1.0)(8193)
        except ValueError as exc:
            print(f"[L0-extra] PASS: UB guard rejects {dtype_str} block_size=8193 ({exc})")
        else:
            failures.append(
                (f"L0-extra/UB-guard-{dtype_str}", AssertionError(f"{dtype_str} 8193 not rejected"))
            )

    # Extra: N defense (N=0 / N<0 -> ValueError at the builder).
    for bad_N in (0, -5):
        try:
            _make_elu_kernel(bad_N, "float16", 1.0)
        except ValueError as exc:
            print(f"[L0-extra] PASS: N defense rejects N={bad_N} ({exc})")
        else:
            failures.append(
                (f"L0-extra/N-defense/{bad_N}", AssertionError(f"N={bad_N} not rejected"))
            )

    # Extra: alpha defense (inf / NaN -> ValueError at the builder,
    # carries test_elu_rejects_nonfinite_alpha semantics).
    for bad_alpha in (float("inf"), float("nan")):
        try:
            _make_elu_kernel(16, "float32", bad_alpha)
        except ValueError as exc:
            print(f"[L0-extra] PASS: alpha defense rejects alpha={bad_alpha} ({exc})")
        else:
            failures.append(
                (f"L0-extra/alpha-defense/{bad_alpha}", AssertionError(f"alpha={bad_alpha} not rejected"))
            )

    # Extra: is_fp8 rejection (dead code on NPU, DESIGN.md 0.5 #5).
    try:
        _make_elu_kernel(16, "float16", 1.0, is_fp8=True)
    except ValueError as exc:
        print(f"[L0-extra] PASS: is_fp8=True rejected ({exc})")
    else:
        failures.append(("L0-extra/is_fp8", AssertionError("is_fp8=True not rejected")))

    # Extra: output_dtype mismatch rejected (manifest same_as(input)).
    try:
        _make_elu_kernel(16, "float16", 1.0, output_dtype="float32")
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
    mlp-hidden [2048,4096]=2^23 / mlp-hidden-wide [2048,8192]=2^24,
    fp16+bf16) and the large block_size sweep (the transit 8192 cases
    sit exactly at the UB cap -- sibling lerp_tensor L1 passes bs=8192
    on its toolchain; they double as the Stage-3 re-verification of the
    stale TRAP-UB-multibuffer entry on the current toolchain).

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

    # Non-default block_size sweep (wrapper config space).
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
    """Boundary: empty-guard / single-element / large N / degenerate
    inputs (warn only, non-blocking)."""
    N = 4096

    # All zeros (negative branch = alpha*(exp(0)-1) = +-0).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-zeros",
            x=torch.zeros(N, dtype=torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # Single element (N=1, num_logical=1 degenerate).
    try:
        _run_case(
            dtype_str="float32",
            tag="Boundary-single",
            N=1,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): single: {e}")

    # Large N (2^25 fp32 = 33.5M elements; multi-hundred serial tasks).
    try:
        _run_case(
            dtype_str="float32",
            tag="Boundary-largeN",
            N=2**25,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): largeN: {e}")

    # alpha=0.0: degenerate to sel(x>0, x, 0) -- negative branch all
    # +-0, positive branch identity (no special-casing in the chain).
    try:
        _run_case(
            dtype_str="float32",
            tag="Boundary-alpha0",
            x=torch.randn(N, dtype=torch.float32).npu(),
            alpha=0.0,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): alpha=0: {e}")

    # All-negative fill (exp underflow floor: alpha*(0-1) = -alpha).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-allneg",
            x=torch.full((N,), -10.0, dtype=torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): allneg: {e}")


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
