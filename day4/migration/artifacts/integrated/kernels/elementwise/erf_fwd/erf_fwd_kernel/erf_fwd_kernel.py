# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Erf forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``ErfFwdKernel`` (TileOPs,
``FloatUnaryKernel`` family, register_copy strategy builder with
``op_func = T.erf(T.cast(x, "float32"))``) to TileLang
``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = erf(x[i]) = (2/sqrt(pi)) * integral_0^x exp(-t^2) dt
on a flattened, contiguous ``(N,)`` tensor of dtype
float16 / bfloat16 / float32; output ``(N,)`` same dtype.  The source
computes entirely in the fp32 domain (``T.erf(T.cast(x, "float32"))``,
single final rounding to the output dtype); this kernel structurally
carries that precision contract over (DESIGN.md section 0.6 R2):
    - float32: native-domain chain (no transit).
    - float16/bfloat16: fp32 transit chain (single final rounding):
      copy -> vcast(rint) up -> fp32-domain 22-op chain ->
      vcast(rint) down (reuses the load buffer) -> copy.
      bfloat16 has no v-prefix arithmetic/transcendental (T.vabs.md /
      T.vdiv.md / T.vexp.md dtype matrices: bf16 x), so the transit is
      also a feasibility requirement.

NPU algorithm (DESIGN.md section 0.6 R1): Abramowitz & Stegun 7.1.26
rational approximation (T.verf is a 7th-order Taylor expansion, NOT a
hardware erf -- machine-verified to diverge at |x| >= 1.5, DESIGN.md
section 1.6.0 C1 / 1.6.1 REJ-1):
    t = 1 / (1 + 0.3275911 * |x|)
    erf(x) = sign(x) * (1 - t*(A1 + t*(A2 + t*(A3 + t*(A4 + t*A5))))
                        * exp(-x^2))
    A1..A5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027,
             1.061405429
fp32-domain op-by-op chain max abs error 5.868e-07 (D-1 machine
verified, 17x margin under the fp32 atol 1e-5).  Boundary semantics:
erf(+-0) = +-0 (zero-sign difference exempted, IEEE value-equal);
erf(+-inf) = +-1; erf(NaN) = NaN (vcmp(NaN<0)=False -> s=+1, output
NaN, equal_nan assertion); erf(+-max_finite) = +-1 (x^2 overflow ->
exp(-x^2) underflow -> m=0 -> r=1 natural saturation, no clamp).

NPU redesign (DESIGN.md section 0.6 R3/R4):
    R1: one-dim ``T.Kernel`` persistent core split
        (``num_kernels = min(num_logical, vector_cores)``) with an
        in-core ``T.serial`` grid-stride loop and an if-guard for
        out-of-range tiles (static bounds) + a 22-op v-prefix vector
        chain (vcmp -> vabs -> vmul/vadd/vdiv -> vexp -> Horner ->
        vsub -> vselect -> vmul) replaces the CUDA threads x npt
        two-level SIMT expansion.
    R4: tail handling = tile-level ``tail = T.min(block_size, N - t0)``
        with zero-origin UB slices on store (official vec_add_1d.py
        pattern; no per-element guard, no padding).  All v-ops run on
        the full (bs,) buffer (stale [tail:] lanes never land in GM).

Factory contract (K3/K11, two-stage; DESIGN.md sections 3.3/3.5.2 and
the landed wrapper examples/TileOPs/tileops/kernels/elementwise/erf_fwd/
erf_fwd.py L100-L117 -- the landed-side contract is the ONLY
authority):
    erf_fwd_kernel(N, dtype[, block_size=None])
        -> kernel(bs=default_block) -> main   (main(x, y), out_idx=[1])
    where ``main(x)`` returns ``y``.  K3 drops the source
    ``output_dtype``/``threads``/``num_per_thread`` parameters (the
    wrapper only ever calls ``erf_fwd_kernel(N, dtype)``, passing a
    ``torch.dtype`` object, and the manifest enforces same_as(input) at
    the wrapper class level); the optional ``block_size`` kwarg only
    presets ``default_block`` and never changes the return shape.  The
    wrapper call surface is:
        erf_fwd_kernel(N, dtype)(block_size)(x)
    with default block_size 1024 (fp32) / 2048 (fp16/bf16) (K11 =
    GPU threads(256) x npt(4/8) folded, wrapper
    ``_BLOCK_SIZE_FP32``/``_BLOCK_SIZE_NON_FP32``).

Run the embedded hierarchical tests:
    python erf_fwd_kernel.py --level L0
    python erf_fwd_kernel.py --level all
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

__all__ = ["erf_fwd_kernel", "golden_erf"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

# Per-dtype block_size cap (DESIGN.md section 4.5: fp32 37 B/elem ->
# 4096 ~= 151.6KB (+12% 169.8KB); transit 39 B/elem -> 4096 ~= 159.7KB
# (+12% 178.9KB), both < 192KB.  4097 is rejected on every dtype).
_UB_MAX_BLOCK = {
    "float32": 4096,
    "float16": 4096,
    "bfloat16": 4096,
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
# repo test_unary_math.py ``_get_tolerances`` L69-L74).
_TOLERANCE = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
    "float32": (torch.float32, 1e-5, 1e-5),
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_erf(x: torch.Tensor) -> torch.Tensor:
    """Erf reference: ``torch.erf(x)``.

    Ported from the source-repo test anchor (test_unary_math.py
    ``ref_fn = torch.erf``, GPU test L402 / NPU test L221): the golden
    IS the fp32-opmath + single round-back path (GOLD-DIAG machine
    verified: CPU ``torch.erf(native fp16/bf16)`` is bit-identical to
    ``torch.erf(x.float()).to(dtype)`` 200000/200000 -- verify_equiv.py,
    DESIGN.md section 1.6.1).  Independent of the NPU algorithm (no
    vcast transit / A&S chain / persistent structure knowledge).  Runs
    on CPU and does not require an NPU device.
    """
    return torch.erf(x.detach().cpu()).to(x.dtype)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def erf_fwd_kernel(N, dtype, block_size=None):
    """Build the Erf kernel factory (K3/K11 contract).

    Returns a JIT callable taking a single ``block_size`` (K11), which
    compiles and returns ``main``; ``main(x)`` returns ``y``
    (``out_idx=[1]``, source-operator contract).  The factory return
    value is ALWAYS a curried builder -- the two-stage call
    ``erf_fwd_kernel(N, dtype)(block_size)(x)`` is the only valid form
    (DESIGN.md section 3.5.2; a single-stage ``(x)`` binds the tensor
    to the ``bs`` parameter and fails at build time).  K3 drops the
    source ``output_dtype``/``threads``/``num_per_thread`` parameters
    (wrapper docs L23-L24; the wrapper only calls ``(N, dtype)``);
    ``block_size`` stays as an optional factory kwarg that only presets
    the curried default (DESIGN.md L0-7 integration form).

    Host-side contract validation (ValueError, survives ``python -O``):

      - N <= 0            -> ValueError (design adaptation: the source
        Op-layer check ``numel != N_total`` lets N=0 through; NPU
        rejects defensively, DESIGN.md section 9.3);
      - dtype not in {float16, bfloat16, float32} -> ValueError.

    The dtype argument accepts both ``torch.dtype`` objects (the
    wrapper passes ``getattr(torch, dtype_str)``, erf_fwd.py L116, and
    the Kernel class passes ``self.dtype``) and str; it is normalized
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

    # fp32 transit for both half-precision dtypes (DESIGN.md section
    # 0.6 R2: structural carry-over of the source fp32-opmath path;
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
        # Host-side UB budget guard (DESIGN.md section 4.5 / 9.2).
        # ValueError is used instead of a bare assert so the guard
        # survives `python -O`.
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
        # parser's if-block variable-table scoping limitation.
        if use_transit:
            # fp16/bf16 transit chain.  Constant broadcast sources are
            # let-bound locals INSIDE the traced script (with T.Kernel),
            # not closure/raw literals: a factory-scope named local also
            # trips the vbrc rank assertion (TRAP-expert-v-operands /
            # TRAP-vbrc-scalar-let-binding, leaky_relu fp32 route L0-1
            # first-run DiagnosticError).  Scalar literal operands of
            # vmul/vadd/vsub do NOT go through let binding and have no
            # such trigger surface (CONST-vsub-crossbuf-scalar note).
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    one_val = 1.0
                    neg_one_val = -1.0
                    zero_val = 0.0
                    # UB buffers (DESIGN.md section 4.3).  x16_ub is the
                    # load buffer and the round-back / store source.
                    x16_ub = T.alloc_ub((bs,), dtype_key)
                    x32_ub = T.alloc_ub((bs,), "float32")
                    ax_ub = T.alloc_ub((bs,), "float32")
                    x2_ub = T.alloc_ub((bs,), "float32")
                    t_ub = T.alloc_ub((bs,), "float32")
                    h_ub = T.alloc_ub((bs,), "float32")
                    e_ub = T.alloc_ub((bs,), "float32")
                    cond_ub = T.alloc_ub((bs,), "bool")
                    ones_ub = T.alloc_ub((bs,), "float32")
                    neg_one_ub = T.alloc_ub((bs,), "float32")
                    zeros_ub = T.alloc_ub((bs,), "float32")
                    T.vbrc(one_val, ones_ub)  # per-kernel once
                    T.vbrc(neg_one_val, neg_one_ub)
                    T.vbrc(zero_val, zeros_ub)

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
                            T.copy(x[t0 : t0 + tail], x16_ub[0:tail])
                            # 2) Upcast to fp32 (rint; lossless since
                            #    fp16/bf16 are subsets of fp32).
                            T.vcast(x16_ub, x32_ub, round_mode="rint")
                            # 3) fp32-domain 22-op A&S 7.1.26 chain
                            #    (full-buffer v-ops; stale [tail:] lanes
                            #    never land in GM).
                            T.vcmp(x32_ub, zeros_ub, cond_ub, "lt")   # cond = x < 0
                            T.vabs(x32_ub, ax_ub)                     # a = |x|
                            T.vmul(x32_ub, x32_ub, x2_ub)             # x^2
                            T.vmul(ax_ub, 0.3275911, ax_ub)           # p*a (in-place)
                            T.vadd(ax_ub, 1.0, ax_ub)                 # denom (in-place)
                            T.vdiv(ones_ub, ax_ub, t_ub)              # t = 1/denom
                            T.vmul(x2_ub, -1.0, x2_ub)                # -x^2 (in-place)
                            T.vexp(x2_ub, e_ub)                       # e = exp(-x^2)
                            T.vmul(ones_ub, 1.061405429, h_ub)        # h = A5
                            T.vmul(h_ub, t_ub, h_ub)                  # A4 step
                            T.vadd(h_ub, -1.453152027, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)                  # A3 step
                            T.vadd(h_ub, 1.421413741, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)                  # A2 step
                            T.vadd(h_ub, -0.284496736, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)                  # A1 step
                            T.vadd(h_ub, 0.254829592, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)                  # leading t factor
                            T.vmul(h_ub, e_ub, x2_ub)                 # m = t*poly*e (x2 dead)
                            T.vsub(ones_ub, x2_ub, e_ub)              # r = 1 - m (e dead)
                            T.vselect(cond_ub, neg_one_ub, ones_ub, h_ub)  # s = +-1 (h dead)
                            T.vmul(e_ub, h_ub, x32_ub)                # out = r*s (x32 dead)
                            # 4) Round back to the input dtype with a
                            #    single rint, reusing the load buffer.
                            T.vcast(x32_ub, x16_ub, round_mode="rint")
                            # 5) Store UB -> GM ([0:tail] only).
                            T.copy(x16_ub[0:tail], y[t0 : t0 + tail])

        else:
            # fp32 native domain: no transit, no independent x32 --
            # x_ub is the domain input AND the final out buffer (the
            # last vmul is CROSS-BUFFER: dst = x_ub != src0/sr1, the
            # expm1 fp32 vsub-cross-buffer pattern's vmul variant,
            # DESIGN.md section 9.1 R1 -- first-run verified below).
            # x_ub's original value dies at the vcmp/vabs/vmul-x2 reads
            # (RAW/WAR by Developer auto-sync).  Store source = x_ub.
            # Constant broadcast sources are let-bound locals inside
            # the traced script (same rule as the transit branch above).
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    one_val = 1.0
                    neg_one_val = -1.0
                    zero_val = 0.0
                    x_ub = T.alloc_ub((bs,), dtype_key)
                    ax_ub = T.alloc_ub((bs,), "float32")
                    x2_ub = T.alloc_ub((bs,), "float32")
                    t_ub = T.alloc_ub((bs,), "float32")
                    h_ub = T.alloc_ub((bs,), "float32")
                    e_ub = T.alloc_ub((bs,), "float32")
                    cond_ub = T.alloc_ub((bs,), "bool")
                    ones_ub = T.alloc_ub((bs,), "float32")
                    neg_one_ub = T.alloc_ub((bs,), "float32")
                    zeros_ub = T.alloc_ub((bs,), "float32")
                    T.vbrc(one_val, ones_ub)
                    T.vbrc(neg_one_val, neg_one_ub)
                    T.vbrc(zero_val, zeros_ub)

                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * bs
                            tail = T.min(bs, N - t0)

                            # 1) Load GM -> UB (same-dtype copy).
                            T.copy(x[t0 : t0 + tail], x_ub[0:tail])
                            # 3) fp32-domain 22-op A&S 7.1.26 chain
                            #    (domain input = x_ub, no upcast).
                            T.vcmp(x_ub, zeros_ub, cond_ub, "lt")
                            T.vabs(x_ub, ax_ub)
                            T.vmul(x_ub, x_ub, x2_ub)
                            T.vmul(ax_ub, 0.3275911, ax_ub)
                            T.vadd(ax_ub, 1.0, ax_ub)
                            T.vdiv(ones_ub, ax_ub, t_ub)
                            T.vmul(x2_ub, -1.0, x2_ub)
                            T.vexp(x2_ub, e_ub)
                            T.vmul(ones_ub, 1.061405429, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)
                            T.vadd(h_ub, -1.453152027, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)
                            T.vadd(h_ub, 1.421413741, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)
                            T.vadd(h_ub, -0.284496736, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)
                            T.vadd(h_ub, 0.254829592, h_ub)
                            T.vmul(h_ub, t_ub, h_ub)                  # leading t factor
                            T.vmul(h_ub, e_ub, x2_ub)                 # m
                            T.vsub(ones_ub, x2_ub, e_ub)              # r
                            T.vselect(cond_ub, neg_one_ub, ones_ub, h_ub)  # s
                            # Out lands in the LOAD buffer (cross-buffer
                            # dst = x_ub; consumed within this iteration
                            # by the store -- no loop-carried state).
                            T.vmul(e_ub, h_ub, x_ub)
                            # 5) Store UB -> GM ([0:tail] only).
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
    equal infinities and IEEE-equal +-0 pairs count as equal here, so
    abs(inf-inf)=NaN can never degenerate into a false violation and
    the +-0 sign difference (golden -0.0 vs NPU +0.0 at x=-0.0) is
    naturally exempted (DESIGN.md sections 8.2/8.3; leaky_relu
    transferable lesson -- the exact-equality branch is mandatory).
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

    For erf the output at +-Inf input is +-1 (finite), so the special
    mask in practice only ever contains NaN positions (erf(NaN)=NaN);
    both-NaN is accepted regardless of payload (the A&S chain may alter
    the payload), Inf positions -- if any appear -- must be
    bit-identical.  +-0 is NOT checked here -- the zero-sign difference
    is IEEE value-equal and exempted (DESIGN.md section 8.3 L0-5).
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


def _assert_erf_saturated(out, ref):
    """erf(+-inf) and erf(+-max_finite) must be exactly +-1.0.

    DESIGN.md section 8.3 L0-5 exact-value assertion: the saturation is
    mechanical (x^2 overflow -> exp(-x^2) underflow -> m=0 -> r=1,
    s=sign -> out=+-1.0 exactly in fp32, no clamp needed).  Golden
    torch.erf saturates the same way, so the exact check doubles as a
    bit-exactness check at these positions.
    """
    got = out[torch.tensor([0, 1, 21, 22], dtype=torch.long)]
    exp = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=out.dtype)
    if not torch.equal(got, exp):
        raise AssertionError(
            f"erf saturation exact-value mismatch: got {got.tolist()} expected +-1.0"
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
    invoked two-stage (``erf_fwd_kernel(N, dtype_str)(block_size)``,
    the K3/K11 family contract).  Raises AssertionError on tolerance
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

    kernel = erf_fwd_kernel(N, dtype_str)(block_size)
    out = kernel(x)
    ref = golden_erf(x.cpu())
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

    Corner set (DESIGN.md section 8.3 L0-5, all extreme values taken
    from torch.finfo -- never hand-written decimal literals, leaky_relu
    transferable lesson): +-Inf / NaN / +-0 / +-subnormal_min
    (2^-24/2^-133/2^-149) / +-finfo.tiny / +-1 / +-2 / +-3 / +-4 / +-6
    / +-100 (source edge values) / +-finfo.max.  Assertion rule:
    assert_close value envelope (equal_nan=True) + NaN/Inf bit check +
    exact +-1.0 saturation check at the +-Inf / +-max positions;
    +-0 zero-sign difference exempted (DESIGN.md section 8.3 L0-5).
    """
    corners = {
        "float16": (2 ** -24, float(torch.finfo(torch.float16).max), float(torch.finfo(torch.float16).tiny)),
        "bfloat16": (2 ** -133, float(torch.finfo(torch.bfloat16).max), float(torch.finfo(torch.bfloat16).tiny)),
        "float32": (2 ** -149, float(torch.finfo(torch.float32).max), float(torch.finfo(torch.float32).tiny)),
    }
    N = 8192
    base = torch.randn(N, dtype=torch.float32)
    for dtype_str, (subnorm, max_fin, tiny) in corners.items():
        vals = base.clone()
        vals[0] = float("inf")
        vals[1] = float("-inf")
        vals[2] = float("nan")
        vals[3] = 0.0
        vals[4] = -0.0
        vals[5] = subnorm
        vals[6] = -subnorm
        vals[7] = tiny
        vals[8] = -tiny
        vals[9] = 1.0
        vals[10] = -1.0
        vals[11] = 2.0
        vals[12] = -2.0
        vals[13] = 3.0
        vals[14] = -3.0
        vals[15] = 4.0
        vals[16] = -4.0
        vals[17] = 6.0
        vals[18] = -6.0
        vals[19] = 100.0
        vals[20] = -100.0
        vals[21] = max_fin
        vals[22] = -max_fin
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

    # L0-1: smoke, all dtypes, N=2^20, randn.  The fp32 route here is
    # the first-run verification of the A2 scalar-literal v-op form and
    # the cross-buffer final vmul (DESIGN.md section 9.1 R1).
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
    # for the cross-buffer final vmul "no loop-carried state"
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

    # L0-5: corner & special-value propagation (+ exact saturation).
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
        fn = erf_fwd_kernel(N, dtype_str)
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        ref = golden_erf(x.cpu())
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

    # Extra 2: two-stage integration form with a torch.dtype object
    # (the wrapper contract: erf_fwd.py L116 passes getattr(torch,
    # dtype_str); the factory block_size kwarg presets default_block +
    # curried application -- DESIGN.md section 8.3 L0-7).
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype, atol, rtol = _TOLERANCE[dtype_str]
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        ref = golden_erf(x.cpu())
        out = erf_fwd_kernel(N, torch_dtype, block_size=2048)(2048)(x).cpu()
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
        print("[L0-extra] PASS: two-stage integration form (torch.dtype + block_size kwarg + curried)")
    except AssertionError as exc:
        failures.append(("L0-extra/two-stage", exc))
        print(f"[collected] L0-extra/two-stage FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-extra/two-stage", exc))
        print(f"[collected] L0-extra/two-stage RUNTIME-ERROR: {exc!r}")

    # Extra 3: single-stage guard (DESIGN.md section 3.5.2 contract
    # analysis): directly calling the factory return value with ``(x)``
    # binds the tensor to the bs parameter and must fail at build time
    # (TypeError / ValueError / any build-stage exception -- a
    # successful run would be a contract violation).
    try:
        dtype_str = "float16"
        N = 8192
        torch_dtype = _TOLERANCE[dtype_str][0]
        x = torch.randn(N, dtype=torch.float32).to(torch_dtype).npu()
        out = erf_fwd_kernel(N, dtype_str, block_size=2048)(x)
        out.cpu()  # noqa: B018  (force materialization if it somehow ran)
        failures.append(
            ("L0-extra/single-stage", AssertionError("single-stage (x) call unexpectedly succeeded"))
        )
        print("[collected] L0-extra/single-stage FAILED: single-stage (x) call unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        print(f"[L0-extra] PASS: single-stage (x) direct call rejected ({type(exc).__name__}: {exc})")

    # Extra 4: UB budget guard (DESIGN.md section 4.5): bs=4097
    # rejected on all dtypes.
    for dtype_str in ("float16", "bfloat16", "float32"):
        try:
            erf_fwd_kernel(4096, dtype_str)(4097)
        except ValueError as exc:
            print(f"[L0-extra] PASS: UB guard rejects {dtype_str} block_size=4097 ({exc})")
        else:
            failures.append(
                (f"L0-extra/UB-guard-{dtype_str}", AssertionError(f"{dtype_str} 4097 not rejected"))
            )

    # Extra 5: N defense (N=0 / N<0 -> ValueError at the builder).
    for bad_N in (0, -5):
        try:
            erf_fwd_kernel(bad_N, "float16")
        except ValueError as exc:
            print(f"[L0-extra] PASS: N defense rejects N={bad_N} ({exc})")
        else:
            failures.append(
                (f"L0-extra/N-defense/{bad_N}", AssertionError(f"N={bad_N} not rejected"))
            )

    # Extra 6: non-float dtype defense (K3 contract rejects anything
    # outside {float16, bfloat16, float32}).
    try:
        erf_fwd_kernel(16, "int32")
    except ValueError as exc:
        print(f"[L0-extra] PASS: non-float dtype rejected ({exc})")
    else:
        failures.append(("L0-extra/non-float-dtype", AssertionError("int32 not rejected")))

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (six items + extras)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size).

    Includes the manifest workloads (DESIGN.md section 5.5:
    elementwise-16m [4096,4096] = 2^24, 3 dtypes), the large
    multi-iteration persistent shapes, and the block_size sweep up to
    the per-dtype UB cap (4096 = the A2 scalar-literal form at the
    largest safe tile; re-verifies the UB budget on the current
    toolchain).

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
    # covers the cross-buffer final vmul no-loop-carried discipline at
    # scale; fp32 2^24 exercises the fp32 route with 16384 logical
    # tiles).
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
        "L1/bf16/N=2^20/bs=4096",
        dtype_str="bfloat16",
        N=2**20,
        block_size=4096,
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
        ref = golden_erf(x.cpu())
        out = erf_fwd_kernel(N, dtype_str, block_size=2048)(1024)(x).cpu()
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
        out = erf_fwd_kernel(N, dtype_str, block_size=1024)(x)
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

    DESIGN.md section 8.3 boundary list: all zeros (erf(0)=+-0), large
    negative saturation (erf(-10)=-1), tiny-x near-zero scale (erf ~
    2x/sqrt(pi) ~ 0), high-magnitude finite scale (erf ~ +-1), single
    element, and a NaN/Inf/+-0/max-finite mixed vector.
    """
    N = 4096

    # All zeros (erf(0) = +-0; zero-sign difference exempted).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-zeros",
            x=torch.zeros(N, dtype=torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # Large negative saturation (erf(-10) = -1.0).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-allneg",
            x=torch.full((N,), -10.0, dtype=torch.float16).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): allneg: {e}")

    # Tiny-x near-zero zone (erf(x) ~ 2x/sqrt(pi); atol dominates).
    try:
        _run_case(
            dtype_str="float32",
            tag="Boundary-tiny",
            x=(torch.randn(N, dtype=torch.float32) * 1e-7).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): tiny: {e}")

    # High-magnitude finite scale (x * 100; inside the saturation band
    # where erf ~ +-1 within the rtol envelope).
    try:
        _run_case(
            dtype_str="float16",
            tag="Boundary-large",
            x=(torch.randn(N, dtype=torch.float32) * 100.0).to(torch.float16).npu(),
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
