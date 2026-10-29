"""verify_equiv.py -- DESIGN.md section 1.6.1 equivalence machine verification (D-1).

Operator: ELU forward (y = x if x > 0 else alpha * (exp(x) - 1), alpha default 1.0).
Source formula taken verbatim from the extracted GPU kernel
(``examples/TileOPs/tileops/kernels/elementwise/elu_fwd/_elu_fwd_kernels.py``
L48-L56, non-fp8 branch):
    zero = 0 (fp32); a = alpha (fp32); one = 1.0 (fp32); v32 = fp32(val)
    y = if_then_else(v32 > zero, val, cast(val.dtype, a * (exp(v32) - one)))
  i.e. the negative branch is computed ENTIRELY in the fp32 domain with a
  single round-back, and the positive branch passes ``val`` through
  unchanged (the source already implements fp32 opmath + one rounding).

Candidates (DESIGN.md section 0.6 R2 / section 1.6.1):
  A1 (adopted, fp32-transit dtype path): NPU chain
      x32 = rint(f32 <- x)            (lossless upcast)
      cond = x32 > 0                  (fp32 domain compare)
      w32 = exp(x32); w32 = w32 - 1.0; w32 = alpha32 * w32
      y32 = select(cond, x32, w32)    (select-then-round)
      y  = rint(y32 -> dtype)         (single round-back)
      vs baseline (source rounding path, round-then-select) and vs golden.
  A3 (adopted, alpha==1.0 specialization): drop the vmul
      (1.0 * v is an IEEE identity -- bit-exact); chain becomes
      w32 = exp(x32) - 1.0 with NO multiply; verified per dtype.
Golden anchor: torch.nn.functional.elu(x.float(), alpha).to(x.dtype)
(the source-repo test ref_program path, test_elu.py L40: fp32 opmath +
single round-back).

Rejected candidates (machine evidence, NOT adopted):
  A2 expm1 form  y = alpha * expm1(x): mathematically equal, but
     (a) no vexpm1 exists in docs/Tilelang.language/ (all 16
     subdirectories enumerated, grep 'expm1' zero hits), and
     (b) it diverges from the golden's straight exp(x)-1 rounding path
     at tiny |x| (shown by nonzero mismatch count).
  A4 max form    y = max(x, alpha*(exp(x)-1)): structurally impossible --
     exp(x) > 1 + x strictly for all x != 0, so the max picks the
     negative-branch expression on the POSITIVE axis too (mismatch shown).
  A5 refactored form  y = alpha*exp(x) - alpha: two roundings instead of
     one for general alpha (diverges from the golden form; shown at
     alpha=0.1; buffer-motivation is void since the 1.0 constant rides
     as a vsub scalar operand, T.vsub.md section 2.1 + mish.py L78
     scalar-operand precedent).

Checks
   * fp64-anchored rounding-path comparison at fp16/bf16/fp32,
   * random sign-balanced data widened across binades + corner values
     (+-Inf/NaN/+-0/subnormal/max-finite via torch.finfo/tiny-x region
     around the exp(x)-1 cancellation zone / fp32-exp underflow zone),
   * verdict EQUIV_PASS iff candidate-vs-golden violations == 0 (source
     test tolerances) AND candidate ulp not worse than baseline.

All computation on torch CPU (no NPU required).
Run:  python verify_equiv.py
"""

from __future__ import annotations

import zlib

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(20261028)
np.random.seed(20261028)

# Source-repo test tolerances (examples/TileOPs/tests/ops/test_elu.py
# _get_tolerances): fp32 (1e-5, 1e-5), fp16 (1e-3, 1e-3), bf16 (1.6e-2, 1.6e-2).
_TOL = {
    "float32": (1e-5, 1e-5),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
}
_DT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
# fp32 exp underflow zone: exp(x) == 0 in fp32 for x <= ~-103.98; the
# transition band (last representable exp, first zero exp) is a corner.
_EXP32_UNDERFLOW_X = -104.0
# Tiny-x region where exp(x)-1 cancellation is worst (|x| << 1).
_TINY_X = (1e-8, -1e-8, 1e-7, -1e-7, 2**-24, -(2.0**-24))


# ---------------------------------------------------------------------------
# Formula implementations (torch CPU)
# ---------------------------------------------------------------------------
def golden(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Source-repo test anchor: F.elu(x.float(), alpha).to(x.dtype)."""
    return F.elu(x.detach().float(), alpha).to(x.dtype)


def baseline_source(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Source kernel rounding path, simulated exactly (round-then-select):
    cond in fp32 (v32 > 0), positive branch = native val passthrough,
    negative branch computed in fp32 then cast once to val.dtype."""
    dt = x.dtype
    x32 = x.float()
    alpha32 = torch.tensor(np.float32(alpha), dtype=torch.float32)
    one32 = torch.tensor(1.0, dtype=torch.float32)
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    neg32 = alpha32 * (torch.exp(x32) - one32)
    return torch.where(x32 > zero32, x, neg32.to(dt))


def candidate_transit(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """DESIGN.md A1 NPU chain simulated (select-then-round): fp32-domain
    compare/mul/sub/select with f32(alpha), single final round-back."""
    dt = x.dtype
    x32 = x.float()
    alpha32 = torch.tensor(np.float32(alpha), dtype=torch.float32)
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    w32 = torch.exp(x32)
    w32 = w32 - torch.tensor(1.0, dtype=torch.float32)
    w32 = alpha32 * w32
    y32 = torch.where(x32 > zero32, x32, w32)
    return y32.to(dt)


def candidate_alpha1_specialized(x: torch.Tensor) -> torch.Tensor:
    """DESIGN.md A3 chain for alpha == 1.0: the vmul is dropped entirely
    (1.0 * v is an IEEE 754 identity for every finite/inf/nan value)."""
    dt = x.dtype
    x32 = x.float()
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    w32 = torch.exp(x32)
    w32 = w32 - torch.tensor(1.0, dtype=torch.float32)
    y32 = torch.where(x32 > zero32, x32, w32)
    return y32.to(dt)


def cand_expm1(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """A2 rejected candidate: select(cond, x, alpha * expm1(x)) -- the
    negative branch only (same select structure), isolating the expm1
    vs straight exp(x)-1 rounding-path divergence."""
    x32 = x.float()
    alpha32 = torch.tensor(np.float32(alpha), dtype=torch.float32)
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    neg32 = alpha32 * torch.expm1(x32)
    return torch.where(x32 > zero32, x32, neg32).to(x.dtype)


def cand_maxform(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """A4 rejected candidate: y = max(x, alpha*(exp(x)-1)) (fp32 domain)."""
    x32 = x.float()
    alpha32 = torch.tensor(np.float32(alpha), dtype=torch.float32)
    neg32 = alpha32 * (torch.exp(x32) - torch.tensor(1.0, dtype=torch.float32))
    return torch.maximum(x32, neg32).to(x.dtype)


def cand_refactored(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """A5 rejected candidate: select(cond, x, alpha*exp(x) - alpha) -- the
    negative branch only (same select structure), isolating the
    two-rounding vs one-rounding divergence of the refactored form."""
    x32 = x.float()
    alpha32 = torch.tensor(np.float32(alpha), dtype=torch.float32)
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    neg32 = alpha32 * torch.exp(x32) - alpha32
    return torch.where(x32 > zero32, x32, neg32).to(x.dtype)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def ulp_diff(a: torch.Tensor, b: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """Element-wise ulp distance between two same-dtype tensors (NaN==NaN
    and +-0 vs +-0 count as 0 -- both are IEEE-equal via `a == b`)."""
    if dt is torch.float32:
        ai, bi = a.view(torch.int32), b.view(torch.int32)
    else:  # float16 / bfloat16 -- both 16-bit layouts
        ai, bi = a.view(torch.int16), b.view(torch.int16)
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    d = (ai.int() - bi.int()).abs()
    d[same] = 0
    return d


def violations(out: torch.Tensor, ref: torch.Tensor, dt: str) -> int:
    """Count elements outside the source-test atol/rtol envelope.

    NaN-aware comparator with THREE branches (leaky_relu task lesson:
    `abs(inf-inf)=NaN` would pierce a bare envelope check and fake
    violations): exact equality (covers +-Inf and +-0 cross-sign) |
    both-NaN | atol/rtol envelope.
    """
    atol, rtol = _TOL[dt]
    o, r = out.float(), ref.float()
    both_nan = torch.isnan(o) & torch.isnan(r)
    ok = (o == r) | both_nan | (torch.abs(o - r) <= (atol + rtol * torch.abs(r)))
    return int((~ok).sum().item())


def max_ulp(a: torch.Tensor, b: torch.Tensor, dt: str) -> int:
    if a.numel() == 0:
        return 0
    return int(ulp_diff(a, b, _DT[dt]).max().item())


def mismatch_count(a: torch.Tensor, b: torch.Tensor) -> int:
    """Value-level mismatch count (IEEE ==; NaN==NaN; +-0 cross-sign equal)."""
    return int((~((a == b) | (torch.isnan(a) & torch.isnan(b)))).sum().item())


# ---------------------------------------------------------------------------
# Test data: random (sign-balanced, binade-widened) + corners
# ---------------------------------------------------------------------------
def make_inputs(dt: str, n: int = 1 << 16, seed_hint: str = "") -> torch.Tensor:
    # Per-case deterministic seed (derived from dtype + hint) so every
    # invocation of the script reproduces the exact same data streams
    # regardless of call order.
    torch.manual_seed(zlib.crc32(f"elu-equiv-{dt}-{seed_hint}".encode()) & 0xFFFFFFFF)
    torch_dt = _DT[dt]
    x = torch.randn(n, dtype=torch.float64)
    # Widen the dynamic range so rounding differences are exercised
    # across binades, then cast to the target dtype.
    x = x * torch.pow(torch.tensor(2.0), torch.randint(-10, 10, (n,), dtype=torch.float64))
    x = x.to(torch_dt)
    finfo = torch.finfo(torch_dt)
    corners = torch.tensor(
        [
            0.0, -0.0,
            float("inf"), float("-inf"), float("nan"),
            1.0, -1.0, 2.0, -2.0,
            float(finfo.max), -float(finfo.max),
            float(finfo.min),            # smallest normal
            float(finfo.min) * 0.5,      # subnormal region
            -(float(finfo.min) * 0.5),
            88.0,                        # exp(88) finite in fp32
            89.0,                        # exp(89) overflows fp32 (discarded lane)
            _EXP32_UNDERFLOW_X,          # exp -> subnormal/0 transition band
            _EXP32_UNDERFLOW_X - 40.0,   # exp -> exactly 0 in fp32
            *_TINY_X,                    # exp(x)-1 cancellation zone
        ],
        dtype=torch.float64,
    ).to(torch_dt)
    return torch.cat([x, corners])


# ---------------------------------------------------------------------------
# A1: adopted fp32-transit (candidate) vs source rounding path (baseline)
#     vs golden  --  general chain (vmul present)
# ---------------------------------------------------------------------------
def run_a1() -> list[dict]:
    rows = []
    n = 1 << 16
    for dt in ("float16", "bfloat16", "float32"):
        for alpha in (1.0, 0.5):
            x = make_inputs(dt, n, seed_hint=f"a1-{alpha}")
            g = golden(x, alpha)
            base = baseline_source(x, alpha)
            cand = candidate_transit(x, alpha)

            cand_vs_gold_viol = violations(cand, g, dt)
            base_vs_gold_viol = violations(base, g, dt)
            cand_vs_gold_ulp = max_ulp(cand, g, dt)
            base_vs_gold_ulp = max_ulp(base, g, dt)
            cand_vs_base_ulp = max_ulp(cand, base, dt)

            # Verdict: candidate must be within tolerance vs golden (0
            # violations expected -- same rounding family) and never
            # worse than the baseline (bit-exact expected: select-then-
            # round == round-then-select here, see DESIGN.md 1.6.1).
            ok = cand_vs_gold_viol == 0 and cand_vs_gold_ulp <= base_vs_gold_ulp
            rows.append({
                "item": "A1 fp32-transit",
                "dtype": dt,
                "alpha": alpha,
                "cand_vs_golden_max_ulp": cand_vs_gold_ulp,
                "cand_vs_golden_viol": cand_vs_gold_viol,
                "baseline_vs_golden_max_ulp": base_vs_gold_ulp,
                "baseline_vs_golden_viol": base_vs_gold_viol,
                "cand_vs_baseline_max_ulp": cand_vs_base_ulp,
                "verdict": "EQUIV_PASS" if ok else "EQUIV_FAIL",
            })
    return rows


# ---------------------------------------------------------------------------
# A3: adopted alpha==1.0 specialization (vmul dropped) vs golden
# ---------------------------------------------------------------------------
def run_a3() -> list[dict]:
    rows = []
    n = 1 << 16
    for dt in ("float16", "bfloat16", "float32"):
        x = make_inputs(dt, n, seed_hint="a3")
        g = golden(x, 1.0)
        cand = candidate_alpha1_specialized(x)
        viol = violations(cand, g, dt)
        ulp = max_ulp(cand, g, dt)
        rows.append({
            "item": "A3 alpha==1 no-mul",
            "dtype": dt,
            "cand_vs_golden_max_ulp": ulp,
            "cand_vs_golden_viol": viol,
            "verdict": "EQUIV_PASS" if (viol == 0 and ulp == 0) else "EQUIV_FAIL",
        })
    return rows


# ---------------------------------------------------------------------------
# A2/A4/A5: rejected-candidate machine evidence (NOT adopted)
# ---------------------------------------------------------------------------
def run_rejected() -> list[dict]:
    rows = []
    n = 1 << 16

    # A2 expm1 form: diverges from the golden straight exp(x)-1 path at
    # tiny |x| (formula domain fp32).
    for alpha in (1.0, 0.1):
        x = make_inputs("float32", n, seed_hint=f"a2-{alpha}")
        g = golden(x, alpha)
        e1 = cand_expm1(x, alpha)
        mism = mismatch_count(e1, g)
        viol = violations(e1, g, "float32")
        rows.append({
            "item": "A2 expm1-form (rejected)",
            "dtype": "float32 (formula domain)",
            "alpha": alpha,
            "mismatch_elems": mism,
            "violation_count": viol,
            "verdict": ("EQUIV_PASS-within-tol but NOT golden-bitwise "
                        f"({mism} mismatches)" if mism > 0 else "bitwise equal"),
        })

    # A4 max form: structurally wrong on the positive axis.
    x = make_inputs("float32", n, seed_hint="a4")
    g = golden(x, 1.0)
    mf = cand_maxform(x, 1.0)
    mism = mismatch_count(mf, g)
    viol = violations(mf, g, "float32")
    rows.append({
        "item": "A4 max-form (rejected)",
        "dtype": "float32 (formula domain)",
        "alpha": 1.0,
        "mismatch_elems": mism,
        "violation_count": viol,
        "verdict": (f"EQUIV_FAIL ({mism} mismatches: exp(x)-1 > x on the "
                     "positive axis, max picks the negative branch)"),
    })

    # A5 refactored form alpha*exp(x)-alpha: two roundings for general
    # alpha -- diverges from the golden one-rounding form.
    x = make_inputs("float32", n, seed_hint="a5")
    g = golden(x, 0.1)
    rf = cand_refactored(x, 0.1)
    mism = mismatch_count(rf, g)
    viol = violations(rf, g, "float32")
    rows.append({
        "item": "A5 alpha*exp-alpha (rejected)",
        "dtype": "float32 (formula domain)",
        "alpha": 0.1,
        "mismatch_elems": mism,
        "violation_count": viol,
        "verdict": (f"EQUIV_PASS-within-tol but NOT golden-bitwise "
                    f"({mism} mismatches at alpha=0.1)" if mism > 0
                    else "bitwise equal at alpha=0.1"),
    })

    # A6 fp16 native-domain chain: diverges from the source fp32-opmath
    # path (source compares and computes the negative branch in fp32).
    x = make_inputs("float16", n, seed_hint="a6")
    g = golden(x, 1.0)
    nat = cand_fp16_native(x, 1.0)
    mism = mismatch_count(nat, g)
    viol = violations(nat, g, "float16")
    ulp = max_ulp(nat, g, "float16")
    rows.append({
        "item": "A6 fp16-native (rejected)",
        "dtype": "float16",
        "alpha": 1.0,
        "mismatch_elems": mism,
        "violation_count": viol,
        "verdict": (f"within-tol but NOT golden-bitwise "
                    f"({mism} mismatches, max {ulp} ulp -- per-op fp16 "
                    "rounding vs source fp32 opmath)"),
    })
    return rows


def cand_fp16_native(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """A6 rejected candidate for fp16: native-domain per-op chain
    (vexp/vsub/vmul all natively fp16 -- T.vexp.md section 2.2.1 marks
    fp16 as supported). Simulated with torch fp16 ops (CPU opmath =
    fp32 compute + per-op round to fp16, the standard model of
    per-op native rounding). Diverges from the source, which does the
    compare AND the whole negative branch in the fp32 domain."""
    dt = x.dtype
    zero16 = torch.tensor(0.0, dtype=dt)
    alpha16 = torch.tensor(np.float16(alpha), dtype=dt)
    one16 = torch.tensor(1.0, dtype=dt)
    e16 = torch.exp(x)                # per-op round to fp16
    t16 = e16 - one16                 # per-op round
    s16 = alpha16 * t16               # per-op round
    return torch.where(x > zero16, x, s16)


def main() -> None:
    print("=" * 104)
    print("D-1 equivalence verification -- ELU forward (DESIGN.md section 1.6.1)")
    print("=" * 104)
    a1 = run_a1()
    a3 = run_a3()
    rej = run_rejected()
    hdr = (f"{'item':<18}{'dtype':<10}{'alpha':>6} | {'cand-gold ulp':>13} "
           f"{'cand-gold viol':>15} | {'base-gold ulp':>13} {'base-gold viol':>15} | "
           f"{'cand-base ulp':>13}  verdict")
    print("-- A1 adopted: fp32-transit (candidate) vs source rounding path (baseline) --")
    print(hdr)
    for r in a1:
        print(f"{r['item']:<18}{r['dtype']:<10}{r['alpha']:>6} | "
              f"{r['cand_vs_golden_max_ulp']:>13} {r['cand_vs_golden_viol']:>15} | "
              f"{r['baseline_vs_golden_max_ulp']:>13} {r['baseline_vs_golden_viol']:>15} | "
              f"{r['cand_vs_baseline_max_ulp']:>13}  {r['verdict']}")

    print()
    print("-- A3 adopted: alpha==1.0 specialization (vmul dropped) vs golden --")
    print(f"{'item':<22}{'dtype':<10} | {'cand-gold ulp':>13} {'cand-gold viol':>15}  verdict")
    for r in a3:
        print(f"{r['item']:<22}{r['dtype']:<10} | "
              f"{r['cand_vs_golden_max_ulp']:>13} {r['cand_vs_golden_viol']:>15}  {r['verdict']}")

    print()
    print("-- A2/A4/A5 rejected-candidate evidence (not adopted items) --")
    print(f"{'item':<30}{'dtype':<26}{'alpha':>6} {'mismatch':>10} {'violations':>11}  verdict")
    for r in rej:
        print(f"{r['item']:<30}{r['dtype']:<26}{r['alpha']:>6} "
              f"{r['mismatch_elems']:>10} {r['violation_count']:>11}  {r['verdict']}")

    fails = [r for r in a1 if r["verdict"] != "EQUIV_PASS"] + [
        r for r in a3 if r["verdict"] != "EQUIV_PASS"
    ]
    print()
    if fails:
        print(f"RESULT: EQUIV_FAIL ({len(fails)} adopted-item row(s) failed)")
        raise SystemExit(1)
    print("RESULT: EQUIV_PASS (all adopted-item rows pass; "
          "A2/A4/A5 rows are rejection evidence, not adopted items)")


if __name__ == "__main__":
    main()
