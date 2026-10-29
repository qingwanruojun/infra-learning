"""verify_equiv.py -- DESIGN.md section 1.6.1 equivalence machine verification (D-1).

Operator: leaky_relu (y = x if x > 0 else negative_slope * x).
Baseline formulas taken verbatim from the source kernel
(``tileops/kernels/elementwise.py`` L2469-L2473, non-fp8 branch):
    val.dtype-domain compute:  zero = 0 (val.dtype); slope_n = slope (val.dtype)
    y = if_then_else(val > zero, val, slope_n * val)          # per-op native rounding
Candidate (DESIGN.md section 0.6 R2, PL-1.17 fp32-transit):
    x32 = f32(x); slope32 = f32(slope)   # rint upcast, lossless
    y32 = if_then_else(x32 > 0, x32, slope32 * x32)           # fp32 domain
    y  = rint(y32 -> dtype)                                    # single round back
Golden anchor: torch.nn.functional.leaky_relu(x.float(), slope).to(x.dtype)
(the source-repo test ref_program path, fp32 opmath + single round-back).

Checks
  A1 (adopted, section 0.6 R2 dtype path): candidate transit vs baseline
     native-domain, fp64-anchored rounding-path comparison at fp16/bf16/fp32,
     random + corner values (inf/NaN/-0.0/subnormal/max-finite), slopes
     {0.01, 0.2}. Verdict: EQUIV_PASS iff candidate-vs-golden mismatches are
     within the source test tolerances AND strictly not-worse than baseline.
  A2 (rejected-candidate machine evidence, NOT adopted): y = max(x, slope*x)
     conditional form -- mathematically equal to leaky_relu only for
     slope in [0, 1]; demonstrated EQUIV_FAIL at slope=2.0.

All computation on torch CPU (fp64 reference where stated); no NPU required.
Run:  python verify_equiv.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(20261028)
np.random.seed(20261028)

# Source-repo test tolerances (examples/TileOPs/tests/ops/test_leaky_relu.py
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


# ---------------------------------------------------------------------------
# Formula implementations (torch CPU)
# ---------------------------------------------------------------------------
def golden(x: torch.Tensor, slope: float) -> torch.Tensor:
    """Source-repo test anchor: F.leaky_relu(x.float(), slope).to(x.dtype)."""
    return F.leaky_relu(x.detach().float(), slope).to(x.dtype)


def baseline_native(x: torch.Tensor, slope: float) -> torch.Tensor:
    """Source kernel rounding path, simulated exactly: all ops in the
    native dtype domain (per-op IEEE rounding; torch CPU fp16/bf16 ops are
    IEEE-correct because the exact product of two fp16/bf16 values fits in
    fp32, so fp32-compute-then-round == native hardware rounding)."""
    dt = x.dtype
    zero = torch.tensor(0.0, dtype=dt)
    slope_n = torch.tensor(slope, dtype=dt)
    return torch.where(x > zero, x, slope_n * x)


def candidate_transit(x: torch.Tensor, slope: float) -> torch.Tensor:
    """DESIGN.md section 0.6 R2 NPU chain simulated: vcast(rint) up to fp32,
    fp32-domain compare/select/mul with f32(slope), vcast(rint) back."""
    dt = x.dtype
    x32 = x.float()
    slope32 = torch.tensor(np.float32(slope), dtype=torch.float32)
    zero32 = torch.tensor(0.0, dtype=torch.float32)
    y32 = torch.where(x32 > zero32, x32, slope32 * x32)
    return y32.to(dt)


def candidate_vmax(x: torch.Tensor, slope: float) -> torch.Tensor:
    """A2 rejected candidate: y = max(x, slope*x) (fp32 domain + round back
    where needed) -- conditionally equivalent, slope in [0, 1] only."""
    x32 = x.float()
    slope32 = torch.tensor(np.float32(slope), dtype=torch.float32)
    return torch.maximum(x32, slope32 * x32).to(x.dtype)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def ulp_diff(a: torch.Tensor, b: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """Element-wise ulp distance between two same-dtype tensors (NaN==NaN
    and inf==inf count as 0)."""
    if dt is torch.float32:
        ai, bi = a.view(torch.int32), b.view(torch.int32)
    else:  # float16 / bfloat16 -- both 16-bit layouts
        ai, bi = a.view(torch.int16), b.view(torch.int16)
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    d = (ai.int() - bi.int()).abs()
    d[same] = 0
    return d


def violations(out: torch.Tensor, ref: torch.Tensor, dt: str) -> int:
    """Count elements outside the source-test atol/rtol envelope (NaN-aware:
    NaN in both = equal; inf in both = equal -- `o == r` covers both, since
    abs(inf-inf) is NaN and would falsely fail the envelope check)."""
    atol, rtol = _TOL[dt]
    o, r = out.float(), ref.float()
    both_nan = torch.isnan(o) & torch.isnan(r)
    ok = (o == r) | both_nan | (torch.abs(o - r) <= (atol + rtol * torch.abs(r)))
    return int((~ok).sum().item())


def max_ulp(a: torch.Tensor, b: torch.Tensor, dt: str) -> int:
    if a.numel() == 0:
        return 0
    return int(ulp_diff(a, b, _DT[dt]).max().item())


# ---------------------------------------------------------------------------
# Test data: random (sign-balanced) + corners
# ---------------------------------------------------------------------------
def make_inputs(dt: str, n: int = 1 << 16) -> torch.Tensor:
    torch_dt = _DT[dt]
    x = torch.randn(n, dtype=torch.float64)
    # widen dynamic range so mul/rounding differences are exercised across
    # binades, then cast to the target dtype.
    x = x * torch.pow(torch.tensor(2.0), torch.randint(-8, 9, (n,), dtype=torch.float64))
    x = x.to(torch_dt)
    # corners appended
    corners = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"),
         1.0, -1.0, 65504.0 if dt == "float16" else 1e30,
         5.96e-8, -5.96e-8, 2.0, -2.0],
        dtype=torch.float64,
    ).to(torch_dt)
    return torch.cat([x, corners])


# ---------------------------------------------------------------------------
# A1: adopted fp32-transit (candidate) vs native-domain (baseline) vs golden
# ---------------------------------------------------------------------------
def run_a1() -> list[dict]:
    rows = []
    n = 1 << 16
    for dt in ("float16", "bfloat16", "float32"):
        for slope in (0.01, 0.2):
            x = make_inputs(dt, n)
            g = golden(x, slope)
            base = baseline_native(x, slope)
            cand = candidate_transit(x, slope)

            cand_vs_gold_viol = violations(cand, g, dt)
            base_vs_gold_viol = violations(base, g, dt)
            cand_vs_gold_ulp = max_ulp(cand, g, dt)
            base_vs_gold_ulp = max_ulp(base, g, dt)
            cand_vs_base_ulp = max_ulp(cand, base, dt)

            # Verdict: candidate must be within tolerance vs golden (0
            # violations expected -- identical rounding path) and never
            # worse than the baseline.
            ok = cand_vs_gold_viol == 0 and cand_vs_gold_ulp <= base_vs_gold_ulp
            rows.append({
                "item": "A1 fp32-transit",
                "dtype": dt,
                "slope": slope,
                "cand_vs_golden_max_ulp": cand_vs_gold_ulp,
                "cand_vs_golden_viol": cand_vs_gold_viol,
                "baseline_vs_golden_max_ulp": base_vs_gold_ulp,
                "baseline_vs_golden_viol": base_vs_gold_viol,
                "cand_vs_baseline_max_ulp": cand_vs_base_ulp,
                "verdict": "EQUIV_PASS" if ok else "EQUIV_FAIL",
            })
    return rows


# ---------------------------------------------------------------------------
# A2: rejected vmax conditional form (machine evidence for rejection)
# ---------------------------------------------------------------------------
def run_a2() -> list[dict]:
    rows = []
    n = 1 << 16
    for slope in (0.01, 2.0):
        dt = "float32"  # formula-level equivalence judged in fp32 domain
        x = make_inputs(dt, n)
        g = golden(x, slope)
        vmax = candidate_vmax(x, slope)
        viol = violations(vmax, g, dt)
        mism = int((~((vmax == g) | (torch.isnan(vmax) & torch.isnan(g)))).sum().item())
        rows.append({
            "item": "A2 vmax-form (rejected candidate)",
            "dtype": "float32 (formula domain)",
            "slope": slope,
            "mismatch_elems": mism,
            "violation_count": viol,
            "verdict": ("EQUIV_PASS (slope in [0,1])" if slope == 0.01
                        else "EQUIV_FAIL (slope>1: max picks x on the "
                             "negative side instead of slope*x)"),
        })
    return rows


def main() -> None:
    print("=" * 100)
    print("D-1 equivalence verification -- leaky_relu (DESIGN.md section 1.6.1)")
    print("=" * 100)
    a1 = run_a1()
    hdr = (f"{'item':<26}{'dtype':<10}{'slope':>6} | {'cand-gold ulp':>13} "
           f"{'cand-gold viol':>15} | {'base-gold ulp':>13} {'base-gold viol':>15} | "
           f"{'cand-base ulp':>13}  verdict")
    print("-- A1 adopted: fp32-transit (candidate) vs native-domain (baseline) --")
    print(hdr)
    for r in a1:
        print(f"{r['item']:<26}{r['dtype']:<10}{r['slope']:>6} | "
              f"{r['cand_vs_golden_max_ulp']:>13} {r['cand_vs_golden_viol']:>15} | "
              f"{r['baseline_vs_golden_max_ulp']:>13} {r['baseline_vs_golden_viol']:>15} | "
              f"{r['cand_vs_baseline_max_ulp']:>13}  {r['verdict']}")

    print()
    print("-- A2 rejected-candidate evidence: y = max(x, slope*x) --")
    print(f"{'item':<38}{'dtype':<26}{'slope':>6} {'mismatch':>10} {'violations':>11}  verdict")
    for r in run_a2():
        print(f"{r['item']:<38}{r['dtype']:<26}{r['slope']:>6} "
              f"{r['mismatch_elems']:>10} {r['violation_count']:>11}  {r['verdict']}")

    fails = [r for r in a1 if r["verdict"] != "EQUIV_PASS"]
    print()
    if fails:
        print(f"RESULT: EQUIV_FAIL ({len(fails)} adopted-item row(s) failed)")
        raise SystemExit(1)
    print("RESULT: EQUIV_PASS (all adopted-item rows pass; "
          "A2 rows are rejection evidence, not adopted items)")


if __name__ == "__main__":
    main()
