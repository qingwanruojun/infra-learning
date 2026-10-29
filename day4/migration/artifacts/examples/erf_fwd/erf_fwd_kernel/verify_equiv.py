#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""D-1 equivalence machine verification for the erf_fwd NPU redesign.

Design-stage verification (DESIGN.md section 1.6.1, gate rule S1-EQUIV-EXEC).
Compares, on torch CPU with fp64 ground truth:

  A1 (adopted)  A&S 7.1.26 rational-approximation chain (op-by-op fp32
                simulation of the NPU v-prefix chain, incl. the fp16/bf16
                fp32-transit single-rounding path)  vs  the baseline
                ``torch.erf`` golden form used by the source-repo tests
                (``ref_fn(x.float()).to(x.dtype)``) and vs fp64 erf.
                Coverage: random tensors at multiple scales + the corner
                set (0 / -0 / +-inf / NaN / +-subnormal / +-tiny / +-1 /
                +-2 / +-3 / +-4 / +-6 / +-100 / +-max-finite from
                torch.finfo -- no hand-written decimal extrema).

  A2 (adopted)  scalar-constant literal path: fp32 rounding of the Python
                literals is identical whether they ride as v-op scalar
                operands or are broadcast through a buffer -- asserted by
                construction (torch fp32 cast round-trip equality).

  REJ-1 (rejection evidence, NOT adopted)  7th-order Taylor expansion
                (the documented lowering shape of ``T.verf`` --
                docs/Tilelang.language/数学操作/T.verf.md section 3:
                ConstantOp + DivFOp + VMulOp + VAddOp, no hardware erf)
                diverges catastrophically for |x| >= 1.5.

  GOLD-DIAG     whether native-dtype ``torch.erf(cpu_tensor)`` is
                bit-identical to the fp32-transit form on CPU (decides
                which golden form the landed kernel embeds).

Run:  python3 verify_equiv.py
Exit code 0 iff every adopted item reports EQUIV_PASS.
"""

import math
import sys

import torch

torch.manual_seed(20261029)

# ---------------------------------------------------------------------------
# A&S 7.1.26 constants (Abramowitz & Stegun, Handbook of Mathematical
# Functions, formula 7.1.26; quoted max abs error 1.5e-7)
# ---------------------------------------------------------------------------
P = 0.3275911
A1 = 0.254829592
A2 = -0.284496736
A3 = 1.421413741
A4 = -1.453152027
A5 = 1.061405429

DT = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
TOL = {  # source-repo test_unary_math.py _get_tolerances (verified both sides)
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
    "float32": (1e-5, 1e-5),
}
MANT = {"float16": 10, "bfloat16": 7, "float32": 23}

SUBNORMAL_MIN = {  # smallest positive subnormal, power-of-two form
    "float16": 2.0 ** -24,
    "bfloat16": 2.0 ** -133,
    "float32": 2.0 ** -149,
}


# ---------------------------------------------------------------------------
# A1: op-by-op fp32 simulation of the NPU v-prefix chain
# ---------------------------------------------------------------------------
def as26_fp32_chain(x32: torch.Tensor) -> torch.Tensor:
    """Simulate the NPU chain in fp32, one statement per v-op.

    Mirrors DESIGN.md section 3.3 exactly: vcmp / vabs / vmul / vadd /
    vdiv / vexp / vsub / vselect, every intermediate rounded to fp32.
    """
    cond = x32 < 0.0          # T.vcmp(x, zeros, "lt")
    a = x32.abs()             # T.vabs
    x2 = x32 * x32            # T.vmul
    a = P * a                 # T.vmul(a, P) scalar literal
    a = 1.0 + a               # T.vadd(a, 1.0)
    t = 1.0 / a               # T.vdiv(ones, a) -- fp32 division
    x2 = -1.0 * x2            # T.vmul(x2, -1.0)
    e = torch.exp(x2)         # T.vexp
    h = torch.full_like(t, A5)  # T.vmul(ones, A5) init (fp32 literal)
    for Ai in (A4, A3, A2, A1):
        h = h * t             # T.vmul(h, t) in-place
        h = h + Ai            # T.vadd(h, Ai) scalar literal
    m = h * e                 # T.vmul(poly, e)
    m = t * m                 # T.vmul(t, poly*e) -- A&S 7.1.26 leading t factor
    r = 1.0 - m               # T.vsub(ones, m)
    s = torch.where(cond, -1.0, 1.0)  # T.vselect(cond, neg_one, ones)
    return r * s              # T.vmul(r, s) -> out


def npu_sim(x: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """Full NPU route simulation incl. fp16/bf16 fp32-transit rounding."""
    x32 = x.float()           # T.vcast(x, x32, "rint") -- exact for fp16/bf16
    out32 = as26_fp32_chain(x32)
    if dtype_str == "float32":
        return out32
    return out32.to(DT[dtype_str])  # single rint round-back


def golden(x: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """Source-repo test anchor form: ref_fn(x.float()).to(x.dtype)."""
    return torch.erf(x.float()).to(DT[dtype_str])


def ulp_diff(sim: torch.Tensor, ref: torch.Tensor, dtype_str: str):
    """Max ulp distance over finite same-sign pairs (0.0 when equal)."""
    s = sim.float()
    r = ref.float()
    ok = torch.isfinite(s) & torch.isfinite(r) & (torch.sign(s) == torch.sign(r))
    if not ok.any():
        return 0.0
    s_o, r_o = s[ok], r[ok]
    mant = MANT[dtype_str]
    exp = torch.floor(torch.log2(torch.clamp(r_o.abs(), min=1e-45)))
    ulp = torch.pow(2.0, exp - mant)
    # subnormal ref: ulp floor at the smallest subnormal
    sub = SUBNORMAL_MIN[dtype_str]
    ulp = torch.clamp(ulp, min=sub)
    d = (s_o - r_o).abs() / ulp
    return float(d.max())


def corner_values(dtype_str: str) -> torch.Tensor:
    """Corner set -- extrema from torch.finfo, never hand-written."""
    dt = DT[dtype_str]
    finfo = torch.finfo(dt)
    sub = SUBNORMAL_MIN[dtype_str]
    vals = [
        0.0, -0.0,
        float("inf"), float("-inf"), float("nan"),
        sub, -sub,
        float(finfo.tiny), -float(finfo.tiny),
        1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0,
        100.0, -100.0,                       # source-repo edge values +-3, 100
        float(finfo.max), -float(finfo.max),
    ]
    return torch.tensor(vals, dtype=dt)


def random_suites(dtype_str: str):
    n = 1_000_000
    g = torch.Generator().manual_seed(20261029)
    suites = {
        "randn": torch.randn(n, generator=g),
        "near-zero(x0.01)": torch.randn(n, generator=g) * 0.01,
        "mid(x3)": torch.randn(n, generator=g) * 3.0,
        "uniform(-6,6)": torch.rand(n, generator=g) * 12.0 - 6.0,
        "tail(+-[3,6])": torch.cat([
            3.0 + torch.rand(n // 2, generator=g) * 3.0,
            -(3.0 + torch.rand(n // 2, generator=g) * 3.0),
        ]),
    }
    if dtype_str == "float32":
        suites["extreme(1e0..1e38)"] = (
            torch.pow(10.0, torch.rand(n, generator=g) * 38.0)
            * (torch.randint(0, 2, (n,), generator=g) * 2.0 - 1.0)
        )
    return suites


def check_dtype(dtype_str: str):
    """Run all suites for one dtype; return the result dict."""
    atol, rtol = TOL[dtype_str]
    total, violations, max_abs64, max_ulp = 0, 0, 0.0, 0.0
    detail = []

    suites = random_suites(dtype_str)
    x_corner = corner_values(dtype_str)
    suites["corners"] = x_corner.float()

    for name, xf in suites.items():
        x = xf.to(DT[dtype_str])
        sim = npu_sim(x, dtype_str)
        ref = golden(x, dtype_str)
        close = torch.isclose(sim, ref, atol=atol, rtol=rtol, equal_nan=True)
        viol = int((~close).sum())
        ref64 = torch.erf(x.double())
        err64 = (sim.double() - ref64).abs()
        finite = torch.isfinite(err64)
        m64 = float(err64[finite].max()) if finite.any() else float("nan")
        mu = ulp_diff(sim, ref, dtype_str)
        total += x.numel()
        violations += viol
        if not math.isnan(m64):
            max_abs64 = max(max_abs64, m64)
        max_ulp = max(max_ulp, mu)
        detail.append((name, x.numel(), viol, m64, mu))

    # special-value bit discipline: +-inf signs and NaN positions must match
    x = corner_values(dtype_str)
    sim = npu_sim(x, dtype_str)
    ref = golden(x, dtype_str)
    spec = torch.isinf(sim) | torch.isinf(ref) | torch.isnan(sim) | torch.isnan(ref)
    bit_ok = True
    if spec.any():
        both_nan = torch.isnan(sim) & torch.isnan(ref)
        bit_ok = bool((both_nan | (sim == ref))[spec].all())

    return {
        "dtype": dtype_str,
        "total": total,
        "violations": violations,
        "max_abs_err_vs_fp64": max_abs64,
        "max_ulp_vs_golden": max_ulp,
        "special_bits_ok": bit_ok,
        "detail": detail,
        "pass": violations == 0 and bit_ok,
    }


def check_a2_constants():
    """A2: literal path and broadcast path carry identical fp32 values."""
    for name, v in (("P", P), ("A1", A1), ("A2", A2), ("A3", A3),
                    ("A4", A4), ("A5", A5), ("1.0", 1.0), ("-1.0", -1.0)):
        lit = torch.tensor(v, dtype=torch.float32)
        brc = torch.full((8,), v, dtype=torch.float32)
        if not torch.equal(lit.expand(8), brc):
            return False, name
    return True, None


def check_taylor_rejection():
    """REJ-1: 7th-order Taylor (= T.verf lowering shape) error table."""
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    rows = []
    for xv in (0.5, 1.0, 1.5, 2.0, 3.0):
        x = torch.tensor(xv, dtype=torch.float64)
        taylor = two_over_sqrt_pi * (
            x - x ** 3 / 3.0 + x ** 5 / 10.0 - x ** 7 / 42.0
        )
        ref = torch.erf(x)
        rows.append((xv, float(taylor), float(ref), float((taylor - ref).abs())))
    return rows


def check_gold_diag():
    """GOLD-DIAG: native-dtype torch.erf vs fp32-transit form on CPU."""
    out = {}
    g = torch.Generator().manual_seed(7)
    x = torch.randn(200_000, generator=g)
    for dtype_str in ("float16", "bfloat16"):
        xd = x.to(DT[dtype_str])
        try:
            native = torch.erf(xd)
        except Exception as exc:  # noqa: BLE001
            out[dtype_str] = f"native-erf-unsupported ({type(exc).__name__})"
            continue
        transit = torch.erf(xd.float()).to(DT[dtype_str])
        same = native.view(torch.int16) == transit.view(torch.int16)
        out[dtype_str] = f"bit-identical {int(same.sum())}/{same.numel()}"
    return out


def main():
    print("=" * 78)
    print("D-1 equivalence verification: erf_fwd A&S 7.1.26 chain vs torch.erf golden")
    print("=" * 78)

    ok_a2, bad = check_a2_constants()
    print(f"\n[A2] scalar-literal fp32 identity: "
          f"{'EQUIV_PASS' if ok_a2 else 'EQUIV_FAIL (' + str(bad) + ')'}")

    print("\n[A1] per-dtype results (random suites + corner set):")
    print(f"{'dtype':<10}{'points':>10}{'viol':>6}{'max|err|vs fp64':>18}"
          f"{'max ulp vs golden':>19}{'special':>9}  verdict")
    all_pass = ok_a2
    for dtype_str in ("float32", "float16", "bfloat16"):
        r = check_dtype(dtype_str)
        verdict = "EQUIV_PASS" if r["pass"] else "EQUIV_FAIL"
        all_pass = all_pass and r["pass"]
        print(f"{r['dtype']:<10}{r['total']:>10}{r['violations']:>6}"
              f"{r['max_abs_err_vs_fp64']:>18.3e}{r['max_ulp_vs_golden']:>19.2f}"
              f"{'ok' if r['special_bits_ok'] else 'BAD':>9}  {verdict}")
        for name, n, viol, m64, mu in r["detail"]:
            print(f"    - {name:<20} n={n:>9} viol={viol:>4} "
                  f"max|err|={m64:.3e} max_ulp={mu:.2f}")

    print("\n[REJ-1] 7th-order Taylor (T.verf lowering shape) error, fp64:")
    print(f"{'x':>6}{'taylor':>16}{'erf':>16}{'abs err':>14}")
    for xv, tv, rv, err in check_taylor_rejection():
        print(f"{xv:>6.1f}{tv:>16.6f}{rv:>16.6f}{err:>14.4g}")

    print("\n[GOLD-DIAG] native-dtype torch.erf vs fp32-transit (CPU):")
    for k, v in check_gold_diag().items():
        print(f"    {k}: {v}")

    print("\nOVERALL:", "EQUIV_PASS" if all_pass else "EQUIV_FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
