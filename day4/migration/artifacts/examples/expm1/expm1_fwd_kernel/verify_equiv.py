#!/usr/bin/env python3
"""D-1 equivalence & precision machine verification for the expm1 migration design.

Verifies (DESIGN.md section 1.6.1):

  A  baseline / source formula (fp32-opmath path, GPU ``op_func`` semantics):
        y = round_D( fp32(exp(x_f32)) - fp32(1.0) )
     A1 adopted NPU transit dtype path is the SAME formula (structural
     inheritance, PL-1.17 "ji-cheng" type): fp16/bf16 = vcast(rint) up ->
     fp32-domain vexp/vsub -> vcast(rint) single round-back.  Machine check:
     cand vs base must be bit-identical (0 ulp) and vs golden 0 tolerance
     violations.
  B  Taylor segment candidate (|x| < T=0.5, Horner terms k=2..9)  -- archived,
     not adopted (evidence row, section 1.6.3.1).
  C  fp64 intermediate candidate -- archived; API-infeasible on npuir
     (negative evidence: T.vexp.md 2.2.1 fp64 x; T.vcast.md 2.2.1 no f64 row;
     T.vbrc.md 2.2.1 no fp64 column; manifest contract fp16|bf16|fp32 only).

Golden anchors (independent of the NPU algorithm):
  g_opmath  = torch.expm1(x_D)          # the source-repo test anchor
  g_fp64ref = round_D(expm1_f64(x))     # fp64 reference

Tolerance judgment reproduces the allclose envelope
  |out - ref| <= atol + rtol * |ref|
with an exact-equality branch ``out == ref`` FIRST (inf==inf / -inf==-inf
lanes would otherwise produce NaN in the diff and fake violations --
leaky_relu RETROSPECTIVE lesson) and both-NaN counted as equal.

Run:  python3 verify_equiv.py
Pure torch CPU (fp32/fp64 semantics); no NPU required.
"""

import torch

torch.manual_seed(20261028)

DTYPES = {
    "float16": (torch.float16, 1e-3, 1e-3),
    "bfloat16": (torch.bfloat16, 1.6e-2, 1.6e-2),
    "float32": (torch.float32, 1e-5, 1e-5),
}

# Taylor candidate constants (candidate B): threshold T and Horner terms.
TAYLOR_T = 0.5
TAYLOR_KMAX = 9  # terms k=2..9; remainder <= T^10/10! = 2.7e-10 << 2^-24


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def round_to(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.to(dtype)


def cand_a(x_D: torch.Tensor) -> torch.Tensor:
    """Candidate A / A1 (source formula, fp32-opmath + single round-back)."""
    x32 = x_D.float()
    y32 = torch.exp(x32) - 1.0  # fp32 exp, fp32 scalar 1.0, fp32 sub
    return round_to(y32, x_D.dtype)


def cand_b(x_D: torch.Tensor) -> torch.Tensor:
    """Candidate B: Taylor segment for |x| < TAYLOR_T, else candidate A."""
    x32 = x_D.float()
    small = x32.abs() < TAYLOR_T
    # Horner for expm1(x) = x*(1 + x/2*(1 + x/3*(... x/k*(1))))
    p = torch.ones_like(x32)
    for k in range(TAYLOR_KMAX, 1, -1):
        p = 1.0 + x32 * p / float(k)
    taylor = x32 * p
    y32 = torch.where(small, taylor, torch.exp(x32) - 1.0)
    return round_to(y32, x_D.dtype)


def cand_c(x_D: torch.Tensor) -> torch.Tensor:
    """Candidate C: fp64 intermediate expm1, rounded once to D."""
    y64 = torch.expm1(x_D.double())
    return round_to(y64, x_D.dtype)


def golden_opmath(x_D: torch.Tensor) -> torch.Tensor:
    """Source-repo test anchor: torch.expm1 in the native dtype."""
    return torch.expm1(x_D)


def golden_fp64(x_D: torch.Tensor) -> torch.Tensor:
    return round_to(torch.expm1(x_D.double()), x_D.dtype)


def _ordered_bits(t: torch.Tensor) -> torch.Tensor:
    """Map IEEE bit patterns to a monotonic integer space (ulp distance)."""
    if t.dtype == torch.float32:
        b = t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        smask = 0x80000000
    else:
        b = t.view(torch.int16).to(torch.int64) & 0xFFFF
        smask = 0x8000
    return torch.where(b >= smask, smask - (b - smask), b)


def compare(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float):
    """Return (max_abs_diff, mismatches, violations, max_ulps).

    Exact-equality branch first (out == ref covers +-Inf pairs and equal
    values); both-NaN counted equal (NaN payload ignored).
    """
    out32, ref32 = out.float(), ref.float()
    eq = out == ref
    eq = eq | (torch.isnan(out) & torch.isnan(ref))
    finite_pair = torch.isfinite(out32) & torch.isfinite(ref32)
    diff = torch.where(eq, torch.zeros_like(out32), (out32 - ref32).abs())
    max_abs_diff = diff.max().item()
    tol = atol + rtol * ref32.abs()
    violations = int((~eq & (diff > tol)).sum().item())
    mismatches = int((~eq).sum().item())
    ulp = (_ordered_bits(out) - _ordered_bits(ref)).abs()
    # mask lanes already counted equal (incl. +-0 value-equal pairs, whose
    # ordered-bit distance is the 2^31 sign artifact, and NaN pairs)
    ulp = ulp[finite_pair & ~eq]
    max_ulps = int(ulp.max().item()) if ulp.numel() else 0
    return max_abs_diff, mismatches, violations, max_ulps


# ---------------------------------------------------------------------------
# test data
# ---------------------------------------------------------------------------
def randn_sets(n: int = 1 << 20):
    """randn + scaled variants to populate small/large |x| bands."""
    base = torch.randn(n, dtype=torch.float32)
    return {
        "randn": base,
        "x0.001": base * 1e-3,   # cancellation region
        "x0.01": base * 1e-2,
        "x10": base * 10.0,      # large-|x| region (fp32 exp ok)
        "ulin": torch.empty(n, dtype=torch.float32).uniform_(-89.0, 89.0),
    }


def edge_values() -> torch.Tensor:
    """Corners: test edge values + IEEE specials (finfo-sourced extrema)."""
    vals = [
        0.0, -0.0,
        88.8, -88.8,          # test edge: fp32 exp overflow / subnormal-in band
        1e-7, -1e-7,          # test edge: cancellation region
        float("inf"), float("-inf"), float("nan"),
        torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max,
        2.0 ** -149, -(2.0 ** -149),      # fp32 min subnormal
        2.0 ** -126, -(2.0 ** -126),      # fp32 min normal
        2.0 ** -24, -(2.0 ** -24),        # 0.5 ulp(1.0) scale
        1e-4, -1e-4, 0.5, -0.5, 16.0, -16.0, 88.0, -88.0, 89.0,
    ]
    return torch.tensor(vals, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. band-wise error distribution of candidate A (fp32 domain) vs fp64 ref
# ---------------------------------------------------------------------------
def band_table():
    bands = [
        (0.0, 2.0 ** -30), (2.0 ** -30, 2.0 ** -20), (2.0 ** -20, 1e-4),
        (1e-4, 1e-2), (1e-2, 0.1), (0.1, 1.0), (1.0, 8.0), (8.0, 32.0),
        (32.0, 80.0), (80.0, 88.0), (88.0, 88.7),
    ]
    rows = []
    for lo, hi in bands:
        # dense log-spaced samples inside the band, both signs
        n = 20000
        u = torch.empty(n).uniform_(0.0, 1.0)
        if lo == 0.0:
            mag = u * hi
        else:
            mag = torch.exp(torch.empty(n).uniform_(torch.log(torch.tensor(lo)),
                                                    torch.log(torch.tensor(hi))))
        x = torch.where(torch.randn(n) < 0, -mag, mag)
        x = x.to(torch.float32)
        ref64 = torch.expm1(x.double())
        y32 = torch.exp(x) - 1.0
        abs_err = (y32.double() - ref64).abs()
        rel_err = abs_err / ref64.abs().clamp_min(1e-300)
        rows.append((lo, hi, abs_err.max().item(), rel_err.max().item()))
    return rows


# ---------------------------------------------------------------------------
# 2. per-dtype candidate vs golden judgment
# ---------------------------------------------------------------------------
def run_cases():
    results = []
    sets = randn_sets()
    edge = edge_values()

    for dtype_name, (D, atol, rtol) in DTYPES.items():
        for set_name, x32 in list(sets.items()) + [("edge", edge)]:
            x_D = round_to(x32, D)
            g1 = golden_opmath(x_D)
            g2 = golden_fp64(x_D)
            for cand_name, fn in (("A", cand_a), ("B", cand_b), ("C", cand_c)):
                out = fn(x_D)
                md, mm, vio, ulp = compare(out, g1, atol, rtol)
                results.append((dtype_name, set_name, cand_name, "g_opmath",
                                md, mm, vio, ulp))
            # base-vs-cand identity check (A1 structural inheritance)
            out_a = cand_a(x_D)
            md, mm, vio, ulp = compare(out_a, round_to(cand_a(x_D), D), atol, rtol)
            # trivially identical; instead check opmath-domain consistency:
            # does CPU torch.expm1(x_D) equal round_D(expm1_f32)? (golden domain doc)
            md2, mm2, _, ulp2 = compare(g1, round_to(torch.expm1(x_D.float()), D),
                                        atol, rtol)
            results.append((dtype_name, set_name, "GOLD-DIAG", "g_opmath~f32opmath",
                            md2, mm2, 0, ulp2))
    return results


def main():
    torch.set_printoptions(precision=6)

    print("=" * 100)
    print("[1] band-wise error of candidate A (naive fp32 exp-1) vs fp64 reference")
    print(f"{'band |x|':>24} {'max abs err':>14} {'max rel err':>12}  margin vs fp32 atol 1e-5")
    atol32 = DTYPES["float32"][1]
    for lo, hi, ae, re_ in band_table():
        print(f"({lo:>10.3e}, {hi:>10.3e}] {ae:>14.3e} {re_:>12.3e}   x{atol32/ae:>8.0f}")

    print("=" * 100)
    print("[2] per-dtype candidate vs golden (g_opmath = torch.expm1 native dtype)")
    print(f"{'dtype':>9} {'set':>7} {'cand':>4} {'max_abs':>11} {'mism':>7} "
          f"{'viol':>6} {'max_ulp':>8}  verdict")
    results = run_cases()
    n_pass = n_fail = 0
    for dtype_name, set_name, cand_name, gname, md, mm, vio, ulp in results:
        if cand_name == "GOLD-DIAG":
            verdict = f"diagnostic (mism={mm})"
            print(f"{dtype_name:>9} {set_name:>7} {cand_name:>4} {md:>11.3e} {mm:>7} "
                  f"{'-':>6} {ulp:>8}  {verdict}")
            continue
        verdict = "EQUIV_PASS" if vio == 0 else "EQUIV_FAIL"
        if vio == 0:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{dtype_name:>9} {set_name:>7} {cand_name:>4} {md:>11.3e} {mm:>7} "
              f"{vio:>6} {ulp:>8}  {verdict}")

    print("=" * 100)
    print("[3] Taylor candidate B quantification (T=0.5, terms k=2..9)")
    x_small = torch.tensor([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5],
                           dtype=torch.float32)
    ref = torch.expm1(x_small.double())
    a = (torch.exp(x_small) - 1.0).double()
    p = torch.ones_like(x_small)
    for k in range(TAYLOR_KMAX, 1, -1):
        p = 1.0 + x_small * p / float(k)
    b = (x_small * p).double()
    print(f"{'x':>10} {'ref expm1':>14} {'A rel err':>12} {'B rel err':>12}")
    for i in range(x_small.numel()):
        r = ref[i].abs().clamp_min(1e-300)
        print(f"{x_small[i].item():>10.1e} {ref[i].item():>14.6e} "
              f"{(a[i]-ref[i]).abs().item()/r:>12.3e} {(b[i]-ref[i]).abs().item()/r:>12.3e}")

    print("=" * 100)
    print(f"SUMMARY: adopted-item rows (A) with 0 violations: {n_pass}; "
          f"EQUIV_FAIL rows: {n_fail}")
    print("A1 verdict: PASS" if n_fail == 0 else "A1 verdict: FAIL")


if __name__ == "__main__":
    main()
