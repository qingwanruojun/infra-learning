"""rmsnorm layered test suite: L0/L1/L2/Boundary + main(--level)."""

import argparse
import os
import sys

import tilelang
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmsnorm import rmsnorm, validate_inputs  # noqa: E402

COVERAGE_CATEGORY = "Normalization"

# ========== Precision standard (mixed tolerance, DESIGN.md section 9.3) ==========
# The user acceptance contract is atol=rtol=1e-4 for float32 and 1e-2 for
# float16; these values are hard-coded on purpose (the skill default table is
# looser for float32) and must not be relaxed.
PRECISION_TABLE = {
    # dtype     : (atol, rtol, max_abs_error_limit, required_matched_ratio)
    "float32": (1e-4, 1e-4, 1e-2, 0.99),
    "float16": (1e-2, 1e-2, 1e-1, 0.99),
}


def get_precision(dtype):
    """Return (atol, rtol, max_abs_error_limit, required_matched_ratio)."""
    return PRECISION_TABLE[dtype]


def _torch_dtype(dtype):
    return torch.float32 if dtype == "float32" else torch.float16


# ========== Golden reference (independent of the NPU torch operators) ==========
def golden_rmsnorm(x, w, eps=1e-5, out_dtype=None):
    """CPU float64 reference: y = x * rsqrt(mean(x^2, -1) + eps) * w."""
    xf = x.detach().cpu().to(torch.float64)
    wf = w.detach().cpu().to(torch.float64)
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd * wf
    return y.to(out_dtype or x.dtype)


def check_precision(actual, golden, dtype):
    """Mixed-tolerance double gate + strict element-wise contract.

    Returns a metrics dict.  ``passed`` requires:
      * NaN / +Inf / -Inf masks identical,
      * every compared element within atol + rtol * |golden| (n_exceed == 0),
      * matched_ratio >= required_matched_ratio and max_abs <= max_abs_error_limit.
    The element-wise condition is strict so that a 1% escape cannot hide behind
    the matched-ratio gate.
    """
    atol, rtol, max_abs_limit, required_ratio = get_precision(dtype)
    a = actual.detach().cpu().to(torch.float64)
    g = golden.detach().cpu().to(torch.float64)

    metrics = {
        "atol": atol,
        "rtol": rtol,
        "max_abs_limit": max_abs_limit,
        "required_ratio": required_ratio,
        "masks_match": True,
        "n_compared": int(g.numel()),
        "n_exceed": 0,
        "matched_ratio": 1.0,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "passed": True,
    }

    masks_match = (
        torch.equal(torch.isnan(a), torch.isnan(g))
        and torch.equal(torch.isposinf(a), torch.isposinf(g))
        and torch.equal(torch.isneginf(a), torch.isneginf(g))
    )
    metrics["masks_match"] = masks_match
    if not masks_match:
        metrics.update(
            passed=False,
            matched_ratio=0.0,
            max_abs=float("inf"),
            max_rel=float("inf"),
            n_exceed=int(g.numel()),
        )
        return metrics

    finite = torch.isfinite(a) & torch.isfinite(g)
    n_finite = int(finite.sum().item())
    metrics["n_compared"] = n_finite
    if n_finite == 0:
        return metrics  # only special values; the mask comparison is the gate

    af, gf = a[finite], g[finite]
    abs_err = (af - gf).abs()
    tol = atol + rtol * gf.abs()
    n_exceed = int((abs_err > tol).sum().item())
    rel_mask = gf.abs() > 1e-6
    if bool(rel_mask.any()):
        max_rel = float((abs_err[rel_mask] / gf[rel_mask].abs()).max().item())
    else:
        max_rel = 0.0

    metrics.update(
        n_exceed=n_exceed,
        matched_ratio=1.0 - n_exceed / n_finite,
        max_abs=float(abs_err.max().item()),
        max_rel=max_rel,
    )
    metrics["passed"] = (
        n_exceed == 0
        and metrics["matched_ratio"] >= required_ratio
        and metrics["max_abs"] <= max_abs_limit
    )
    return metrics


def _format_metrics(level, name, M, N, dtype, eps, m):
    marker = "[PRECISION_PASS]" if m["passed"] else "[PRECISION_FAIL]"
    verdict = "PASS" if m["passed"] else "FAIL"
    print(
        f"{marker} {level} {name} shape=({M},{N}) dtype={dtype} "
        f"eps={eps:g} | atol={m['atol']:g} rtol={m['rtol']:g} "
        f"max_abs_limit={m['max_abs_limit']:g} req_ratio={m['required_ratio']:g} | "
        f"matched_ratio={m['matched_ratio']:.6f} max_abs={m['max_abs']:.6e} "
        f"max_rel={m['max_rel']:.6e} n_exceed={m['n_exceed']}/{m['n_compared']} "
        f"mask_match={m['masks_match']} -> {verdict}"
    )


# ========== Test data (built on CPU, one H2D copy per case) ==========
def _make_tensors(M, N, dtype, vrange=(-1.0, 1.0), seed=0, normal=False):
    gen = torch.Generator().manual_seed(seed)
    td = _torch_dtype(dtype)
    if normal:
        x = torch.randn((M, N), generator=gen, dtype=torch.float32)
        w = torch.randn((N,), generator=gen, dtype=torch.float32)
    else:
        lo, hi = vrange
        x = torch.rand((M, N), generator=gen, dtype=torch.float32) * (hi - lo) + lo
        w = torch.rand((N,), generator=gen, dtype=torch.float32) * 2.0 - 1.0
    return x.to(td), w.to(td)


def _make_special(M, N, dtype, kind, seed=0):
    """CPU tensors with finite base values plus sparse special values."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand((M, N), generator=gen, dtype=torch.float32) * 2.0 - 1.0
    w = torch.rand((N,), generator=gen, dtype=torch.float32) * 2.0 - 1.0
    flat = x.view(-1)
    if kind == "inf":
        flat[0] = float("inf")
        flat[1] = float("-inf")
        flat[2] = 0.5  # keep at least one finite value (inside the same row)
    elif kind == "nan":
        flat[0] = float("nan")
        flat[1] = 0.5
    elif kind == "zero":
        x.zero_()
    elif kind == "db_fp16":
        flat[0] = 65504.0
        flat[1] = -65504.0
        flat[2] = 1.0
    elif kind == "db_fp32":
        flat[0] = 1e17
        flat[1] = -1e17
        flat[2] = 1.0
    else:
        raise ValueError(f"unknown special kind {kind!r}")
    return x.to(_torch_dtype(dtype)), w.to(_torch_dtype(dtype))


# ========== L0: gate test (regular shape, block-aligned) ==========
L0_CASES = [
    # name, M, N, dtype, eps, vrange, normal
    ("l0_basic", 1024, 1024, "float32", 1e-5, (-1.0, 1.0), True),
]


def _run_precision(
    level, name, M, N, dtype, eps=1e-5, vrange=(-1.0, 1.0), normal=False, seed=0
):
    try:
        x_cpu, w_cpu = _make_tensors(M, N, dtype, vrange, seed, normal)
        ref = golden_rmsnorm(x_cpu, w_cpu, eps, out_dtype=_torch_dtype(dtype))
        kernel = rmsnorm(M, N, dtype, eps)
        y = kernel(x_cpu.npu(), w_cpu.npu())
        metrics = check_precision(y, ref, dtype)
        _format_metrics(level, name, M, N, dtype, eps, metrics)
        return metrics["passed"]
    except Exception as exc:  # noqa: BLE001 - a failing case must not stop the layer
        print(
            f"[PRECISION_FAIL] {level} {name} shape=({M},{N}) dtype={dtype} "
            f"eps={eps:g}: {type(exc).__name__}: {exc}"
        )
        return False


def test_rmsnorm_l0():
    """L0 gate test: DESIGN.md section 9.2 (regular shape, block-aligned)."""
    ok = True
    for name, M, N, dtype, eps, vrange, normal in L0_CASES:
        ok &= _run_precision("l0", name, M, N, dtype, eps, vrange, normal)
    return ok


# ========== L1: functional test (regular + irregular shapes) ==========
# (name, M, N, dtype, eps, vrange, normal, tags)
L1_CASES = [
    (
        "l1_fp32_base",
        1024,
        1024,
        "float32",
        1e-5,
        (-1.0, 1.0),
        False,
        ["D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    ),
    (
        "l1_fp16_base",
        1024,
        1024,
        "float16",
        1e-5,
        (-1.0, 1.0),
        False,
        ["D-DTYPE-fp16", "D-SHAPE-ALIGNED", "D-VALRANGE-S"],
    ),
    (
        "l1_tail_1023",
        1023,
        1024,
        "float32",
        1e-5,
        (-1.0, 1.0),
        False,
        ["D-SHAPE-TAIL-MID"],
    ),
    (
        "l1_fp16_tail_1023",
        1023,
        1024,
        "float16",
        1e-5,
        (-1.0, 1.0),
        False,
        ["D-DTYPE-fp16", "D-SHAPE-TAIL-MID"],
    ),
    (
        "l1_tail_1025",
        1025,
        1024,
        "float32",
        1e-5,
        (-1.0, 1.0),
        False,
        ["D-SHAPE-TAIL-1"],
    ),
    (
        "l1_tail_1032",
        1032,
        1024,
        "float32",
        1e-5,
        (-1.0, 1.0),
        False,
        ["D-SHAPE-TAIL-MID"],
    ),
    (
        "l1_prime_1009",
        1009,
        1024,
        "float32",
        1e-5,
        (-1.0, 1.0),
        False,
        ["D-SHAPE-PRIME"],
    ),
    ("l1_edge_1xN", 1, 1024, "float32", 1e-5, (-1.0, 1.0), False, ["D-SHAPE-EDGE"]),
    ("l1_edge_Nx1", 1024, 1, "float32", 1e-5, (-1.0, 1.0), False, ["D-SHAPE-EDGE"]),
    (
        "l1_valrange_M",
        1024,
        1024,
        "float32",
        1e-5,
        (-10.0, 10.0),
        False,
        ["D-VALRANGE-M"],
    ),
    (
        "l1_valrange_L",
        1024,
        1024,
        "float32",
        1e-5,
        (-100.0, 100.0),
        False,
        ["D-VALRANGE-L"],
    ),
    (
        "l1_valrange_ASYM",
        1024,
        1024,
        "float32",
        1e-5,
        (-5.0, 10.0),
        False,
        ["D-VALRANGE-ASYM"],
    ),
    ("l1_eps_1e-3", 1024, 1024, "float32", 1e-3, (-1.0, 1.0), False, ["D-PARAM-eps"]),
    (
        "l1_eps_1e-8",
        1024,
        1024,
        "float16",
        1e-8,
        (-1.0, 1.0),
        False,
        ["D-PARAM-eps", "D-DTYPE-fp16"],
    ),
    ("l1_n_1000", 1024, 1000, "float32", 1e-5, (-1.0, 1.0), False, ["D-VALRANGE-S"]),
]


def test_rmsnorm_l1():
    """L1 functional test: parameter combinations incl. tail / prime shapes."""
    ok = True
    for name, M, N, dtype, eps, vrange, normal, _tags in L1_CASES:
        ok &= _run_precision("l1", name, M, N, dtype, eps, vrange, normal)
    return ok


# ========== L2: negative tests (illegal input must be rejected) ==========
# (name, tags)
L2_CASES = [
    ("l2_dtype_bf16", ["D-EXC-DTYPE"]),
    ("l2_mixed_dtype", ["D-EXC-DTYPE"]),
    ("l2_rank3", ["D-EXC-SHAPE"]),
    ("l2_w_len", ["D-EXC-SHAPE"]),
    ("l2_n_too_large", ["D-EXC-SHAPE"]),
    ("l2_m_zero", ["D-EXC-SHAPE"]),
]


def _run_exception(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - rejection is the expected outcome
        print(
            f"[BOUNDARY_PASS] l2 {name}: correctly rejected ({type(exc).__name__}: {exc})"
        )
        return True
    print(f"[BOUNDARY_WARN] l2 {name}: illegal input was silently accepted")
    return False


def test_rmsnorm_l2():
    """L2 negative tests: unsupported dtype / illegal shapes must be rejected."""
    x = torch.zeros((1024, 1024), dtype=torch.float32)
    w = torch.zeros((1024,), dtype=torch.float32)
    x3 = torch.zeros((2, 1024, 1024), dtype=torch.float32)
    w_short = torch.zeros((512,), dtype=torch.float32)
    w_f16 = torch.zeros((1024,), dtype=torch.float16)
    x_empty = torch.zeros((0, 1024), dtype=torch.float32)

    cases = {
        "l2_dtype_bf16": lambda: rmsnorm(1024, 1024, "bfloat16", 1e-5),
        "l2_mixed_dtype": lambda: validate_inputs(x, w_f16),
        "l2_rank3": lambda: validate_inputs(x3, w),
        "l2_w_len": lambda: validate_inputs(x, w_short),
        "l2_n_too_large": lambda: rmsnorm(1024, 16384, "float32", 1e-5),
        "l2_m_zero": lambda: validate_inputs(x_empty, w),
    }
    ok = True
    for name, _tags in L2_CASES:
        ok &= _run_exception(name, cases[name])
    return ok


# ========== Boundary: legal special values (precision compared, blocking) ==========
# (name, M, N, dtype, eps, kind, tags)
BOUNDARY_CASES = [
    ("bnd_inf", 128, 1024, "float32", 1e-5, "inf", ["D-SPECIAL-INF"]),
    ("bnd_nan", 128, 1024, "float32", 1e-5, "nan", ["D-SPECIAL-NAN"]),
    ("bnd_zero", 128, 1024, "float32", 1e-5, "zero", ["D-SPECIAL-ZERO"]),
    ("bnd_db_fp16", 128, 1024, "float16", 1e-5, "db_fp16", ["D-SPECIAL-DBOUND"]),
    ("bnd_db_fp32", 128, 1024, "float32", 1e-5, "db_fp32", ["D-SPECIAL-DBOUND"]),
]


def _run_boundary(name, M, N, dtype, eps, kind, seed=0):
    try:
        x_cpu, w_cpu = _make_special(M, N, dtype, kind, seed)
        ref = golden_rmsnorm(x_cpu, w_cpu, eps, out_dtype=_torch_dtype(dtype))
        kernel = rmsnorm(M, N, dtype, eps)
        y = kernel(x_cpu.npu(), w_cpu.npu())
        m = check_precision(y, ref, dtype)
        marker = "[BOUNDARY_PASS]" if m["passed"] else "[BOUNDARY_WARN]"
        print(
            f"{marker} boundary {name} shape=({M},{N}) dtype={dtype} "
            f"eps={eps:g} | atol={m['atol']:g} rtol={m['rtol']:g} "
            f"max_abs_limit={m['max_abs_limit']:g} | "
            f"matched_ratio={m['matched_ratio']:.6f} max_abs={m['max_abs']:.6e} "
            f"max_rel={m['max_rel']:.6e} n_exceed={m['n_exceed']}/{m['n_compared']} "
            f"mask_match={m['masks_match']} -> "
            f"{'PASS' if m['passed'] else 'WARN'}"
        )
        return bool(m["passed"])
    except Exception as exc:  # noqa: BLE001 - a boundary warning must fail the run
        print(
            f"[BOUNDARY_WARN] boundary {name} dtype={dtype}: {type(exc).__name__}: {exc}"
        )
        return False


def test_rmsnorm_boundary():
    """Boundary tests: INF/NAN/zero/dtype-bound values (precision-compared)."""
    ok = True
    for name, M, N, dtype, eps, kind, _tags in BOUNDARY_CASES:
        ok &= _run_boundary(name, M, N, dtype, eps, kind)
    return ok


# ========== Coverage bookkeeping (machine-checkable) ==========
COVERAGE_MANIFEST = {
    "D-DTYPE-fp32": 1,
    "D-DTYPE-fp16": 2,
    "D-SHAPE-ALIGNED": 2,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 3,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 2,
    "D-VALRANGE-S": 3,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-eps": 2,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 2,
    "D-EXC-DTYPE": 2,
    "D-EXC-SHAPE": 4,
}

# Rank coverage: the operator only declares rank-2 tensors (proto.yaml), so
# D-SHAPE-RANK-* is not an applicable dimension.
COVERAGE_NA = {
    "D-SHAPE-RANK-2": "proto.yaml declares a single rank (rank-2 X, rank-1 W); no multi-rank support",
}


# ========== main: --level dispatch + exit code ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="l0", choices=["l0", "l1", "l2", "boundary", "all"]
    )
    args = parser.parse_args()

    tilelang.disable_cache()  # avoid stale compiled artifacts
    torch.manual_seed(0)
    torch.npu.set_device(0)

    blocking_ok = True  # all four layers feed the exit code
    if args.level in ("l0", "all"):
        blocking_ok &= test_rmsnorm_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_rmsnorm_l1()
    if args.level in ("l2", "all"):
        blocking_ok &= test_rmsnorm_l2()
    if args.level in ("boundary", "all"):
        blocking_ok &= test_rmsnorm_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
