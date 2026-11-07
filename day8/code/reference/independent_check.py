"""Independent RMSNorm checker used for Day 8 acceptance.

This file deliberately does NOT import the generated kernel or its golden.
It computes the reference in float64 on CPU from the raw inputs and compares
the kernel output. It also provides a negative control that must fail.

Reference:
    rstd = 1 / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    y    = x * rstd * w
"""

from __future__ import annotations

import torch


def torch_ref_fp64(x: torch.Tensor, w: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """CPU float64 reference, independent of any NPU/torch_npu op."""
    xf = x.detach().cpu().double()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd
    if w is not None:
        y = y * w.detach().cpu().double()
    return y


def compare(
    y: torch.Tensor,
    ref: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict:
    """Mixed-tolerance comparison in float64, with a masked relative error.

    max_rel here is measured only where |ref| > 1e-6 * max|ref| so that
    near-zero denominators do not manufacture huge relative errors. The
    unmasked value is also returned for transparency.
    """
    actual = y.detach().cpu().double()
    golden = ref.detach().cpu().double()
    if actual.shape != golden.shape:
        return {
            "pass": False,
            "reason": f"shape mismatch {tuple(actual.shape)} vs {tuple(golden.shape)}",
        }
    diff = (actual - golden).abs()
    tol = atol + rtol * golden.abs()
    ok = diff <= tol
    matched_ratio = ok.double().mean().item()
    max_abs = diff.max().item()
    denom = golden.abs().clamp_min(1e-12)
    max_rel_unmasked = (diff / denom).max().item()
    mask = golden.abs() > (1e-6 * golden.abs().max().clamp_min(1e-12))
    max_rel_masked = (diff[mask] / golden[mask].abs()).max().item() if mask.any() else 0.0
    return {
        "pass": bool(matched_ratio == 1.0),
        "max_abs": max_abs,
        "max_rel": max_rel_masked,
        "max_rel_unmasked": max_rel_unmasked,
        "matched_ratio": matched_ratio,
        "n_elements": golden.numel(),
        "n_mismatch": int((~ok).sum().item()),
        "atol": atol,
        "rtol": rtol,
    }


def negative_control(y: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    """Mutate one value by 10x its own tolerance; the checker must fail it.

    A fixed absolute perturbation is not enough: with rtol, a large-magnitude
    element has a large tolerance, and for fp16 a 0.1 step can still be inside
    it. Scaling by ``atol + rtol*|golden|`` at the chosen element guarantees
    the injected error is 10x the local tolerance.
    """
    mutated = y.clone()
    flat = mutated.reshape(-1)
    idx = int(flat.abs().argmax().item())
    g = ref.reshape(-1)[idx].item()
    delta = 10.0 * (atol + rtol * abs(g))
    flat[idx] = flat[idx] + delta
    return compare(mutated, ref, atol, rtol)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1023, 1024)
    w = torch.randn(1024)
    ref = torch_ref_fp64(x, w)
    good = compare(ref.float(), ref, 1e-4, 1e-4)
    bad = negative_control(ref.float(), ref, 1e-4, 1e-4)
    print("identity:", good)
    print("negative control:", bad)
    assert good["pass"], good
    assert not bad["pass"], bad
    print("SELF_CHECK_PASS")
