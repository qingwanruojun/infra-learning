"""Minimal MixCV kernel: Cube GEMM followed by a Vector post-op (exp).

Day 4 §6 Q5: "a minimal MixCV example (Cube then Vector post-processing) that
runs" -- this is the skeleton Day 5's FlashAttention needs (Q*K^T on the Cube,
softmax on the Vector).

The kernel is written in Developer mode (the high-level form used by
examples/mixcv/example_mixcv.py): ``T.alloc_shared`` lands in L1, the
``T.alloc_fragment`` accumulator in L0C, ``T.gemm`` is Cube-only, ``T.vexp``
is Vector-only, and TileLang inserts the cross-core sync.

Run:  python day4/code/own_ops/mixcv_min.py
"""

import os
import re
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.isdir(os.path.join(_REPO_ROOT, "tilelang")) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.abspath(os.path.join(HERE, "..", "logs"))

DTYPE = "float16"
INNER = "float32"


def mixcv(M, N, K, block_M, block_N):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), DTYPE),
        B: T.Tensor((K, N), DTYPE),
        C: T.Tensor((M, N), INNER),
        D: T.Tensor((M, N), INNER),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            off_m = (cid // n_num) * block_M
            off_n = (cid % n_num) * block_N

            A_BUF = T.alloc_shared((block_M, K), DTYPE)      # L1
            B_BUF = T.alloc_shared((K, block_N), DTYPE)      # L1
            C_BUF = T.alloc_fragment((block_M, block_N), INNER)  # L0C
            D_BUF = T.alloc_fragment((block_M, block_N), INNER)  # L0C

            T.copy(A[off_m:off_m + block_M, 0:K], A_BUF[0:block_M, 0:K])
            T.copy(B[0:K, off_n:off_n + block_N], B_BUF[0:K, 0:block_N])

            T.gemm(A_BUF, B_BUF, C_BUF, [block_M, K, block_N], initC=True)
            T.vexp(C_BUF, D_BUF)  # Vector core post-processing

            T.copy(C_BUF[0:block_M, 0:block_N],
                   C[off_m:off_m + block_M, off_n:off_n + block_N])
            T.copy(D_BUF[0:block_M, 0:block_N],
                   D[off_m:off_m + block_M, off_n:off_n + block_N])

    return main


def build(M, N, K, block_M, block_N):
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    # examples/mixcv/example_mixcv.py sets this too; the L0C->GM staging for the
    # Vector op needs workspace.
    os.environ.setdefault("TILELANG_ASCEND_WORKSPACE_SIZE", str(M * N))
    return tilelang.compile(mixcv(M, N, K, block_M, block_N), target="npuir")


def dump_ops(kernel, tag):
    """Show that one module carries both a Cube op and a Vector op."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{tag}.mlir")
    with open(path, "w") as fh:
        fh.write(kernel.mlir_content)
    counts = {}
    for line in kernel.mlir_content.splitlines():
        m = re.search(r"\bhivm\.hir\.(\w+)|\bhivm\.(\w+)", line)
        if m:
            name = m.group(1) or m.group(2)
            counts[name] = counts.get(name, 0) + 1
    print(f"wrote {path}")
    print("  ops in module:", ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())))
    for line in kernel.mlir_content.splitlines():
        if re.search(r"func.func|mmad|vexp|set_flag|wait_flag|sync", line):
            print("   ", line.strip()[:150])


def run_case(kernel, label, M, N, K, scale):
    a = (torch.randn(M, K, dtype=torch.float16) * scale).npu()
    b = (torch.randn(K, N, dtype=torch.float16) * scale).npu()
    c = torch.zeros(M, N, dtype=torch.float32).npu()
    d = torch.zeros(M, N, dtype=torch.float32).npu()

    kernel(a, b, c, d)
    torch.npu.synchronize()

    # fp32 reference: upcast the fp16 inputs, matmul in fp32, then exp.
    ref_c = a.float() @ b.float()
    ref_d = torch.exp(ref_c)
    n_inf = int(torch.isinf(ref_d).sum().item())
    finite = torch.isfinite(ref_d)
    print(f"\n[{label}] scale={scale:.4g}  |C|max={ref_c.abs().max().item():.3g}  "
          f"exp 溢出 {n_inf} 个元素（占 {n_inf / ref_d.numel():.2%}）")
    for name, got, ref in (("C=gemm", c, ref_c), ("D=exp(gemm)", d, ref_d)):
        if name.startswith("D") and n_inf:
            # inf - inf = nan would poison the metric; measure on finite entries.
            got_f, ref_f = got[finite], ref[finite]
        else:
            got_f, ref_f = got, ref
        err = (got_f - ref_f).abs()
        scale_ref = ref_f.abs().max().clamp_min(1e-12)
        denom = ref_f.abs().clamp_min(scale_ref * 1e-2)
        print(f"  {name}: max_abs={err.max().item():.4e} "
              f"max_abs/max|C|={(err.max() / scale_ref).item():.3e} "
              f"max_rel(|ref|>=1%max)={(err / denom).max().item():.3e} "
              f"rmse={err.pow(2).mean().sqrt().item():.3e}")
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)
    print(f"  {label}: All check passed (rtol=1e-2, atol=1e-2)")


def main():
    M, N, K = 1024, 1024, 512
    block_M, block_N = 128, 256
    kernel = build(M, N, K, block_M, block_N)
    dump_ops(kernel, "mixcv_min")

    # 1) unscaled, exactly like examples/mixcv/example_mixcv.py: C ~ N(0, K), so
    #    exp(C) overflows fp32 (max ~1e49) and the official assert degenerates
    #    into "inf == inf".  Kept only as evidence of that degeneracy.
    run_case(kernel, "official (unscaled)", M, N, K, 1.0)
    # 2) scaled by 0.36: std(C) = sqrt(K)*0.36^2 ~= 3, so |C|max ~ 16 and
    #    exp stays finite -- rtol=1e-2 now tests actual values.
    run_case(kernel, "scaled x0.36", M, N, K, 0.36)


if __name__ == "__main__":
    main()
