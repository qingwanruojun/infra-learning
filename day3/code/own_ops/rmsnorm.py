"""Hand-written RMSNorm kernel in the npuir dialect (own_ops/rmsnorm.py).

Tiling choice: one block owns ``block_M`` full rows, and the complete hidden
dimension ``N`` lives in UB.  This keeps the reduction inside one tile, so
``T.reduce`` can produce ``(block_M, 1)`` directly and no cross-tile
accumulation is needed.

Compute sequence (matches the Day-2 notes):

    xsq    = x * x                          # T.vmul
    sumsq  = reduce(xsq, dims=[1], sum)     # (block_M, N) -> (block_M, 1)
    rstd   = rsqrt(sumsq / N + eps)         # scalars go to src2 of vmul/vadd
    y      = x * rstd [* weight]            # broadcast (block_M,N)x(block_M,1)
    y      = cast(y, dtype)

Buffer budget for M=N=1024, block_M=8 (UB = 192 KiB on 910B).  Every buffer is
allocated unconditionally (see the parser-scope note below), so the fp32 path
uses x_in 32 + x_f32 32 + xsq 32 + y_out 32 + weight 8 KiB ≈ 136 KiB.
The fp16 path swaps the in/out buffers to 16 KiB each: ≈ 102 KiB.
Both fit, but the fp32 margin is only ~56 KiB; adding block_M or multi-buffering
will need the x_f32/x_in aliasing optimisation.
"""

import contextlib
import os

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


@contextlib.contextmanager
def ascend_mode(mode: str):
    prev = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = prev


def _mode_for_memory(memory: str) -> str:
    if memory not in ("ub", "shared"):
        raise ValueError(f"memory must be 'ub' or 'shared', got {memory!r}")
    return "Expert" if memory == "ub" else "Developer"


def rmsnorm_kernel(
    M,
    N,
    block_M,
    dtype="float32",
    eps=1e-5,
    use_weight=True,
    memory="ub",
    rstd_mode="sqrt_rec",
):
    """Build ``Y = X * rstd(mean(X^2, dim=1) + eps) [* W]``.

    Args:
        M, N: matrix shape.
        block_M: rows per block; ``N`` full rows are kept in UB.
        dtype: input/output dtype; accumulation is always fp32.
        eps: numerical stabilizer.
        use_weight: multiply by a per-column weight after normalization.
        memory: ``"ub"`` (Expert / alloc_ub) or ``"shared"``
            (Developer / alloc_shared).
        rstd_mode: how to compute 1/sqrt(x):
            ``"sqrt_rec"`` = ``vsqrt`` then ``vrec`` (accurate, fp32 default);
            ``"rsqrt"``    = hardware ``vrsqrt`` (~3e-3 relative error,
                             acceptable for fp16);
            ``"newton"``   = ``vrsqrt`` + one Newton refinement (~1e-5).
    """
    if dtype not in ("float32", "float16"):
        raise ValueError(f"unsupported dtype {dtype!r}")
    if rstd_mode not in ("sqrt_rec", "rsqrt", "newton"):
        raise ValueError(f"unsupported rstd_mode {rstd_mode!r}")
    if N <= 0 or block_M <= 0:
        raise ValueError("N and block_M must be positive")

    accum_dtype = "float32"
    alloc = T.alloc_ub if memory == "ub" else T.alloc_shared
    num_blocks = T.ceildiv(M, block_M)

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        W: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            # All buffers are allocated unconditionally: the TVMScript parser
            # scopes allocations made inside a conditional, so a later use
            # outside that branch would be "Undefined variable".
            x_in = alloc((block_M, N), dtype)
            x_f32 = alloc((block_M, N), accum_dtype)
            xsq = alloc((block_M, N), accum_dtype)
            y_out = alloc((block_M, N), dtype)
            sumsq = alloc((block_M, 1), accum_dtype)
            rstd = alloc((block_M, 1), accum_dtype)
            rstd_tmp = alloc((block_M, 1), accum_dtype)
            rstd_refined = alloc((block_M, 1), accum_dtype)
            w_in = alloc((1, N), dtype)
            w_f32 = alloc((1, N), accum_dtype)

            # Both scalars live inside T.Kernel but outside any T.Scope, and
            # are always passed as src2 of the binary op.
            inv_n = 1.0 / float(N)
            m_start = cid * block_M
            rows = T.min(block_M, M - m_start)

            T.copy(X[m_start : m_start + rows, :], x_in[0:rows, :])
            # The condition is a Python bool, so the parser emits exactly one
            # branch.  Only ops live inside these ifs -- never allocations,
            # whose names would fall out of the parser's variable scope.
            if dtype == accum_dtype:
                T.copy(x_in, x_f32)
            else:
                T.vcast(x_in, x_f32, round_mode="rint")

            # The weight load/cast belongs to the use_weight ablation: with
            # use_weight=False the block must not even read W.  Otherwise the
            # "no weight" timing still pays 4 KiB of per-block copy traffic
            # (~12.5% of the 32 KiB row read) and is not a clean ablation.
            if use_weight:
                T.copy(W, w_in)
                if dtype == accum_dtype:
                    T.copy(w_in, w_f32)
                else:
                    T.vcast(w_in, w_f32, round_mode="rint")

            # (1) x^2 into a dedicated fp32 intermediate buffer.
            T.vmul(x_f32, x_f32, xsq)

            # (2) row sum: (block_M, N) -> (block_M, 1).
            T.reduce(xsq, sumsq, dims=[1], reduce_mode="sum")

            # (3) rstd = rsqrt(sumsq * (1/N) + eps); scalars are src2.
            T.vmul(sumsq, inv_n, sumsq)
            T.vadd(sumsq, eps, sumsq)
            if rstd_mode == "rsqrt":
                # Hardware approximation: measured max rel err ~2.8e-3.
                T.vrsqrt(sumsq, rstd)
            elif rstd_mode == "sqrt_rec":
                T.vsqrt(sumsq, rstd_tmp)
                T.vrec(rstd_tmp, rstd)
            else:  # "newton": r0 * (1.5 - 0.5 * s * r0^2)
                T.vrsqrt(sumsq, rstd_tmp)  # r0
                T.vmul(rstd_tmp, rstd_tmp, rstd)  # r0^2
                T.vmul(rstd, sumsq, rstd)  # s * r0^2
                T.vmul(rstd, -0.5, rstd)  # -0.5 * s * r0^2
                T.vadd(rstd, 1.5, rstd)  # 1.5 - 0.5 * s * r0^2
                T.vmul(rstd_tmp, rstd, rstd_refined)  # r0 * (...)
                T.copy(rstd_refined, rstd)

            # (4) broadcast (block_M, N) * (block_M, 1) -> (block_M, N).
            T.vmul(x_f32, rstd, xsq)

            # (5) optional per-column weight, broadcast (block_M,N)*(1,N).
            if use_weight:
                T.vmul(xsq, w_f32, xsq)

            if dtype == accum_dtype:
                T.copy(xsq, y_out)
            else:
                T.vcast(xsq, y_out, round_mode="rint")
            T.copy(y_out[0:rows, :], Y[m_start : m_start + rows, :])

    return main


def compile_rmsnorm(
    M,
    N,
    block_M,
    dtype="float32",
    eps=1e-5,
    use_weight=True,
    memory="ub",
    rstd_mode="sqrt_rec",
    clear_cache=True,
):
    if clear_cache:
        tilelang.cache.clear_cache()
    with ascend_mode(_mode_for_memory(memory)):
        func = rmsnorm_kernel(
            M,
            N,
            block_M,
            dtype=dtype,
            eps=eps,
            use_weight=use_weight,
            memory=memory,
            rstd_mode=rstd_mode,
        )
        return tilelang.compile(func, target="npuir")


def make_tensors(M, N, dtype="float32", use_weight=True):
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(M, N, dtype=torch_dtype, device="npu")
    # The compiled signature always carries W; it is simply unused when
    # use_weight=False, so allocate a dummy tensor in that case.
    if use_weight:
        w = torch.randn(N, dtype=torch_dtype, device="npu")
    else:
        w = torch.zeros(N, dtype=torch_dtype, device="npu")
    y = torch.zeros(M, N, dtype=torch_dtype, device="npu")
    return x, w, y


def torch_ref(x, weight, eps=1e-5):
    """CPU float32 reference (independent of the NPU torch kernels)."""
    xf = x.detach().cpu().float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd
    if weight is not None:
        y = y * weight.detach().cpu().float()
    return y.to(x.dtype)


def check(
    M,
    N,
    block_M,
    dtype="float32",
    eps=1e-5,
    use_weight=True,
    memory="ub",
    rstd_mode="sqrt_rec",
    rtol=None,
    atol=None,
):
    kernel = compile_rmsnorm(
        M,
        N,
        block_M,
        dtype=dtype,
        eps=eps,
        use_weight=use_weight,
        memory=memory,
        rstd_mode=rstd_mode,
    )
    x, w, y = make_tensors(M, N, dtype=dtype, use_weight=use_weight)
    kernel(x, w, y)

    ref = torch_ref(x, w if use_weight else None, eps=eps)
    y_cpu = y.cpu()
    ref_cpu = ref.cpu()
    if rtol is None:
        rtol = 1e-4 if dtype == "float32" else 1e-2
    if atol is None:
        atol = 1e-4 if dtype == "float32" else 1e-2
    torch.testing.assert_close(y_cpu, ref_cpu, rtol=rtol, atol=atol)

    diff = (y_cpu.float() - ref_cpu.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref_cpu.float().abs() + 1e-12)).max().item()
    return y, max_abs, max_rel


def benchmark(kernel, x, w, y, warmup=3, iters=10, rounds=5):
    """Per-call latency in microseconds after warmup (compile excluded).

    Returns median / min / max over ``rounds`` windows of ``iters`` calls.
    """
    kernel(x, w, y)
    torch.npu.synchronize()
    for _ in range(warmup):
        kernel(x, w, y)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            kernel(x, w, y)
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    samples.sort()
    return {
        "median_us": samples[len(samples) // 2],
        "min_us": samples[0],
        "max_us": samples[-1],
        "samples_us": samples,
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_rmsnorm_fp32():
    """M=N=1024, block_M=8, fp32, with weight. Expect ~1e-4 or better."""
    _, max_abs, max_rel = check(1024, 1024, 8, dtype="float32", use_weight=True)
    print(f"\nfp32 max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
    assert max_abs < 1e-4, max_abs


def test_rmsnorm_fp32_no_weight():
    check(1024, 1024, 8, dtype="float32", use_weight=False)


def test_rmsnorm_fp32_tail_rows():
    """M=1023 with block_M=8 -> last block owns 7 rows."""
    check(1023, 1024, 8, dtype="float32", use_weight=True)


def test_rmsnorm_fp16():
    """fp16 I/O with fp32 accumulation; tolerance relaxed to 1e-2.

    Hardware vrsqrt (~3e-3 relative) is good enough at this tolerance and is
    the cheapest rstd path, so use it here.
    """
    _, max_abs, max_rel = check(
        1024,
        1024,
        8,
        dtype="float16",
        use_weight=True,
        rstd_mode="rsqrt",
        rtol=1e-2,
        atol=1e-2,
    )
    print(f"\nfp16 max_abs={max_abs:.3e} max_rel={max_rel:.3e}")


def test_rmsnorm_fp16_sqrt_rec():
    """Lock the fp16 + sqrt_rec precision at 5e-3.

    Measured max_abs is 9.8e-4 to 3.9e-3 depending on the input, well below
    the 1e-2 fp16 tolerance and far below vrsqrt's ~2.3e-2; asserting 5e-3
    catches a regression to vrsqrt.
    """
    _, max_abs, max_rel = check(
        1024,
        1024,
        8,
        dtype="float16",
        use_weight=True,
        rstd_mode="sqrt_rec",
        rtol=1e-2,
        atol=5e-3,
    )
    print(f"\nfp16/sqrt_rec max_abs={max_abs:.3e} max_rel={max_rel:.3e}")


def test_rmsnorm_newton():
    """vrsqrt + one Newton step is accurate enough for the fp32 1e-4 target.

    Uses the same M=N=1024 shape as ``test_rmsnorm_fp32`` so the two rstd
    modes are compared on identical inputs; max_abs is sensitive to the
    number of elements, so M=64 would not be comparable.
    """
    _, max_abs, max_rel = check(
        1024, 1024, 8, dtype="float32", use_weight=True, rstd_mode="newton"
    )
    print(f"\nnewton max_abs={max_abs:.3e} max_rel={max_rel:.3e}")


def test_rmsnorm_shared():
    """Developer mode (alloc_shared) parity on a small case."""
    check(64, 1024, 8, dtype="float32", use_weight=True, memory="shared")


# ---------------------------------------------------------------------------
# measurement entry point
# ---------------------------------------------------------------------------


def main():
    torch.npu.set_device(0)
    M, N, block_M = 1024, 1024, 8
    for dtype in ("float32", "float16"):
        for use_weight in (False, True):
            for rstd_mode in ("sqrt_rec", "rsqrt"):
                kernel = compile_rmsnorm(
                    M,
                    N,
                    block_M,
                    dtype=dtype,
                    use_weight=use_weight,
                    rstd_mode=rstd_mode,
                )
                x, w, y = make_tensors(M, N, dtype=dtype, use_weight=use_weight)
                st = benchmark(kernel, x, w, y)
                ref = torch_ref(x, w if use_weight else None)
                diff = (y.cpu().float() - ref.cpu().float()).abs()
                gbps = M * N * (4 if dtype == "float32" else 2) * 2 / (
                    st["median_us"] * 1e-6
                ) / 1e9
                print(
                    f"dtype={dtype:7s} weight={str(use_weight):5s} "
                    f"rstd={rstd_mode:8s} {st['median_us']:8.2f} us "
                    f"[min {st['min_us']:.1f}, max {st['max_us']:.1f}] "
                    f"{gbps:6.1f} GB/s  max_abs={diff.max().item():.3e}"
                )


if __name__ == "__main__":
    main()
