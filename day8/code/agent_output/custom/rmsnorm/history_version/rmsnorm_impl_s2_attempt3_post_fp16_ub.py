"""RMSNorm kernel for Ascend910 (Developer mode, target="npuir").

Y[i, j] = X[i, j] * rsqrt(mean(X[i, :]^2) + eps) * W[j]

Pure Vector operator: one block owns ``block_m`` full rows and the complete
hidden dimension ``N`` lives in UB, so the row reduction is a single
``T.reduce_sum``.  The design is taken from custom/rmsnorm/DESIGN.md.
"""

import tilelang
from tilelang import language as T

_MAX_N = 8192
_SUPPORTED_DTYPES = ("float32", "float16")


def pick_block_m(N, dtype, ub_bytes=192 * 1024):
    """Host-side integer tiling choice (compile-time constant, reads no data).

    The fp16 path materialises more UB than the two logical buffers
    (``x_sh`` 2 B + ``xsq`` 4 B per element): the two fp32 promotions and the
    fp32 result add cast temporaries.  Measured on Ascend910B3 / CANN 9.0.0 for
    N=1024: block_m=16 needs 264,192 B, block_m=12 needs 198,720 B,
    block_m=10 / 8 fit - i.e. ~16 B per element per row.  The fp16 block is
    therefore sized with that coefficient and capped at 8 rows, which keeps
    the tile at ~133 KiB (68% of the 192 KiB UB).  The fp32 path has no cast
    temporaries and keeps the design's block_m <= 16 (135,296 B at N=1024).
    """
    N = int(N)
    if dtype == "float32":
        bm = (ub_bytes - 4 * N) // (8 * N)
        return max(1, min(16, int(bm)))
    bm = (ub_bytes - 6 * N) // (16 * N)
    return max(1, min(8, int(bm)))


def validate_inputs(x, w, eps=1e-5):
    """Pure Python metadata validation of the operator inputs (no compute)."""
    import torch

    if x.dim() != 2:
        raise ValueError(f"rmsnorm expects a rank-2 X, got rank {x.dim()}")
    if w.dim() != 1:
        raise ValueError(f"rmsnorm expects a rank-1 W, got rank {w.dim()}")
    if x.shape[0] <= 0:
        raise ValueError(f"rmsnorm requires M > 0, got M={x.shape[0]}")
    if x.shape[1] <= 0 or x.shape[1] > _MAX_N:
        raise ValueError(f"rmsnorm requires 1 <= N <= {_MAX_N}, got N={x.shape[1]}")
    if w.shape[0] != x.shape[1]:
        raise ValueError(
            f"W.shape[0] ({w.shape[0]}) must equal X.shape[1] ({x.shape[1]})"
        )
    if x.dtype != w.dtype:
        raise ValueError(f"X and W must share a dtype, got {x.dtype} and {w.dtype}")
    if x.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"unsupported dtype {x.dtype}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return True


@tilelang.jit(out_idx=[2], target="npuir")
def rmsnorm(M, N, dtype="float32", eps=1e-5, block_m=None):
    """Build the RMSNorm kernel for a fixed (M, N, dtype, eps) configuration.

    Returns a compiled callable ``kernel(x, w) -> y`` (``Y`` is allocated by
    the JIT wrapper because of ``out_idx=[2]``).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {dtype!r}; expected {_SUPPORTED_DTYPES}")
    M = int(M)
    N = int(N)
    if M <= 0:
        raise ValueError(f"rmsnorm requires M > 0, got M={M}")
    if N <= 0 or N > _MAX_N:
        raise ValueError(f"rmsnorm requires 1 <= N <= {_MAX_N}, got N={N}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")
    if block_m is None:
        block_m = pick_block_m(N, dtype)
    block_m = int(block_m)
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got block_m={block_m}")

    num_blocks = T.ceildiv(M, block_m)
    inv_n = 1.0 / float(N)

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),  # type: ignore
        W: T.Tensor((N,), dtype),  # type: ignore
        Y: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            # GM entry/exit tile, weight tile and fp32 compute buffers.
            x_sh = T.alloc_shared((block_m, N), dtype)
            w_sh = T.alloc_shared((N,), dtype)
            xsq = T.alloc_fragment((block_m, N), "float32")
            acc = T.alloc_fragment((block_m, 1), "float32")
            rstd = T.alloc_fragment((block_m, 1), "float32")

            m0 = cid * block_m
            rows = T.min(block_m, M - m0)

            # Load a full static tile; only the valid rows are copied in.
            T.copy(X[m0 : m0 + rows, 0:N], x_sh[0:rows, 0:N])
            T.copy(W, w_sh)

            if dtype == "float16":
                # The fp16 tile is promoted to fp32 inline; no cached fp32 copy
                # is kept (x_sh / w_sh still hold the originals for step 7).
                for i, j in T.Parallel(block_m, N):
                    xsq[i, j] = T.cast(x_sh[i, j], "float32") * T.cast(
                        x_sh[i, j], "float32"
                    )
            else:
                for i, j in T.Parallel(block_m, N):
                    xsq[i, j] = x_sh[i, j] * x_sh[i, j]

            # Row reduction of x^2: (block_m, N) -> (block_m, 1).
            T.reduce_sum(xsq, acc, dim=1, clear=True)

            # mean + eps, then rstd = 1 / sqrt(.).  vsqrt + vrec keeps the
            # fp32 relative error far below the 1e-4 tolerance (vrsqrt does not).
            for i in T.Parallel(block_m):
                acc[i, 0] = acc[i, 0] * inv_n + eps
            T.vsqrt(acc, rstd)
            T.vrec(rstd, rstd)

            if dtype == "float16":
                for i, j in T.Parallel(block_m, N):
                    xsq[i, j] = (
                        T.cast(x_sh[i, j], "float32")
                        * rstd[i, 0]
                        * T.cast(w_sh[j], "float32")
                    )
                T.vcast(xsq, x_sh, round_mode="rint")
            else:
                for i, j in T.Parallel(block_m, N):
                    x_sh[i, j] = x_sh[i, j] * rstd[i, 0] * w_sh[j]

            # Write back only the valid rows (no host padding / cropping).
            T.copy(x_sh[0:rows, 0:N], Y[m0 : m0 + rows, 0:N])

    return main
