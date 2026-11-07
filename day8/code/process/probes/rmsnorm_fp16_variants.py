"""Compile-only probe for the fp16 UB budget (does not run on the NPU)."""

import sys

import tilelang
from tilelang import language as T

sys.path.insert(0, "/mnt/workspace/learn/infra-learning/day8/code/agent_output/custom/rmsnorm")


def build(M, N, block_m, dtype, eps, pass_configs):
    @tilelang.jit(out_idx=[2], target="npuir", pass_configs=pass_configs)
    def factory(M, N, block_m, dtype, eps):
        num_blocks = T.ceildiv(M, block_m)
        inv_n = 1.0 / float(N)

        @T.prim_func
        def main(
            X: T.Tensor((M, N), dtype),  # type: ignore
            W: T.Tensor((N,), dtype),  # type: ignore
            Y: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(num_blocks, is_npu=True) as (cid, _):
                x_sh = T.alloc_shared((block_m, N), dtype)
                w_sh = T.alloc_shared((N,), dtype)
                xsq = T.alloc_fragment((block_m, N), "float32")
                acc = T.alloc_fragment((block_m, 1), "float32")
                rstd = T.alloc_fragment((block_m, 1), "float32")

                m0 = cid * block_m
                rows = T.min(block_m, M - m0)
                T.copy(X[m0 : m0 + rows, 0:N], x_sh[0:rows, 0:N])
                T.copy(W, w_sh)

                if dtype == "float16":
                    for i, j in T.Parallel(block_m, N):
                        xsq[i, j] = T.cast(x_sh[i, j], "float32") * T.cast(
                            x_sh[i, j], "float32"
                        )
                else:
                    for i, j in T.Parallel(block_m, N):
                        xsq[i, j] = x_sh[i, j] * x_sh[i, j]

                T.reduce_sum(xsq, acc, dim=1, clear=True)
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

                T.copy(x_sh[0:rows, 0:N], Y[m0 : m0 + rows, 0:N])

        return main

    return factory(M, N, block_m, dtype, eps)


def probe(label, block_m, pass_configs):
    try:
        build(1024, 1024, block_m, "float16", 1e-5, pass_configs)
        print(f"PROBE {label} bm={block_m}: COMPILE_OK")
        return True
    except Exception as exc:  # noqa: BLE001
        text = str(exc).replace("\n", " ")
        print(f"PROBE {label} bm={block_m}: FAIL {text[:300]}")
        return False


if __name__ == "__main__":
    key = tilelang.PassConfigKey
    probe("default", 16, None)
    probe("default", 8, None)
    probe("default", 10, None)
    probe("default", 12, None)
    probe("nomultibuf", 16, {key.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False})
    probe("nomultibuf", 12, {key.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False})
