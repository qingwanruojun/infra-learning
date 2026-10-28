import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


def build(M, N, block_M):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), "float32"),
        R1: T.Tensor((M, 1), "float32"),
        R2: T.Tensor((M, 1), "float32"),
        R3: T.Tensor((M, 1), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            x = T.alloc_ub((block_M, N), "float32")
            xsq = T.alloc_ub((block_M, N), "float32")
            s = T.alloc_ub((block_M, 1), "float32")
            r = T.alloc_ub((block_M, 1), "float32")
            t = T.alloc_ub((block_M, 1), "float32")
            t2 = T.alloc_ub((block_M, 1), "float32")
            inv_n = 1.0 / float(N)
            eps = 1e-5

            T.copy(X[cid * block_M : (cid + 1) * block_M, :], x)
            T.vmul(x, x, xsq)
            T.reduce(xsq, s, dims=[1], reduce_mode="sum")
            T.vmul(s, inv_n, s)
            T.vadd(s, eps, s)

            # (1) hardware rsqrt
            T.vrsqrt(s, r)
            T.copy(r, R1[cid * block_M : (cid + 1) * block_M, :])

            # (2) sqrt + reciprocal
            T.vsqrt(s, t)
            T.vrec(t, r)
            T.copy(r, R2[cid * block_M : (cid + 1) * block_M, :])

            # (3) one Newton step on top of hardware rsqrt
            T.vrsqrt(s, t)  # t = r0
            T.vmul(t, t, r)  # r = r0^2
            T.vmul(r, s, r)  # r = s*r0^2
            T.vmul(r, -0.5, r)  # r = -0.5*s*r0^2
            T.vadd(r, 1.5, r)  # r = 1.5 - 0.5*s*r0^2
            T.vmul(t, r, t2)  # t2 = r0 * (...)
            T.copy(t2, R3[cid * block_M : (cid + 1) * block_M, :])

    return main


torch.npu.set_device(0)
M, N, block_M = 1024, 1024, 8
tilelang.cache.clear_cache()
k = tilelang.compile(build(M, N, block_M), target="npuir")
x = torch.randn(M, N, device="npu")
outs = [torch.zeros(M, 1, device="npu") for _ in range(3)]
k(x, *outs)
ref = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-5)
for i, o in enumerate(outs, 1):
    rel = ((o.cpu() - ref.cpu()).abs() / ref.cpu().abs()).max().item()
    print(f"method{i} max_rel={rel:.3e}")
