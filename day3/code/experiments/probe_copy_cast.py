import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


def kern(M, N, idtype, odtype):
    @T.prim_func
    def main(A: T.Tensor((M, N), idtype), B: T.Tensor((M, N), odtype)):
        with T.Kernel(1, is_npu=True) as (cid, _):
            a = T.alloc_ub((M, N), idtype)
            b = T.alloc_ub((M, N), odtype)
            T.copy(A, a)
            T.copy(a, b)
            T.copy(b, B)

    return main


for idt, odt in [("float16", "float32"), ("float32", "float16"), ("float32", "float32")]:
    tilelang.cache.clear_cache()
    k = tilelang.compile(kern(64, 64, idt, odt), target="npuir")
    a = torch.randn(64, 64, dtype=getattr(torch, idt), device="npu")
    b = torch.zeros(64, 64, dtype=getattr(torch, odt), device="npu")
    k(a, b)
    d = (b.cpu().float() - a.cpu().float()).abs().max().item()
    print(idt, "->", odt, "max diff", d)
