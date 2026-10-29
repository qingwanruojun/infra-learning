"""Diagnose uint8/bool handling for BitwiseNotFwdOp.

    python3 -u spike_bitnot_diag.py --case uint8diag
    python3 -u spike_bitnot_diag.py --case bool_i8
"""
import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402


def build_vnot(dtype):
    @tilelang.jit(out_idx=[1], target="npuir")
    def kern(N, block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                offset = cid * block_size
                tail = T.min(block_size, shape - offset)
                T.copy(x[offset : offset + tail], x_ub[0:tail])
                T.vnot(x_ub, y_ub)
                T.copy(y_ub[0:tail], y[offset : offset + tail])

        return main

    return kern


def build_vnot_prealloc(dtype):
    """out_idx=[] so the caller allocates the output tensor (uint8 workaround)."""
    @tilelang.jit(out_idx=[], target="npuir")
    def kern(N, block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                offset = cid * block_size
                tail = T.min(block_size, shape - offset)
                T.copy(x[offset : offset + tail], x_ub[0:tail])
                T.vnot(x_ub, y_ub)
                T.copy(y_ub[0:tail], y[offset : offset + tail])

        return main

    return kern


def case_uint8_prealloc(n=1000, block=256):
    f = build_vnot_prealloc("uint8")(n, block)
    x = torch.randint(0, 256, (n,), dtype=torch.int64).to(torch.uint8).npu()
    y = torch.empty_like(x)
    f(x, y, n)
    ref = torch.bitwise_not(x.cpu())
    print("uint8_prealloc", "PASS" if torch.equal(y.cpu(), ref) else "MISMATCH", y.dtype)


def build_bool_xor(dtype):
    @tilelang.jit(out_idx=[1], target="npuir")
    def kern(N, block_size):
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype), shape: T.int32):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((block_size,), dtype)
                ones_ub = T.alloc_ub((block_size,), dtype)
                y_ub = T.alloc_ub((block_size,), dtype)
                offset = cid * block_size
                tail = T.min(block_size, shape - offset)
                T.copy(x[offset : offset + tail], x_ub[0:tail])
                T.fill(ones_ub, T.cast(1, dtype))
                T.vxor(x_ub, ones_ub, y_ub)
                T.copy(y_ub[0:tail], y[offset : offset + tail])

        return main

    return kern


def case_uint8diag(n=1024, block=256):
    fn = build_vnot("uint8")
    f = fn(n, block)
    pf = f.prim_func
    print("prim_func params:", [(p.name, str(p.dtype)) for p in pf.params])
    print("jit out_idx dtype:", f.out_idx if hasattr(f, "out_idx") else "n/a")
    x = torch.randint(0, 256, (n,), dtype=torch.int64).to(torch.uint8).npu()
    y = f(x, n)
    print("out dtype:", y.dtype, "numel:", y.numel(), "shape:", tuple(y.shape))
    ref = torch.bitwise_not(x.cpu())
    if y.dtype == torch.uint8:
        print("equal:", torch.equal(y.cpu(), ref))
    else:
        raw = y.cpu().view(torch.uint8)
        print("raw bytes:", raw.numel(), "head x:", x.cpu()[:8].tolist(),
              "head y bytes:", raw[:8].tolist(), "head ref:", ref[:8].tolist())
        print("first n bytes equal:", torch.equal(raw[:n], ref))


def case_bool_i8(n=1024, block=256):
    dtype = "int8"
    fn = build_bool_xor(dtype)
    xb = torch.randint(0, 2, (n,), dtype=torch.int64).to(torch.bool).npu()
    y = fn(n, block)(xb.view(torch.int8), n)
    print("out dtype:", y.dtype)
    got = y.view(torch.bool).cpu()
    ref = torch.bitwise_not(xb.cpu())
    print("bool_i8 PASS" if torch.equal(got, ref) else "bool_i8 MISMATCH", got.dtype)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    choices=["uint8diag", "bool_i8", "uint8_prealloc"])
    ap.add_argument("--n", type=int, default=1024)
    a = ap.parse_args()
    if a.case == "uint8diag":
        case_uint8diag()
    elif a.case == "uint8_prealloc":
        case_uint8_prealloc(a.n, 256)
    else:
        case_bool_i8(a.n, 256)
