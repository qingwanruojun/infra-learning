#!/usr/bin/env python
"""Benchmark the Developer-mode FlashAttention npuir kernel.

The upstream example (examples/flash_attn_npuir_dev.py) hardcodes seq_len=512
and only checks correctness. This script keeps the same kernel body but makes
seq_len a compile-time argument, then reports host / device / e2e timing for a
seq_len sweep.

Usage:
    python day5-flashattn/code/bench_fa_dev.py --seq-lens 512 1024 2048 4096
"""

import argparse
import csv
import os
import statistics
import time

import torch
import torch_npu

import tilelang
import tilelang.language as T


def build_kernel(seq_len, dim, block_M, block_N, block_K, dtype, accum_dtype):
    # seq_len/dim are explicit jit args so the kernel cache key includes them;
    # a closure over a module global would silently reuse the wrong kernel.
    @tilelang.jit(out_idx=[-1], target="npuir")
    def online_flash_attention(seq_len, dim, block_M, block_N, block_K,
                               dtype, accum_dtype):
        shape = [seq_len, dim]

        @T.prim_func
        def flash_attention(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), is_npu=True) as (cid, _):
                offset = cid * block_M
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                T.copy(Q[offset : offset + block_M, 0:dim], Q_shared)

                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                scores = T.alloc_fragment([block_M, block_N], accum_dtype)
                scores_cast = T.alloc_fragment([block_M, block_N], dtype)
                correction = T.alloc_fragment([block_M, 1], accum_dtype)
                local_max = T.alloc_fragment([block_M, 1], accum_dtype)
                local_sum = T.alloc_fragment([block_M, 1], accum_dtype)
                acc_m = T.alloc_fragment([block_M, 1], accum_dtype)
                acc_l = T.alloc_fragment([block_M, 1], accum_dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                tmp = T.alloc_fragment([block_M, block_N], accum_dtype)
                tmp1 = T.alloc_fragment([block_M, 1], accum_dtype)
                new_max = T.alloc_fragment([block_M, 1], accum_dtype)
                scales = T.alloc_fragment([block_M, block_N], accum_dtype)

                value_zero = 0
                scale = (1.0 / dim) ** 0.5
                value_min = -T.infinity(accum_dtype)
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, acc_l)
                T.vbrc(value_min, acc_m)
                T.vbrc(scale, scales)

                for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=2):
                    T.copy(K[k * block_N : (k + 1) * block_N, 0:dim], K_shared)
                    T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)

                    T.vmul(scores, scales, scores)
                    T.reduce_max(scores, local_max, dim=1)
                    T.vmax(acc_m, local_max, new_max)
                    T.vsub(acc_m, new_max, tmp1)
                    T.vexp(tmp1, correction)
                    T.vsub(scores, new_max, tmp)
                    T.vexp(tmp, scores)
                    T.reduce_sum(scores, local_sum, dim=1)
                    T.vmul(acc_l, correction, acc_l)
                    T.vadd(acc_l, local_sum, acc_l)
                    T.vmul(acc_o, correction, acc_o)
                    T.vcast(scores, scores_cast, round_mode="rint")
                    T.vbrc(value_zero, tmp1)
                    T.vadd(tmp1, new_max, acc_m)

                    T.copy(V[k * block_N : (k + 1) * block_N, 0:dim], V_shared)
                    T.gemm(scores_cast, V_shared, acc_o, initC=False)

                T.vdiv(acc_o, acc_l, acc_o)
                O_cast = T.alloc_shared([block_M, dim], dtype)
                T.vcast(acc_o, O_cast, round_mode="rint")
                real_m = T.min(block_M, seq_len - cid * block_M)
                T.copy(O_cast, Output[cid * block_M : cid * block_M + real_m, 0:dim])

        return flash_attention

    return online_flash_attention(seq_len, dim, block_M, block_N, block_K,
                                  dtype, accum_dtype)


def stats(ms):
    med = statistics.median(ms)
    q1, q3 = statistics.quantiles(ms, n=4)[0], statistics.quantiles(ms, n=4)[2]
    return med, q3 - q1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--limit", type=int, default=8192,
                        help="skip seq_len > limit")
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    torch.npu.set_device(0)

    rows = []
    for seq_len in args.seq_lens:
        if seq_len > args.limit:
            print(f"skip seq_len={seq_len} (> --limit {args.limit})")
            continue

        print(f"\n=== seq_len={seq_len} dim={args.dim} "
              f"block={args.block_m}/{args.block_n}/{args.block_k} ===", flush=True)
        t_compile = time.perf_counter()
        kernel = build_kernel(seq_len, args.dim, args.block_m, args.block_n,
                              args.block_k, "float16", "float32")
        compile_s = time.perf_counter() - t_compile
        print(f"compile: {compile_s:.1f} s", flush=True)

        shape = [seq_len, args.dim]
        q = torch.randn(shape, dtype=torch.float16).npu()
        k = torch.randn(shape, dtype=torch.float16).npu()
        v = torch.randn(shape, dtype=torch.float16).npu()

        out = kernel(q, k, v)

        scale = (1.0 / args.dim) ** 0.5
        ref = (torch.nn.functional.softmax(
            (q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        max_abs = (out.float() - ref.float()).abs().max().item()
        print(f"correctness: assert_close passed, max_abs={max_abs:.3e}", flush=True)

        for _ in range(args.warmup):
            kernel(q, k, v)
        torch.npu.synchronize()

        # host: enqueue 20 calls, sync once at the end
        n_host = 20
        t0 = time.perf_counter()
        for _ in range(n_host):
            kernel(q, k, v)
        host_us = (time.perf_counter() - t0) / n_host * 1e6
        torch.npu.synchronize()

        # device: npu events around each call
        dev_ms = []
        for _ in range(args.rounds):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            kernel(q, k, v)
            end.record()
            torch.npu.synchronize()
            dev_ms.append(start.elapsed_time(end))

        # e2e: perf_counter with a sync per call
        e2e_ms = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            kernel(q, k, v)
            torch.npu.synchronize()
            e2e_ms.append((time.perf_counter() - t0) * 1e3)

        # torch reference: manual 3-op chain, same measurement
        for _ in range(args.warmup):
            torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v
        torch.npu.synchronize()
        ref_dev_ms = []
        for _ in range(args.rounds):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v
            end.record()
            torch.npu.synchronize()
            ref_dev_ms.append(start.elapsed_time(end))
        ref_e2e_ms = []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1).to(torch.float16) @ v
            torch.npu.synchronize()
            ref_e2e_ms.append((time.perf_counter() - t0) * 1e3)

        med, iqr = stats(dev_ms)
        ref_med, ref_iqr = stats(ref_dev_ms)
        e2e_med, e2e_iqr = stats(e2e_ms)
        ref_e2e_med, _ = stats(ref_e2e_ms)
        flops = 4 * seq_len * seq_len * args.dim  # QK + PV, each 2*M*N*K
        tflops = flops / (med * 1e-3) / 1e12

        row = {
            "seq_len": seq_len,
            "dim": args.dim,
            "block": f"{args.block_m}/{args.block_n}/{args.block_k}",
            "compile_s": round(compile_s, 1),
            "host_us": round(host_us, 1),
            "device_ms_med": round(med, 3),
            "device_ms_iqr": round(iqr, 3),
            "e2e_ms_med": round(e2e_med, 3),
            "e2e_ms_iqr": round(e2e_iqr, 3),
            "torch_device_ms_med": round(ref_med, 3),
            "torch_device_ms_iqr": round(ref_iqr, 3),
            "torch_e2e_ms_med": round(ref_e2e_med, 3),
            "ratio_device": round(med / ref_med, 2),
            "tflops": round(tflops, 2),
            "max_abs": f"{max_abs:.3e}",
        }
        rows.append(row)
        print(
            f"host={row['host_us']} us/call  device={row['device_ms_med']} ms "
            f"[IQR {row['device_ms_iqr']}]  e2e={row['e2e_ms_med']} ms  "
            f"torch device={row['torch_device_ms_med']} ms  "
            f"ratio={row['ratio_device']}x  {row['tflops']} TFLOPs",
            flush=True,
        )

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
