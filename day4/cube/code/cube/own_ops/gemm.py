"""Day 4: hand-written Cube (matrix) kernels in the npuir dialect.

Three tiers, one file:

* A -- ``gemm_single_core``: one *global* block (one AIC core), M=N=K=256,
  fp16 in / fp32 accumulate / fp16 out.  Correctness only.
* B -- ``gemm_blocked``: divisible shapes; the (m, n) tile indices are mapped
  onto a *flattened 1-D* grid (``cid // n_tiles`` / ``cid % n_tiles``) because
  the NPU launcher only parallelises ``blockIdx.x`` -- a 2-D grid is accepted
  by the frontend but serialised into ``scf.for`` inside each block (see
  ``probe_grid.py``).  K loop inside the block.
* C -- ``gemm_dynamic``: symbolic M/N/K, ``T.ceildiv`` grid and runtime
  ``T.min`` tails on both M/N and K.  No ``//`` anywhere, so no tail can be
  silently dropped.

API provenance (all line numbers are in this repo unless noted):

* ``T.alloc_L1/L0C``          tilelang/language/allocate.py:92,104
* ``T.load_nd2nz``            tilelang/language/customize_npuir.py:565
* ``T.gemm == npuir_dot``     tilelang/language/__init__.py:99,
                              tilelang/language/customize_npuir.py:519-562
* ``T.store_fixpipe``         tilelang/language/customize_npuir.py:604
* ``T.Scope("Cube")``         tilelang/language/customize_npuir.py:1939
* address-space <- scope     tilelang/language/allocate.py:79-90,
  (shared.dyn=L1, wmma.accumulator=L0C)   src/target/codegen_npuir.cc:1339-1340
* blockIdx.x -> get_block_idx src/target/codegen_npuir.cc:1292-1300
* capacity constants          see ``arch_report`` / the day-4 note.

Run the correctness tests::

    python -m pytest day4/code/own_ops/gemm.py -s -q

Run the benchmark (device minutes are long; see --rounds)::

    python day4/code/own_ops/gemm.py bench

Dump the generated NPUIR for the three kernels::

    python day4/code/own_ops/gemm.py dump-ir
"""

import argparse
import collections
import contextlib
import csv
import glob
import os
import statistics
import sys
import time

# The tlvenv also carries a pip-installed tilelang that predates the npuir
# Cube ops (no T.load_nd2nz).  Prefer the checkout this file lives in.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.isdir(os.path.join(_REPO_ROOT, "tilelang")) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

if not hasattr(T, "load_nd2nz"):
    raise ImportError(
        f"tilelang at {tilelang.__file__} has no T.load_nd2nz; "
        f"expected the npuir checkout at {_REPO_ROOT}"
    )

DTYPE = "float16"
ACCUM = "float32"

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.abspath(os.path.join(HERE, "..", "logs"))


# ---------------------------------------------------------------------------
# mode helper (same shape as notes/day2 own_ops/abs.py)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def ascend_mode(mode: str = "Expert"):
    """TILELANG_ASCEND_MODE is read at compile time."""
    prev = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = prev


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def gemm_single_core(
    M=256,
    N=256,
    K=256,
    block_M=128,
    block_N=128,
    block_K=64,
    dtype=DTYPE,
    accum_dtype=ACCUM,
):
    """Tier A: the whole C is produced by one grid block (one AIC core).

    The L0C accumulator is 128 KiB on dav-c220 (see ``arch_report``), so a
    256x256 fp32 tile does not fit.  The block therefore walks 2x2 output
    tiles serially; it is still one *global* block, one core.

    L0C clearing: ``initC=(ki == 0)`` resets the accumulator on the first K
    chunk of *each* (mt, nt) tile.  No explicit memset is needed.
    """
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0, (
        "tier A is the exact-division path; use gemm_dynamic for tails"
    )
    m_tiles = M // block_M
    n_tiles = N // block_N
    k_iters = K // block_K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                a_buf = T.alloc_L1((block_M, block_K), dtype)
                b_buf = T.alloc_L1((block_K, block_N), dtype)
                c_buf = T.alloc_L0C((block_M, block_N), accum_dtype)

                for mt in T.serial(m_tiles):
                    for nt in T.serial(n_tiles):
                        for ki in T.serial(k_iters):
                            T.load_nd2nz(
                                A[mt * block_M, ki * block_K],
                                a_buf,
                                [block_M, block_K],
                            )
                            T.load_nd2nz(
                                B[ki * block_K, nt * block_N],
                                b_buf,
                                [block_K, block_N],
                            )
                            T.gemm(
                                a_buf,
                                b_buf,
                                c_buf,
                                initC=(ki == 0),
                                b_transpose=False,
                                size=[block_M, block_K, block_N],
                            )
                        # fixpipe is outside the K loop: one store per tile.
                        T.store_fixpipe(
                            c_buf,
                            C[mt * block_M, nt * block_N],
                            size=[block_M, block_N],
                            enable_nz2nd=True,
                        )

    return main


def gemm_blocked(
    M,
    N,
    K,
    block_M=128,
    block_N=128,
    block_K=64,
    dtype=DTYPE,
    accum_dtype=ACCUM,
):
    """Tier B: static, divisible shapes; one logical block per (M,N) tile.

    The grid is *flattened to 1D*: ``T.Kernel(m_tiles*n_tiles)`` and
    ``by = cid // n_tiles``, ``bx = cid % n_tiles`` -- the same shape as
    examples/gemm/matmul.py:25-28.  A 2D grid is accepted by the frontend
    (src/ir.cc:186-199) but the launcher only parallelises blockIdx.x
    ("only 1D parallelization is supported for NPU",
    tilelang/jit/jit_npu.py:520); the second grid dim is emitted as a
    serial scf.for inside every block (verified on the IR: 1024^3 with 2D
    grid produced ``scf.for 0..8`` around the whole tile body).  So a 2D
    grid would leave ``n_tiles`` blocks on chip, each looping over all M
    tiles -- not what "multi-core split over M/N" means.
    """
    assert M % block_M == 0 and N % block_N == 0, (
        "tier B is the exact-division path; use gemm_dynamic for tails"
    )
    m_tiles = (M + block_M - 1) // block_M
    n_tiles = (N + block_N - 1) // block_N
    k_iters = (K + block_K - 1) // block_K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(n_tiles * m_tiles, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                a_buf = T.alloc_L1((block_M, block_K), dtype)
                b_buf = T.alloc_L1((block_K, block_N), dtype)
                c_buf = T.alloc_L0C((block_M, block_N), accum_dtype)

                off_m = (cid // n_tiles) * block_M
                off_n = (cid % n_tiles) * block_N
                for ki in T.serial(k_iters):
                    T.load_nd2nz(
                        A[off_m, ki * block_K], a_buf, [block_M, block_K]
                    )
                    T.load_nd2nz(
                        B[ki * block_K, off_n], b_buf, [block_K, block_N]
                    )
                    T.gemm(
                        a_buf,
                        b_buf,
                        c_buf,
                        initC=(ki == 0),
                        b_transpose=False,
                        size=[block_M, block_K, block_N],
                    )
                T.store_fixpipe(
                    c_buf, C[off_m, off_n], size=[block_M, block_N], enable_nz2nd=True
                )

    return main


def gemm_dynamic(
    block_M=128,
    block_N=128,
    block_K=64,
    dtype=DTYPE,
    accum_dtype=ACCUM,
):
    """Tier C: symbolic M/N/K plus runtime tails for M, N *and* K.

    Grid is the flattened 1D ``ceildiv(M,bm) * ceildiv(N,bn)`` -- T.ceildiv,
    never ``//``, so the tail tile is launched rather than dropped.
    Every load and the mmad get the true extent ``min(block, dim - offset)``,
    so a shape such as 257x263x271 is legal without padding the input.
    """
    M = T.symbolic("M")
    N = T.symbolic("N")
    K = T.symbolic("K")

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True
        ) as (cid, _):
            with T.Scope("Cube"):
                a_buf = T.alloc_L1((block_M, block_K), dtype)
                b_buf = T.alloc_L1((block_K, block_N), dtype)
                c_buf = T.alloc_L0C((block_M, block_N), accum_dtype)

                n_tiles = T.ceildiv(N, block_N)
                off_m = (cid // n_tiles) * block_M
                off_n = (cid % n_tiles) * block_N
                remain_M = T.min(block_M, M - off_m)
                remain_N = T.min(block_N, N - off_n)

                for ki in T.serial(T.ceildiv(K, block_K)):
                    off_k = ki * block_K
                    remain_K = T.min(block_K, K - off_k)
                    T.load_nd2nz(
                        A[off_m, off_k], a_buf, [remain_M, remain_K]
                    )
                    T.load_nd2nz(
                        B[off_k, off_n], b_buf, [remain_K, remain_N]
                    )
                    T.gemm(
                        a_buf,
                        b_buf,
                        c_buf,
                        initC=(ki == 0),
                        b_transpose=False,
                        size=[remain_M, remain_K, remain_N],
                    )
                T.store_fixpipe(
                    c_buf,
                    C[off_m, off_n],
                    size=[remain_M, remain_N],
                    enable_nz2nd=True,
                )

    return main


# ---------------------------------------------------------------------------
# compilation cache (tilelang also has an on-disk cache)
# ---------------------------------------------------------------------------

_KERNELS = {}


def get_kernel(kind, **kwargs):
    key = (kind, tuple(sorted(kwargs.items())))
    if key in _KERNELS:
        return _KERNELS[key]

    builder = {
        "single": gemm_single_core,
        "blocked": gemm_blocked,
        "dynamic": gemm_dynamic,
    }[kind]
    with ascend_mode("Expert"):
        func = builder(**kwargs)
        kernel = tilelang.compile(func, target="npuir")
    _KERNELS[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# reference + error metrics
# ---------------------------------------------------------------------------


def make_tensors(M, N, K, device="npu"):
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(K, N, dtype=torch.float16, device=device)
    c = torch.zeros(M, N, dtype=torch.float16, device=device)
    return a, b, c


def reference_fp16(a, b):
    """CPU float64 matmul, then cast to the target dtype.

    The cast is deliberate: it isolates compiler/accumulator error from the
    unavoidable fp16 output quantisation.
    """
    return torch.matmul(a.cpu().double(), b.cpu().double()).to(torch.float16)


def error_stats(c, ref):
    """Error metrics computed on the host.

    ``max_rel`` alone is a misleading number for a GEMM: a row/column that
    cancels to ~1e-6 (while rms|C| is ~30) makes err/|ref| arbitrarily
    large without saying anything about kernel quality.  So report:

    * ``max_abs``            -- absolute worst element
    * ``max_abs_over_scale`` -- max_abs / max|C|, the dtype-independent one
    * ``max_rel_nonzero``    -- max relative error over elements that are not
                                cancellation-dominated (|ref| >= 1% of max|C|)
    * ``rmse``, ``n_gt_1ulp`` -- bulk quality and the count of elements that
                                differ by more than one fp16 ulp of themselves
    """
    c32 = c.detach().float().cpu()
    r32 = ref.detach().float().cpu()
    err = (c32 - r32).abs()
    scale = r32.abs().max().clamp_min(1e-12)
    ulp = torch.pow(2.0, torch.floor(torch.log2(c32.abs().clamp_min(2.0**-24))) - 10)
    mask = r32.abs() >= 0.01 * scale
    rel_nonzero = (
        float((err[mask] / r32[mask].abs()).max())
        if bool(mask.any())
        else float("nan")
    )
    return {
        "max_abs": float(err.max()),
        "max_abs_over_scale": float(err.max() / scale),
        "max_rel": float((err / r32.abs().clamp_min(1e-6)).max()),
        "max_rel_nonzero": rel_nonzero,
        "rmse": float(err.pow(2).mean().sqrt()),
        "n_gt_1ulp": int((err > ulp).sum()),
        "n": int(err.numel()),
    }


def fmt_stats(s):
    return (
        f"max_abs={s['max_abs']:.4e} (max_abs/max|C|={s['max_abs_over_scale']:.2e}) "
        f"max_rel={s['max_rel']:.4e} max_rel(|ref|>=1%max)={s['max_rel_nonzero']:.4e} "
        f"rmse={s['rmse']:.2e} n_err>1ulp={s['n_gt_1ulp']}/{s['n']}"
    )


def check(kernel, M, N, K, atol=5e-3, rtol=1e-2):
    """Run one shape, assert the tolerance, return ``error_stats`` as a dict.

    Tolerance rationale
    -------------------
    * accumulation is fp32, so the GEMM's own error grows like
      ``eps_f32 * sqrt(K) * |C|`` -- for K <= 4096 that is ~1e-4 * |C|,
      i.e. rtol=1e-2 leaves two orders of magnitude of margin;
    * the output cast to fp16 contributes at most half an ulp,
      ``|C| * 2^-11 ~= 5e-4 * |C|``, also under rtol=1e-2;
    * atol=5e-3 only matters for near-zero outputs, where |C| is small and
      the fp32 cancellation error is bounded well below 5e-3 for |C| ~ 4*sqrt(K).
    """
    a, b, c = make_tensors(M, N, K)
    kernel(a, b, c)
    torch.npu.synchronize()
    ref = reference_fp16(a, b)
    stats = error_stats(c, ref)
    torch.testing.assert_close(
        c.detach().cpu().float(),
        ref.detach().cpu().float(),
        atol=atol,
        rtol=rtol,
        msg=lambda m: f"{m}\n{fmt_stats(stats)}",
    )
    return stats


# ---------------------------------------------------------------------------
# timing: host / device / end-to-end (the Day-3 three-column protocol)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def pushd(path):
    old = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def device_median(fn, iters=20, log_dir=LOG_DIR, tag="op"):
    """Median device duration from CANN's op_summary CSV.

    ``export_chrome_trace`` is what triggers CANN's own parse pass; without
    it the run directory contains no ``op_summary_*.csv``.
    """
    from torch_npu.profiler import ProfilerActivity, profile

    for _ in range(3):  # warmup, discarded
        fn()
    torch.npu.synchronize()

    with pushd(log_dir):
        before = set(glob.glob("export_only_prof_dir/*"))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
            for _ in range(iters):
                fn()
            torch.npu.synchronize()
        # The tag holds the human label ("compute-bound 4096^3_t1"), which has
        # spaces and '/' -- CANN rejects that as an output path.
        trace_name = f"trace_{abs(hash(tag)) % (1 << 32):08x}.json"
        prof.export_chrome_trace(os.path.join("/tmp", trace_name))
        new = sorted(set(glob.glob("export_only_prof_dir/*")) - before)

    if not new:
        raise RuntimeError(f"[{tag}] profiler produced no run directory")
    csvs = glob.glob(
        os.path.join(log_dir, new[-1], "PROF_*", "mindstudio_profiler_output",
                     "op_summary_*.csv")
    )
    if not csvs:
        raise RuntimeError(f"[{tag}] no op_summary csv under {new[-1]}")

    agg = collections.defaultdict(list)
    for path in csvs:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                agg[row["Op Name"]].append(float(row["Task Duration(us)"].strip()))

    name, durs = max(agg.items(), key=lambda kv: len(kv[1]))
    if len(durs) < iters:
        print(f"  [{tag}] warning: op '{name}' has {len(durs)}/{iters} samples")
    return statistics.median(durs), name, len(durs)


def device_medians_pair(fns, iters=10, log_dir=LOG_DIR, tag="pair"):
    """Median device duration of several callables measured in ONE session.

    Profile start/stop costs ~4 s (CANN parse), so 4 sessions per ABBA round
    would cost ~16 s per round and ~15 min per configuration.  Instead the
    calls are interleaved inside one session as A B B A per iteration, which
    keeps the ABBA structure and cuts the overhead 4x.  Per-op durations are
    separated afterwards by ``Op Name``: torch kernels are prefixed
    ``aclnn...``, ours carries the prim_func name.

    Returns ``({key: median_us}, {key: op_name})``.
    """
    from torch_npu.profiler import ProfilerActivity, profile

    for _ in range(3):  # warmup, discarded
        for fn in fns.values():
            fn()
    torch.npu.synchronize()

    with pushd(log_dir):
        before = set(glob.glob("export_only_prof_dir/*"))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
            for _ in range(iters):
                for fn in fns.values():  # A B
                    fn()
                for fn in reversed(list(fns.values())):  # B A
                    fn()
            torch.npu.synchronize()
        trace_name = f"trace_{abs(hash(tag)) % (1 << 32):08x}.json"
        prof.export_chrome_trace(os.path.join("/tmp", trace_name))
        new = sorted(set(glob.glob("export_only_prof_dir/*")) - before)

    if not new:
        raise RuntimeError(f"[{tag}] profiler produced no run directory")
    csvs = glob.glob(
        os.path.join(log_dir, new[-1], "PROF_*", "mindstudio_profiler_output",
                     "op_summary_*.csv")
    )
    if not csvs:
        raise RuntimeError(f"[{tag}] no op_summary csv under {new[-1]}")

    agg = collections.defaultdict(list)
    for path in csvs:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                agg[row["Op Name"]].append(float(row["Task Duration(us)"].strip()))

    # Each callable ran 2*iters times; drop stray ops with a different count.
    cands = {n: d for n, d in agg.items() if len(d) == 2 * iters}
    if len(cands) != len(fns):
        raise RuntimeError(
            f"[{tag}] expected {len(fns)} ops with {2 * iters} samples, got "
            f"{ {n: len(d) for n, d in agg.items()} }"
        )
    torch_names = [n for n in cands if "aclnn" in n.lower()]
    if len(torch_names) != 1:
        raise RuntimeError(f"[{tag}] cannot tell torch op apart: {list(cands)}")
    torch_name = torch_names[0]
    our_name = next(n for n in cands if n != torch_name)
    med = {"torch": statistics.median(cands[torch_name]),
           "ours": statistics.median(cands[our_name])}
    names = {"torch": torch_name, "ours": our_name}
    return med, names


def host_submit_us(fn, iters=100):
    """Per-call host enqueue cost: timed WITHOUT synchronising each call."""
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) / iters * 1e6


def e2e_us(fn, iters=20):
    """End-to-end: host dispatch + device, one synchronise per call."""
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e6


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q3 - q1


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def _gemm_or_zeros(kind, M, N, K, block_M, block_N, block_K):
    if kind == "dynamic":
        return get_kernel("dynamic", block_M=block_M, block_N=block_N,
                          block_K=block_K)
    return get_kernel(kind, M=M, N=N, K=K, block_M=block_M, block_N=block_N,
                      block_K=block_K)


def abba(label, M, N, K, block_M, block_N, block_K, kind="dynamic",
         rounds=10, iters=10, dtype=torch.float16):
    """A(torch) B(ours) B(ours) A(torch) per round; median + IQR across rounds.

    Reported columns, per Day-3 protocol:
      host_us    -- enqueue only, no per-call sync
      device_us  -- profiler Task Duration (kernel own time)
      e2e_us     -- one sync per call
    """
    kernel = _gemm_or_zeros(kind, M, N, K, block_M, block_N, block_K)
    a, b, c = make_tensors(M, N, K)

    kernel(a, b, c)
    torch.npu.synchronize()
    if M * N * K <= 512 ** 3:
        # CPU float64 reference: measures absolute accuracy (see reference_fp16).
        ref = reference_fp16(a, b)
        ref_note = "fp64 CPU ref"
    else:
        # Device-side reference: cheap enough for 4096^3 and up.  It catches
        # structural bugs (wrong tile offset, dropped tail, stale L0C) but does
        # NOT measure absolute accuracy -- both sides are fp16 with different
        # accumulation order, so the error shown is roughly one output ulp.
        ref = torch.matmul(a, b).cpu()
        ref_note = "torch.matmul on NPU ref (structural check only)"
    stats = error_stats(c, ref)
    ok = torch.allclose(
        c.detach().cpu().float(), ref.float(), atol=5e-3, rtol=1e-2
    )
    print(f"\n=== {label}  M={M} N={N} K={K}  tile={block_M}x{block_N}x{block_K}"
          f"  kind={kind} ===")
    print(f"    correctness: {fmt_stats(stats)} allclose={ok}  [{ref_note}]")
    assert ok, f"{label}: correctness failed"

    # fixed costs, measured once
    host_t = host_submit_us(lambda: torch.matmul(a, b, out=c))
    host_o = host_submit_us(lambda: kernel(a, b, c))
    e2e_t = e2e_us(lambda: torch.matmul(a, b, out=c))
    e2e_o = e2e_us(lambda: kernel(a, b, c))

    per_round = {"torch": [], "ours": []}
    names = {}
    for rnd in range(rounds):
        # One profiler session per round, A B B A inside it (see
        # device_medians_pair: 4 sessions/round cost ~16 s of CANN parsing).
        med, names = device_medians_pair(
            {"torch": lambda: torch.matmul(a, b, out=c),
             "ours": lambda: kernel(a, b, c)},
            iters, tag=f"{label}_r{rnd}")
        t_us, o_us = med["torch"], med["ours"]
        per_round["torch"].append(t_us)
        per_round["ours"].append(o_us)
        print(f"  round {rnd:>2}  torch {t_us:9.2f} us  ours {o_us:9.2f} us"
              f"  ratio {o_us / t_us:5.3f}x")

    flops = 2.0 * M * N * K
    traffic = (M * K + K * N + M * N) * 2  # fp16 bytes, no reuse assumption
    for who in ("torch", "ours"):
        vals = per_round[who]
        med = statistics.median(vals)
        print(f"  {who:<5} device median {med:9.2f} us  IQR {iqr(vals):7.2f}"
              f"  min {min(vals):7.2f}  max {max(vals):7.2f}"
              f"  -> {flops / med / 1e6:7.2f} TFLOPs"
              f"  | equiv HBM {traffic / med / 1e3:7.1f} GB/s")
    ratios = [o / t for o, t in zip(per_round["ours"], per_round["torch"])]
    print(f"  ratio median {statistics.median(ratios):.3f}x  "
          f"IQR {iqr(ratios):.3f}")
    print(f"  host: torch {host_t:.1f} us/call, ours {host_o:.1f} us/call")
    print(f"  e2e : torch {e2e_t:.1f} us/call, ours {e2e_o:.1f} us/call")
    print(f"  device op names: torch={names.get('torch')!r} "
          f"ours={names.get('ours')!r}")
    return {
        "label": label, "M": M, "N": N, "K": K, "kind": kind,
        "tile": (block_M, block_N, block_K),
        "torch_us": statistics.median(per_round["torch"]),
        "our_us": statistics.median(per_round["ours"]),
        "our_iqr": iqr(per_round["ours"]),
        "torch_iqr": iqr(per_round["torch"]),
        "host_torch_us": host_t, "host_our_us": host_o,
        "e2e_torch_us": e2e_t, "e2e_our_us": e2e_o,
        "tflops": flops / statistics.median(per_round["ours"]) / 1e6,
        "hbm_gbs": traffic / statistics.median(per_round["ours"]) / 1e3,
    }


def arch_report():
    """Query the device instead of copying a core count from a note."""
    from tilelang.utils import NPUUtils

    u = NPUUtils.get()
    print("=== device / arch (runtime query) ===")
    print(f"  get_aicore_num()          = {u.get_aicore_num()}")
    print(f"  get_aicube_core_num()     = {u.get_aicube_core_num()}")
    print(f"  get_aivector_core_num()   = {u.get_aivector_core_num()}")
    print(f"  get_arch()                = {u.get_arch()}")
    print("  capacity constants in source (tilelang/utils/npu_arch.py:16-36):")
    print("    Ascend910B: cores=30 UB=192K L1=1M L0A=64K L0B=64K L0C=512K "
          "cube=[16,16,16]")
    print("  CANN dav-c220 constants (cann-9.0.0/.../utils/"
          "kernel_utils_constants.h:127-135):")
    print("    L1=512K-128B UB=192K L0C=128K  L0A=L0B=64K")
    print("  Ascend910B3.ini: cube_core_cnt=20, l1_size=524288, "
          "cube_m/n/k_size=16/16/16")
    print("  -> the two L0C numbers disagree (512K vs 128K); the kernels below "
          "assume the conservative 128K.")


def bench_main(rounds=10, iters=10):
    arch_report()
    rows = []

    # Regime 1: compute-bound (large K).
    rows.append(abba("compute-bound 4096^3", 4096, 4096, 4096, 128, 128, 64,
                     kind="blocked", rounds=rounds, iters=iters))

    # Regime 2: memory-bound (small K, large M/N).
    # 8192^2x64 is kept as a *record only*: its 128 MiB output stays inside the
    # 192 MiB L2, so torch's 69.58 us implies ~2.26 TB/s write bandwidth, above
    # the 1.27 TB/s HBM steady state -- not a HBM measurement (see
    # probe_membound_torch.py).  The usable memory-bound point is 16384^2x64
    # (C = 512 MiB), measured by probe_pair_bias.py.
    rows.append(abba("memory-bound 8192x8192x64", 8192, 8192, 64, 128, 128, 64,
                     kind="blocked", rounds=rounds, iters=iters))

    # K sweep at fixed M=N=4096: where does TFLOPs stop scaling?
    print("\n\n######## K sweep, M=N=4096 (dynamic kernel, one compile) ########")
    for K in (64, 128, 256, 512, 1024, 2048, 4096):
        rows.append(abba(f"K-sweep K={K}", 4096, 4096, K, 128, 128, 64,
                         kind="dynamic", rounds=rounds, iters=iters))

    # Machine-readable summary next to the raw log.
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, "bench_summary.csv")
    fields = ["label", "M", "N", "K", "kind", "tile", "torch_us", "torch_iqr",
              "our_us", "our_iqr", "host_torch_us", "host_our_us",
              "e2e_torch_us", "e2e_our_us", "tflops", "hbm_gbs"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\nwrote {csv_path}")
    print("\n=== summary (device time, median of per-round ABBA medians) ===")
    print(f"{'config':<26}{'ours_us':>10}{'IQR':>8}{'torch_us':>10}"
          f"{'ratio':>8}{'TFLOPs':>9}{'eq.HBM':>10}")
    for r in rows:
        print(f"{r['label']:<26}{r['our_us']:>10.2f}{r['our_iqr']:>8.2f}"
              f"{r['torch_us']:>10.2f}{r['our_us'] / r['torch_us']:>8.3f}"
              f"{r['tflops']:>9.2f}{r['hbm_gbs']:>10.1f}")


# ---------------------------------------------------------------------------
# IR dump
# ---------------------------------------------------------------------------


def dump_ir():
    os.makedirs(LOG_DIR, exist_ok=True)
    jobs = [
        ("tierA_single_256x256x256", get_kernel(
            "single", M=256, N=256, K=256, block_M=128, block_N=128,
            block_K=64)),
        ("tierB_blocked_1024x1024x1024", get_kernel(
            "blocked", M=1024, N=1024, K=1024, block_M=128, block_N=128,
            block_K=64)),
        ("tierC_dynamic", get_kernel("dynamic", block_M=128, block_N=128,
                                     block_K=64)),
    ]
    for name, kernel in jobs:
        path = os.path.join(LOG_DIR, f"{name}.mlir")
        with open(path, "w") as fh:
            fh.write(kernel.mlir_content)
        print(f"wrote {path} ({len(kernel.mlir_content)} bytes)")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_single_block_256():
    """Tier A: one global block, exact division 256^3."""
    kernel = get_kernel("single", M=256, N=256, K=256, block_M=128,
                        block_N=128, block_K=64)
    stats = check(kernel, 256, 256, 256)
    print(f"\n[single 256^3] {fmt_stats(stats)}")


def test_blocked_divisible_1024():
    """Tier B: 8x8 = 64 logical blocks on a flattened 1-D grid, K loop inside."""
    kernel = get_kernel("blocked", M=1024, N=1024, K=1024, block_M=128,
                        block_N=128, block_K=64)
    stats = check(kernel, 1024, 1024, 1024)
    print(f"\n[blocked 1024^3] {fmt_stats(stats)}")


def test_dynamic_k_tail():
    """Tier C: K is not a multiple of block_K (271 = 4*64 + 15)."""
    kernel = get_kernel("dynamic", block_M=128, block_N=128, block_K=64)
    stats = check(kernel, 256, 256, 271)
    print(f"\n[dynamic K tail 256x256x271] {fmt_stats(stats)}")


def test_dynamic_mn_tail():
    """Tier C: M and N are not multiples of the tile (tail 1 row, 7 cols)."""
    kernel = get_kernel("dynamic", block_M=128, block_N=128, block_K=64)
    stats = check(kernel, 257, 263, 256)
    print(f"\n[dynamic MN tail 257x263x256] {fmt_stats(stats)}")


def test_dynamic_all_tail():
    """Tier C: M=257, N=263, K=271 -- nothing divides anything."""
    kernel = get_kernel("dynamic", block_M=128, block_N=128, block_K=64)
    stats = check(kernel, 257, 263, 271)
    print(f"\n[dynamic all tail 257x263x271] {fmt_stats(stats)}")


def test_dynamic_single_row():
    """Tier C stress: M=1 (a single row), N=17, K=15 (< one fractal on K)."""
    kernel = get_kernel("dynamic", block_M=128, block_N=128, block_K=64)
    stats = check(kernel, 1, 17, 15)
    print(f"\n[dynamic 1x17x15] {fmt_stats(stats)}")


def test_dynamic_rectangular_large():
    """Tier C: rectangular, multi-block, K divisible -- regression guard."""
    kernel = get_kernel("dynamic", block_M=128, block_N=128, block_K=64)
    stats = check(kernel, 512, 768, 2048)
    print(f"\n[dynamic 512x768x2048] {fmt_stats(stats)}")


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="bench",
                        choices=["bench", "dump-ir", "check", "arch"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.command == "bench":
        bench_main(rounds=args.rounds, iters=args.iters)
    elif args.command == "dump-ir":
        dump_ir()
    elif args.command == "arch":
        arch_report()
    else:
        for fn in (
            test_single_block_256,
            test_blocked_divisible_1024,
            test_dynamic_k_tail,
            test_dynamic_mn_tail,
            test_dynamic_all_tail,
            test_dynamic_single_row,
            test_dynamic_rectangular_large,
        ):
            fn()
            print(f"{fn.__name__}: PASS")
