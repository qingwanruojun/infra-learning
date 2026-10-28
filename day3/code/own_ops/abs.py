"""Hand-written abs kernel in the npuir dialect (own_ops/abs.py).

The structure intentionally mirrors ``examples/elementwise/vec_add_1d.py``:

* one block per ``block_N`` elements, so the grid is ``T.ceildiv(N, block_N)``
  (NOT ``N // block_N`` -- the example silently drops the tail when ``N`` is
  not a multiple of ``block_N``);
* a runtime ``shape`` scalar drives ``tail = min(block_N, shape - offset)`` so
  the last block only touches the elements that actually exist;
* the element-wise op is the npuir intrinsic ``T.vabs`` (``npui_abs``).

Two buffer flavours are provided so the same kernel can be measured in both
Ascend modes:

* ``memory="ub"``    -> ``T.alloc_ub``    + ``TILELANG_ASCEND_MODE=Expert``
* ``memory="shared"``-> ``T.alloc_shared``+ ``TILELANG_ASCEND_MODE=Developer``

Run the tests with::

    python -m pytest own_ops/abs.py -s -q

Run the benchmark / knob table with::

    python own_ops/abs.py
"""

import os
import contextlib

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


def _mode_for_memory(memory: str) -> str:
    if memory not in ("ub", "shared"):
        raise ValueError(f"memory must be 'ub' or 'shared', got {memory!r}")
    return "Expert" if memory == "ub" else "Developer"


@contextlib.contextmanager
def ascend_mode(mode: str):
    """Temporarily set TILELANG_ASCEND_MODE (read at compile time)."""
    prev = os.environ.get("TILELANG_ASCEND_MODE")
    os.environ["TILELANG_ASCEND_MODE"] = mode
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILELANG_ASCEND_MODE", None)
        else:
            os.environ["TILELANG_ASCEND_MODE"] = prev


def abs_kernel(N, block_N, dtype="float32", memory="ub"):
    """Build the TIR prim_func for ``B = abs(A)`` over a 1-D tensor of length N.

    Args:
        N: logical number of elements (also the tensor length).
        block_N: elements handled by one AI-vector block.
        dtype: element dtype.
        memory: ``"ub"`` (Expert) or ``"shared"`` (Developer).
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if block_N <= 0:
        raise ValueError(f"block_N must be positive, got {block_N}")

    alloc = T.alloc_ub if memory == "ub" else T.alloc_shared
    num_blocks = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, _):
            a_buf = alloc((block_N,), dtype)
            b_buf = alloc((block_N,), dtype)

            offset = cid * block_N
            # Runtime tail: the last block may own fewer than block_N elements.
            tail = T.min(block_N, shape - offset)

            T.copy(A[offset : offset + tail], a_buf[0:tail])
            T.vabs(a_buf, b_buf)
            T.copy(b_buf[0:tail], B[offset : offset + tail])

    return main


def compile_abs(N, block_N, dtype="float32", memory="ub", clear_cache=True):
    """Compile an abs kernel for the given shape / buffer flavour."""
    if clear_cache:
        tilelang.cache.clear_cache()
    with ascend_mode(_mode_for_memory(memory)):
        func = abs_kernel(N, block_N, dtype=dtype, memory=memory)
        return tilelang.compile(func, target="npuir")


def make_tensors(N, dtype="float32"):
    torch_dtype = getattr(torch, dtype)
    a = torch.randn(N, dtype=torch_dtype, device="npu")
    # Include negative values / zeros so abs is actually exercised.
    a[::7] *= -1
    b = torch.zeros(N, dtype=torch_dtype, device="npu")
    return a, b


def run_abs(kernel, a, b, N):
    kernel(a, b, N)


def torch_ref(a):
    return torch.abs(a)


def check(N, block_N, dtype="float32", memory="ub", rtol=None, atol=None):
    kernel = compile_abs(N, block_N, dtype=dtype, memory=memory)
    a, b = make_tensors(N, dtype=dtype)
    run_abs(kernel, a, b, N)
    torch.npu.synchronize()

    ref = torch_ref(a)
    if rtol is None:
        rtol = 1e-4 if dtype == "float32" else 1e-2
    if atol is None:
        atol = 1e-4 if dtype == "float32" else 1e-2
    torch.testing.assert_close(b, ref, rtol=rtol, atol=atol)

    # Extra guard: the tail element of a non-divisible shape must be written.
    if N % block_N != 0:
        tail = N % block_N
        torch.testing.assert_close(
            b[-tail:],
            ref[-tail:],
            rtol=rtol,
            atol=atol,
            msg=f"tail of {tail} element(s) was not computed",
        )
    return b


def benchmark(kernel, a, b, N, warmup=3, iters=10, rounds=5):
    """Per-call latency after compile, in microseconds.

    The first call includes compilation / launcher setup, so warm up first and
    synchronize before every timing window.  Each round averages ``iters``
    calls; the returned median over ``rounds`` is far less noisy than a single
    window (observed spread is ~5%).
    """
    kernel(a, b, N)
    torch.npu.synchronize()
    for _ in range(warmup):
        kernel(a, b, N)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            kernel(a, b, N)
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end) / iters * 1000.0)  # us
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


def test_abs_divisible():
    """N=4096, block_N=4096 -> exactly one block, no tail."""
    check(N=4096, block_N=4096, dtype="float32", memory="ub")


def test_abs_tail_one():
    """N=4097, block_N=4096 -> 2 blocks, the second handles exactly 1 element."""
    check(N=4097, block_N=4096, dtype="float32", memory="ub")


def test_abs_tail_multi_block():
    """N=4100, block_N=1024 -> 5 blocks, tail = 4 elements."""
    check(N=4100, block_N=1024, dtype="float32", memory="ub")


def test_abs_shared():
    """Same op with alloc_shared in Developer mode."""
    check(N=4097, block_N=4096, dtype="float32", memory="shared")


def test_abs_fp16():
    check(N=4097, block_N=1024, dtype="float16", memory="ub", rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# knob-table measurements
# ---------------------------------------------------------------------------


_CACHE = {}


def _kernel_for(N, block_N, memory="ub", dtype="float32"):
    key = (N, block_N, memory, dtype)
    if key not in _CACHE:
        _CACHE[key] = compile_abs(N, block_N, dtype=dtype, memory=memory)
    a, b = make_tensors(N, dtype=dtype)
    return _CACHE[key], a, b


def _measure(N, block_N, memory="ub", dtype="float32"):
    kernel, a, b = _kernel_for(N, block_N, memory=memory, dtype=dtype)
    stats = benchmark(kernel, a, b, N, iters=10, rounds=5)
    us = stats["median_us"]
    itemsize = 4 if dtype == "float32" else 2
    gbps = N * itemsize * 2 / (us * 1e-6) / 1e9
    return us, gbps, stats


def _print_row(label, st):
    us = st["median_us"]
    gbps = st["gbps"]
    print(
        f"  {label:22s} {us:8.2f} us  {gbps:6.1f} GB/s"
        f"  [min {st['min_us']:.1f}, max {st['max_us']:.1f}]"
    )


def main():
    torch.npu.set_device(0)

    print("== abs: tile size (N=1048576, alloc_ub) ==")
    for block_N in (1024, 2048, 4096):
        _, gbps, st = _measure(1048576, block_N)
        st["gbps"] = gbps
        _print_row(f"block_N={block_N}", st)

    print("== abs: alignment (block_N=4096, alloc_ub) ==")
    # 1048575 -> last block has 4095 elements; 1048577 -> tail of 1 element
    # (the 32 B-alignment "performance cliff" case from the Day-2 notes).
    for N in (1048576, 1048575, 1048577):
        _, gbps, st = _measure(N, 4096)
        st["gbps"] = gbps
        _print_row(f"N={N}", st)

    print("== abs: block-start alignment (N=1048576, alloc_ub) ==")
    # The N sweep above changes only the tail length; every block still starts
    # at cid*4096 (16 KiB aligned).  With block_N=1023/1025 the start of every
    # block is shifted by a non-multiple of the vector width, which is the
    # closest user-level proxy for vectorization/alignment in this DSL.
    for block_N in (1023, 1024, 1025):
        _, gbps, st = _measure(1048576, block_N)
        st["gbps"] = gbps
        _print_row(f"block_N={block_N}", st)

    print("== abs: buffer strategy (N=1048576, block_N=4096) ==")
    for memory in ("ub", "shared"):
        _, gbps, st = _measure(1048576, 4096, memory=memory)
        st["gbps"] = gbps
        _print_row(f"memory={memory}", st)


if __name__ == "__main__":
    main()
