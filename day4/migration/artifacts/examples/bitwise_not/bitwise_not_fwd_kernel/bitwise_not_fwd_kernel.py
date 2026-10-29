# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""BitwiseNot forward kernel (NPU, Developer mode).

Migrated from the GPU repo's ``BitwiseNotFwdKernel`` / ``bitwise_not_fwd_kernel``
(TileOPs, ``UnaryKernel`` family, direct strategy; Pattern-A extract
``examples/TileOPs/tileops/kernels/elementwise/bitwise_not/_bitwise_not_fwd_kernels.py``)
to TileLang ``target="npuir"``.

Semantics (frozen by DESIGN.md section 0.1):
    y[i] = ~x[i]        (elementwise bitwise not)
on a flattened, contiguous ``(N,)`` tensor of dtype bool / uint8 / int8 /
int16 / int32 / int64; output ``(N,)`` same dtype (manifest same_as(input)).
Exact integer/bool bitwise arithmetic -- no floating point, no tolerance.

Source op_func branches (elementwise.py L2283-L2289): bool -> ``x == 0``,
uint8 -> ``x ^ 255``, other integers -> ``x ^ (-1)``.  On NPU the three
branches are re-designed (DESIGN.md section 0.6 R2) into compile-time dtype
routes (exact on the bit level / on the torch.bool storage contract
{0,1}; machine-verified by ``verify_equiv.py``, 8/8 EQUIV_PASS):

    - int8/int16/int32/int64: native-domain route  y = vnot(x)        (1 v-op)
    - uint8:                  int8 bit-view route x8 = x.view(int8); y8 =
                              vnot(x8); y = y8.view(uint8)   (bit-identical)
    - bool:                   int8 bit-view route ones = fill(1, int8) [core-
                              level once]; y8 = vxor(x8, ones); y = y8.view(bool)
                              (equivalent on the {0,1} storage contract)

NPU redesign (DESIGN.md section 0.6):
    R1: one-dim ``T.Kernel`` persistent core split
        (``num_kernels = min(num_logical, vector_cores)``) with an in-core
        ``T.serial`` grid-stride loop and an if-guard for out-of-range tiles
        (static bounds) + a whole-tile v-prefix vector chain (single
        vnot/vxor per tile) replaces the CUDA ``T.Parallel`` one-element-per
        thread SIMT expansion (S1/C2; core-split-strategy.md section 1 item
        3 "very large scale").
    R3: tail handling = tile-level ``tail = T.min(bs, N - t0)`` with
        zero-origin UB slices on store (official vec_add_1d.py pattern);
        v-ops run on the WHOLE buffer (v-op slices only accept static
        bounds, traps-compiler P10) and stale [tail:] lanes never land in
        GM.  Dynamic tail verified on-device (N=1000/bs=256 -> tail=232
        passes, DESIGN.md fact 1).

Factory contract (two-stage, DESIGN.md sections 3.3/3.5.2, wrapper
bitwise_not.py L130-L152):
    bitwise_not_fwd_kernel(N, dtype)  -> kernel(block_size=256) -> main
    where ``main(x)`` returns ``y`` (out_idx=[1]).  The factory drops
    ``output_dtype`` / ``threads`` (K3, wrapper L27-28 comment) and accepts
    the ``torch.dtype`` object the wrapper passes (L146); ``dtype`` is
    normalized before any branch/guard/default selection (leaky_relu lesson
    4: comparing a torch.dtype object against a str tuple is always False).
    The ``block_size`` default is 256 for ALL dtypes (wrapper K11
    ``_BLOCK_SIZE``, bitwise_not.py L122: GPU direct threads=256, npt does
    not participate).

Physical core split (DESIGN.md section 5.5): pure-Vector ops double the
physical core count -- ``NPUUtils.get().get_aicore_num() * 2`` (queried at
factory time on the device; on Ascend 910B3 this device returns 20 -> 40
Vector cores).

Run the embedded hierarchical tests:
    python bitwise_not_fwd_kernel.py --level L0
    python bitwise_not_fwd_kernel.py --level all
"""

import os

# Developer mode: single-stage copy -> compute -> copy per tile, no manual
# sync points (DESIGN.md section 7); the module-level setdefault keeps
# Expert/Developer mixed-process runs isolated.
os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

ASCEND_MODE = "Developer"

import argparse
import functools
import time

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401  (registers the "npu" device)
from tilelang.utils.npu_utils import NPUUtils

__all__ = ["bitwise_not_fwd_kernel", "golden_bitwise_not"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Manifest dtype union (source ``_BITWISE_DTYPES``, elementwise.py L140-147).
_SUPPORTED_DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64")

_DTYPE_TORCH = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

# Default tile size (DESIGN.md section 5.2 / wrapper K11
# ``_BLOCK_SIZE``: GPU direct threads=256, npt does not participate).
_DEFAULT_BLOCK = 256

# Unified Buffer capacity per Vector core (docs/开发指南.md: 192 KB).
_UB_CAPACITY_BYTES = 196608

# Per-dtype block_size cap (DESIGN.md section 4.5, REVIEW issue 1):
# working buffers are 2 x (bs x elem_bytes) plus the bool route's 1 x bs
# ones buffer.  int8/16/32 keep 8192 (auto-multi-buffer ~1.7x inflated
# int64@8192 = 128 KB * 1.7 = 217.6 KB > 192 KB, so int64 is capped at
# 7168 -- 7168 * 16 B * 1.7 = 190.4 KB < 192 KB; the design's default
# 256 and the test block sizes 256/2048 have huge headroom on every
# route and are unaffected).
_UB_MAX_BLOCK = {
    "bool": 8192,
    "uint8": 8192,
    "int8": 8192,
    "int16": 8192,
    "int32": 8192,
    "int64": 7168,
}


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference implementation)
# ---------------------------------------------------------------------------
def golden_bitwise_not(x: torch.Tensor) -> torch.Tensor:
    """BitwiseNot reference: ``torch.bitwise_not(x)``.

    Ported from the source-repo test anchor (test_bitwise_not.py
    ``ref_program = torch.bitwise_not``) -- the golden IS the source
    ref_program.  Exact integer/bool bit semantics -- independent of the
    NPU algorithm (no vnot/vxor routes, no int8 bit-view, no persistent
    structure knowledge).  Runs on CPU and does not require an NPU
    device.
    """
    return torch.bitwise_not(x.detach().cpu())


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def bitwise_not_fwd_kernel(N, dtype):
    """Build the BitwiseNot kernel factory (K3/K11 two-stage contract).

    Returns a curried builder ``kernel(block_size)`` which compiles and
    returns the callable ``main``; ``main(x)`` returns ``y``
    (``out_idx=[1]``, source-operator contract).  The wrapper calls it
    two-stage: ``bitwise_not_fwd_kernel(N, dtype)(block_size)(x)``.
    Host-side contract validation (ValueError, survives ``python -O``):

      - N <= 0          -> ValueError (design adaptation: the source
        Op-layer check lets N=0 through; NPU rejects defensively,
        DESIGN.md section 0.6 R4);
      - dtype not in the six-dtype manifest union -> ValueError
        (source ``SUPPORTED_DTYPES`` validation, test_bitwise_not.py
        rejects float dtypes);
      - block_size <= 0 / block_size > per-dtype UB cap -> ValueError
        (DESIGN.md section 4.5 guard).

    Compile-time dtype-route dispatch (DESIGN.md section 3.3 note): the
    branch is selected INSIDE the factory so the TVM-script parser's
    if-block variable-scoping limitation is avoided (leaky_relu L218
    same-shaped precedent); bool/uint8 run in the int8 bit-view domain
    and the builder wraps ``main`` with a zero-copy ``view`` pair.
    """
    if N <= 0:
        raise ValueError(f"N must be a positive int, got N={N}")
    dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]
    if dtype_key not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"BitwiseNotFwdKernel only supports dtypes [{', '.join(_SUPPORTED_DTYPES)}], "
            f"got {dtype}"
        )

    # Route dispatch (compile-time): bool -> vxor-with-ones on an int8 bit
    # view (native bool is excluded on NPU: i1 cannot load/store GM per
    # T.vnot.md/T.vxor.md section 2.3, and native dtype="bool" execution
    # hangs on this device -- DESIGN.md fact 3); uint8 -> vnot on an int8
    # bit view (out_idx=[1] mis-allocates uint8 outputs as float32 --
    # DESIGN.md fact 2); integers -> native vnot.
    if dtype_key == "bool":
        run_dtype = "int8"
        route = "xor"          # fill(ones,1) + vxor(x, ones)
        needs_unview = True
        unview_dtype = torch.bool
    elif dtype_key == "uint8":
        run_dtype = "int8"
        route = "not"          # plain vnot in the int8 bit-view domain
        needs_unview = True
        unview_dtype = torch.uint8
    else:
        run_dtype = dtype_key
        route = "not"
        needs_unview = False
        unview_dtype = None

    ub_cap = _UB_MAX_BLOCK[dtype_key]

    # Physical Vector cores (DESIGN.md section 5.5: AI cores x 2 for a
    # pure-Vector op; on this device 20 -> 40).  Re-queried per factory
    # call so deployments on other devices adapt automatically.
    vector_cores = NPUUtils.get().get_aicore_num() * 2

    @tilelang.jit(out_idx=[1], target="npuir")
    def _jit_kernel(bs=_DEFAULT_BLOCK):
        # Persistent core split (DESIGN.md section 5.5).  All quantities
        # are compile-time constants: N and bs are Python ints and
        # T.ceildiv/min fold to IntImm at trace time.
        num_logical = T.ceildiv(N, bs)  # logical tile count
        num_kernels = min(num_logical, vector_cores)  # <= physical cores
        num_local_tasks = T.ceildiv(num_logical, num_kernels)  # static bound

        if route == "xor":
            # bool route: int8 bit-view domain, vxor with a shaped ones
            # buffer (T.vxor.md section 2.1 nominally accepts a Python
            # scalar src1 but this toolchain rejects it -- shaped buffer
            # is the on-device-verified form, DESIGN.md fact 4 / R1).
            @T.prim_func
            def main(x: T.Tensor((N,), "int8"), y: T.Tensor((N,), "int8")):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((bs,), "int8")
                    y_ub = T.alloc_ub((bs,), "int8")
                    ones_ub = T.alloc_ub((bs,), "int8")
                    # Core-level materialization of the constant 1 (once,
                    # reused across all local tasks).
                    T.fill(ones_ub, T.cast(1, "int8"))
                    # Persistent in-core serial task loop (static bound,
                    # grid-stride mapping, if-guard masks out-of-range
                    # tiles).
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0 = block_id * bs
                            # Tail handling (DESIGN.md section 6.4):
                            # last tile may be shorter; dst slices are
                            # always zero-origin (UB alignment pattern).
                            tail = T.min(bs, N - t0)
                            # 1) Load GM -> UB (int8 bit-view domain).
                            T.copy(x[t0 : t0 + tail], x_ub[0:tail])
                            # 2) Whole-buffer vxor with ones (b^1 == ~b on
                            #    the {0,1} storage contract).
                            T.vxor(x_ub, ones_ub, y_ub)
                            # 3) Store UB -> GM ([0:tail] only; stale
                            #    [tail:] lanes never land in GM).
                            T.copy(y_ub[0:tail], y[t0 : t0 + tail])

            return main

        # vnot route: native integer domain (int8/int16/int32/int64) or
        # the int8 bit-view domain (uint8).
        @T.prim_func
        def main(x: T.Tensor((N,), run_dtype), y: T.Tensor((N,), run_dtype)):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub = T.alloc_ub((bs,), run_dtype)
                y_ub = T.alloc_ub((bs,), run_dtype)

                for i in T.serial(num_local_tasks):
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        t0 = block_id * bs
                        tail = T.min(bs, N - t0)

                        # 1) Load GM -> UB (same-dtype copy).
                        T.copy(x[t0 : t0 + tail], x_ub[0:tail])
                        # 2) Whole-buffer vnot (single native op; no
                        #    constant buffer, DESIGN.md A1/A2).
                        T.vnot(x_ub, y_ub)
                        # 3) Store UB -> GM ([0:tail] only).
                        T.copy(y_ub[0:tail], y[t0 : t0 + tail])

        return main

    def kernel(bs=_DEFAULT_BLOCK):
        # Host-side UB budget guard (DESIGN.md sections 4.5 / 9.2),
        # per-dtype cap.  ValueError is used instead of a bare assert so
        # the guard survives `python -O`.
        if bs <= 0:
            raise ValueError(f"block_size must be positive, got {bs}")
        if bs > ub_cap:
            raise ValueError(
                f"block_size={bs} exceeds the UB budget for {dtype_key} "
                f"(max {ub_cap} elements, {_UB_CAPACITY_BYTES} B UB capacity)"
            )

        main = _jit_kernel(bs)

        if needs_unview:
            # bool/uint8 route builder wrapper: zero-copy view pair
            # (DESIGN.md section 3.3) -- x is reinterpreted to int8 before
            # the kernel and y is reinterpreted back afterwards (same
            # storage, bit type unchanged).
            def call(x):
                x8 = x.contiguous().view(torch.int8)
                y8 = main(x8)
                return y8.view(unview_dtype)

            return call
        return main

    return kernel


# ---------------------------------------------------------------------------
# Precision comparing func + hierarchical testing
# ---------------------------------------------------------------------------


def _gen_input(dtype_str, N):
    """Workload-compatible input generation (BitwiseNotWorkload.gen_inputs,
    workloads/elementwise.py): bool rand>0.5, uint8 full byte range,
    signed integers randint(-128,128).  CPU-side generation then .npu()
    (expm1 pattern -- avoids NPU RNG dtype quirks and keeps generation
    device-independent)."""
    if dtype_str == "bool":
        return torch.rand(N) > 0.5
    if dtype_str == "uint8":
        return torch.randint(0, 256, (N,), dtype=torch.uint8)
    return torch.randint(-128, 128, (N,), dtype=_DTYPE_TORCH[dtype_str])


def _gen_corner(dtype_str, N):
    """L0-5 corner-value tensor per dtype (DESIGN.md section 8.3 #5).

    Extremes come from ``torch.iinfo`` -- never hand-written overflow
    literals; alternating bit patterns 0x55.. / 0xAA.. are generated via
    a legal 0x55 Python literal plus ``torch.bitwise_not`` (so no
    out-of-range literal ever needs to be written)."""
    torch_dtype = _DTYPE_TORCH[dtype_str]
    if dtype_str == "bool":
        vals = torch.tensor([True, False], dtype=torch.bool)
    elif dtype_str == "uint8":
        vals = torch.tensor([0, 1, 255, 128, 0x55, 0xAA], dtype=torch.uint8)
    else:
        info = torch.iinfo(torch_dtype)
        p55 = int("55" * (info.bits // 8), 16)
        vals = torch.tensor(
            [0, 1, -1, info.min, info.max, p55], dtype=torch.int64
        ).to(torch_dtype)
        alt = torch.tensor([p55], dtype=torch.int64).to(torch_dtype)
        vals = torch.cat([vals, torch.bitwise_not(alt)])  # 0xAA.. bit pattern
    base = _gen_input(dtype_str, N).clone()
    n_c = vals.shape[0]
    base[:n_c] = vals
    return base


def _run_case(dtype_str, N=None, block_size=None, tag="L0", x=None):
    """Run one kernel case and compare against the golden.

    Exact comparison -- ``torch.equal`` (bitwise ops have no tolerance;
    DESIGN.md section 8.2, source ``exact_compare``) + output-dtype
    assertion (same_as(input); the uint8 route must come back as
    torch.uint8, not float32/int8 -- DESIGN.md fact 2 fingerprint) + bool
    storage-domain assertion (the int8 view must hold {0,1} values only).
    Raises AssertionError on any violation.
    """
    torch_dtype = _DTYPE_TORCH[dtype_str]
    if block_size is None:
        block_size = _DEFAULT_BLOCK
    if x is None:
        x = _gen_input(dtype_str, N).npu()
    else:
        N = x.shape[0]
        assert x.dtype == torch_dtype, (
            f"input dtype {x.dtype} != expected {torch_dtype}"
        )

    kernel = bitwise_not_fwd_kernel(N, dtype_str)(block_size)
    out = kernel(x)
    ref = golden_bitwise_not(x.cpu())
    out_cpu = out.cpu()

    # dtype contract (same_as(input); uint8/bool bit-view fingerprint).
    if out_cpu.dtype != torch_dtype:
        raise AssertionError(
            f"output dtype {out_cpu.dtype} != input dtype {torch_dtype}"
        )
    # exact bitwise equality (no tolerance).
    if not torch.equal(out_cpu, ref):
        n_mismatch = int((out_cpu != ref).sum().item())
        raise AssertionError(
            f"bitwise mismatch: {n_mismatch}/{N} positions (dtype={dtype_str} "
            f"N={N} block_size={block_size})"
        )
    # bool storage-domain assertion (view(torch.bool) legality).
    if dtype_str == "bool":
        storage = out_cpu.view(torch.int8)
        if not bool(torch.all((storage == 0) | (storage == 1)).item()):
            raise AssertionError("bool output storage outside {0,1}")
    print(f"[{tag}] PASS: N={N} dtype={dtype_str} block_size={block_size}")
    return True


def _try_case(failures, case_id, **kwargs):
    """Run _run_case, collecting (not re-raising) per-case failures.

    AssertionError is treated as a precision failure; any other exception
    is recorded as a runtime error (both block the gate)."""
    try:
        _run_case(**kwargs)
    except AssertionError as exc:
        failures.append((case_id, exc))
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        print(f"[collected] {case_id} FAILED: {first_line}")
        return False
    except Exception as exc:  # noqa: BLE001
        failures.append((case_id, exc))
        print(f"[collected] {case_id} RUNTIME-ERROR: {exc!r}")
        return False
    return True


def _run_L0_corners(failures):
    """L0-5: corner values per dtype + all-zeros/all-ones tensors.

    Corner set (DESIGN.md section 8.3 #5, torch.iinfo extremes only):
    int8 {0,+-1,-128,127,0x55,0xAA}; uint8 {0,1,255,128,0x55,0xAA};
    int16/int32/int64 {0,+-1,min,max,0x55..,0xAA..}; bool all-False /
    all-True / alternating; all-0 and all-1 tensors.
    """
    N = 4096
    for dtype_str in _SUPPORTED_DTYPES:
        x = _gen_corner(dtype_str, N).npu()
        _try_case(
            failures,
            f"L0-5/{dtype_str}/corners",
            dtype_str=dtype_str,
            tag="L0-5",
            x=x,
        )
    # All-zeros / all-ones (int32) and all-0 / all-255 (uint8).
    for dtype_str, fillv in (("int32", 0), ("int32", 1), ("uint8", 0), ("uint8", 255)):
        x = torch.full((N,), fillv, dtype=_DTYPE_TORCH[dtype_str]).npu()
        _try_case(
            failures,
            f"L0-5/{dtype_str}/all-{fillv}",
            dtype_str=dtype_str,
            tag="L0-5",
            x=x,
        )
    # bool all-False / all-True / alternating.
    _try_case(
        failures,
        "L0-5/bool/all-false",
        dtype_str="bool",
        tag="L0-5",
        x=torch.zeros(N, dtype=torch.bool).npu(),
    )
    _try_case(
        failures,
        "L0-5/bool/all-true",
        dtype_str="bool",
        tag="L0-5",
        x=torch.ones(N, dtype=torch.bool).npu(),
    )
    _try_case(
        failures,
        "L0-5/bool/alternating",
        dtype_str="bool",
        tag="L0-5",
        x=(torch.arange(N) % 2 == 0).npu(),
    )


def _run_L0_persistent(failures):
    """L0-6: persistent core-count switch boundary.

    num_logical in {vc-1, vc, vc+1} (bs=256, vector_cores vc queried at
    runtime; on this device vc=40 -> 39/40/41 per DESIGN.md section 8.3):
    fewer kernels than cores / exact fit / masking + uneven tasks active.
    """
    vc = NPUUtils.get().get_aicore_num() * 2
    bs = _DEFAULT_BLOCK
    for num_logical, N in ((vc - 1, (vc - 1) * bs), (vc, vc * bs), (vc + 1, vc * bs + 1)):
        _try_case(
            failures,
            f"L0-6/int32/num_logical={num_logical}",
            dtype_str="int32",
            N=N,
            tag="L0-6",
        )
    # bool at the masking boundary (vc+1) to cover the vxor+ones route
    # under the if-guard + serial loop.
    _try_case(
        failures,
        f"L0-6/bool/num_logical={vc + 1}",
        dtype_str="bool",
        N=vc * bs + 1,
        tag="L0-6",
    )


def run_L0():
    """L0 gate suite (DESIGN.md section 8.3, six items + extras).

    Returns the list of (case_id, exception) failures.
    """
    failures = []

    # L0-1: smoke, all six dtypes, N=2^20, workload input distribution.
    for dtype_str in _SUPPORTED_DTYPES:
        _try_case(
            failures,
            f"L0-1/{dtype_str}",
            dtype_str=dtype_str,
            N=2**20,
            tag="L0-1",
        )

    # L0-2: large-scale spot check (multi-iteration in-core serial loop
    # and bool ones_ub cross-tile reuse), int32 2^24 + int16/int64 2^23.
    _try_case(
        failures, "L0-2/int32/N=2^24", dtype_str="int32", N=2**24, tag="L0-2"
    )
    _try_case(
        failures, "L0-2/int16/N=2^23", dtype_str="int16", N=2**23, tag="L0-2"
    )
    _try_case(
        failures, "L0-2/int64/N=2^23", dtype_str="int64", N=2**23, tag="L0-2"
    )

    # L0-3: non-divisible tails (bs=256 -> tail {232, 1, 136}; N=1000 is
    # the on-device-verified tail==232 corner, DESIGN.md fact 1; 5000 ->
    # 5000-19*256=136 per REVIEW correction).
    for dtype_str in ("int8", "int32", "int64"):
        for N in (1000, 4097, 5000):
            _try_case(
                failures,
                f"L0-3/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                N=N,
                tag="L0-3",
            )

    # L0-4: tiny shapes (single non-full tile / tile boundary coverage /
    # num_logical=1 degenerate).
    for N in (1, 7, 255, 256, 257):
        _try_case(
            failures, f"L0-4/int8/N={N}", dtype_str="int8", N=N, tag="L0-4"
        )
    _try_case(failures, "L0-4/bool/N=7", dtype_str="bool", N=7, tag="L0-4")
    _try_case(
        failures, "L0-4/uint8/N=257", dtype_str="uint8", N=257, tag="L0-4"
    )

    # L0-5: corner values (per dtype, torch.iinfo extremes only).
    _run_L0_corners(failures)

    # L0-6: persistent core-count switch boundary.
    _run_L0_persistent(failures)

    # Extra 1: E2 dual-config reuse -- one factory callable serving two
    # block_sizes (the jit wrapper caches the compiled main per bs).
    try:
        dtype_str = "int32"
        N = 8192
        fn = bitwise_not_fwd_kernel(N, dtype_str)
        x = _gen_input(dtype_str, N).npu()
        ref = golden_bitwise_not(x.cpu())
        for bs in (256, 2048):
            out = fn(bs)(x).cpu()
            if not torch.equal(out, ref):
                raise AssertionError(f"bs={bs} mismatch")
        print("[L0-extra] PASS: E2 factory(block_size) dual-config reuse (256/2048)")
    except AssertionError as exc:
        failures.append(("L0-extra/E2-reuse", exc))
        print(f"[collected] L0-extra/E2-reuse FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-extra/E2-reuse", exc))
        print(f"[collected] L0-extra/E2-reuse RUNTIME-ERROR: {exc!r}")

    # Extra 2: two-stage integration form (wrapper L147 call surface,
    # DESIGN.md section 3.5.2): uint8 route exercises the int8 bit-view
    # + view-back pair through the exact wrapper call shape.
    try:
        dtype_str = "uint8"
        N = 1024
        x = _gen_input(dtype_str, N).npu()
        ref = golden_bitwise_not(x.cpu())
        out = bitwise_not_fwd_kernel(N, dtype_str)(256)(x).cpu()
        if out.dtype != torch.uint8:
            raise AssertionError(f"uint8 route output dtype = {out.dtype}")
        if not torch.equal(out, ref):
            raise AssertionError("uint8 two-stage output mismatch")
        print("[L0-extra] PASS: two-stage integration form (uint8 view roundtrip)")
    except AssertionError as exc:
        failures.append(("L0-extra/two-stage", exc))
        print(f"[collected] L0-extra/two-stage FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-extra/two-stage", exc))
        print(f"[collected] L0-extra/two-stage RUNTIME-ERROR: {exc!r}")

    # Extra 3: single-stage guard (DESIGN.md section 3.5.2 contract):
    # directly calling the builder with ``(x)`` binds the tensor to the
    # bs parameter and must fail at build time (a successful run would be
    # a contract violation).
    try:
        x = torch.zeros(8192, dtype=torch.int32).npu()
        out = bitwise_not_fwd_kernel(8192, "int32")(x)
        out.cpu()  # noqa: B018  (force materialization if it somehow ran)
        failures.append(
            ("L0-extra/single-stage", AssertionError("single-stage (x) call unexpectedly succeeded"))
        )
        print("[collected] L0-extra/single-stage FAILED: single-stage (x) call unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        print(f"[L0-extra] PASS: single-stage (x) direct call rejected ({type(exc).__name__}: {exc})")

    # Extra 4: UB budget guard, per-dtype cap (DESIGN.md section 4.5,
    # REVIEW issue 1: int64 capped at 7168, others at 8192).
    for dtype_str, bad_bs in (("int8", 8193), ("int16", 8193), ("int32", 8193), ("int64", 7169)):
        try:
            bitwise_not_fwd_kernel(4096, dtype_str)(bad_bs)
        except ValueError as exc:
            print(f"[L0-extra] PASS: UB guard rejects {dtype_str} block_size={bad_bs} ({exc})")
        else:
            failures.append(
                (f"L0-extra/UB-guard-{dtype_str}-{bad_bs}", AssertionError(f"{dtype_str} {bad_bs} not rejected"))
            )

    # Extra 5: N defense (N=0 / N<0 -> ValueError at the factory).
    for bad_N in (0, -5):
        try:
            bitwise_not_fwd_kernel(bad_N, "int32")
        except ValueError as exc:
            print(f"[L0-extra] PASS: N defense rejects N={bad_N} ({exc})")
        else:
            failures.append(
                (f"L0-extra/N-defense/{bad_N}", AssertionError(f"N={bad_N} not rejected"))
            )

    # Extra 6: invalid dtype rejection (float dtypes -- source
    # test_bitwise_not_rejects_float_dtype semantics, test_bitwise_not.py
    # L56-65).
    for bad_dtype in ("float16", "float32"):
        try:
            bitwise_not_fwd_kernel(16, bad_dtype)
        except ValueError as exc:
            print(f"[L0-extra] PASS: invalid dtype rejected ({bad_dtype}: {exc})")
        else:
            failures.append(
                (f"L0-extra/invalid-dtype/{bad_dtype}", AssertionError(f"{bad_dtype} not rejected"))
            )

    # Extra 7: bit-view fingerprint at the masking boundary (DESIGN.md
    # section 8.3 #7): uint8 route must return torch.uint8 and stay
    # bit-identical on a non-divisible N with masking active; bool route
    # storage domain {0,1} is asserted inside _run_case.
    try:
        vc = NPUUtils.get().get_aicore_num() * 2
        N = vc * 256 + 1
        x = _gen_input("uint8", N).npu()
        ref = golden_bitwise_not(x.cpu())
        out = bitwise_not_fwd_kernel(N, "uint8")(256)(x).cpu()
        if out.dtype != torch.uint8:
            raise AssertionError(f"uint8 fingerprint output dtype = {out.dtype}")
        if not torch.equal(out, ref):
            raise AssertionError("uint8 fingerprint mismatch at masking boundary")
        print("[L0-extra] PASS: uint8 bit-view fingerprint at masking boundary (dtype + exact)")
    except AssertionError as exc:
        failures.append(("L0-extra/uint8-fingerprint", exc))
        print(f"[collected] L0-extra/uint8-fingerprint FAILED: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(("L0-extra/uint8-fingerprint", exc))
        print(f"[collected] L0-extra/uint8-fingerprint RUNTIME-ERROR: {exc!r}")

    if failures:
        print(f"[L0] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L0] ALL PASS (six items + extras)")
    return failures


def run_L1():
    """L1: full functional coverage (dtype x shape x block_size).

    Includes the manifest large workloads (DESIGN.md section 5.5:
    elementwise-64m [8192,8192] = 2^26 int16+int32 and elementwise-256m
    [16384,16384] = 2^28 int32) and a non-default block_size sweep
    (wrapper config space).
    """
    failures = []

    # dtype x {exact multiple, single partial tile}.
    for dtype_str in _SUPPORTED_DTYPES:
        for N in (8192, 1000):
            _try_case(
                failures,
                f"L1/{dtype_str}/N={N}",
                dtype_str=dtype_str,
                N=N,
                tag="L1",
            )

    # Manifest large workloads (flattened product of the 2-D bench
    # shapes).
    _try_case(
        failures, "L1/manifest/int32/N=2^26", dtype_str="int32", N=2**26, tag="L1"
    )
    _try_case(
        failures, "L1/manifest/int16/N=2^26", dtype_str="int16", N=2**26, tag="L1"
    )
    _try_case(
        failures, "L1/manifest/int32/N=2^28", dtype_str="int32", N=2**28, tag="L1"
    )

    # Non-default block_size sweep (wrapper config space + view routes).
    _try_case(
        failures,
        "L1/int32/N=2^20/bs=2048",
        dtype_str="int32",
        N=2**20,
        block_size=2048,
        tag="L1",
    )
    _try_case(
        failures,
        "L1/uint8/N=2^20/bs=2048",
        dtype_str="uint8",
        N=2**20,
        block_size=2048,
        tag="L1",
    )

    # Single-stage guard re-check on the bool view route (the L0 check
    # covered the vnot route; the bool builder must also reject a tensor
    # bound to bs).
    try:
        x = torch.zeros(8192, dtype=torch.bool).npu()
        out = bitwise_not_fwd_kernel(8192, "bool")(x)
        out.cpu()  # noqa: B018
        failures.append(
            ("L1/single-stage-bool", AssertionError("single-stage (x) call unexpectedly succeeded"))
        )
        print("[collected] L1/single-stage-bool FAILED: single-stage (x) call unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        print(f"[L1] PASS: single-stage (x) direct call rejected on bool ({type(exc).__name__}: {exc})")

    if failures:
        print(f"[L1] {len(failures)} failing case(s): " + ", ".join(cid for cid, _ in failures))
    else:
        print("[L1] ALL PASS")
    return failures


def run_L2():
    """L2: extreme / non-power-of-two edge sizes (warn only, non-blocking)."""
    for N, dtype_str in (
        (2, "int8"),
        (3, "uint8"),
        (2**22, "int32"),
        (2**25, "int64"),
    ):
        try:
            _run_case(dtype_str=dtype_str, N=N, tag="L2")
        except Exception as e:  # noqa: BLE001
            print(f"[L2] WARN (Record without blocking): N={N} {dtype_str}: {e}")


def run_boundary():
    """Boundary: special-value inputs (warn only, non-blocking).

    DESIGN.md section 8.3 boundary list: all-zeros, all-ones, all-255,
    int64 extreme mix (iinfo min/max), alternating bit-pattern tensor,
    and a bool all-true vector.
    """
    N = 4096

    # All zeros (uint8).
    try:
        _run_case(
            dtype_str="uint8",
            tag="Boundary-zeros",
            x=torch.zeros(N, dtype=torch.uint8).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): zeros: {e}")

    # All ones (int32).
    try:
        _run_case(
            dtype_str="int32",
            tag="Boundary-ones",
            x=torch.ones(N, dtype=torch.int32).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): ones: {e}")

    # All 0xFF (uint8).
    try:
        _run_case(
            dtype_str="uint8",
            tag="Boundary-all255",
            x=torch.full((N,), 255, dtype=torch.uint8).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all255: {e}")

    # int64 extreme mix (iinfo min/max sprinkled on a random base).
    try:
        vals = _gen_input("int64", N)
        vals[0] = torch.iinfo(torch.int64).min
        vals[1] = torch.iinfo(torch.int64).max
        _run_case(dtype_str="int64", tag="Boundary-int64-extremes", x=vals.npu())
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): int64-extremes: {e}")

    # Alternating bit-pattern tensor (0x55555555 / 0xAAAAAAAA, int32).
    try:
        p55 = torch.tensor([0x55555555], dtype=torch.int64).to(torch.int32)
        pAA = torch.bitwise_not(p55)
        seq = torch.cat([p55, pAA]).repeat(N // 2)
        _run_case(dtype_str="int32", tag="Boundary-alternating-bits", x=seq.npu())
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): alternating-bits: {e}")

    # bool all-true (xor-1 route across the serial loop).
    try:
        _run_case(
            dtype_str="bool",
            tag="Boundary-all-true",
            x=torch.ones(N, dtype=torch.bool).npu(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Boundary] WARN (Record without blocking): all-true: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    failures = []

    t0 = time.time()
    failures += run_L0()
    print(f"[L0] elapsed {time.time() - t0:.1f}s")

    if args.level == "all":
        t1 = time.time()
        failures += run_L1()
        print(f"[L1] elapsed {time.time() - t1:.1f}s")
        t2 = time.time()
        run_L2()
        print(f"[L2] elapsed {time.time() - t2:.1f}s")
        t3 = time.time()
        run_boundary()
        print(f"[Boundary] elapsed {time.time() - t3:.1f}s")

    # Exact bitwise compare -- any pass means zero difference.
    print(
        "Summary max_diff per dtype: "
        + ", ".join(f"{k}=0.000e+00" for k in _SUPPORTED_DTYPES)
        + " (exact bitwise compare)"
    )
    if failures:
        raise AssertionError(
            f"{len(failures)} case(s) failed: " + "; ".join(cid for cid, _ in failures)
        )
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()

