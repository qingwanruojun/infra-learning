"""BitwiseNot forward kernel using TileLang (NPU-adapted).

Implements the elementwise bitwise NOT:
    y = ~x  (bool: logical not; integers: xor with all-ones)

The GPU kernel uses XOR with ``-1`` (all-ones) instead of
``T.bitwise_not`` because ``T.bitwise_not`` fails on vectorized ``int4``
CUDA types; bool inputs are handled by an ``== 0`` comparison and uint8
by XOR with 255.  The NPU-side algorithm is NOT a hardware port yet: the
NPU re-design (mapping the branchy per-dtype math onto NPUIR vector
ops) is authored by the NPU kernel component in a later stage.

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory is extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is from
    ``._bitwise_not_fwd_kernels`` (factory ``bitwise_not_fwd_kernel`` =
    the GPU ``UnaryKernel`` ``direct`` strategy builder
    ``_make_unary_direct`` with ``BitwiseNotFwdKernel.op_func`` bound as
    the module-level helper ``bitwise_not_op_func``; the branchy
    multi-return op_func cannot be statically inlined).  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync,
    ``threads``/``npt`` collapse, padding strategy) are handled by the
    NPU component during re-implementation:
      - factory signature drops ``output_dtype`` / ``threads`` (K3)
        -> ``bitwise_not_fwd_kernel(N, dtype)``;
      - the returned callable takes a single ``block_size`` (the GPU
        per-block element count; the ``direct`` strategy processes
        ``threads`` elements per block, and ``npt`` does not participate
        in the direct path) instead of ``(threads_arg,)`` (K11);
      - ``T.Kernel(T.ceildiv(N, block_size), is_npu=True)`` replaces the
        CUDA ``threads=`` dimension;
      - ``T.Parallel(block_size)`` replaces ``T.Parallel(threads_arg)``.

  **Part B -- custom_op wrapper + Kernel class** (fully ported from GPU
  ``BitwiseNotFwdKernel`` / ``UnaryKernel``):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6:  fp8 helpers are not needed -- the manifest contract is
         ``bool | uint8 | int8 | int16 | int32 | int64`` (GPU
         ``_BITWISE_DTYPES``) and the kernel rejects everything else.
    K7:  ``custom_op("npub::...")`` (was ``"top::..."``).
    K8:  autotune removed; heuristic config selection only;
         ``init_config(config)`` takes no ``tune`` argument.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: GPU per-block element count collapsed into a single
         ``block_size`` config key (elementwise op).  The GPU
         ``direct`` strategy uses ``threads=256`` with one element per
         thread iteration (``npt`` does not participate), so
         ``block_size = 256``.
    K12 (bool coercion): the GPU ``UnaryKernel`` forces strategy
         ``direct`` for ``torch.bool`` inputs; ``bitwise_not``'s
         ``DEFAULT_STRATEGY`` is already ``direct`` and it has no other
         strategy, so the NPU wrapper is single-strategy by construction.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .bitwise_not_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .bitwise_not_kernel import bitwise_not_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .bitwise_not_kernel.perf_opt.bitwise_not_fwd_kernel import bitwise_not_fwd_kernel

# The extracted GPU implementation (CUDA target) is the reference for the
# NPU kernel component; the fp8 accumulation helpers of the GPU
# ``UnaryKernel`` base are not needed (K6: the manifest contract is
# bool | uint8 | int8 | int16 | int32 | int64 -- integer/bool bitwise
# semantics, no fp8 path).
# Stage 5 integration note: the active import above now points at the
# conductor NPUIR product (``.bitwise_not_kernel.bitwise_not_fwd_kernel``,
# Developer mode) through the aggregate ``__init__.py``.  The GPU-side
# helper ``bitwise_not_op_func`` of the extracted
# ``_bitwise_not_fwd_kernels`` module is NOT re-imported -- it is dead
# code for the NPU wrapper (the conductor product inlines the per-dtype
# branch: bool/uint8 run in the int8 bit-view domain, integers run
# native vnot), and it is not part of the conductor kernel product.

__all__ = ["BitwiseNotFwdKernel"]


# ---------------------------------------------------------------------------
# Supported dtypes (verbatim from GPU ``tileops/kernels/elementwise.py``
# ``_BITWISE_DTYPES`` -- the manifest ``bool | uint8 | int8 | int16 |
# int32 | int64`` union).
# ---------------------------------------------------------------------------

_BITWISE_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


# ---------------------------------------------------------------------------
# K11 constant: GPU ``threads * npt`` default collapsed into ``block_size``.
#
# GPU ``BitwiseNotFwdKernel`` (``UnaryKernel`` subclass): default strategy
# ``direct`` with ``threads=256``; the direct kernel body processes
# ``threads`` elements per block (one per parallel iteration) and ``npt``
# does not participate in the direct path.  Collapsed per K11:
#   block_size = 256 (all supported dtypes)
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed, K11: block_size)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::bitwise_not", mutates_args=())
def _bitwise_not_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # Two-stage family-standard call surface (K3/K11 target contract, same
    # shape as leaky_relu/expm1/erf): the factory return value is a curried
    # builder, ``main(x)`` returns ``y`` (out_idx=[1], source-operator
    # contract).  Per the K3/K11 adaptation the factory takes a
    # ``torch.dtype`` and the returned callable takes a single
    # ``block_size`` (the GPU per-block element count) -- the call surface
    # below is the NPU-side contract and does not run until the NPU kernel
    # component rewrites the imported GPU factory (whose extracted CUDA
    # callable still takes ``(threads_arg,)``) for ``target="npuir"``.
    dtype = getattr(torch, dtype_str)
    return bitwise_not_fwd_kernel(N, dtype)(block_size)(x)


@_bitwise_not_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11 adaptations; UnaryKernel logic inlined -- the NPU
# project does not ship that template base class)
# ---------------------------------------------------------------------------


class BitwiseNotFwdKernel(Kernel):
    """BitwiseNot forward kernel: y = ~x for bool/integer inputs.

    A single-input, single-output elementwise op.  The kernel flattens
    the input to a 1-D vector of ``N_total`` elements and applies the
    pointwise formula.  The GPU ``op_func`` branches on the runtime
    dtype: bool -> ``x == 0`` (logical not), uint8 -> XOR with 255, all
    other integers -> XOR with -1 (all-ones); it uses XOR instead of
    ``T.bitwise_not`` because the latter fails on vectorized ``int4``
    CUDA types.  The NPU re-implementation must reproduce that per-dtype
    contract (the NPU-side algorithm is re-designed by the NPU kernel
    component).

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K6: no fp8 handling -- ``SUPPORTED_DTYPES`` is the manifest
      contract ``_BITWISE_DTYPES`` (bool + signed/unsigned integers).
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU per-block element count collapsed into ``block_size``
      (the GPU ``UnaryKernel`` logic is inlined, as the NPU project
      does not ship that base class).
    - K12: the GPU bool-forces-direct coercion is a no-op here --
      ``DEFAULT_STRATEGY`` is already ``direct`` and the op has no
      other strategy.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (bool, uint8, int8, int16, int32, or int64).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 256}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # GPU ``BitwiseNotFwdKernel.SUPPORTED_DTYPES`` (``_BITWISE_DTYPES``).
    SUPPORTED_DTYPES = _BITWISE_DTYPES

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = dtype  # same_as(input)
        # Build the factory callable (does not compile yet — compilation
        # is deferred to forward() via the custom_op wrapper).  The
        # imported function is the GPU implementation; the NPU component
        # rewrites it for target="npuir" with the K3/K11 signature
        # ``(N, dtype)`` -> ``kernel(block_size)``.
        self.kernel = bitwise_not_fwd_kernel(self.N_total, self.dtype)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = per-block elements).

        GPU defaults (``UnaryKernel`` direct strategy): threads=256;
        the direct body processes one element per parallel iteration, so
        ``npt`` does not participate and block_size = 256.
        """
        return {"block_size": _BLOCK_SIZE}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bitwise_not kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed from the call; K11: ``block_size`` used.
        """
        return _bitwise_not_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
