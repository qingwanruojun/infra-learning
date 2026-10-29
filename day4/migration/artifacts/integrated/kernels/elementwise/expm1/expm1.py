"""Expm1 forward kernel using TileLang (NPU-adapted).

Implements the elementwise exponential-minus-one:
    y = exp(x) - 1

Adaptation summary (GPU -> NPU), per docs/gpu_to_npu_adaptation.md §3:

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``expm1_fwd_kernel``) is extracted
    from the GPU repo via ``extract_tl_kernel.py`` and imported as-is.
    It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 and K11 adaptations
    (decorator, grid/sync, ``threads``/``npt`` collapse, padding strategy)
    are handled by the NPU component during re-implementation:
      - factory signature drops ``output_dtype`` / ``threads`` /
        ``num_per_thread`` (K3) -> ``expm1_fwd_kernel(N, dtype,
        output_dtype=None, block_size=...)``;
      - the returned callable takes a single ``block_size`` (the GPU
        ``threads * npt`` product) instead of ``(threads_arg, npt_arg)``
        (K11);
      - ``T.Kernel(T.ceildiv(N, block_size), is_npu=True)`` replaces the
        CUDA ``threads=`` dimension;
      - ``T.Parallel(block_size)`` replaces
        ``T.Parallel(threads_arg, npt_arg)``.

  **Part B -- custom_op wrapper + Kernel class** (fully ported from GPU
  ``Expm1FwdKernel`` / ``UnaryKernel`` / ``FloatUnaryKernel``):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6:  fp8 helpers are not needed -- the manifest contract is
         ``float16 | bfloat16 | float32`` and the GPU
         ``FloatUnaryKernel.SUPPORTED_DTYPES`` already rejects fp8.
    K7:  ``custom_op("npub::...")`` (was ``"top::..."``).
    K8:  autotune removed; heuristic config selection only;
         ``init_config(config)`` takes no ``tune`` argument.
    K9:  ``threads`` removed from ``default_config`` and ``forward``.
    K11: GPU ``threads * npt`` product collapsed into a single
         ``block_size`` config key (elementwise op).
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# The extracted GPU implementation (CUDA target) is the reference for the
# NPU kernel component; the fp8 accumulation helpers of the GPU
# ``UnaryKernel`` base are not needed (K6: the manifest contract is
# float16 | bfloat16 | float32 and ``FloatUnaryKernel`` rejects fp8).
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .expm1_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .expm1_kernel import expm1_fwd_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .expm1_kernel.perf_opt.expm1_fwd_kernel import expm1_fwd_kernel

__all__ = ["Expm1FwdKernel"]


# ---------------------------------------------------------------------------
# K11 constants: GPU ``threads * npt`` defaults collapsed into ``block_size``.
#
# GPU ``UnaryKernel`` defaults (default strategy ``register_copy``):
# threads=256, npt=4 (fp32) or npt=8 (fp16/bf16).  Collapsed per K11
# ("block_size = the original threads * npt product"):
#   fp32:      256 * 4 = 1024
#   fp16/bf16: 256 * 8 = 2048
# ---------------------------------------------------------------------------

_BLOCK_SIZE_FP32 = 1024
_BLOCK_SIZE_NON_FP32 = 2048


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed, K11: block_size)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::expm1_fwd", mutates_args=())
def _expm1_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    # Two-stage family-standard call surface (DESIGN.md section 3.5.2,
    # same shape as leaky_relu L98 / elu_fwd L99): the factory return
    # value is always a curried builder, ``main(x)`` returns ``y``
    # (out_idx=[1], source-operator contract).  A single-stage
    # ``(x)`` would bind the tensor to the ``bs`` parameter (build-time
    # TypeError) and the trailing ``, N`` cannot bind in the jit
    # callable contract.
    return expm1_fwd_kernel(N, dtype_str, block_size=block_size)(block_size)(x)


@_expm1_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor):
    return torch.empty((N,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9, K11 adaptations; UnaryKernel/FloatUnaryKernel logic
# inlined -- the NPU project does not ship those template base classes)
# ---------------------------------------------------------------------------


class Expm1FwdKernel(Kernel):
    """Expm1 forward kernel: y = exp(x) - 1.

    A single-input, single-output elementwise math op.  The kernel
    flattens the input to a 1-D vector of ``N_total`` elements and
    applies the pointwise formula (computed in fp32 intermediate
    precision, matching the GPU ``op_func``:
    ``T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")``).

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures).
    - K6: no fp8 handling -- ``SUPPORTED_DTYPES`` is the manifest
      contract ``(float16, bfloat16, float32)`` (GPU
      ``FloatUnaryKernel`` rejects fp8 the same way), so
      ``output_dtype == dtype`` always.
    - K8: autotune removed; heuristic config selection only.
      ``init_config(config)`` takes no ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and ``forward``.
    - K11: GPU ``threads * npt`` product collapsed into ``block_size``
      (the GPU ``UnaryKernel`` logic is inlined, as the NPU project
      does not ship that base class).

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g.
            ``{"block_size": 2048}``).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # Developer-form vector kernel; invocations are scoped to Developer
    # mode regardless of ambient TILELANG_ASCEND_MODE (CG-2026-0010).
    ascend_mode = "Developer"

    # GPU ``FloatUnaryKernel.SUPPORTED_DTYPES`` (float-only ops).
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

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
        self.kernel = expm1_fwd_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K11: block_size = threads * npt).

        GPU defaults (``UnaryKernel``, strategy ``register_copy``):
        threads=256, npt=4 (fp32) or npt=8 (fp16/bf16).  Collapsed per
        K11: block_size = 1024 (fp32) or 2048 (fp16/bf16).
        """
        if self.dtype == torch.float32:
            return {"block_size": _BLOCK_SIZE_FP32}
        return {"block_size": _BLOCK_SIZE_NON_FP32}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the expm1 kernel.

        Accepts a 1-D contiguous tensor of ``N_total`` elements.
        K9: ``threads`` removed from the call; K11: ``block_size`` used.
        """
        return _expm1_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
