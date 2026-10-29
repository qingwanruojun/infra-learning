"""LeakyReluFwdOp -- elementwise LeakyReLU activation (NPU-adapted).

Adaptation from GPU (TileOPs) to NPU:

- O1: ``input.is_cuda`` -> ``backend.is_device_tensor(input)``.
- O3: ``tune`` parameter removed; kernel constructor takes
      ``(N_total, dtype, negative_slope, config=None)``.
- O4: ``from tileops.device import get_device_backend``.
- O5: ``_OP_REGISTRY`` / ``_wrapped`` custom_op dispatch and
      ``compile_boundary.register_instance`` removed; ``forward``
      dispatches directly through ``_eager_forward``.
- O6: Op-specific flow preserved -- scalar-param validation, flatten ->
      kernel -> reshape, inplace copy-back, roofline (flops = ``2 * N``,
      compare-and-select(1) + mul = 2 per elem, per roofline.md §1.3).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.leaky_relu import LeakyReluFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["LeakyReluFwdOp"]


_MANIFEST_INT_SCALAR_DTYPES = (
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
)


def _validate_scalar_param_repr(
    param_name: str, value, dtype: torch.dtype, op_name: str,
    *, allow_nonfinite_float: bool = False,
) -> None:
    """Reject scalar params that cannot be represented in the user dtype.

    Verbatim port from GPU ``tileops/ops/elementwise/_base.py`` (O6:
    validation logic preserved).  Validates against the *user-facing*
    ``dtype``, not any kernel intermediate dtype.

    Integer and bool mirror PyTorch ``Tensor.masked_fill`` coercion:

    - bool: any int/float, reduced to ``{0, 1}``.
    - Signed int: any value in ``[iinfo.min, iinfo.max]``, truncated toward
      zero. NaN/Inf and out-of-range raise.
    - ``uint8``: ints in ``[-255, 255]``, negatives wrapping via ``& 0xFF``;
      float scalars must be in ``[0, 255]``.

    Floats always accept ``NaN`` and require finite values in ``finfo`` range.
    ``+/-Inf`` passes only under ``allow_nonfinite_float``.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; treat explicitly so the int
        # range checks below operate on the integer/float branch.
        return
    if not isinstance(value, (int, float)):
        raise TypeError(f"{op_name} expected scalar {param_name} to be int/float, got {type(value)}")

    if dtype == torch.bool:
        return

    if dtype in _MANIFEST_INT_SCALAR_DTYPES:
        iinfo = torch.iinfo(dtype)
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(
                    f"{op_name} received {param_name}={value!r}, but {param_name} must be finite "
                    f"and representable in dtype {dtype}"
                )
            # PyTorch range-checks the real float value, then truncates
            # toward zero. Negative float scalars never wrap into uint8
            # (``uint8.masked_fill(mask, -1.0)`` raises in PyTorch).
            if not (iinfo.min <= value <= iinfo.max):
                raise ValueError(
                    f"{op_name} received {param_name}={value!r}, which is not representable in "
                    f"dtype {dtype} (valid finite range: [{iinfo.min}, {iinfo.max}])"
                )
            return
        # Python int branch. uint8 wraps negatives in [-255, 255] via
        # two's complement, matching PyTorch.
        if dtype == torch.uint8 and value < 0:
            if value < -255:
                raise ValueError(
                    f"{op_name} received {param_name}={value!r}, which is not representable in "
                    f"dtype {dtype} (valid integer range: [-255, 255] with wraparound, "
                    f"or [0, 255] direct)"
                )
            return
        if not (iinfo.min <= value <= iinfo.max):
            raise ValueError(
                f"{op_name} received {param_name}={value!r}, which is not representable in "
                f"dtype {dtype} (valid integer range: [{iinfo.min}, {iinfo.max}])"
            )
        return

    finfo = torch.finfo(dtype)
    value_f64 = float(value)
    if math.isnan(value_f64):
        return
    if math.isinf(value_f64):
        # PyTorch preserves +/-Inf for float tensor scalars. Ops needing a
        # finite scalar (elu alpha, softplus beta) reject here; masked_fill
        # writes the scalar into storage and opts in.
        if allow_nonfinite_float:
            return
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, but {param_name} must be finite and "
            f"representable in dtype {dtype}"
        )
    if not (finfo.min <= value_f64 <= finfo.max):
        raise ValueError(
            f"{op_name} received {param_name}={value!r}, which is not representable in "
            f"dtype {dtype} (valid finite range: "
            f"[{finfo.min}, {finfo.max}])"
        )


class LeakyReluFwdOp(Op):
    """Leaky ReLU: y = x if x > 0 else negative_slope * x.

    Single-input, single-output elementwise activation.  The input is
    flattened to a 1-D vector of ``N_total`` elements, the kernel is
    dispatched, and the output is reshaped back to the original shape.

    Args:
        N_total: Total number of elements (flattened input).
        dtype: Data type (float16, bfloat16, or float32).
        negative_slope: Slope for negative inputs (default 0.01).
        inplace: When True, copy the result back into ``input`` and
            return ``input`` (preserving tensor identity). The kernel
            still computes into a fresh buffer; only the user-visible
            tensor is mutated, mirroring ``torch.nn.functional.leaky_relu``.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "leaky_relu"
    kernel_cls = LeakyReluFwdKernel
    # Manifest: flops = "2 * N". Per roofline.md §1.3:
    # compare-and-select(1) + mul = 2 per elem.
    FLOPS_PER_ELEM = 2

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        negative_slope: float = 0.01,
        inplace: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        _validate_scalar_param_repr("negative_slope", negative_slope, dtype, self._op_name)
        self.N_total = N_total
        self.dtype = dtype
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.output_dtype = dtype  # same_as(input)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            N_total, dtype, negative_slope=negative_slope,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"leaky_relu": LeakyReluFwdKernel}

    def _validate_input(self, input: torch.Tensor) -> None:
        """Validate input tensor.

        NPU adaptation (O1): device check uses ``backend.is_device_tensor``
        instead of ``input.is_cuda``.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(input):
            raise ValueError(f"input must be a {backend.name} tensor, got device {input.device}")
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if input.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {input.numel()}")

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Direct kernel call: flatten -> kernel -> reshape."""
        orig_shape = input.shape
        flat = input.contiguous().reshape(-1)
        return self.kernel(flat).reshape(orig_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_input(input)
        if self.inplace:
            result = self._eager_forward(input)
            input.copy_(result.reshape(input.shape))
            return input
        return self._eager_forward(input)

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * (self.dtype.itemsize + self.output_dtype.itemsize)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this unary elementwise op instance.

        Mirrors the manifest roofline:
        ``flops = 2 * N`` and ``bytes = 2 * N * elem_bytes``.
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)
