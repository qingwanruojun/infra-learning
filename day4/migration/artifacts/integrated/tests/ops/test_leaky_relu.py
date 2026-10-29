"""Correctness tests for LeakyReluFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_activation.py`` — LeakyReLU subset
(``test_leaky_relu`` + ``test_independent_activation_rejects_non_float_dtype``).
Uses the device backend instead of hard-coded ``"cuda"`` (T1).
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.kernels.elementwise.leaky_relu import LeakyReluFwdKernel
from tileops.ops.elementwise.leaky_relu import LeakyReluFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase, RandnWorkload


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


def _device() -> str:
    return get_device_backend().name


class LeakyReluTest(RandnWorkload, TestBase):
    """Test fixture for LeakyReluFwdOp — flat 1-D randn input."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        super().__init__((n_total,), dtype)
        self.n_total = n_total

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x.float(), 0.01).to(x.dtype)


class LeakyReluFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(
                    1_048_576, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
                ),
                pytest.param(1_048_576, torch.float16, marks=[pytest.mark.smoke]),
                pytest.param(1_048_576, torch.bfloat16, marks=[pytest.mark.smoke]),
                # Full: larger follow-up coverage
                pytest.param(4_000_000, torch.float16, marks=[pytest.mark.full]),
                pytest.param(4_000_000, torch.bfloat16, marks=[pytest.mark.full]),
            ],
        ),
    ]


@LeakyReluFixture
def test_leaky_relu_op(n_total: int, dtype: torch.dtype) -> None:
    test = LeakyReluTest(n_total, dtype)
    op = LeakyReluFwdOp(N_total=n_total, dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@LeakyReluFixture
def test_leaky_relu_param(n_total: int, dtype: torch.dtype) -> None:
    """Non-default negative_slope is honored (constructor-time scalar)."""
    test = LeakyReluTest(n_total, dtype)
    op = LeakyReluFwdOp(N_total=n_total, dtype=dtype, negative_slope=0.2)
    x = test.gen_inputs()[0]
    ref = F.leaky_relu(x.float(), 0.2).to(x.dtype)
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(op(x), ref, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_leaky_relu_inplace() -> None:
    """inplace=True mutates the input and returns it (tensor identity kept)."""
    dtype = torch.float32
    n_total = 4096
    test = LeakyReluTest(n_total, dtype)
    op = LeakyReluFwdOp(N_total=n_total, dtype=dtype, inplace=True)
    x = test.gen_inputs()[0]
    ref = F.leaky_relu(x.float(), 0.01).to(x.dtype)
    out = op(x)
    assert out is x
    torch.testing.assert_close(x, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.smoke
def test_leaky_relu_rejects_non_float_dtype() -> None:
    with pytest.raises(ValueError, match="only supports dtypes"):
        LeakyReluFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_leaky_relu_rejects_nonfinite_negative_slope() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        LeakyReluFwdOp(N_total=16, dtype=torch.float32, negative_slope=float("inf"))


@pytest.mark.smoke
def test_leaky_relu_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = LeakyReluFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 2 * N, f"flops {flops} != 2 * N = {2 * N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
