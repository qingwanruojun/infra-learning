"""Correctness tests for Expm1FwdOp and ErfFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_unary_math.py`` — expm1 + erf
subsets (``test_expm1`` / ``test_expm1_edge`` / ``test_erf`` /
``test_erf_edge``, per the shared-file porting rule: only these ops'
cases are ported).  Uses the device backend instead of hard-coded
``"cuda"`` (T1); no CUDA skip markers (T2); no ``tune`` params (T3) or
autotune try/except (T4).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.erf_fwd import ErfFwdKernel
from tileops.kernels.elementwise.expm1 import Expm1FwdKernel
from tileops.ops.elementwise.erf_fwd import ErfFwdOp
from tileops.ops.elementwise.expm1 import Expm1FwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.workload_base import FixtureBase, RandnWorkload


def _device() -> str:
    return get_device_backend().name


def _get_tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.float16:
        return {"atol": 1e-3, "rtol": 1e-3}
    if dtype == torch.bfloat16:
        return {"atol": 1.6e-2, "rtol": 1.6e-2}
    return {"atol": 1e-5, "rtol": 1e-5}


def _randn(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, device=_device(), dtype=dtype)


def _repeat_values(values: list[float], n: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor(values, device=_device(), dtype=dtype)
    repeats = (n + len(values) - 1) // len(values)
    return base.repeat(repeats)[:n]


class Expm1Test(RandnWorkload, TestBase):
    """Test fixture for Expm1FwdOp — flat 1-D input (port of GPU
    ``UnaryMathTest``; ``gen_fn`` lets the edge case substitute a
    domain-restricted draw)."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None, ref_fn=None):
        super().__init__((n_total,), dtype)
        self.n_total = n_total
        self._gen_fn = gen_fn
        self._ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        return (torch.randn(self.n_total, device=_device(), dtype=self.dtype),)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch ground truth, upcast to fp32 and downcast back
        # (skill-sanctioned NPU reference form; mirrors the GPU
        # floor/ceil/round refs and avoids bf16 npu-dispatch gaps).
        return self._ref_fn(x.float()).to(x.dtype)


def _make_expm1_test(n_total, dtype, gen_fn, ref_fn):
    """Build test, instantiate op, and run check (port of GPU
    ``_make_math_test`` restricted to Expm1FwdOp)."""
    test = Expm1Test(n_total, dtype, gen_fn=gen_fn, ref_fn=ref_fn)
    op = Expm1FwdOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_get_tolerances(dtype))


class Expm1Fixture(FixtureBase):
    """L1 correctness across supported float dtypes.

    Port of GPU ``MathFixture`` — first 3 params are smoke
    (fp32 / fp16 / bf16), fp32 also covers packaging.
    """

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(
                1_048_576, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
            ),
            pytest.param(1_048_576, torch.float16, marks=[pytest.mark.smoke]),
            pytest.param(1_048_576, torch.bfloat16, marks=[pytest.mark.smoke]),
            # Full: larger follow-up coverage
            pytest.param(4_000_000, torch.float16, marks=[pytest.mark.full]),
            pytest.param(4_000_000, torch.bfloat16, marks=[pytest.mark.full]),
        ]),
    ]


class Expm1EdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements (port of GPU ``MathEdgeFixture``)."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=[pytest.mark.smoke]),
        ]),
    ]


@Expm1Fixture
def test_expm1(n_total: int, dtype: torch.dtype) -> None:
    _make_expm1_test(n_total, dtype, _randn, torch.expm1)


@Expm1EdgeFixture
def test_expm1_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_expm1_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 88.8, -88.8, 1e-7], n, d),
        torch.expm1,
    )


@pytest.mark.smoke
def test_expm1_rejects_non_float_dtype() -> None:
    # Port of GPU ``test_math_ops_reject_non_float_dtype`` (expm1 subset):
    # FloatUnaryKernel rejects non-float dtypes.
    with pytest.raises(ValueError, match="only supports dtypes"):
        Expm1FwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_expm1_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = Expm1FwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == 2 * N, f"flops {flops} != 2 * N = {2 * N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


# ---------------------------------------------------------------------------
# ErfFwdOp (erf subset; port of GPU ``test_erf`` / ``test_erf_edge``)
# ---------------------------------------------------------------------------


class ErfTest(RandnWorkload, TestBase):
    """Test fixture for ErfFwdOp — flat 1-D input (port of GPU
    ``UnaryMathTest``; ``gen_fn`` lets the edge case substitute a
    domain-restricted draw)."""

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None, ref_fn=None):
        super().__init__((n_total,), dtype)
        self.n_total = n_total
        self._gen_fn = gen_fn
        self._ref_fn = ref_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        return (torch.randn(self.n_total, device=_device(), dtype=self.dtype),)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch ground truth, upcast to fp32 and downcast back
        # (skill-sanctioned NPU reference form; mirrors the GPU
        # floor/ceil/round refs and avoids bf16 npu-dispatch gaps).
        return self._ref_fn(x.float()).to(x.dtype)


def _make_erf_test(n_total, dtype, gen_fn, ref_fn):
    """Build test, instantiate op, and run check (port of GPU
    ``_make_math_test`` restricted to ErfFwdOp)."""
    test = ErfTest(n_total, dtype, gen_fn=gen_fn, ref_fn=ref_fn)
    op = ErfFwdOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_get_tolerances(dtype))


class ErfFixture(FixtureBase):
    """L1 correctness across supported float dtypes.

    Port of GPU ``MathFixture`` — first 3 params are smoke
    (fp32 / fp16 / bf16), fp32 also covers packaging.
    """

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(
                1_048_576, torch.float32, marks=[pytest.mark.smoke, pytest.mark.packaging]
            ),
            pytest.param(1_048_576, torch.float16, marks=[pytest.mark.smoke]),
            pytest.param(1_048_576, torch.bfloat16, marks=[pytest.mark.smoke]),
            # Full: larger follow-up coverage
            pytest.param(4_000_000, torch.float16, marks=[pytest.mark.full]),
            pytest.param(4_000_000, torch.bfloat16, marks=[pytest.mark.full]),
        ]),
    ]


class ErfEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements (port of GPU ``MathEdgeFixture``)."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=[pytest.mark.smoke]),
        ]),
    ]


@ErfFixture
def test_erf(n_total: int, dtype: torch.dtype) -> None:
    _make_erf_test(n_total, dtype, _randn, torch.erf)


@ErfEdgeFixture
def test_erf_edge(n_total: int, dtype: torch.dtype) -> None:
    _make_erf_test(
        n_total,
        dtype,
        lambda n, d: _repeat_values([0.0, 3.0, -3.0, 100.0], n, d),
        torch.erf,
    )


@pytest.mark.smoke
def test_erf_rejects_non_float_dtype() -> None:
    # Port of GPU ``test_math_ops_reject_non_float_dtype`` (erf subset):
    # FloatUnaryKernel rejects non-float dtypes.
    with pytest.raises(ValueError, match="only supports dtypes"):
        ErfFwdKernel(N_total=16, dtype=torch.int32)


@pytest.mark.smoke
def test_erf_eval_roofline() -> None:
    N = 1_048_576
    dtype = torch.float16
    op = ErfFwdOp(N_total=N, dtype=dtype)
    flops, mem_bytes = op.eval_roofline()
    elem_bytes = dtype.itemsize
    assert flops == N, f"flops {flops} != N = {N}"
    assert mem_bytes == 2 * N * elem_bytes, (
        f"bytes {mem_bytes} != 2 * N * elem_bytes = {2 * N * elem_bytes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
