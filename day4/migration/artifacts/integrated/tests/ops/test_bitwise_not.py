"""Correctness tests for BitwiseNotFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_bitwise.py`` — bitwise_not subset
(``BitwiseFixture`` / ``BitwiseNotTest`` / ``test_bitwise_not`` /
``test_bitwise_not_rejects_float_dtype``; per the shared-file porting
rule only this op's cases are ported — the and/or/xor ops are not yet
migrated).  Bitwise NOT is exact integer/bool arithmetic, so the
reference uses exact comparison (no fp32 upcast, no tolerance).  Uses
the device backend instead of hard-coded ``"cuda"`` (T1); no CUDA skip
markers (T2); no ``tune`` params (T3) or autotune try/except (T4).
"""

import pytest
import torch

from tileops.kernels.elementwise.bitwise_not import BitwiseNotFwdKernel
from tileops.ops.elementwise.bitwise_not import BitwiseNotFwdOp
from tileops.testing.test_base import TestBase, exact_compare
from tileops.workloads.elementwise import BitwiseNotWorkload
from tileops.workloads.workload_base import FixtureBase


class BitwiseFixture(FixtureBase):
    """Parametrize over torch-supported bitwise_not dtypes.

    Port of GPU ``BitwiseFixture`` — all six bool/integer dtypes are
    smoke-level (the op is exact integer/bool arithmetic).
    """

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.bool, marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(1_048_576, torch.uint8, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int8, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int32, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.int64, marks=pytest.mark.smoke),
        ]),
    ]


class BitwiseNotTest(BitwiseNotWorkload, TestBase):
    """Test fixture for bitwise_not."""

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bitwise_not(x)


@BitwiseFixture
def test_bitwise_not(n_total: int, dtype: torch.dtype) -> None:
    test = BitwiseNotTest((n_total,), dtype)
    op = BitwiseNotFwdOp(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.float16, marks=pytest.mark.smoke),
    pytest.param(torch.bfloat16, marks=pytest.mark.smoke),
    pytest.param(torch.float32, marks=pytest.mark.smoke),
])
def test_bitwise_not_rejects_float_dtype(dtype: torch.dtype) -> None:
    # Float dtypes are outside the manifest contract (bool | uint8 | int8
    # | int16 | int32 | int64) and must be rejected by the kernel.
    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotFwdKernel(N_total=16, dtype=dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
