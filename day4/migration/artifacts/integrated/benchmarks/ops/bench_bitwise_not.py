"""Benchmarks for BitwiseNotFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_unary_elementwise.py`` —
bitwise_not subset (``test_bitwise_not_bench``, per the shared-file
porting rule: only this op's case is ported).  Workloads and roofline
formulas are loaded from the standalone manifest.  Single-input op: the
manifest's ``single_input_workload_contract`` holds (1 tensor input, no
params), so ``workloads_to_params(..., include_extra=True)`` applies.

The GPU ``torch.bitwise_not`` baseline is a native op, but the NPU
project's established bench convention records the TileOPs kernel only
(see bench_mish / bench_elu / bench_unary_elementwise); the accuracy
reference stays in the correctness tests and is not benchmarked here.
No autotune try/except (T4) — the NPU project removed autotune.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.bitwise_not import BitwiseNotFwdOp
from tileops.workloads.elementwise import BitwiseNotWorkload

_BITWISE_NOT_OP = "BitwiseNotFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_BITWISE_NOT_OP, include_extra=True),
)
def test_bitwise_not_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = BitwiseNotWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = BitwiseNotFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_BITWISE_NOT_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
