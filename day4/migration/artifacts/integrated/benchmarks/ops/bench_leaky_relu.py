"""Benchmarks for LeakyReluFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` —
LeakyReLU subset (``test_leaky_relu_manifest_bench``).  Workloads and
roofline formulas are loaded from the standalone manifest.  Single-input
op: the manifest's ``single_input_workload_contract`` holds (1 tensor
input + ``negative_slope`` / ``inplace`` params).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.leaky_relu import LeakyReluFwdOp
from tileops.workloads.elementwise import LeakyReluWorkload

_LEAKY_RELU_OP = "LeakyReluFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_LEAKY_RELU_OP, include_extra=True),
)
def test_leaky_relu_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = LeakyReluWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = LeakyReluFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_LEAKY_RELU_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
