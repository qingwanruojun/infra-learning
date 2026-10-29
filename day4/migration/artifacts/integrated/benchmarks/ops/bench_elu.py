"""Benchmarks for EluFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_elementwise_manifest.py`` —
ELU subset (``test_elu_manifest_bench``).  Workloads and roofline
formulas are loaded from the standalone manifest.  Single-input op: the
manifest's ``single_input_workload_contract`` holds (1 tensor input +
``alpha`` / ``inplace`` params).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.elu_fwd import EluFwdOp
from tileops.workloads.elementwise import EluWorkload

_ELU_OP = "EluFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ELU_OP, include_extra=True),
)
def test_elu_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = EluWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = EluFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_ELU_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
