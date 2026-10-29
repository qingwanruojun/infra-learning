"""Benchmarks for Expm1FwdOp and ErfFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_unary_elementwise.py`` —
expm1 + erf subsets (``test_expm1_bench`` / ``test_erf_bench``, per the
shared-file porting rule: only these ops' cases are ported).  Workloads
and roofline formulas are loaded from the standalone manifest.
Single-input ops: the manifest's ``single_input_workload_contract``
holds (1 tensor input, no params).

The GPU ``torch.expm1`` / ``torch.erf`` baselines are native ops, but
the NPU project's established bench convention records the TileOPs
kernel only (see bench_mish / bench_elu / bench_leaky_relu); the
accuracy reference stays in the correctness tests and is not
benchmarked here.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.elementwise.erf_fwd import ErfFwdOp
from tileops.ops.elementwise.expm1 import Expm1FwdOp
from tileops.workloads.elementwise import ErfWorkload, Expm1Workload

_EXPM1_OP = "Expm1FwdOp"
_ERF_OP = "ErfFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_EXPM1_OP, include_extra=True),
)
def test_expm1_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = Expm1Workload(shape, dtype)
    inputs = test.gen_inputs()

    op = Expm1FwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_EXPM1_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_ERF_OP, include_extra=True),
)
def test_erf_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    test = ErfWorkload(shape, dtype)
    inputs = test.gen_inputs()

    op = ErfFwdOp(N_total=test.n_total, dtype=dtype, **op_params)
    bm = ManifestBenchmark(_ERF_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
