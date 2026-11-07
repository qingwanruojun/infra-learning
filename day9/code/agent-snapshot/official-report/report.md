# TileOPs Evaluation Report: AminFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-11-05T19:04:06.943646+08:00 |
| evaluation_scope | AminFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261105_190051_728147_AminFwdOp |
| git_commit | a83796be77897cfb57274d0758f00ef50b9c27a0 |
| correctness_target | tests/ops/test_amin_fwd.py |
| benchmark_target | benchmarks/ops/bench_amin_fwd.py |

### Environment / 运行环境

| Item | Value |
|---|---|
| npu | Ascend910B3 x 1 |
| cpu | aarch64 |
| cann | 9.0.0 |
| driver | 25.5.0 |
| pytorch | 2.7.1+cpu |
| pytorch_npu | 2.7.1.post4 |
| tilelang | 0.1.1.10+ubuntu.20.4.cann900 |
| python | 3.11.4 |
| os | Linux-5.10.0-182.0.0.95.r2673_211.hce2.aarch64-aarch64-with-glibc2.31 |
| container | container |

## Results Overview / 结果总览

| Pass Rate | Operators | Total Cases | Failed Cases |
|---:|---:|---:|---:|
| 100.0% | 1 | 14 | 0 |

## Operator Analysis / 算子分析

| Operator | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|
| AminFwdOp | 100.0% (14/14) | 0.00e+00 | 4 | 33.66% – 68.19% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### AminFwdOp

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| hidden-state-reduce | 14.0600 | 68.19 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| hidden-state-reduce | 15.0200 | 63.83 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| long-seq-reduce | 7.1200 | 33.66 | [64, 32768] | bfloat16 | msprof | N/A | 1.7500 |
| 3d-multidim-reduce | 5.6800 | 42.20 | [4, 128, 4096] | float16 | msprof | main | 1.7500 |
