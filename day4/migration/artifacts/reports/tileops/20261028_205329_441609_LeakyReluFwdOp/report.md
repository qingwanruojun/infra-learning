# TileOPs Evaluation Report: LeakyReluFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-28T21:08:11.354650+08:00 |
| evaluation_scope | LeakyReluFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261028_205329_441609_LeakyReluFwdOp |
| git_commit | 88ee8c302e7c5bc2153d68f02b613cae04449351 |
| correctness_target | tests/ops/test_leaky_relu.py |
| benchmark_target | benchmarks/ops/bench_leaky_relu.py |

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

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| LeakyReluFwdOp | N/A | 100.0% (14/14) | 0.00e+00 | 6 | 13.06% – 13.55% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### LeakyReluFwdOp

<details>
<summary>Session Timing / Session 耗时 — N/A</summary>

> Timing data unavailable: breakdown[10].duration_s must be a number

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| gan-feat | 141.4628 | 13.55 | [16, 256, 64, 64] | float16 | msprof | main | 1.7500 |
| gan-feat | 144.8429 | 13.24 | [16, 256, 64, 64] | bfloat16 | msprof | main | 1.7500 |
| gan-feat-deep | 71.3614 | 13.43 | [16, 512, 32, 32] | float16 | msprof | main | 1.7500 |
| gan-feat-deep | 72.0814 | 13.30 | [16, 512, 32, 32] | bfloat16 | msprof | main | 1.7500 |
| fc-wide | 71.1014 | 13.48 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| fc-wide | 73.4215 | 13.06 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
