# TileOPs Evaluation Report: BitwiseNotFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-29T06:04:12.062167+08:00 |
| evaluation_scope | BitwiseNotFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261029_055054_204932_BitwiseNotFwdOp |
| git_commit | 2a94a29845e556ba0d75a07f699ceb9886544f63 |
| correctness_target | tests/ops/test_bitwise_not.py |
| benchmark_target | benchmarks/ops/bench_bitwise_not.py |

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
| 100.0% | 1 | 9 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| BitwiseNotFwdOp | 1h21m57s | 100.0% (9/9) | 0.00e+00 | 5 | 5.57% – 19.10% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### BitwiseNotFwdOp

<details>
<summary>Session Timing / Session 耗时 — 1h21m57s</summary>

| Phase / 环节 | Duration / 耗时 | Ratio / 占比 | Detail / 说明 |
|---|---:|---:|---|
| Stage 0 脚手架 | 9m41s | 11.80% | scaffolder ses_e7bbb589 20:47:34→20:57:16，7 文件移植 + Tier 1 + meta |
| Stage 1 算子设计 | 18m30s | 22.60% | designer ses_e7bb13139 20:58:39→21:17:10，DESIGN.md 617 行 + verify_equiv |
| Stage 1 门禁修复 | 41.400s | 0.80% | designer ses_e7b9fb513 21:17:45→21:18:26，2 处 S1-PLACEHOLDER 字面量修复 |
| Stage 2 设计检视 | 12m26s | 15.20% | reviewer ses_e7b9ed5ba 21:18:42→21:31:09，9 维度 REVIEW.md 结论通过 |
| Stage 3 算子开发 | 13m37s | 16.60% | developer ses_e7b93184b 21:31:32→21:45:13，kernel + L0/L1/L2/Boundary 全过 |
| Stage 5 集成验证 | 20m34s | 25.10% | integrator ses_e7b85aa9a 21:46:12→22:06:46，集成 + 单算子 report passed |
| 编排与校验开销 | 6m24s | 7.80% | conductor 状态迁移/门禁/二次精度校验/kb 检索/export 导出等 20:45:32→22:07:29 |

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 455.5091 | 8.42 | [4096, 4096] | int32 | msprof | main | 1.7500 |
| elementwise-16m | 501.9700 | 19.10 | [4096, 4096] | int64 | msprof | main | 1.7500 |
| elementwise-64m | 1699.7540 | 5.57 | [8192, 8192] | int16 | msprof | main | 1.7500 |
| elementwise-64m | 1801.0160 | 13.80 | [8192, 8192] | int32 | msprof | main | 1.7500 |
| elementwise-256m | 7220.3647 | 16.20 | [16384, 16384] | int32 | msprof | main | 1.7500 |
