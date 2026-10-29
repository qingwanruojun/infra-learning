# TileOPs Evaluation Report: EluFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-28T22:58:58.153135+08:00 |
| evaluation_scope | EluFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261028_224432_491740_EluFwdOp |
| git_commit | f388fab185ec7f59d827045ccec3c9e647efe09f |
| correctness_target | tests/ops/test_elu.py |
| benchmark_target | benchmarks/ops/bench_elu.py |

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
| EluFwdOp | 1h52m37s | 100.0% (14/14) | 0.00e+00 | 6 | 16.30% – 18.53% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### EluFwdOp

<details>
<summary>Session Timing / Session 耗时 — 1h52m37s</summary>

| Phase / 环节 | Duration / 耗时 | Ratio / 占比 | Detail / 说明 |
|---|---:|---:|---|
| 编排预检与场景路由 | 3m12s | 2.80% | 21:10:51→21:14:03：环境探测/harness 探测/状态 init/kb_stale_check（conductor Primary） |
| Stage 0 迁移脚手架 | 10m40s | 9.50% | 21:14:03→21:24:44：tileops-scaffolder 7 文件移植+Tier1 校验+meta |
| Stage 1 算子设计（first_design） | 33m49s | 30.00% | 21:26:06→21:59:55：tilelang-op-designer 产出 DESIGN.md+verify_equiv.py |
| Stage 1 门禁修复重试 | 8m10s | 7.30% | 22:00:39→22:08:50：S1-EQUIV-EXEC 弃选证据行从 §1.6.1 移入 §1.6.3 |
| Stage 2 设计检视 | 16m26s | 14.60% | 22:09:16→22:25:42：tilelang-design-reviewer 9 维度检视+D-1/D-5 机械复核，结论通过 |
| Stage 3 算子开发 | 11m59s | 10.60% | 22:26:10→22:38:09：tilelang-op-developer 实现+分层测试全过，[PRECISION_PASS] |
| Stage 5 迁移集成 | 21m23s | 19.00% | 22:41:03→23:02:26：tilelang-op-integrator 集成+单算子 report（正确性 14/14、bench 9/9） |
| 终态钩子与报告 | 1m54s | 1.70% | 23:02:26→23:03:28 后：opencode export+耗时分析+报告渲染（conductor Primary） |

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| mlp-hidden | 78.7616 | 16.30 | [2048, 4096] | float16 | msprof | main | 1.7500 |
| mlp-hidden | 80.5616 | 18.20 | [2048, 4096] | bfloat16 | msprof | main | 1.7500 |
| mlp-hidden-wide | 156.3031 | 16.40 | [2048, 8192] | float16 | msprof | main | 1.7500 |
| mlp-hidden-wide | 158.7032 | 18.45 | [2048, 8192] | bfloat16 | msprof | main | 1.7500 |
| gan-feat | 155.9031 | 16.45 | [16, 256, 64, 64] | float16 | msprof | main | 1.7500 |
| gan-feat | 157.9832 | 18.53 | [16, 256, 64, 64] | bfloat16 | msprof | main | 1.7500 |
