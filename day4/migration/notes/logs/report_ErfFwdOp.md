# TileOPs Evaluation Report: ErfFwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-29T04:12:47.611065+08:00 |
| evaluation_scope | ErfFwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261029_034150_398117_ErfFwdOp |
| git_commit | d7f990d0f2d1899a45f46af76ab12f4bd5de0d15 |
| correctness_target | tests/ops/test_unary_math.py |
| benchmark_target | benchmarks/ops/bench_unary_elementwise.py |

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
| 100.0% | 1 | 16 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| ErfFwdOp | 1h59m39s | 100.0% (10/10) | 4.33e-05 | 7 | 20.21% – 32.22% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### ErfFwdOp

<details>
<summary>Session Timing / Session 耗时 — 1h59m39s</summary>

| Phase / 环节 | Duration / 耗时 | Ratio / 占比 | Detail / 说明 |
|---|---:|---:|---|
| 场景路由与准备 | 1m45s | 1.50% | 02:16:05-02:17:50 场景识别+harness 探测+场景文件载入+leaky_relu 教训读取+statectl init+start 0+kb_stale_check(154 stale) |
| Stage 0 脚手架 | 10m10s | 8.50% | 02:17:50-02:28:00 @tileops-scaffolder：add-npu-op 机器模式 7 文件移植+Tier1 校验+meta |
| Stage 0 收束 | 1m06s | 0.90% | 02:28:00-02:29:06 gate0+complete0+migration init(聚合状态) |
| Stage 1 首次设计 | 38m37s | 32.30% | 02:29:06-03:07:43 @tilelang-op-designer：M0→R→M1 全流程（T.verf 禁用双证+A&S 7.1.26 机器验证+persistent 分核三要素+verify_equiv EQUIV_PASS） |
| Stage 1 收束 | 22.000s | 0.30% | 03:07:43-03:08:05 gate1+complete1+start2 |
| Stage 2 检视 | 14m16s | 11.90% | 03:08:05-03:22:21 @tilelang-design-reviewer：9 维度全 pass+机械复核（verify_equiv 重跑/design_calc_check/kb_stale_check 对账） |
| Stage 2 收束 | 23.000s | 0.30% | 03:22:21-03:22:44 complete2+start3 |
| Stage 3 开发 | 13m53s | 11.60% | 03:22:44-03:36:37 @tilelang-op-developer：A&S 22-op v-prefix 链+fp32 transit+persistent，L0/L1/L2/Boundary 全过→PRECISION_PASS |
| Stage 3 收束+二次校验 | 2m53s | 2.40% | 03:36:37-03:39:30 complete3+conductor 二次校验 --level all 全过+func done+phase INTEGRATE+start5 |
| Stage 5 集成 | 35m47s | 29.90% | 03:39:30-04:15:17 @tilelang-op-integrator：integrate_kernel.py+单算子 report（correctness 16/16+msprof 20/20 bench） |
| 终态收束 | 27.000s | 0.40% | 04:15:17-04:15:44 complete5+integrate pass+phase DONE+opencode export full.json+Session 耗时分析 |

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 294.8059 | 31.24 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 298.4460 | 32.08 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 442.6689 | 20.21 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| elementwise-64m | 1176.0835 | 31.31 | [8192, 8192] | float16 | msprof | main | 1.7500 |
| elementwise-64m | 1187.9637 | 32.22 | [8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| elementwise-256m | 4702.4741 | 31.32 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| elementwise-256m | 4761.3149 | 32.16 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
