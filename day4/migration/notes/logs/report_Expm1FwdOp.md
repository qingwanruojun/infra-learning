# TileOPs Evaluation Report: Expm1FwdOp

## Experiment Setup / 评测配置

### Metadata / 元信息

| Item | Value |
|---|---|
| framework | TileOPs Reporting 0.1.0 |
| date | 2026-10-29T01:38:24.115147+08:00 |
| evaluation_scope | Expm1FwdOp |
| benchmark | TileOPs pytest operator suite |
| profiler | msprof |
| run_id | 20261029_012203_068316_Expm1FwdOp |
| git_commit | ef4078a697b36c575c8c258179be3fac293325e9 |
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
| 100.0% | 1 | 8 | 0 |

## Operator Analysis / 算子分析

| Operator | Total Time | Correctness | Avg Max Abs Error | Performance Shapes | Ratio Range |
|---|---:|---:|---:|---:|---:|
| Expm1FwdOp | 2h39m01s | 100.0% (8/8) | 0.00e+00 | 7 | 14.71% – 27.08% |

> Correctness 格式为通过率 (通过用例/总用例)；Avg Max Abs Error 来自正确性测试；Ratio 为各 shape 的范围。

## Operator Details / 算子明细

### Expm1FwdOp

<details>
<summary>Session Timing / Session 耗时 — 2h39m01s</summary>

| Phase / 环节 | Duration / 耗时 | Ratio / 占比 | Detail / 说明 |
|---|---:|---:|---|
| SCAFFOLD (Stage 0) | 12m18s | 7.70% | 23:05:41→23:18:00 tileops-scaffolder 完成 7 文件脚手架 + .migration_meta.json + 迁移 prompt（日志：task 会话 ses_e7cf45728... |
| DESIGN first (Stage 1) | 36m09s | 22.70% | 23:19:50→23:55:58 tilelang-op-designer first_design 生成 DESIGN.md + verify_equiv.py（54/54 行精度机器验证） |
| REVIEW first (Stage 2) | 23m34s | 14.80% | 23:56:46→00:20:21 tilelang-design-reviewer 首轮检视：结论不通过（3 阻塞：内嵌表不可复现 / wrapper 契约断言错误 / UB 口径误用） |
| DESIGN revision (Stage 1) | 28m58s | 18.20% | 00:21:10→00:50:08 designer mode=revision 修复 3 阻塞 + 4 建议（表格重填 / 两段式契约 / 平台值口径+guard 分 dtype） |
| REVIEW re-review (Stage 2) | 13m32s | 8.50% | 00:50:28→01:04:01 复审：结论通过（0 阻塞，修订闭环逐项验证到决策行） |
| DEVELOP (Stage 3) | 11m59s | 7.50% | 01:04:28→01:16:27 tilelang-op-developer first_impl：[PRECISION_PASS]，L0/L1/L2/Boundary 全过（含二次校验重跑全量） |
| INTEGRATE (Stage 5) | 23m57s | 15.10% | 01:17:28→01:41:25 tilelang-op-integrator 集成 + wrapper 两段式适配 + 单算子 report（correctness 8/8、benchmark 7/7 有效） |
| conductor 编排（Primary 内联） | 8m32s | 5.40% | 23:03:06→01:42:08 场景路由/状态机/门禁/知识预注入/耗时分析等内联工具调用与推理，含各阶段间编排间隙 |

</details>

| Label | Latency (us) | Ratio (%) | Shape / Parameters | DType | Mode | Kernel | Bandwidth (TB/s) |
|---|---:|---:|---|---|---|---|---:|
| elementwise-16m | 128.1226 | 14.97 | [4096, 4096] | float16 | msprof | main | 1.7500 |
| elementwise-16m | 130.3226 | 14.71 | [4096, 4096] | bfloat16 | msprof | main | 1.7500 |
| elementwise-16m | 219.7044 | 17.45 | [4096, 4096] | float32 | msprof | main | 1.7500 |
| elementwise-64m | 507.1902 | 18.90 | [8192, 8192] | float16 | msprof | main | 1.7500 |
| elementwise-64m | 515.8703 | 18.47 | [8192, 8192] | bfloat16 | msprof | main | 1.7500 |
| elementwise-256m | 2053.0210 | 27.08 | [16384, 16384] | float16 | msprof | main | 1.7500 |
| elementwise-256m | 2090.6418 | 26.60 | [16384, 16384] | bfloat16 | msprof | main | 1.7500 |
