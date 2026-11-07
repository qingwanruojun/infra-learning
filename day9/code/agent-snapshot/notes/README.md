# Day 9：TileOPs 五算子 NPU 迁移与调优（agent 迁移 → agent 调优）

日期：2026-11-05（UTC+8）
环境：Ascend 910B3 / CANN 9.0.0 / torch 2.7.1+cpu + torch_npu 2.7.1 / TileLang npuir
工作目录：`refs/tilelang-mlir-ascend/examples/TileOPs`（迁移 harness）、`refs/TileOPs`（GPU 源仓）

## 目标（用户给的口径）

1. 评估 TileLang 对 NPU 的表达能力与缺口，给出语言扩展路线与演进方案。
2. 给出需要新增的优化原语及其目标场景。
3. 典型算子迁移 + 调优：**纯 vector 类平均 Roofline Ratio ≥ 60%（挑战 80%）**；**mix 类 ≥ 40%（挑战 60%）**。
4. 记录“迁移后 ratio”和“目标 ratio”。

## 算子与分类

| 算子 | 分类 | 状态 |
|---|---|---|
| AminFwdOp | 纯 vector | 迁移已存在，收尾修复中（UB 边界溢出） |
| MaxPool1dIndicesFwdOp | 纯 vector | 排队 |
| FusedAddRMSNormFwdOp | 纯 vector | agent 迁移中（Stage 0） |
| FFTC2COp | 纯 vector | 排队；c128 受 fp64 缺口阻塞 |
| GatedDeltaNetPrefillFwdOp | mix（cube+vector） | 排队；workload 需裁剪 |

## 流程（用户指定）

先用 agent 迁移（conductor `migration` 场景，harness 模式），迁移完成后在同一实现上起 `optimize` 场景调优。
单卡 + msprof 采样冲突 ⇒ 迁移/调优串行执行，不并行跑多个 conductor。

## 目录

```text
day9/
├── README.md                  # 本文件（状态 + 结论索引）
├── STATUS.md                  # 实时进度表（随执行更新）
├── note/
│   ├── learning-report.md     # 学习报告（缺口、原语、ratio 汇总）
│   └── gap-analysis.md        # TileLang/NPU 表达能力缺口与演进路线
└── code/
    ├── logs/                  # conductor 原始日志（按时刻快照）
    └── raw/                   # 基准 JSON、ratio 表快照
```
