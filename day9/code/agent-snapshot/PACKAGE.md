# Day 9 报告包（2026-11-05）

AminFwdOp 在 Ascend 910B3 上的迁移 + Stage 4 调优终态。

## 头条数字

| 项 | 数值 |
|---|---|
| 迁移后平均 Roofline Ratio | 42.73% |
| 调优后平均 Roofline Ratio | **51.97%** |
| 提升 | +9.24 pp |
| 要求（纯 vector 类平均） | ≥60%（挑战 80%） |
| 判定 | 未达标，缺口 8.03 pp；80% 物理不可达 |
| 正确性 | 14/14，max_diff = 0 |
| Stage 4 gate | pass（failures = []） |

逐例（等权平均，排除 smoke）：

| case | 迁移版 | 调优后 |
|---|---:|---:|
| `[2048,4096]` fp16（控制组） | 14.64 µs / 65.49% | 14.06 µs / 68.19% |
| `[2048,4096]` bf16 | 19.62 µs / 48.86% | 15.02 µs / 63.83% |
| `[64,32768]` bf16 | 7.50 µs / 31.96% | 7.12 µs / 33.66% |
| `[4,128,4096]` fp16 dim=[0,2] | 9.74 µs / 24.61% | 5.68 µs / 42.20% |

## 目录

```
notes/                     学习报告与调优笔记
  learning-report.md       ← 主报告：结论、口径、逐例、有效/无效改动、能力缺口
  amin-tuning-estimate.md  调优前估算 + Stage 4 中间/终态账面（§4b/§4c）
  gap-analysis.md          TileLang/NPU 表达能力缺口与演进路线
  roofline-calibration.md  分母标定（1.75 vs 1.20/1.27 TB/s 的溯源）
  queue-prep.md            其余算子的迁移准备
  README.md / STATUS.md    当日目标、队列、实时进度
raw/                       当日原始件：基准 JSON 快照、带宽标定输出、baseline ratio
logs/                      当日 conductor 日志与调优 opt_log 副本
official-report/           harness 官方报告 20261105_190051_728147_AminFwdOp
  report.md / report.html  人读版
  benchmark.json           逐例 Ratio（msprof 口径）
  run.json                 正确性 14/14 + 环境信息
  pytest/                  junit xml
  artifacts/               msprof 原始 profile（14 MB）
kernel/                    调优产物
  final/                   ← 最终提交的两个 kernel 源（wrapper 已切到这版）
  candidates/              8 轮全部候选实现
  opt_log.md               调优全过程（现象/假设/改动/A-B 裁决/plateau 判定）
  perf_records.jsonl       33 条测量记录（含 final 四条）
  workload_inventory.json  workload 清单与 tune 标记
  ab-logs/                 各轮 A-B-B-A 的 20 次逐次 Task Duration（裁决依据）
  regression-logs/         L0/L1/L2/boundary 回归日志（final 模块 + 两个 baseline）
gaps/capability-gaps.md    回写的语言/工具链能力缺口条目（含 Amin 实证）
```

## 关键结论

1. 有效改动只有两处：3d 的 `eff_bm` 上调（9.74→5.68 µs）、bf16 的 fp32 中转收敛（19.62→15.02 µs）。
2. 其余 8 个方向（reduce-on-shared、均衡分核、连续映射+合并 store、`T.Pipelined`、
   tn 下调、2-tile 交错、N-split、单 tile tn=32768）全部 tie / 回退 / UB 溢出。
3. 缺口集中在 long-seq（33.66%）与 3d（42.20%）两例；两个 16 MB 的 case 已过 60% 线。
4. 优化器判定 plateau，未触发 `[DESIGN_LIMIT]`（结构加速估计 long-seq 1.7×、
   3d 1.3×，未同时满足 ≥2× 门槛）；能力缺口记录在 `gaps/capability-gaps.md`。
5. 需要新增的原语（详见 `notes/learning-report.md` §6）：bf16 原生 reduce/min、
   显式分支/谓词向量原语（防同 dst 双 vcast 折叠）、按 shape/dtype 的 tile 策略分派、
   可声明地址聚簇的 task 映射、双缓冲口径的 UB 预算查询。
