# infra-learning

昇腾 910B3 上的 TileLang NPUIR 学习与实验记录。学习主线从环境与 DSL 起步，经过 Vector、Cube、MixCV、MLIR 编译和 Agent 生成，最后归纳设备性能测量与验收方法。算子迁移及随后的交付、调优和复核归入 [工作任务](任务记录.md)，按任务口径汇报。

## 学习路线

| 阶段 | 学习内容 | 实验与记录 |
|---|---|---|
| Day 1 · 10/26 | Agent 循环、工具权限与仓库分工 | [Agent 与技术链路](day1/day1学习指标.md) |
| Day 2 · 10/27 | CANN/TileLang 环境、DSL 与运行链 | [环境与 DSL](day2/day2学习指标.md)，VecAdd 冒烟通过 |
| Day 3 · 10/28 | Vector 算子、UB 分块、尾块与数值校验 | [abs/RMSNorm](day3/note/day3学习指标.md)，12 个测试通过 |
| Day 4 · 10/29 | Cube GEMM、分块与最小 MixCV 协作 | [Cube 实验](day4/cube/note/day4报告-Cube.md) |
| Day 5 · 10/30 | FlashAttention 数据流、Expert/Developer 与 CANN 兼容 | [实验与复现](day5/day5-flashatten/README.md) |
| Day 6 · 10/31–11/02 | MLIR 基础、HFusion/HIVM、设备产物与 host 验收 | [MLIR 到 NPU](day6/README.md) |
| Day 7 · 11/02 | 跟踪同一次 VecAdd 的 TIR、NPUIR、pass 和设备产物 | [编译链追踪](day7/README.md) |
| Day 8 · 11/04 | 按 CANNBot 工作流材料组织 Agent 生成 RMSNorm，修复 UB 并独立验收 | [生成与验收](day8/README.md) |

公开学习记录截至 Day 8（11 月 4 日）。阶段收获见 [学习总结](学习总结.md)。

## 复现与归档

- 算子和实验代码按原日期保存在对应目录。运行前按各实验 README 核对 CANN、TileLang 及设备配置。
- Day 6 从 `day6/code/repro.sh` 开始，检查设备运行与独立数值校验。Day 7 从 `day7/code/run_vecadd_dev.sh` 开始，按该日 README 提取 IR 并核对原 JIT 和复编产物。
- 早期 PDF、周会和飞书整理稿保留各自记录时间。
- [Skills](skills/README.md)汇总算子开发、性能测量、报告与任务迁移流程。

## 相关项目

- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)
- [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)
- [cannbot-skills](https://gitcode.com/cann/cannbot-skills)

## 联系方式

如果对我的学习记录、实验结果或复现步骤有疑问，欢迎发邮件联系。[2133292721@qq.com](mailto:2133292721@qq.com)
