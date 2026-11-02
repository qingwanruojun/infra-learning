# infra-learning

昇腾 910B3 上的算子开发与编译链学习记录。主线是从 Agent/Skills 辅助写算子，走到 TileLang NPUIR、AscendNPU-IR、NPU 执行和 profiler 反馈。仓库按 Day 1–10 归档；**截至 2026-11-02，Day 1–6 已有实验与报告，Day 7–10 仍按计划推进**。

```text
Agent / CANNBot Skills → TileLang Python DSL → TVM TIR → NPUIR / HFusion / HIVM
                         → 设备二进制 → NPU 运行与校验 → profiler 反馈
```

这是一张学习路线图，各天验证到的阶段不同。尤其是 Day 6：HIVM VecAdd 与 HFusion 来源的 VecAdd 都完成了上卡正确性验证；HFusion 来源的路径使用两段工具链，尚未验证包内编译器的单命令路径或多算子融合。

## 从哪里开始看

| 天 | 内容与入口 | 当前结果 |
|---|---|---|
| Day 1 · 10/26 | [Agent 原理与仓库地图](day1/day1学习指标.md) | 梳理 Codex 式 agent loop、三个仓库的分工和学习链路 |
| Day 2 · 10/27 | [环境与 TileLang DSL](day2/day2学习指标.md) | CANN/TileLang 环境验证、`vec_add_1d` 冒烟、DSL 速读 |
| Day 3 · 10/28 | [Vector 算子笔记](day3/note/day3学习指标.md) · [代码与日志索引](day3/code/README.md) | 手写 abs/RMSNorm，正确性与精度测试；另复核 host、device、e2e 测量口径 |
| Day 4 · 10/29 | [TileOPs 五算子迁移](day4/migration/迁移报告.md) · [Cube GEMM 报告](day4/cube/note/day4报告-Cube.md) | 迁移五个算子并做独立复验；Cube GEMM 三档实现和 MixCV 最小例子 |
| Day 5 · 10/30 | [FlashAttention / MixCV 交付包](day5/day5-flashatten/README.md) · [学习笔记](day5/day5-flashatten/note/day5学习指标.md) | Expert 与 Developer 正确性链路通过；Developer 模式完成首轮性能对拍 |
| Day 6 · 10/31–11/02 | [MLIR→NPU 总入口](day6/README.md) · [学习报告](day6/note/day6学习报告.md) · [代码和原始日志](day6/code/) | 最小 MLIR pass 实验；HIVM 和 HFusion 来源的 VecAdd 上卡；host 强校验与负例 |
| Day 7 · 计划 11/03 | 编译链路追踪 | 待做：用同一次计算追 TVM IR、NPUIR/HIVM，核实 TileLangIR 是否实际参与 |
| Day 8–10 · 计划 11/04–11/06 | Agent 生成、profiling 调优、最终整合 | 待做；不预写实验结果 |

## 已完成阶段的关键收获

**Day 1–2：把工具链放到机器上。**理解 Agent、CANNBot Skills、TileLang 和 AscendNPU-IR 各负责哪一段；编译 TileLang 并用 `examples/elementwise/vec_add_1d.py` 做最初的 NPU 冒烟。环境和 DSL 细节以 Day 2 笔记为准。

**Day 3：手写 Vector 算子，也修正了性能口径。**abs 和 RMSNorm 覆盖 fp16/fp32 与尾块等情况；最终正确性记录为 12 个 pytest 用例通过。早期 4/64 MiB 或 128 MiB 的对拍数字受 host、缓存与会话波动影响，不能直接当作 HBM kernel 的优劣；完整数据和口径见 [Day 3 笔记](day3/note/day3学习指标.md)。

**Day 4：迁移与 Cube 两条线。**BitwiseNot、Erf、Expm1、Elu、LeakyRelu 五个 TileOPs 算子通过框架验收，独立复验记录为 53 passed；性能记录只作为未调优基线。Cube 侧完成单块、多块和动态尾块 GEMM，并验证最小 Cube+Vector 同核例子。迁移边界和有效性能点分别见两份 Day 4 报告。

**Day 5：FlashAttention 的正确性与首轮对拍。**Expert 模式因 `syn_instr_mode` 与 CANN 9.0.0 后端不兼容而做 JIT 适配，修复后能运行；Developer 模式完成多形状正确性和 profiler 记录。性能数据仅属于 Developer 模式，64×64 tile 等原因仍是待验证假设，见 [Day 5 交付包](day5/day5-flashatten/README.md)。

**Day 6：从 IR 走到可核对的运行结果。**`mlir-opt` 小例子展示常量折叠和 CSE；HIVM `add.mlir` 编译成设备 ELF 后，两组各 16 个 int16 结果正确。新的 HFusion/Linalg 加法输入也经转换、设备编译和上卡得到 32/32 正确结果。host 会检查发射返回码并逐元素比较；故意换成乘法内核时检出 30/32 处不匹配、退出码为 1。这里没有性能或多算子融合结论。详见 [Day 6 README](day6/README.md)。

## 目录与复现入口

```text
day1/、day2/               初期笔记与环境记录
day3/                       note/ 报告；code/ 算子、探针、日志
day4/                       migration/ 五算子迁移；cube/ GEMM
day5/day5-flashatten/        README、note/、code/、日志
day6/                       README；code/ 源码/IR/产物/日志；note/ 报告与技术笔记
skills/                     六个可复用 skill，见 skills/README.md
```

在已经配置好原 NPU 环境的机器上，Day 6 的复现入口为：

```bash
bash day6/code/repro.sh
```

它会重新编译和上卡，最终应输出 `ALL_CHECKS=PASS`。脚本默认引用原机器的 CANN 和 AscendNPU-IR 构建路径；换机器先核对路径。Day 3、Day 4、Day 5 的命令、依赖及原始日志分别从各自的入口文档进入，避免把不同时期的环境配置混在一起。

## Skills

仓库的 [skills 清单](skills/README.md) 目前有六个：

| Skill | 解决的问题 |
|---|---|
| [ascend-npuir-bench-guardrails](skills/ascend-npuir-bench-guardrails/SKILL.md) | NPUIR kernel 的正确性、边界和性能结论门槛 |
| [ascend-npu-benchmark](skills/ascend-npu-benchmark/SKILL.md) | host/device/e2e 分离与设备 profiler 对拍 |
| [tilelang-npuir](skills/tilelang-npuir/SKILL.md) | 910B 上写 TileLang NPUIR 算子 |
| [ascend-tileops-migration](skills/ascend-tileops-migration/SKILL.md) | 用 Agent 框架迁移 TileOPs 算子 |
| [study-notes-daily](skills/study-notes-daily/SKILL.md) | 每日笔记、结论修正和版本记录 |
| [experiment-report-editor](skills/experiment-report-editor/SKILL.md) | 对照代码与日志写学习报告、整理交付包 |

## 证据与范围

- 每一天的数字和结论以对应 `code/logs/`、报告中的证据索引及运行日期为准。编译、生成二进制、上卡执行、数值验证和性能测量是不同阶段。
- Day 6 的当前运行记录是 `day6/code/logs/step6-*`；`step4-*`、`step5-*` 保留为 10 月 31 日的历史材料。Day 7–10 尚未完成。
- 部分 PDF 是按当时版本导出的快照，阅读最新结论优先看对应 Markdown 和 README。

## 相关项目

- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)（本项目使用 `npuir` 分支）
- [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)
- [cannbot-skills](https://gitcode.com/cann/cannbot-skills)
