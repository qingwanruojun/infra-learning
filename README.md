# infra-learning

> 昇腾（Ascend）算子自动生成与优化 · 10 天学习笔记

主线链路：

```
Agent（Codex 式循环） → CANNBot Skills → TileLang(npuir) 生成 kernel
    → AscendNPU-IR（HFusion / HIVM） → NPU 执行 → profiler 反馈调优
```

## 目录结构

```
day1/   Agent 原理 / 三个仓库地图 / 全链路梳理
day2/   环境搭建、TileLang 编译、L2 冒烟、DSL 全量速读
day3/   手写 Vector 算子（abs、rmsnorm）
  code/own_ops/        abs.py、rmsnorm.py（含 pytest 用例与 benchmark）
  code/experiments/    运行时探针（copy dtype 转换、rstd 算法对比）
  code/logs/           原始实测日志
```

## 进度

| 天 | 日期 | 主题 | 产出 | 状态 |
|---|---|---|---|---|
| D1 | 10/26 | Codex/Agent 原理、申请机器、读三个仓库 README | `day1/day1学习指标.md` | ✅ |
| D2 | 10/27 | 编译 TileLang、跑通 L2、TileLang DSL 全量速读 | `day2/day2学习指标.md` | ✅ |
| D3 | 10/28 | 手写 abs / rmsnorm，精度对齐 + 参数实验，测量口径复盘 | [`day3/note/day3学习指标-融合版.md`](day3/note/day3学习指标-融合版.md)、`day3/code/` | ✅ |
| D4 | 10/29 | Cube 算子（GEMM） | — | 未开始 |
| D5 | 10/30 | MixCV / FlashAttention、第一周收口 | — | 未开始 |
| D6 | 11/02 | MLIR 与 AscendNPU-IR 入门 | — | 未开始 |
| D7 | 11/03 | 编译链路追踪（TVM IR → TileLangIR → HIVM） | — | 未开始 |
| D8 | 11/04 | CANNBot Skills 与 Agent 生成算子 | — | 未开始 |
| D9 | 11/05 | Profiling 与性能优化 | — | 未开始 |
| D10 | 11/06 | 整合交付（demo + 汇报） | — | 未开始 |

## Day 1 · 全景与环境

- **Codex / Agent 原理**：agent loop（输入 → 模型 → tool call → 执行 → 结果回填）、harness 职责（上下文压缩 / 轮数上限 / 权限 / 沙箱）、三种典型失败模式（反复打转、猜 API、长任务跑偏）。
- **机器与仓库**：CANNLab 申请 A2 环境（910B3，1 卡），克隆并通读 `tilelang-ascend`、`AscendNPU-IR`、`cannbot-skills`。
- **三个仓库的定位**：
  - `AscendNPU-IR`：昇腾的 MLIR 方言集合与编译工具链（CANN 栈的编译器层），分 HFusion（硬件无关融合）与 HIVM（硬件相关降级）。
  - `cannbot-skills`：CANNBot 智能体层的技能仓，Plugin → Agent → Skill 三层，不参与编译。
  - `tilelang-ascend`：昇腾版 TileLang，Python DSL 写、TVM 编译、最终经 AscendNPU-IR 落二进制。
- **链路图**：见 `day1学习笔记.pdf`。

## Day 2 · 工具链落地 + DSL 速读

编译与冒烟：

```bash
git submodule update --init --recursive
bash install_npuir.sh --bishengir-path <AscendNPU-IR>/build
python examples/elementwise/vec_add_1d.py
```

L2 通过以四个信号为准：`AscendNPU IR compile success`、`Precompiling NPU launcher header..`、输出 `device='npu:0'`（防静默 fallback）、`All check passed`。

DSL 速读中的几个关键发现（细节见 `day2/day2学习指标.md`）：

- `TILELANG_ASCEND_MODE` 不设即为 **Expert**；Expert 下 `alloc_shared` 确定落在 L1（`cbuf`、AIC），因此"选内存空间等于选计算单元"，写向量算子必须用 `alloc_ub`。
- `T.Pipelined` 的 `num_stages` 默认 **0（不启用流水）**，但 NPU 路径默认开启 `--enable-auto-multi-buffer`，不写也会有双缓冲。
- `T.Kernel` 实际支持 1D/2D 网格，文档里"必须且仅有 1 个 block 维度"是过期描述。
- 对齐是主路径约束：加载要求 32B 对齐，fp16 尾轴取 16 倍数、fp32 取 8；尾块不是边界情况。
- `vsigmoid` 展开为 `VAdd + VSub + VExp + VDiv` 四条指令，调优应按 IR 数估成本。

## Day 3 · 手写 Vector 算子

`day3/code/own_ops/abs.py` 和 `rmsnorm.py`：按 `vec_add_1d` 的结构重写，并修正了示例里 `N // block_N` 会丢尾块的问题（改为 `T.ceildiv` + 运行时 `shape` 驱动的 tail）。两个算子都支持 `alloc_ub`（Expert）与 `alloc_shared`（Developer）两种 buffer 策略做对比。

测试（`logs/pytest_all.log`）：

```
11 passed in 473.63s
fp32 abs   max_abs=1.907e-06  max_rel=3.503e-07
fp16 abs/输出 max_abs=2.344e-02 max_rel=3.030e-02
newton 法 rstd max_abs=5.865e-05 max_rel=8.586e-06
```

abs 参数实验（N=1048576，`logs/abs_bench_final.log`）：

| 实验 | 配置 | 耗时 | 有效带宽 |
|---|---|---|---|
| tile 尺寸 | block_N=1024 / 2048 / 4096 | 295 / 290 / 262 µs | 28.4 / 28.9 / 32.1 GB/s |
| 非整除形状 | N=1048575 / 1048577 | 269 / 265 µs | 31.2 / 31.7 GB/s（与整除持平，尾块逻辑有效） |
| buffer 策略 | alloc_ub / alloc_shared | 264 / 342 µs | 31.8 / 24.5 GB/s |

rmsnorm（`logs/rmsnorm_bench.log`）：fp32 下裸 `rsqrt` 的 max_abs 是 1.3e-02，比 `sqrt+reciprocal` 的 1.9e-06 差 4 个数量级，超出对齐 torch 的容差；加一步 Newton 后回到 5.9e-05。耗时数据见下面的口径说明。

> **⚠ 测量口径复盘（v1.4）**：v1.3 用 profiler + 同进程交替对拍确认——N=1M 的耗时由 **host 提交**主导（256 块时 host 239 µs / device 13 µs）；kernel 在 HBM 尺度（≥128 MiB）与 torch 打平（0.93–0.94x，双方 1.24–1.32 TB/s），此前“慢 3.7 倍”是小尺寸缓存内现象。绝对耗时与“快 11% / 慢 29%”这类幅度仍不可引用，tile / buffer 结论需在 device 时间域复测。正确性、精度与 DSL 结论不受影响。完整修正对照与重测计划见 [day3 笔记](day3/note/day3学习指标-融合版.md) 第 4.4、6 节与附录 D。

## 环境

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B3，1 卡（CANNLab A2 环境） |
| CANN | 9.0.0（机器预装；仓库文档基线 8.5.0） |
| Python | 3.11 |
| AscendNPU-IR | 19.1.7（commit `31f6903`） |
| TileLang | `tilelang-ascend` `npuir` 分支 |
| torch / torch_npu | 版本需与 `tilelang-ascend/requirements.txt` 对齐 |

## 如何复现 Day 3

在已装好 tilelang-ascend 的 NPU 机器上，于 `day3/code/` 目录：

```bash
python -m pytest own_ops/abs.py -s -q     # 精度测试
python own_ops/abs.py                     # tile 尺寸 / 对齐 / buffer 策略实验
python own_ops/rmsnorm.py                 # dtype × weight × rstd 算法实验
```

## 外部资料

- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)（`npuir` 分支）
- [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)
- [cannbot-skills](https://gitcode.com/cann/cannbot-skills)

## 说明

- `day*/` 下的 PDF 是导出的完整笔记（含截图）；Markdown 中的图片指向本地 Typora 路径，在 GitHub 上不会渲染。
- `day3/code/logs/` 是原始实测输出，未做整理，便于对照正文数字。
