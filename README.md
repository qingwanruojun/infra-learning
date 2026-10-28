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
  note/               学习指标（当前版 day3学习指标.md，v1.7 口径；旧版本见 git 历史）
  code/README.md      代码索引与冗余审计（24 个探针逐一标注用途 / 状态 / 归档日志）
  code/own_ops/        abs.py、rmsnorm.py（含 pytest 用例与 benchmark）
  code/experiments/    24 个运行时探针（host 预算、ABBA 对拍、device 阶梯、rstd 精度…）
  code/logs/           43 份实测日志 + 61 份 op_summary CSV
skills/ 4 个 Claude Code skill：算子 guardrails / 对拍协议 / TileLang 套路 / 每日笔记
```

## 进度

| 天 | 日期 | 主题 | 产出 | 状态 |
|---|---|---|---|---|
| D1 | 10/26 | Codex/Agent 原理、申请机器、读三个仓库 README | `day1/day1学习指标.md` | ✅ |
| D2 | 10/27 | 编译 TileLang、跑通 L2、TileLang DSL 全量速读 | `day2/day2学习指标.md` | ✅ |
| D3 | 10/28 | 手写 abs / rmsnorm，精度对齐 + 参数实验，测量口径复盘 | [`day3/note/day3学习指标.md`](day3/note/day3学习指标.md)、`day3/code/`、`skills/` | ✅ |
| D4 | 10/29 | **TileOPs 5 算子迁移**（Agent 框架 `migration-harness`）：BitwiseNot / Erf / Expm1 / Elu / LeakyRelu → 910B；CUBE 顺带 | `day4/`（待建） | 未开始 |
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

测试（`logs/pytest_all_v2.log`，12 个用例含 fp16 + `sqrt_rec`）：

```
12 passed in 502.22s
fp32 rmsnorm sqrt_rec        max_abs=1.907e-06  max_rel=3.455e-07
fp32 rmsnorm rsqrt+Newton    max_abs=9.441e-05  max_rel=1.249e-05   （M=1024，余量 6%）
fp16 rmsnorm rsqrt           max_abs=2.344e-02  max_rel=3.448e-02
```

更早的 `11 passed in 473.63s`（`logs/pytest_all.log`）是加 `fp16+sqrt_rec` 之前的快照；
里面的 Newton `5.865e-05` 是 **M=64** 的形状，与上表 M=1024 不可比。

abs 参数实验（N=1048576，`logs/abs_bench_final.log`）：

| 实验 | 配置 | 耗时 | 有效带宽 |
|---|---|---|---|
| tile 尺寸 | block_N=1024 / 2048 / 4096 | 295 / 290 / 262 µs | 28.4 / 28.9 / 32.1 GB/s |
| 非整除形状 | N=1048575 / 1048577 | 269 / 265 µs | 31.2 / 31.7 GB/s（与整除持平，尾块逻辑有效） |
| buffer 策略 | alloc_ub / alloc_shared | 264 / 342 µs | 31.8 / 24.5 GB/s |

rmsnorm（`logs/rmsnorm_bench.log`）：fp32 下裸 `rsqrt` 的 max_abs 是 1.3e-02，比 `sqrt+reciprocal` 的 1.9e-06 差 4 个数量级，超出对齐 torch 的容差；加一步 Newton 后（M=1024）回到 9.4e-05。耗时数据见下面的口径说明。

> **⚠ 测量口径复盘（v1.7）**：用 profiler + ABBA ×10 轮 + 五点 host 探针关闭了早期的三个待核点。
> ① N=1M 的耗时由 **host 提交**主导，且是两段式（1–256 块平坦 216–232 µs，到 1024 为 0.04 µs/块，到 4096 为 0.26 µs/块）。
> ② **≥256 MiB 张量与 torch 打平**：256 MiB 1.000x（422.57 vs 422.37 µs）、512 MiB 1.008x（845.21 vs 838.46 µs），双方 1.27–1.28 TB/s。
> ③ **128 MiB 的 1.070x 不可外推**：该尺寸 torch 自己跨进程就漂 167–214 µs（27%），比值会落到 0.96–1.09x，因此不作对拍点；4 / 64 MiB 的 3.66x / 1.21x 来自块调度与缓存，不是 HBM kernel 质量。
> ④ “快 11% / 慢 29%”这类幅度仍不可引用，tile / buffer 结论需在 device 时间域复测。正确性、精度与 DSL 结论不受影响。
> 完整版见 [day3 学习指标](day3/note/day3学习指标.md)，结论沿革见其附录 C。

## Skills（可被 Claude Code 直接加载）

`skills/` 下的四个 skill 是把 Day3 的返工固化成检查表，不是总结文档。复制到
`~/.claude/skills/`（全局）或项目的 `.claude/skills/`（推荐，随仓库走）即可使用，
`SKILL.md` 的 `description` 决定触发时机。

| Skill | 何时触发 | 核心内容 |
|---|---|---|
| `ascend-npuir-bench-guardrails` | 写/改 npuir kernel、要给"快/慢多少"的结论 | 10 条铁律、形状矩阵、三栏计时、ABBA、证据分级 E0–E4；附 7 个 A 类 + 10 个 B 类失败案例 |
| `ascend-npu-benchmark` | 要证明"比 torch 快还是慢" | host / device / e2e 三时钟分离、`op_summary`、≥256 MiB、IQR/MAD |
| `tilelang-npuir` | 在 910B 上新增/修改 TileLang 算子 | UB 预算与多缓冲、Expert/Developer、动态 shape 尾部、launcher 陷阱 |
| `study-notes-daily` | 一天收尾、写汇报、结论被推翻要同步多处 | logs/worklog/report 三层结构、笔记版本化与修正对照表 |

有冲突时以 `ascend-npuir-bench-guardrails` 为准（它是 Day3 复盘的最终版）。
详见 [`skills/README.md`](skills/README.md)。

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
python -m pytest own_ops/abs.py own_ops/rmsnorm.py -s -q   # 12 个用例，约 8.5 min（编译占大头）
python own_ops/abs.py                     # 第一版旋钮表（Event 口径，耗时列不可引用）
python own_ops/rmsnorm.py                 # dtype × weight × rstd；只有 max_abs 可引用

# device 域复核（笔记 §3.2 / §4.4 的数据来源）
python experiments/probe_hostbudget.py            # 五点 host 预算闭合
python experiments/probe_profiler_kernel_time.py  # device 尺寸阶梯
python experiments/probe_blockcount_16M.py        # 16M：4096 块 vs 40 块串行
python experiments/probe_abba_hbm.py 67108864     # 256 MiB，ABBA×10
python experiments/probe_abba_hbm.py 134217728    # 512 MiB，ABBA×10
```

`probe_abba_hbm.py` 的 `BLOCK, NCORE, ROUNDS` 写死在文件头，改配置要改源码重编译；
`NCORE=40` 是 910B3 的实测值，换卡先查接口。

## 外部资料

- [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend)（`npuir` 分支）
- [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)
- [cannbot-skills](https://gitcode.com/cann/cannbot-skills)

## 说明

- `day*/` 下的 PDF 是导出的完整笔记（含截图）；Markdown 中的图片指向本地 Typora 路径，在 GitHub 上不会渲染。
- `day3/code/logs/` 是原始实测输出，未做整理，便于对照正文数字。
