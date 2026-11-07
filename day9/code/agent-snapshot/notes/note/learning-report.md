# Day 9 学习报告：TileOPs 算子 NPU 迁移与调优

日期：2026-11-05（UTC+8）
硬件/栈：Ascend 910B3 / CANN 9.0.0 / torch 2.7.1+cpu + torch_npu 2.7.1 / TileLang npuir
状态：**AminFwdOp 迁移 + 调优已完成**；其余四算子未启动（见 §7）

## 1. 结论先行

| 项 | 数值 |
|---|---|
| 迁移后平均 Ratio | **42.73%** |
| 调优后平均 Ratio | **51.97%** |
| 提升 | **+9.24 pp** |
| 要求（纯 vector 类平均） | ≥60%（挑战 80%） |
| 判定 | **未达标，缺口 8.03 pp**；80% 物理不可达 |
| 正确性 | 14/14，max_diff = 0 |
| Stage 4 gate | pass（failures = []） |

四例等权平均，口径见 §2。原始数据：`code/raw/amin_final_benchmark_20261105.json`、
`code/raw/amin_final_report_20261105.md`；调优全过程：`code/logs/amin_opt_log_20261105.md`。

## 2. Ratio 口径（已核到源码）

1. `ManifestBenchmark.profile` 用 msprof 采 kernel，读 `visualize_data.bin` 的 CANN roofline 条目
   （`title="GM/L2"` + `bw_name` 含 `GM Read + Write`）。
2. `Ratio = Perf / min(AI × bw, computility)`，其中 **bw = 1.75 TB/s**（CANN 表值；
   `SOC_CONFIG["gm_bw"]=1.8` 只用于 CSV fallback 路径，不参与本报告）。
3. 代数化简：`Ratio = Perf/(AI×bw) = (F/t)/((F/B)×bw) = B/(t×bw)`，fops 消掉。
   Amin 的计算分支从不 binding（AI≈0.5，计算上限 8.6 TOps/s 比带宽分支大一个量级），
   所以 **Ratio ≡ 实测 GM 读带宽 ÷ 1.75 TB/s**。bf16 按 2× fops 计数不影响比值。
4. `total_bytes` 只统计 L2 读 miss，不含写。Amin 写占 0.4% 可忽略；
   **写=读的算子（RMSNorm）必须先修这个口径**，否则 AI 被高估约 2×、ratio 被系统性压低。
5. 平均口径：该算子 manifest 全部 workload（4 例，排除 smoke）**等权算术平均**，不按字节/耗时加权。

## 3. 逐例结果

| case | 迁移版 | 调优后 | Ratio | Δ |
|---|---:|---:|---:|---:|
| `[2048,4096]` fp16（控制组） | 14.64 µs / 65.49% | 14.06 µs / **68.19%** | 68.19% | +2.70 |
| `[2048,4096]` bf16 | 19.62 µs / 48.86% | 15.02 µs / **63.83%** | 63.83% | +14.97 |
| `[64,32768]` bf16 | 7.50 µs / 31.96% | 7.12 µs / **33.66%** | 33.66% | +1.70 |
| `[4,128,4096]` fp16 dim=[0,2] | 9.74 µs / 24.61% | 5.68 µs / **42.20%** | 42.20% | +17.59 |
| **等权平均** | **42.73%** | **51.97%** | | **+9.24** |

缺口集中度：long-seq 一例贡献 −6.58 pp（33.66 vs 60），3d 贡献 −4.45 pp，两例即全部缺口；
两个 16 MB 的 case 已过 60% 线。

## 4. 有效改动（A/B 双跑确认才计入）

1. **`r2a_bmup`（3d）**：工厂内放大 wrapper 的 effective `block_m`（`_uplift_block_m`），
   把 128 个输出行摊到更多核、减少每核 tiny-store 条数。9.74 → 5.68 µs，**−42%**。
2. **`r4a_bfcast`（bf16）**：bf16 无原生向量 min，走 fp32 中转；收敛中转的 `vcast` 次数与
   数据复用，减少指令数。19.62 → 15.02 µs，**−23%**。
3. long-seq 在 final 合并时被动拿到 7.50 → 7.12（−5%），不是独立优化点。

## 5. 无效/负向结果（同样是结论）

| 方向 | 结果 | 原因 |
|---|---|---|
| reduce-on-shared（省 frag copy） | tie → 组合后回退 | 高 UB 占用下负收益 |
| 均衡分核 64→32 核 × 2 满波 | tie（−0.9%） | HBM 波形模型高估，mte2 读非纯 HBM 串行波 |
| 连续任务映射 + 合并 store（r7a） | **bf16 +18.5% 回退** | 读局部性损失 > store 合并收益 |
| tiled 版连续映射（r7b） | +1.9% | 同上 |
| 任务循环 `T.Pipelined(num_stages=2)` | tie（+1.1%） | auto-multi-buffer 已覆盖 |
| tiled `tn` 下调 / 2-tile 交错 / N-split | 无收益 / 编译器折叠 / 数学否决 | — |
| 单 tile `tn=32768` | UB 溢出 | bf16 vcast fp32 frag 4 B/elem，192 KB 放不下 |

**方法论教训**：单次 profile 反复给出假信号（例如 13.43 µs 实际是 +8.6% 回退、
17.11 µs 实际中性）。只有同进程 A-B-B-A、3 对 × 20 次、取中位数的比值可信；
本算子存在 ±3% 噪声和 ±12% 的运行态双态（同 kernel 8.52 vs 7.59 µs）。

## 6. TileLang 表达能力缺口（对应交付件 1、2）

1. **bf16 原域归约缺失**：`T.reduce` 的 bf16 路径不可用（文档 §2.2.1），只能 fp32 中转，
   每元素指令数翻倍。长序列 bf16 的 33.66% 主要卡在这里。→ 需要 **bf16 原生 reduce/min 原语**。
2. **同 dst 双 vcast 被编译器折叠**：两个分支写同一 UB 目标时直线代码被折叠成一条，
   丢失并行（`r3b` 实证）。→ 需要**显式的分支/谓词向量原语**，或在 IR 层禁止该折叠。
3. **缺按 shape/dtype 选择 tile 策略的分派原语**：最优策略随 dtype 变化
   （fp16 走 rowblock、bf16 走 bfcast）；`r7a` 在 fp16 领先、bf16 回退 18.5%，
   最终只能靠 host 侧 Python 分支拼装。→ 需要**编译期特化/分派原语**。
4. **task 映射的地址聚簇不可声明**：`r7a` 的负结果说明 stride 映射让 40 核同波读相邻行块、
   获得 L2/DDR 局部性，而连续映射把 40 个核打散到 40 个大区间。
   语言里没有表达“波内地址聚簇”的手段。→ 需要**可声明 locality 的 task 映射原语**。
5. **auto-multi-buffer 的 UB 预算不含双缓冲**：`r1b` 因预算公式少算 32 B 直接 ub overflow。
   → 需要在编译期给出**双缓冲口径的 UB 预算查询**。
6. **小 shape 是发射/流水受限，不是带宽受限**：4 MB 两例 GM 读命中率 ≈0（bench 每次刷 cache），
   属于“未命中 L2 + 指令受限”。任何用 wall-clock 算小 shape 带宽的做法都会失真。

> 优化器按 skill 标准判定 plateau 成立（连续 3 轮 tie/回退；结构加速估计 long-seq 1.7×、3d 1.3×，
> 未同时满足 ≥2× 门槛），因此**未触发 [DESIGN_LIMIT]、未产出 perf_feedback.md**；
> 能力缺口改为记录在 `perf_opt/capability-gaps.md`。这一点在最终评估里要如实说明：
> 缺口来自语言/工具链能力，不是没有调。

## 7. 未完成项

- FusedAddRMSNormFwdOp：暂停在 Stage 1（Stage 0 工件 7 件已落盘）；Amin 调优已结束，可续跑。
- MaxPool1dIndices / FFTC2C / GDN：只跑 Stage 0–2。FFTC2C 的 c128（atol 1e-8）受无 fp64 向量路径阻塞。
- 类均值口径：60% 要求是“纯 vector 类平均”，Amin 单算子 51.97% 不能直接代表类均值，
  需在其余算子跑完后合并计算；报告中应分开列“单算子均值”和“类均值”。

## 8. 复现入口

```bash
# 官方报告（正确性 + 性能）
cd refs/tilelang-mlir-ascend/examples/TileOPs
python3 -m pytest tests/ops/test_amin_fwd.py -sv
python3 -m pytest benchmarks/ops/bench_amin_fwd.py -sv -m "not smoke"
# 调优回归
python3 tileops/kernels/reduction/amin_fwd/amin_fwd_kernel/perf_opt/_simple_reduce_kernel.py --level all
```
