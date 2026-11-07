# 实时进度（2026-11-05）

最后更新：19:10

> **Amin Stage 4 完成**（18:56 DONE，19:04 gate pass）。
> 官方报告 `reports/tileops/20261105_190051_728147_AminFwdOp`：
> Ratio 68.19 / 63.83 / 33.66 / 42.20，**均值 51.97%**（迁移版 42.73%，+9.24 pp）；
> 正确性 14/14，max_diff=0；wrapper 已切 perf_opt，baseline 未改。
> **60% 未达成（差 8.03 pp），80% 不可达。** 明细见 `note/amin-tuning-estimate.md` §4c、
> `note/learning-report.md`（含逐例数、有效/无效改动、6 条能力缺口）。
> 原始件已归档：`code/raw/amin_final_benchmark_20261105.json`、`code/logs/amin_opt_log_20261105.md`。

> **执行模式（10:45 起）**：**只跑 AminFwdOp**（迁移 → 调优）。其余算子全部暂停/不启动，Stage 3 准入条件 = Amin 调优结束。
>
> **迁移已完成（14:30，phase=DONE，14/14 精度）**。调优前估算已落盘：
> `note/amin-tuning-estimate.md`（三项可达时间基线，均值按实测分母 58.9%）。

## 队列

| # | 算子 | 阶段 | 开始 | 最近变化 | 备注 |
|---|---|---|---|---|---|
| 0 | FusedAddRMSNormFwdOp | **暂停在 Stage 1**（10:44 停止） | 09:52 | Stage 0 done；DESIGN 未落盘 | 续跑：`migrate-op FusedAddRMSNormFwdOp`（状态 DESIGN/1） |
| 1 | AminFwdOp | Stage 0（**从头迁移**，scaffolder） | 10:34 | 唯一在跑的 conductor | 默认 `[0,(1→2→3)×N,5]`；旧件备份 `/tmp/amin-fromscratch-1105/` |
| 2 | MaxPool1dIndicesFwdOp | 排队（只跑 0–2） | — | — | 先验：改映射顺序有效（−5.9%），双缓冲无效 |
| 3 | FFTC2COp | 排队（只跑 0–2） | 10:40 启动后 10:42 取消 | 未产生工件 | c128 受 fp64 缺口阻塞 |
| 4 | GatedDeltaNetPrefillFwdOp | 排队（只跑 0–2） | 10:40 启动后 10:42 取消 | 未产生工件 | 16 workloads 需裁剪 |

## Stage 0 交付件（FusedAddRMSNorm，7 文件）

- [x] `tileops/kernels/norm/fused_add_rms_norm/_fused_add_r_m_s_norm_fwd_kernels.py`（GPU 提取件）
- [x] `tileops/manifest/norm.yaml`（FusedAddRMSNormFwdOp，7 workloads）
- [x] `tileops/workloads/norm.py`
- [x] `tileops/kernels/norm/fused_add_rms_norm/fused_add_rms_norm.py`（wrapper）
- [x] `tileops/ops/norm/fused_add_rms_norm.py`（Op）
- [x] `tests/ops/test_fused_add_rms_norm.py`
- [x] `benchmarks/ops/bench_norm.py`
- [x] `.migration_meta.json`（`extracted_functions`）

## 监测

- `bthcvrkt0`：FusedAddRMSNorm 阶段信号 + 进程退出（10:22 重挂，30 min）
- `b4bz2q6ae`：FusedAddRMSNorm 工件落盘（15 s）
- `bw3cpqxzz`：Amin 阶段信号 + 进程退出

## 已知风险

1. 7 个 workload 中 3 个是 `M=1` decode（4096/8192/16384），会显著拉低平均 ratio；需按 prefill/decode 分组报。
2. bf16 只能 fp32 transit；bf16 格子按 f32 口径解读（G1）。
3. FFTC2C 的 complex128（atol 1e-8）在无 fp64 向量路径下不可达（G2）。
4. 两个 conductor 并行 + 单卡：Stage 5 bench（msprof）可能互相污染，需单跑复核。
