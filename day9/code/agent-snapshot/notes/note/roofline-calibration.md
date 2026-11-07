# Roofline 分母标定（实测带宽）

日期：2026-11-05 10:55
硬件：Ascend 910B3 / CANN 9.0.0 / torch_npu 2.7.1
脚本：`/tmp/npu-work/bw_calib.py`、`/tmp/npu-work/bw_calib_small.py`
原始输出：`day9/code/raw/bw_calib_20261105.txt`

## 结论

1. **大 shape（≥1 GiB）纯拷贝实测 ≈ 1.20 TB/s**（fp16 1201.6 / bf16 1212.1 / fp32 1208.3 GB/s）。
   理论峰值 1.75 TB/s ⇒ 实测可达率 68.6%。
2. **256 MiB 只有 ≈1.07 TB/s**，比 1 GiB 还低（DRAM 页/核间争抢），不是单调曲线。
3. **64 MiB 反弹到 1.37–1.55 TB/s，32 MiB 的 burst 到 3.3 TB/s** ⇒ 已进入 L2。
4. **单次调用的固定开销 ≈75–86 µs**，与 shape 无关；连续 25 次背靠背拷贝摊薄到 ≈18 µs/次。
   小于 ~32 MiB 的 workload，wall-clock 时间几乎全是发射开销，不能用来算带宽。

## 对分母的影响

| 口径 | 分母 | 60% 目标等价于 | 80% 挑战等价于 |
|---|---|---|---|
| 理论峰值（原口径） | 1.75 TB/s | 1.05 TB/s | 1.40 TB/s |
| 实测 HBM（大 shape） | 1.20 TB/s | **0.72 TB/s** | 0.96 TB/s |
| 实测 per-shape（小 shape，L2 内） | 需 msprof 逐 shape 标定 | 待定 | 待定 |

- 按理论峰值算，80% 要求硬件跑到 1.40 TB/s，比大 shape 实测的 1.20 TB/s 还高 17%，**物理上不可达**（只有 L2 命中情形可能）。
- 按实测 HBM 算，60% 需要 0.72 TB/s、80% 需要 0.96 TB/s —— 都在可达范围内，是有意义的工程目标。

## 建议口径（待用户确认）

- **主口径**：ratio_meas = 吞吐 / 实测可达，其中实测可达按 shape 分档：
  - ≥256 MiB：1.20 TB/s（HBM）
  - <32 MiB：用同 shape 的纯拷贝 kernel 经 msprof 测 kernel 时间，得到 per-shape 上限（避免 wall-clock 的发射开销污染）
- **附表**：保留按 1.75 TB/s 理论峰值的 ratio，用于和交付要求原文对账。
- 换算关系：ratio_theory = ratio_meas × (实测可达 / 1.75 TB/s)。

## 口径溯源（2026-11-05 14:40 补）

- 报告里的 `Bandwidth(TB/s)=1.75` **不是** `msprof_roofline.py` 的 `SOC_CONFIG["gm_bw"]=1.8`。
  实际路径：`benchmark_base.py` 读 msprof 的 `visualize_data.bin`，取 CANN 自己生成的 roofline 条目
  （`title="GM/L2"` + `bw_name` 含 `GM Read + Write`），其中 `bw` 由 CANN 表给出 = **1.75 TB/s**。
  `SOC_CONFIG` 只在 `compute_from_csvs` 这条 CSV fallback 路径生效。1.8 vs 1.75 的疑问至此关闭。
- `total_bytes` = `L2Cache.csv` 的 read_cache_miss_allocate × 128 B，**只计读**。
  对 Amin 写占比 0.4% 无影响；对写=读的算子（RMSNorm）会高估 AI 约 2×。

## 待办

- [x] 口径溯源：1.75 来自 CANN bin 的 roofline，不是 SOC_CONFIG。
- [ ] 用 msprof 对 4/8/16/32 MiB 的纯拷贝取 kernel 时间，补齐小 shape 的 per-shape 上限。
      （补充：单次 wall-clock 被 ~78 µs 发射地板污染，burst 18 µs/次 仍含 host 入队，
      小 shape 的 device 上限只能靠 msprof；Amin 实测 fp16 16 MB 已给出 1.15 TB/s 的实例上界。）
- [ ] Amin 的各 workload（16.8 MB / 4.2 MB / 4.2 MB / 16 KB）分别落到哪一档，bench 后按档取分母。
