# Day 4 交付包：Cube（GEMM）

日期 2026-10-28 · Ascend910B3（20 AIC / 40 AIV，运行时查询）· CANN 9.0.0 ·
tilelang-ascend npuir 分支 · AscendNPU-IR 19.1.7

## 包内容

```
day4/
  note/
    day4学习指标.md      主笔记：22 行 API 速查表（file:line）、三档设计、
                        正确性表、性能表、Q1–Q5、Vector vs Cube、未验项
    learning-report.md   当天学习报告（含三条口径修正）
    day4报告-Cube.md     汇总报告（做了什么 / 结果 / 学到了什么，含证据分级与本次代码改动）
    worklog.md           时间线、错误与修复
  code/
    own_ops/
      gemm.py                    三档 GEMM + 7 个 pytest + ABBA 基准工具
      mixcv_min.py               最小 MixCV（Cube gemm → Vector exp）
      probe_shapes.py            边缘形状探针（fractal / 对齐，Q1/Q2）
      probe_grid.py              2D grid 静默串行化的最小复现
      probe_membound_torch.py    复核 torch“超过 HBM”的 8192²×64（C 扫 L2 边界）
      probe_pair_bias.py         单独测量 vs ABBA 配对测量的偏差（4096³ / 16384²×64）
    logs/
      *.log                    全部原始输出（失败的那次也在，未覆盖）
      *.mlir                   三档 IR / 2D grid 证据 / MixCV IR
      bench_summary.csv        9 行机读性能表
      profiler_op_summary/     109 份 CANN op_summary CSV（device 时间原始来源）
  PACKAGE-README.md          本文件
```

**未包含**：`logs/export_only_prof_dir/` 下的 profiler 原始目录（89 MB，
每份 ~1 MB 的 json/trace）。需要时用 `gemm.bench_main()` 重跑即可再生成；
device 时间的关键列已抽成 `profiler_op_summary/*.csv`。

## 一键复现

```bash
# 正确性：7 个用例（整除 / K 尾 / M,N 尾 / 全尾 / 单行 / 长方形）
python -m pytest day4/code/own_ops/gemm.py -q -s

# 三档 IR
python -c "import sys; sys.path.insert(0,'day4/code'); from own_ops.gemm import dump_ir; dump_ir()"

# 全量基准（10 配置 × 10 轮 ABBA，实测约 7 分钟）
python -c "import sys; sys.path.insert(0,'day4/code'); from own_ops.gemm import bench_main; bench_main(10,10)"

# 探针
python day4/code/own_ops/probe_shapes.py
python day4/code/own_ops/probe_grid.py
python day4/code/own_ops/probe_membound_torch.py
python day4/code/own_ops/probe_pair_bias.py
python day4/code/own_ops/mixcv_min.py
```

## 结论摘要（设备域，10 轮 ABBA，中位数 [IQR]）

| 配置 | ours device µs | torch device µs | 比值 | ours TFLOPs |
|------|---------------:|----------------:|-----:|------------:|
| 4096³（compute-bound，配对=单跑） | 1312.60 [1.40] | 501.54 [0.52] | 2.617 | **104.71** |
| 16384²×64（memory-bound，C=512 MiB） | 1778.34（单独）/ 2114.43 [0.38]（配对） | 401.91（单独）/ 571.77 [1.73]（配对） | 4.425 / 3.698 | **19.32** |
| 8192²×64 ❌作废 | 405.98 [0.24] | 69.58 [0.73]（L2 常驻） | 5.835 | 21.16 |

K sweep（M=N=4096）：20.08 → 24.80 → 47.33 → 68.89 → 86.44 → 99.13 → 105.04 TFLOPs，
K≈1024 后进入平台；小 K 掉点 = 算术强度 K/3 不够，时间由搬运主导。

## 已知缺口

- 大形状基准的正确性检查已从“跳过”改为设备侧参考（`gemm.abba()`），
  该改动尚未上机重跑，`bench_*.log` 里仍是旧的 “skipped”；重跑 `bench_main` 后才是新口径。
- fixpipe 耗时占比未单独量化。
- 未做软件流水（`T.Pipelined`/多 buffer）对照。
- L0C 容量冲突（npu_arch 512K vs CANN 128K）未实测，按 128K 保守设计。
- K≤128 未做小 K 调优，该段比值（7.6–8.6x）不构成 kernel 质量结论。
- 16384²×64 的绝对时间随会话/分配变化 12–30%，只报区间。
