---
name: ascend-npu-benchmark
description: 在 Ascend NPU 上正确测量/对拍算子性能（三时钟分离、op_summary 取设备时长、10 轮 ABBA、尺寸≥256 MiB、IQR/MAD）。当需要证明"我的 kernel 比 torch/CANN 快还是慢"、或者 host 开销/探针数字看起来不对时使用。
---

# Ascend NPU 性能测量

本方法论在 910B3 + CANN 9.0.0 + torch_npu + TileLang npuir 上踩过全部坑，每条都有日志。

## 0. 铁律：三个时钟，分开量、分开报、永不混算

| 时钟 | 怎么量 | 量到的是什么 | 这台机器上的典型值 |
|---|---|---|---|
| host | `perf_counter` 包住 N 次入队、中间不 sync，除以 N | 提交路径 | 我们的 kernel 216–450 µs；`torch.abs` 14–17 µs |
| device | CANN `op_summary_*.csv` 的 `Task Duration(us)` | kernel 真跑多久 | 13–845 µs |
| e2e | `torch.npu.Event` 包住 N 次调用再除以 N | ≈ max(host, device) | 4096 块时 1085 µs（device 只有 158 µs） |

- **`Event` 包 N 次除以 N 量到的是 host，不是 kernel。** 只要 grid 大（≥1024 块）就彻底失真。
- 任何"快 x 倍"必须标明是哪个时钟。三个数不能放进同一张表却不标口径。
- ours 的 host 是 torch 的 15–30 倍：**端到端慢不代表 kernel 慢**，反之亦然。

## 1. 设备时长怎么取

这个 build 里 `profile.key_averages()` / `events()` **不存在**，chrome trace 只有 host 事件。唯一可靠路径：

```python
from torch_npu.profiler import ProfilerActivity, profile

before = set(glob.glob("export_only_prof_dir/*"))
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
    for _ in range(iters):          # iters=20，每次 CSV 一行
        fn()
    torch.npu.synchronize()
prof.export_chrome_trace("/tmp/x.json")   # 必须调，否则 CANN 不解析、不出 CSV
new = sorted(set(glob.glob("export_only_prof_dir/*")) - before)
csvs = glob.glob(os.path.join(new[-1], "PROF_*",
                              "mindstudio_profiler_output", "op_summary_*.csv"))
```

- 每份 CSV = 一次调用；取 20 次的中位数。
- `Block Num` 列恒为 0，**不能**用它核对 grid 大小。
- 导出目录会长到 127 MB / 176 个 CSV；跑完删掉，只把中位数写进 `.log`。
- 先 `assert torch.equal(...)` 再计时。UB 未初始化会让"错的 kernel"跑出漂亮数字。

## 2. 对拍协议（照抄，别创新）

1. **同进程交叉**，顺序 A-B-B-A，10 轮。
2. 显式预热：60 s 负载或 ≥5 次 + `synchronize()`；**丢弃前 2 轮**。
3. 报 **median + IQR + MAD**，不报单次、不报均值。
4. **只信同进程的比值。** 跨进程绝对带宽不可比（冷启动 1.25 TB/s vs 预热后 1.45 TB/s）。
5. **尺寸门槛：≥256 MiB 张量**（流量 ≥512 MiB）。
6. 同一结论至少两个独立进程复现。

### 为什么是 256 MiB

| 尺寸 | 结果 | 说明 |
|---|---|---|
| 4 MiB（N=1M，8 MiB 流量） | 我们 13.0 µs vs torch 3.6 µs → **3.7x 慢** | 完全在 cache 内，比的是访存路径 |
| 128 MiB | 不作数 | 正压 L2 边界。同一个 `torch.abs` 跨进程量到 **167–213 µs（27%）**，因为它能跑 1.45 TB/s > 稳态 HBM 1.28 TB/s |
| 256 MiB | **1.000x**（422.57 vs 422.37，stdev 0.09/0.12） | 纯 HBM |
| 512 MiB | **1.008x**（845.21 vs 838.46，stdev 0.11/0.31） | 纯 HBM，1.27–1.28 TB/s |

128 MiB 上得出的任何比值（0.93x、1.07x……）都会随进程翻盘。**先选尺寸，再谈结论。**

## 3. 硬件基线（910B3）

- 20 cube / 40 vector core，192 KiB UB/core，CANN 9.0.0。
- HBM 稳态 **1.27–1.28 TB/s**（spec 1.6 TB/s）。达到这个数就算打满。
- 核数 40→20 只慢 5–7% ⇒ 40 核早已吃满，别再调并行度。

## 4. 提交前自检

- [ ] 三个时钟是否分列？
- [ ] 张量是否 ≥256 MiB？
- [ ] 是否有 10 轮 + IQR/MAD + 丢前 2 轮？
- [ ] 冷/热、跨进程是否比较过？
- [ ] 每个数字能否指到一份 `.log`？
- [ ] 有没有把"未解释"写成"已解释"？

## 5. 现成探针（本仓库）

`day3/code/experiments/` 下可直接改参数复用（每个探针的 docstring 写了用途/跑法/归档日志/状态）：

- `probe_abba_hbm.py <N>` — 单尺寸 10 轮 ABBA（入口最常用）
- `probe_ladder_bidir.py` — 多尺寸升序+降序，检查尺寸依赖
- `probe_config_sweep.py` — block_N × 核数扫描
- `probe_hostbudget.py` / `probe_launch_taskqueue.py` — host 口径
- `probe_npugraph.py` — 图捕获天花板（本 build 不可用）

结论与日志索引见 [`day3/note/day3学习指标.md`](../../day3/note/day3学习指标.md) §3.2 / §4.4、
`day3/code/logs/profiler_summary.log`；探针清单见 [`day3/code/README.md`](../../day3/code/README.md)。

## 6. 不适用范围

迁移任务（conductor 的 Stage 5 TileOPs report）跑的是默认配置 + 占位 bench，
**不是**本 skill 的对拍：只说明"能集成"，不能拿来写快慢。要对拍另起 `optimize` 场景，
或按本 skill 重测。流程见 [`ascend-tileops-migration`](../ascend-tileops-migration/SKILL.md) §6。
