# day3/code — 索引与冗余审计

Day3 的全部可运行资产：两个手写算子、24 个测量探针、43 份实测日志 + 61 份设备时长 CSV。
正文结论见 [day3 学习指标](../note/day3学习指标.md)（v1.7 口径）；本文件只回答两个问题：
**每个文件是干什么的、哪些已经被取代**。

所有命令都在 `day3/code/` 目录下执行（探针靠相对路径 `own_ops.abs` 取 `ascend_mode`），
且只在装了 `tilelang-ascend` + `torch_npu` 的 NPU 机器上能跑。

```
own_ops/       abs.py  rmsnorm.py          两个算子 + pytest + Event 口径旋钮表
experiments/   24 个探针                   测量体系与带宽对拍的一次性实验
logs/          43 个 .log                  探针 stdout 重定向的原始输出
               profiler_raw/ 61 个 CSV     op_summary 设备时长原始数据
```

---

## 1. own_ops — 算子本体

| 文件 | 内容 | 入口 |
|---|---|---|
| `abs.py` | `abs_kernel(N, block_N, dtype, memory)`；`T.ceildiv` + 运行时 `tail` 处理尾块；`alloc_ub` / `alloc_shared` 两种 buffer | `pytest abs.py`（5 例）/ `python abs.py`（tile、对齐、block 起始对齐、buffer 四张旋钮表） |
| `rmsnorm.py` | `rmsnorm_kernel(M, N, block_M, dtype, eps, use_weight, memory, rstd_mode)`；`rstd_mode ∈ {rsqrt, sqrt_rec, newton}` | `pytest rmsnorm.py`（7 例）/ `python rmsnorm.py`（dtype × weight × rstd 扫描） |

两者共用一个上下文管理器 `ascend_mode()`（切 Expert/Developer），探针里也复用它。
`main()` 打出来的所有耗时都是 **Event 口径**，只用于看旋钮趋势，绝对值不可引用（见笔记 §3.1）。

典型跑法：

```bash
python -m pytest own_ops/abs.py own_ops/rmsnorm.py -s -q    # 12 例，约 8.5 min，编译占大头
python own_ops/abs.py
python own_ops/rmsnorm.py
```

---

## 2. experiments — 24 个探针

每个探针的模块 docstring 已经写明：要问的问题、怎么跑、归档在哪个日志、结论状态。
**它们全部在 import 时执行、没有 `__main__` 保护**，当时的归档方式就是 `python x.py > logs/y.log 2>&1`。
这是有意保留的：探针是一次性实验，加 argparse / main 只会让人误以为它们是可以长期复用的工具。

### 2.1 host 成本与测量体系（8 个）

| 探针 | 问的问题 | 日志 | 状态 |
|---|---|---|---|
| `probe_hostbudget.py` | 单次调用的 host 成本能不能沿 block 数闭合？ | `hostbudget.log` | **主证据**，笔记 §4.4a 五点表 |
| `probe_host_vs_blockcost.py` | 240 µs 是 enqueue 还是 sync？ | `host_vs_blockcost.log` | **主证据**，§4.2c |
| `probe_hostprofile.py` | 哪个 host 函数随 block 数线性增长？ | `hostprofile.log` | **主证据** |
| `probe_hostcost_4096blocks.py` | 4096 块那 1095 µs 怎么拆？ | `hostcost_4096blocks.log`；早期 N=1M 版是 `hostcost_1M.log` | **主证据**，§4.4 |
| `probe_launch_taskqueue.py` | taskqueue 包装开/关，单次 launch 差多少？ | `launch_taskqueue_on.log` / `_off.log` | **主证据** |
| `probe_launch_multibuffer.py` | 自动多缓冲会不会改变 host 成本？ | `launch_multibuffer.log` | **主证据** |
| `probe_npugraph.py` | 图捕获最多能省掉多少 host 成本？ | `npugraph.log`、`npugraph_capture.log`、`npugraph_capture_noqueue.log`、`npugraph_direct.log` | **主证据（负结果）**：上限是 device 时间，不是 50 µs |
| `probe_profiler_kernel_time.py` | device 时长到底是多少？ | `profiler_kernel_time.log`、`profiler_summary.log`、`profiler_raw/` | **主证据**，§3.2 全部 device 数字的来源 |

### 2.2 设备时间与 HBM 带宽（7 个）

| 探针 | 问的问题 | 日志 | 状态 |
|---|---|---|---|
| `probe_abba_hbm.py` | 256 / 512 MiB 上我们 vs torch，10 轮 ABBA | `abba_hbm_256MiB.log`、`abba_hbm_512MiB.log`、`abba_hbm.log`（128 MiB，已退役） | **主证据**，§3.2 / §4.4 |
| `probe_ladder_bidir.py` | 阶梯上的差距是尺寸效应还是时钟效应？ | `ladder_bidir_v2.log`（v1 作废） | **主证据** |
| `probe_hbm_peak.py` | torch 自己在 128 / 256 / 512 MiB 的带宽阶梯 | `hbm_peak.log` | **部分有效**：256/512 MiB 仍可用，128 MiB 点已退役 |
| `probe_config_sweep.py` | block_N × NCORE 在 device 时间域的扫描 | `config_sweep_128MiB.log` | **主证据**（128 MiB 工作集） |
| `probe_clock_state.py` | 128 MiB 的读数是不是时钟态依赖？ | `clock_state.log` | **主证据**：128 MiB 不能做对拍点的根因 |
| `probe_blockcount_16M.py` | 4096 块大网格 vs 40 块串行 | `blockcount_16M.log` | **主证据**，§4.4b |
| `probe_static_vs_dynamic_tail.py` | 运行时 `tail`（动态切片长度）要不要付带宽代价？ | `static_vs_dynamic_tail.log` | **主证据**：不要 |

### 2.3 已被取代（5 个，保留作过程证据）

| 探针 | 日志 | 为什么保留 / 被谁取代 |
|---|---|---|
| `probe_bandwidth_scaling.py` | `bandwidth_scaling.log` | 第一次尝试把 per-launch 与带宽分开，后被 `host_vs_blockcost` + `hostbudget` 取代 |
| `probe_bw_floor.py` | `bw_floor.log` | 被 `host_vs_blockcost` 取代；它的 ~41 µs 读数也被 `timing_crosscheck` 修正过 |
| `probe_hbm_compare.py` | `hbm_compare.log` | 被 `hbm_interleaved` → `abba_hbm` 取代；其中 torch 166.93 µs 是孤点 |
| `probe_hbm_interleaved.py` | `hbm_interleaved.log` | 2 轮交替 → 10 轮 ABBA；round 0 的 0.93x 不可用 |
| `probe_launch_overhead.py` | `launch_overhead.log` | 被 `host_vs_blockcost` + `probe_profiler_kernel_time` 取代 |

### 2.4 正确性与 IR 旁证（4 个，不进对拍链）

| 探针 | 用途 | 日志 |
|---|---|---|
| `probe_timing_crosscheck.py` | Event 计时与 wall clock 交叉验证 + 正确性复核 | `timing_crosscheck.log` |
| `probe_vector_width.py` | dump NPUIR 里编译器选的向量宽度 | `vector_width_ir.log` |
| `probe_rstd.py` | 一条 kernel 内三种 rstd 算法对比（§1.3） | 无（数字直接进正文） |
| `probe_copy_cast.py` | `T.copy` 跨 dtype 报错的最小复现（§2.3） | 无（编译器报错原文进正文） |

---

## 3. logs — 原始数据

**43 个 `.log`** 按来源分四类：

- **探针直出（主流）**：对应关系见上面三张表；`profiler_summary.log` 是从 `profiler_raw/*.csv`
  汇总的手工表，不是脚本输出。
- **`own_ops` 的 `main()`（Event 口径，耗时列不可引用）**：`abs_bench_run1/2/3.log`、
  `abs_bench_v2.log`、`abs_bench_final.log`（§3.1 的 295/290/262 µs 出自这里）、
  `rmsnorm_bench.log`、`rmsnorm_bench_v2.log`。
- **pytest**：`pytest_all.log`（11 passed, 473.63s）、`pytest_all_v2.log`（12 passed, 502.22s，当前口径）。
- **作废/一次性**：`abs_bench_interleaved_aborted.log`（中途 KeyError 中断的旧版 main）、
  `blockN_1M.log` / `blockN_large.log`（早期手搓 block_N 扫描）、`hostcost_1M.log`（脚本现已固定 N=16M）、
  `ladder_bidir_v1_invalid.log`（v1 口径错误，正文只用 v2）。

**`profiler_raw/` 61 份 `op_summary_*.csv`**：每份 20 次调用的 `Task Duration(us)`，
16:28–16:38 的 21 份 + 16:55 ABBA 十轮的 40 份，逐文件对应关系见 `profiler_raw/README.md`。
CANN 的临时目录每次用完即删，需要原始 trace 时重跑对应探针即可再生成。

---

## 4. 冗余审计结论

24 个探针**没有一个是纯重复**：它们是一条按时间推进的研究链，后面的问题由前面的结果引出。
例如 128 MiB 的 1.070x → 先怀疑尺寸（`hbm_peak`）→ 再怀疑时钟（`clock_state`）→
最后定位到 host 提交（`hostbudget`），每一步都产生了正文里要引用的数字。

因此：

1. **不删文件**。附录 B 和正文按文件引用了这些日志，删掉就断链；而且"当初错在哪"本身是笔记 §4 的内容。
2. **被取代的只在 docstring 里标 `Status: superseded by …`**，并在上面 §2.3 集中列出，写到目录里反而难查。
3. **固定网格 kernel（`ncore=40`、core 内 `T.serial` 循环）在 13 个探针里各内联一份**，是有意重复：
   从 `own_ops` 只借 `ascend_mode()` 一个上下文管理器，kernel 本身不共享，
   这样改一个探针的编译参数不会波及其他探针的读数。
4. **唯一真正的重复是日志**（`rmsnorm_bench.log` 与 `_v2`、`abs_bench` 五份、`ladder_bidir_v1_invalid.log`），
   但它们记录的正是踩坑当时的状态，同样保留。
