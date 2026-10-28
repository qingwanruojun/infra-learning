# Day 3 · 手写 Vector 算子，以及一次测量口径事故（10/28 周三）

### 今天学习总结

1. 手写 `abs` 与 `rmsnorm` 两个 Vector 算子，修正了示例里丢尾块的问题，精度对齐 torch（11/11 通过）。
2. 跑了 tile 尺寸 / 对齐 / buffer 策略三组实验，产出第一版旋钮表。
3. 复核时发现：N=1M 下的性能数字被固定开销主导，同一配置跨 run 漂 10–30%，原有结论互相矛盾。
4. 拆解后得到模型：耗时 ≈ 固定开销 + 块数 × 每块成本 + 数据搬运；块粒度才是主因。给出测量与 kernel 两层改法。

---

## 1. abs 算子

实现照 `examples/elementwise/vec_add_1d.py` 的结构重写，两处关键差异：

```python
num_blocks = T.ceildiv(N, block_N)          # 示例用 N // block_N，非整除时丢尾块
...
offset = cid * block_N
tail = T.min(block_N, shape - offset)       # 运行时 shape 驱动尾块
T.copy(A[offset : offset + tail], a_buf[0:tail])
T.vabs(a_buf, b_buf)
T.copy(b_buf[0:tail], B[offset : offset + tail])
```

- `shape` 作为运行时标量传入，同一个编译产物可以跑非整除形状；`offset` 越界时 `tail` 为负，`T.copy` 空操作，不会写坏内存。
- 两种 buffer：`alloc_ub` + Expert / `alloc_shared` + Developer，用来对比落点（UB vs L1）。

测试用例覆盖：整除、尾块 1 个元素、尾块 4 个元素、shared 路径、fp16 路径。

## 2. rmsnorm 算子

Tiling：一个 block 负责 `block_M` 整行，完整 hidden 维 `N` 放在 UB 内，规约不出块，`T.reduce` 直接得到 `(block_M, 1)`。

UB 预算（M=N=1024, block_M=8）：

| 路径 | buffer 合计 | UB 192 KiB 余量 |
|---|---|---|
| fp32 | ≈ 136 KiB | ~56 KiB（加 block_M 或多缓冲必须做 aliasing） |
| fp16 | ≈ 102 KiB | 充足 |

rstd 三种实现都保留：

1. `vrsqrt`（硬件倒数平方根）——最快；
2. `vsqrt + vrec`——最准；
3. `vrsqrt` + 一步 Newton——折中。

精度实测（`logs/rmsnorm_bench.log`，fp32）：

| 实现 | max_abs |
|---|---|
| sqrt + reciprocal | 1.907e-06 |
| rsqrt + 1 步 Newton | 5.865e-05 |
| 裸 rsqrt | 1.285e-02 |

结论：**裸 rsqrt 在 fp32 下精度不够**（差 4 个数量级），必须 Newton 或 sqrt+reciprocal。fp16 下三种都在 1e-2 量级，与 fp16 本身精度相当。

## 3. 第一版旋钮表（⚠ 已被第 4 节降级）

abs（N=1048576，alloc_ub，`logs/abs_bench_final.log`）：

| 实验 | 配置 | 耗时 | 等效带宽 |
|---|---|---|---|
| tile 尺寸 | block_N=1024 / 2048 / 4096 | 295 / 290 / 262 µs | 28.4 / 28.9 / 32.1 GB/s |
| 非整除 | N=1048575 / 1048577 | 269 / 265 µs | 31.2 / 31.7 GB/s |
| buffer | ub / shared | 264 / 342 µs | 31.8 / 24.5 GB/s |

rmsnorm（M=N=1024, block_M=8，`logs/rmsnorm_bench.log`）：

| dtype | weight | rstd | 耗时 | 等效带宽 | max_abs |
|---|---|---|---|---|---|
| fp32 | False | sqrt_rec | 318.03 µs | 26.4 GB/s | 7.153e-07 |
| fp32 | False | rsqrt | 255.64 µs | 32.8 GB/s | 1.285e-02 |
| fp32 | True | sqrt_rec | 235.18 µs | 35.7 GB/s | 1.907e-06 |
| fp32 | True | rsqrt | 253.40 µs | 33.1 GB/s | 2.525e-02 |
| fp16 | False | sqrt_rec | 273.49 µs | 15.3 GB/s | 1.953e-03 |
| fp16 | False | rsqrt | 267.54 µs | 15.7 GB/s | 9.766e-03 |
| fp16 | True | sqrt_rec | 268.77 µs | 15.6 GB/s | 3.906e-03 |
| fp16 | True | rsqrt | 255.85 µs | 16.4 GB/s | 2.344e-02 |

**注意：表中的耗时全部不可信**，原因见下节；`max_abs` 一列是可信的。

## 4. 曲折：测到的不是带宽

### 4.1 现象

同一份代码、同一台机器，连续三轮跑，同一配置：

| 配置 | run1 | run2 | run3 | final |
|---|---|---|---|---|
| N=1048575 | 248 µs | 248 µs | 321 µs | 269 µs |
| N=1048576 | 239 µs | 257 µs | 253 µs | 265 µs |
| memory=shared | 245 µs | 270 µs | 271 µs | 342 µs |
| memory=ub | 243 µs | 263 µs | 258 µs | 264 µs |

漂移 10–30%，而待比较的差异只有 5–15%——信噪比接近 1。直接后果是结论互相推翻：`shared` 在一轮里快 4 µs，另一轮慢 78 µs；"4096 最优"在下一轮就变成"1024 最优"。

### 4.2 归因

**（a）块粒度是大头。** 在 N=16M 上扫 block_N：

| block_N | 块数 | 耗时（16M） |
|---|---|---|
| 4096 | 4096 | 1120 µs |
| 8192 | 2048 | 800 µs |
| 16384 | 1024 | 274 µs |

耗时随块数近似线性：差 846 µs / 差 3072 块 ≈ **0.27 µs per block**，且该尺寸下截距接近 0。数据量相同，只改块数就有 4 倍差距，说明瓶颈是每块的固定成本，不是搬运带宽。

**（b）小尺寸被常数项吞掉。** 补充扫描 N = 1M / 4M / 16M 做线性拟合【补充分析，未在本轮重跑】：截距 ≈ 178 µs，斜率 ≈ 148 GB/s（block_N=4096 口径）。N=1M 的 234–335 µs 里约 75% 是这个截距，真正随数据量增长的部分只有几十 µs。tile / 对齐 / buffer 的差异全叠在这个常数上。

**（c）计时方式放大了常数。** `benchmark()` 用 NPU Event 包住了 **10 次 Python 调用的循环**，测的是设备时间轴上的总跨度。如果 host 侧每次入队（Python → TileLang wrapper → ACL）比设备执行还慢，设备会在两次 kernel 之间空转，而这段空隙会被 Event 计入。所以那 178 µs 很可能是 host 节奏，而不是硬件 launch 成本。【未验：需要 msprof 的 kernel 级 duration 来归属 host / device】

**（d）和 torch.abs 比不公平。** 8 MiB 的输入大概率整块落在片上缓存；torch 的算子 kernel 高度优化，且 host 开销低一个量级。实测同一份 8 MiB 数据 torch.abs 约 18–30 µs（≈470 GB/s），128 MiB 时 223 µs（≈1.2 TB/s）。这个 8 倍差距不能当成"我们的 kernel 差 8 倍"。

### 4.3 哪些结论还成立

| 结论 | 状态 |
|---|---|
| 两个算子精度对齐 torch，尾块不丢 | ✅ 成立（11/11） |
| 裸 rsqrt 精度不够、需要 Newton | ✅ 成立（精度数据与耗时无关） |
| 16M 下 block_N 越大越快，主因是块数 | ✅ 成立（趋势单调、幅度 4x，超出噪声） |
| 1M 下 4096 最优 / shared 有 30% 惩罚 / 对齐无影响 | ❌ 不成立：全部落在噪声内 |
| rmsnorm 各配置的耗时排序 | ❌ 不成立：同样处于固定开销主导区 |

### 4.4 改法

**测量层（先做，约 20 分钟）**

1. 用 `torch_npu.profiler` / `msprof` 取 **kernel 级 device duration**，不再用 Event 包 Python 循环。
2. 报告口径二选一：≥64 MiB 的绝对耗时；或扫 N 做回归、**报斜率（GB/s）并扣除截距**。
3. 噪声控制：每组 ≥5 次 run 取中位，差异 < 2σ 就写"无差别"，不写"更优"。

**kernel 层**

1. 固定核数 + 核内串行，替代"每 tile 一个 block"：

```python
NCORE = 40  # 实际值通过接口查，示例里的 20 只是示意
tiles = T.ceildiv(N, block_N)
with T.Kernel(NCORE, is_npu=True) as (cid, _):
    a_buf = T.alloc_ub((block_N,), dtype)
    b_buf = T.alloc_ub((block_N,), dtype)
    for i in T.serial(T.ceildiv(tiles, NCORE)):
        t = cid + i * NCORE
        if t * block_N < shape:
            off = t * block_N
            tail = T.min(block_N, shape - off)
            T.copy(A[off:off + tail], a_buf[0:tail])
            T.vabs(a_buf, b_buf)
            T.copy(b_buf[0:tail], B[off:off + tail])
```

2. 放大 block_N 到 8192，配 `T.Pipelined(..., num_stages=2)` 让搬运与计算重叠。UB 预算：8192 × 4 B × 2 buffer × 2 stage = 128 KiB < 192 KiB，可行；16384 双缓冲是 256 KiB，必须先做 aliasing。
3. 现在是 `copy → 计算 → copy` 全串行，单 buffer 下链路利用率上限约 50%，这是结构性的，调参解决不了。

### 4.5 教训

- **先验证测量口径，再调参**。同一配置跨 run 不稳定时，任何参数结论都是噪声。
- 性能表必须回答"这个尺寸下这套方法有没有分辨力"；没有就写没有。
- 对比高度优化库时控制数据规模（出片上缓存），并区分 host 开销与 kernel 时间。
- 正确性结论和性能结论要分开下：walk 通了就是通了，慢不慢是另一件事。

## 5. 下一步

1. msprof 归属那 178 µs（host / device），10 分钟。
2. 按 4.4 改 abs kernel，在 ≥64 MiB 上重测。
3. 旋钮表升到 v2：每格 5 run、附 σ、标注"可分辨 / 不可分辨"。
4. D8 / D9 的性能对比沿用同一口径，避免第二次翻车。

## 附录：文件与复现

```bash
python -m pytest own_ops/abs.py own_ops/rmsnorm.py -s -q   # 精度
python own_ops/abs.py                                     # 第一版旋钮表（已降级）
python own_ops/rmsnorm.py
python experiments/probe_rstd.py                          # 三种 rstd 精度探针
python experiments/probe_copy_cast.py                     # copy 跨 dtype 是否插 cast
```

- 原始输出：`logs/abs_bench_final.log`、`logs/abs_bench_run{1,2,3}.log`、`logs/rmsnorm_bench.log`、`logs/pytest_all.log`
- 中途崩溃的一次：`logs/abs_bench_interleaved_aborted.log`（脚本 KeyError + 设备异常，与 kernel 无关，记录过程用）
