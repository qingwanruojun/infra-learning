### 今天学习总结

1. 完成 abs 与 RMSNorm 两个手写 npuir kernel，11 个测试全部通过（`11 passed in 473.63s`），fp32 RMSNorm 对 torch 参考达到 `max_abs=1.9e-6`。
2. 踩到 4 个真实的 DSL 坑（`N // block_N` 丢尾、条件分支作用域、`vcast` 不是恒等转换、硬件 `vrsqrt` 只有 ~3e-3 精度），并据此填出第一版旋钮表。
3. 复核性能数据时发现：N=1M 下每次调用被固定开销主导，跨 run 漂 10–30%，多组结论互相矛盾。
4. 拆出模型：耗时 ≈ 固定开销 + 块数 × 每块成本 + 数据搬运；并给出测量与 kernel 两层改法。精度结论全部保留，性能幅度结论降级。
5. **v1.4 复核**（profiler + ABBA 10 轮 + 五点 host 预算）确认三件事：① 瓶颈在 host 提交，且是两段式（1–256 块平坦 216–232 µs，到 1024 为 0.04 µs/块，到 4096 为 0.26 µs/块）；② device 时长域中位 torch 185.67 µs vs 我们 200.02 µs = **1.070x（我们慢约 7%）**，1.45 vs 1.34 TB/s；③ v1.3 的"0.93–0.94x 持平"是 round 0 暖机样本。此前"慢 3.7 倍"是小尺寸缓存内现象，不是 kernel 质量差距。详见 4.4。

## 1. 正确性结果

### 1.1 abs

```
test_abs_divisible        N=4096  block_N=4096   1 block       PASS
test_abs_tail_one         N=4097  block_N=4096   2 blocks,尾1  PASS
test_abs_tail_multi_block N=4100  block_N=1024   5 blocks,尾4  PASS
test_abs_shared           N=4097  alloc_shared   Developer     PASS
test_abs_fp16             N=4097  fp16            Expert        PASS
```

关键点：把示例的 `N // block_N` 换成 `T.ceildiv(N, block_N)` 后，N=4097 的第 2 个 block 只处理 1 个元素，`T.copy` 的动态 slice 正确工作；这正是原示例静默漏掉的路径。

### 1.2 RMSNorm

（M=N=1024，block_M=8，eps=1e-5，有 weight）

| dtype | rstd | max_abs | max_rel | 判定 |
|---|---|---|---|---|
| fp32 | `vsqrt + vrec` | 1.907e-6 | 3.492e-7 | 远优于 1e-4 |
| fp32 | `vrsqrt` + 1 步 Newton | 5.865e-5 | 8.586e-6 | 通过 1e-4，但比 `sqrt_rec` 差约 30 倍，多 4 条向量指令 |
| fp16 | `vrsqrt` | 2.344e-2 | 1.389e-2 | 在 `rtol=atol=1e-2` 下通过（fp16 输出量化主导） |
| fp16 | `vsqrt + vrec` | 3.906e-3 | — | 更稳，建议作为 Day 8 基线 |

另通过：无 weight、M=1023（最后一块 7 行）、`alloc_shared` Developer 模式。合计 11 个用例。

### 1.3 rstd 精度探针

`experiments/probe_rstd.py` 对同一批 fp32 行单独测 rstd 的相对误差【探针】：

| 方法 | rstd 相对误差 |
|---|---|
| `T.vrsqrt` | 2.8e-3 |
| `T.vsqrt + T.vrec` | ~1e-7（与 torch 对拍为 0） |
| `T.vrsqrt` + 1 次 Newton | 1.2e-5 |

误差定位方法：逐行算 `y / ref`。行内标准差只有 4e-8、整行是同一个比例，说明误差全在 rstd，而不是 reduce 顺序或 cast。探针测的是 rstd 相对误差，1.2 节的 `max_abs` 是端到端 y 的绝对误差，两者量级关系由 `|x|` 放大，排序一致。

**结论**：fp32 对 1e-4 默认用 `sqrt_rec`（Newton 可作备选，精度差约 30 倍、指令更多）；fp16 对 1e-2 可以用 `vrsqrt`，但基线建议用 `sqrt_rec`。

## 2. 学到的 DSL 事实

### 2.1 tail-safe 的一维分块

```python
num_blocks = T.ceildiv(N, block_N)
...
offset = cid * block_N
tail = T.min(block_N, shape - offset)
T.copy(A[offset : offset + tail], a_buf[0:tail])
T.vabs(a_buf, b_buf)
T.copy(b_buf[0:tail], B[offset : offset + tail])
```

- grid 必须 `ceildiv`；`N // block_N` 在非整除形状下少算尾块，且**不会报错**。
- `tail` 依赖运行时 `shape` 参数，而不是 Python 常量；同一个编译产物能吃不同 N。
- `offset` 越界时 `tail` 为负，`T.copy` 空操作，不会写坏内存。
- `T.vabs` 就是 `npui_abs`，在 Expert 模式直接落 `hivm.hir.vabs`。

### 2.2 条件分支：只在里面放算子，不要放分配

TVMScript 对 Python bool 的 `if` 会在解析期求值（`visit_if` 里 `isinstance(condition, bool)` 分支），但整段 `if` 包在 `var_table.with_frame()` 里。在 `if` 里 `T.alloc_ub` 得到的名字出了 `if` 就 `Undefined variable`。所以：

```python
# 对：分配在外面
x_f32 = alloc(...)
if dtype == accum_dtype:   # 解析期只留一个分支
    T.copy(x_in, x_f32)
else:
    T.vcast(x_in, x_f32, round_mode="rint")
```

### 2.3 `T.copy` / `T.vcast` 的 dtype 语义

- `T.copy` **不做** dtype 转换：`Check failed: npuirop.src->dtype == npuirop.dst->dtype`。
- `T.vcast(f32→f32, round_mode="rint")` 不是恒等转换，而是**取整到整数**；实测 `-1.0321 → -1`，整个 RMSNorm 输出变成整数。
- `T.vcast(f16→f32, "rint")`、`T.vcast(f32→f16, "rint")` 正常。
- 同 dtype 用 `T.copy`，跨 dtype 用 `T.vcast`，并按 Python bool 分支二选一。

### 2.4 硬件 `vrsqrt` 是近似

精度数据见 1.3。fp32 达不到 1e-4，必须 Newton 或 `sqrt+reciprocal`；fp16 在 1e-2 容差内可直接用。

### 2.5 2D tile + 行内 reduce

RMSNorm 选择 `(block_M, N)` 整行进 UB，而不是按 N 切 tile：

```python
T.vmul(x_f32, x_f32, xsq)                          # x^2
T.reduce(xsq, sumsq, dims=[1], reduce_mode="sum")  # (block_M,N)->(block_M,1)
T.vmul(sumsq, inv_n, sumsq)                        # 标量在 src2
T.vadd(sumsq, eps, sumsq)
T.vsqrt(sumsq, rstd_tmp); T.vrec(rstd_tmp, rstd)
T.vmul(x_f32, rstd, xsq)                           # 广播 (bM,N)*(bM,1)
T.vmul(xsq, w_f32, xsq)                            # 广播 (bM,N)*(1,N)
```

- `T.reduce` 的硬约束：src/dst 同 rank；恰好一维不同，且 dst 该维为 1。`(block_M, N) → (block_M, 1)` 合法。
- N 全进 UB 后，reduce 不需要跨 tile 累加；这是 1.5 h 内能写完的关键。
- 标量只能放在二元算子的 `src2`；`inv_n` / `eps` 在 `T.Kernel` 内、`T.Scope` 外定义。
- 广播 `(bM,N) × (bM,1)` 和 `(bM,N) × (1,N)` 都可用；后者用于 weight。

### 2.6 UB 预算

原计划 4 个 buffer ≈128 KiB。实际为了绕开解析器作用域，所有 buffer 无条件分配，fp32 路径：

```
x_in 32K + x_f32 32K + xsq 32K + y_out 32K + w_in 4K + w_f32 4K + tiny
≈ 136 KiB < 192 KiB
```

仍安全，但余量只剩 ~56 KiB。block_M 加到 16 或开多 buffer 会顶到上限；届时把 `x_f32` 别名到 `x_in`，或为 fp32/fp16 各写一个 prim_func。

## 3. 第一版旋钮表（被证明数据有一定问题）

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

**耗时与带宽列不可引用**，原因见下节；`max_abs` 一列可信。

## 4. 测量口径复盘

### 4.1 现象

同一份代码、同一台机器，连续四轮跑，同一配置：

| 配置 | run1 | run2 | run3 | final |
|---|---|---|---|---|
| block_N=1024 | 278 µs | 270 µs | 270 µs | 295 µs |
| block_N=4096（align 段） | 239 µs | 257 µs | 253 µs | 265 µs |
| N=1048575（尾 1 对齐） | 248 µs | 248 µs | 321 µs | 269 µs |
| memory=shared | 245 µs | 270 µs | 271 µs | 342 µs |
| memory=ub | 243 µs | 263 µs | 258 µs | 264 µs |

单配置跨 run 漂移约 10%，个别 27–40% 离群；待比较的差异只有 5–15%，信噪比接近 1。

### 4.2 "单轮区间不重叠"也不等于可复现

第二轮稿据 final run 的区间不重叠（1024 的 min 287.5 > 4096 的 max 264.6）得出"4096 快 11.3%"、"shared 慢 29%"。这组观察本身没算错，但有三点让幅度不能引用：

1. **跨 run 合并后区间几乎相接**：1024 的中位在 270–295 µs（最小 263.5）；4096 在 239–265 µs（最大 267.8）。方向稳定，但效应量只有几个到十几 µs。
2. **配置测量顺序固定**：每轮都是 tile 段 1024→2048→4096、buffer 段 ub→shared。设备暖机 / 降频会让"后测的更快 / 更慢"，与参数效应同向，无法区分。交错顺序的脚本尝试过但崩了（`logs/abs_bench_interleaved_aborted.log`，脚本 KeyError），这项控制尚未完成。
3. **单轮自带噪声**：同一配置在同一轮里测两次可以差 1.3%（261.65 vs 265.02 µs），这是单轮比较的分辨率下限。

所以：**方向性结论保留为"下一轮优先验证的假设"，百分比一律不写进结论**。

### 4.3 归因

**（a）块粒度是大头。** 在 N=16M 上扫 block_N：

| block_N | 块数 | 耗时（16M） |
|---|---|---|
| 4096 | 4096 | 1120 µs |
| 8192 | 2048 | 800 µs |
| 16384 | 1024 | 274 µs |

耗时随块数近似线性：差 846 µs / 差 3072 块 ≈ **0.27 µs per block**，该尺寸下截距接近 0。数据量相同，只改块数就有 4 倍差距，瓶颈是每块固定成本，不是搬运带宽。这一组本身也是升序测量，存在顺序混杂；但 4 倍幅度远超观察到的 10–30% 漂移，判断成立【实测，可信度较高】。profiler 与五点探针进一步把这条成本归到 **host 提交**，不是设备调度成本，见 4.4。

**（b）小尺寸被常数项吞掉。** 补充扫描 N = 1M / 4M / 16M 做线性拟合【补充分析】：截距 ≈ 178 µs、斜率 ≈ 148 GB/s（block_N=4096 口径）。**【已修正】"截距≈0"是两点拟合的假象**：五点（1 / 64 / 256 / 1024 / 4096 块）实测是**两段式**——1→256 块平坦在 216–232 µs，256→1024 只有 0.04 µs/块，1024→4096 才涨到 0.26 µs/块。N=1M（256 块）整段落在平坦区，234–335 µs 几乎全是固定项。

**（c）计时方式放大了常数。** `benchmark()` 用 NPU Event 包住 **10 次 Python 调用的循环**，测的是设备时间轴的总跨度。如果 host 侧每次入队（Python → TileLang wrapper → ACL）比设备执行还慢，设备会在两次 kernel 之间空转，这段空隙会被计入。那 178 µs 是 host 节奏，而非硬件 launch 成本。**【已验证，见 4.4】256 块时 host 239.2 µs / device 12.99 µs；4096 块时 host 1091.8 µs / device 156–159 µs。**

**（d）和 torch.abs 比不公平。** 8 MiB 输入整块落在片上缓存；torch 的 kernel 高度优化，host 开销低一个量级。此前 8 MiB 上约 18–30 µs、我们 240 µs 的差距是小尺寸局部现象。**【已修正】128 MiB 上 ABBA 10 轮、设备时长中位：我们比 torch 慢约 7%（1.070x，详见 4.4）。"慢 3.7 倍 / 8 倍"不成立，但"打平"同样不成立。**

### 4.4 v1.4 复核：host 是主瓶颈，device 时长比 torch 慢 7%（10/28 晚）

> 数据来源：远端机器的补充实验（profiler + ABBA 10 轮 + 五点 host 探针）。原始 CSV / profiler 日志尚未同步到本仓库，以下为汇总值。所有表格标注口径：`Event` 是 NPU Event 包多次调用的跨度，`device` 是 profiler 的 kernel 级时长。

**（a）HBM 尺度 ABBA 对拍（device 时长）。** N=32M（128 MiB 张量），40 块 + 核内串行，block_N=8192，同进程 A-B-B-A 交替 10 轮：

| 指标 | torch.abs | 我们 |
|---|---|---|
| 中位（device） | 185.67 µs | 200.02 µs |
| 等效带宽（读+写 256 MiB） | 1.45 TB/s | 1.34 TB/s |
| 比值 | — | **1.070x（我们慢约 7%）** |

9/10 轮落在 1.05–1.09，只有一个交点。v1.3 的 0.93–0.94x 正是取了 **round 0 的冷启动样本**：torch 从 218.21 µs 掉到 184–186，我们从 204 掉到 197，只有第一轮相交。所以 **"持平或略优"不能再写**，对外表述是"device 时长慢约 7%"。

注意口径归属：这个 1.070x 是 **device 时长**；端到端在 4096 块时仍是 host 主导（1084.6 µs，见下表），两者要分开报。剩下的 7% 属于访问模式 / 每核占用，不是提交成本，可在 NPUGraph 之后做一次 HBM 尺度参数扫描。

**（b）带宽基线（device duration）。**

| 张量 | 耗时 | 等效带宽 |
|---|---|---|
| 8 MiB | — | 4.7 TB/s（片上缓存） |
| 64 MiB | — | 5.8 TB/s（片上缓存） |
| 128 MiB | 214.4 µs | 1.25 TB/s |
| 256 MiB | 421.9 µs | 1.27 TB/s |
| 512 MiB | 841.8 µs | 1.28 TB/s |

128 / 256 / 512 MiB 严格线性，确认从 128 MiB 起已是 HBM 主导；8 / 64 MiB 的高带宽说明 N=1M 的结果不代表 kernel 质量。

> **口径内不一致（未解决）**：这张表里 128 MiB 是 214.4 µs（1.25 TB/s），而 (a) 的 ABBA 复测是 200.02 µs（1.34 TB/s），同为 device 时长却差 7%。两者是不同批次、可能不同配置；引用时以 ABBA 中位为准，重测时把两组的 block_N / NCORE 对齐后再合并。

**（c）host 预算闭合（五点探针，`logs/hostbudget.log`，单位 µs）。**

| 块数 | wall | cProfile 总 | launcher.launch | load_kernel_binary | grid eval |
|---|---|---|---|---|---|
| 1 | 231.9 | 252.4 | 113.3 | 53.2 | 36.6 |
| 64 | 219.0 | 254.6 | 107.7 | 59.0 | 36.7 |
| 256 | 215.9 | 248.6 | 105.2 | 54.2 | 37.1 |
| 1024 | 252.4 | 286.4 | 135.2 | 59.4 | 37.1 |
| 4096 | 1084.6 | 1122.8 | 941.9 | 73.7 | 47.3 |

两段式 host 定律：1→256 块平坦在 216–232 µs（常数项 = launcher ~105 µs + `load_kernel_binary` ~55 µs + grid eval ~37 µs 等），256→1024 只有 0.04 µs/块，1024→4096 才涨到 0.26 µs/块。v1.3 的 `max(≈240, 0.27×块数)` 在 1024 块处高估（预测 276，实测 252），缺的正是 `launcher.launch` 那个常数。

**launcher 随块数增长这条线查过了，是死路**：`tilelang/jit/jit_npu.py:575-665` 只发一次 `rtKernelLaunch(func, blockNum, &args, ...)`；`rtMalloc(workspace_size × blockNum)` 和 `syncBlockLock` 都有 `>0` 守卫，这两个 kernel 都不触发。增量在 ACL 内部，工具链层没有可改的点，直接用 NPUGraph 绕开整条提交路径。

cProfile 总比 wall 高 20–38 µs（walker 自身开销）：**总耗引用 wall，cProfile 只看各项比例**。

固定 NCORE + 核内串行使端到端 1095→299 µs（3.7x），但 device 只快 5.8x（158→27 µs）——收益主要来自少提交。

**（d）被推翻的说法。**

- torch 166.93 µs（1608 GB/s ≈ 理论峰值）是孤点，另三次 214–217 µs；"我们慢 1.27 倍"不成立。
- "4096 块设备约 800 µs、比 40 块快 30 倍"不成立：device 158 vs 27 µs，只有 5.8x；那 1095 µs 里约 1092 µs 是 host。
- "截距≈0"是两点拟合的假象（见 4.3b）。
- **v1.3 的"0.93–0.94x 持平或略优"不成立**：那是 round 0 暖机样本，10 轮 ABBA 后是 1.070x（见 4.4a）。
- `T.Pipelined` / "单 buffer 只有半链路带宽"对 abs 不适用：计算量≈0，read+write 共享同一条链路，HBM 尺度已经喂满（见 4.5 修正）。

**（e）v1.3 的三个【待核】已全部关闭。**

1. **计时语义**：所有表都带口径标签（`Event` / `device`）；203 µs 是设备时长，231.9 µs 是 1 块的墙钟，两者不矛盾，但不可混引。
2. **host 预算**：五点探针闭合，缺项就是 `launcher.launch` 的常数（~105 µs）与两段斜率。
3. **样本量**：ABBA ×10 轮，9/10 落在 1.05–1.09。

剩余未解决项只有两个，都不影响上面的结论：cProfile 与 wall 的 20–38 µs 系统性差（用 wall 做总耗即可）；N=1048575 那个偶发慢点尚未在 profiler 域复测。

### 4.5 改法

**测量层**

1. 用 `torch_npu.profiler` / `msprof` 取 **kernel 级 device duration**，不再用 Event 包 Python 循环。
2. 配置顺序随机化或交错（修好 `benchmark` 的 KeyError），避免暖机与参数效应同向。
3. 报告口径：≥64 MiB 的绝对耗时；或扫 N 做回归、**报斜率（GB/s）并扣除截距**。
4. 每组 ≥5 次 run 取中位，差异 < 2σ 写"无差别"。
5. 每个配置同时记录 **host 提交 / device duration / end-to-end** 三列，避免把提交成本算进 kernel。

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

2. 放大 block_N 到 8192，与 40 块 + 核内串行组合；v1.3 已验证这条路径在 128 MiB 上跑到 1.32 TB/s。**【修正】不要把 `T.Pipelined` 当作 abs 的主优化**：计算量≈0，流水不可能超过链路带宽，双缓冲只在低占用时改善延迟隐藏。
3. **NPUGraph** 是当前收益最大的一步：4096 块时 host 是设备的 7 倍，graph replay 可吃掉 `load_kernel_binary` 常数和大部分提交成本。注意静态 shape，运行时 `tail` 版本需单独固化。
4. ~~查 `launcher.launch` 为何随块数增长~~ —— 已查，死路（见 4.4c）：单一 `rtKernelLaunch`，`rtMalloc` / `syncBlockLock` 都有 `>0` 守卫，增量在 ACL 内部。改为用 NPUGraph 绕开整条提交路径。

### 4.6 学到的经验

- **先验证测量口径，再调参**。同一配置跨 run 不稳定时，参数结论都是噪声。
- **单轮区间不重叠 ≠ 跨 run 可复现**：固定顺序会让暖机效应伪装成参数效应。
- 性能表必须回答"这个尺寸下这套方法有没有分辨力"；没有就写没有。
- 对比高度优化库时控制数据规模（出片上缓存），并区分 host 开销与 kernel 时间。
- 正确性结论和性能结论分开下：跑通了就是跑通了，慢不慢是另一件事。
- **host 与 device 必须分开计时**：否则提交成本会被当成 kernel 性能，结论会差一个数量级。
- **拟合至少三点**：两点拟合会造出"过原点"的假象；有平台期时用 `max` / 分段模型。
- **对拍要在带宽主导区**，多次交替取中位；单次孤点不能下结论。

## 5. 与计划的偏差

| 计划 | 实际 | 原因 |
|---|---|---|
| rstd 用 `T.vrsqrt` | fp32 默认 `vsqrt+vrec`，fp16 基线也改用 `vsqrt+vrec` | 硬件 vrsqrt 只有 ~3e-3，达不到 fp32 1e-4 |
| 4 个 buffer ≈128 KiB | 无条件分配 6 个，≈136 KiB | 解析器不允许在 `if` 里分配后被外部使用 |
| 对齐 vs 非对齐预期"悬崖" | 1M 下三点差异 <2%，未复现稳定悬崖；一次 27% 为双峰噪声 | 尾块占比小；需要更细的 N 扫描和 PMU |
| 计划 4.5 h | 实际约 60 min（含 11 个测试的全量回归 7 min 53 s） | 示例与测试组织可复用；主要时间花在编译（每次 ~45–50 s）和排错 |

## 6. 下一步

1. **统一三列口径**：每个配置记录 host / device / end-to-end，表头标注口径；每组 ≥10 轮 ABBA 取中位。
2. **NPUGraph**（当前收益最大）：静态 shape 版先跑，能吃掉整条提交路径（含 `load_kernel_binary`）；分别量 device 是否不变、端到端降多少。
3. **在 device 时间域重做旋钮表 v2**：乱序、≥5 run、附 σ；再下 tile / buffer 结论。
4. **追那 7%**：HBM 尺度上扫 block_N / NCORE / 核内串行 vs 每 tile 一块，看是否能追平 torch 的 1.45 TB/s。
5. ~~查 `launcher.launch` 的 O(块数) 成本~~：已排除，见 4.4c。在此之前保持 40 块 + 核内串行作为默认调度。
6. **block_M 4/8 扫描**（带宽主导区）；block_M=16 先做 aliasing；block_N=16384 同理编不过。
7. **对齐细扫**：N = 1048576 ± {1,2,4,8,16,32}，每点 20+ 轮，确认尾块长度 / 32 B 对齐与慢路径的关系；顺便把 N=1048575 的偶发慢点在 profiler 域复测。
8. **把 reduce 推出 UB**：N > UB 时按 N 切 tile、跨 tile 累加（参考 `examples/norm/rms_norm.py` 的 split-K）。
9. **补测试**：fp16 + `sqrt_rec` 精度锁定；M/N 都不整除的 2D 尾块组合。
10. **减少无条件分配**：fp32/fp16 各写一个 prim_func（或用 `T.macro` 共享 body），把 fp32 的 UB 占用从 136 KiB 降回 ~100 KiB。
11. **D8 / D9 沿用 v2 协议**：以 `sqrt_rec` 版本为手写基线，与 agent 生成结果对比正确性、IR 和端到端耗时。

## 附录 A：复现

```bash
cd day3/code
python -m pytest own_ops/abs.py own_ops/rmsnorm.py -s -q   # 精度，约 8 分钟（编译占大头）
python own_ops/abs.py                                     # 第一版旋钮表（结论已降级）
python own_ops/rmsnorm.py
python experiments/probe_rstd.py                          # 三种 rstd 精度探针
python experiments/probe_copy_cast.py                     # copy 跨 dtype 是否插 cast
```

## 附录 B：文件

- 原始输出：`logs/abs_bench_final.log`、`logs/abs_bench_run{1,2,3}.log`、`logs/rmsnorm_bench.log`、`logs/pytest_all.log`
- 失败记录：`logs/abs_bench_interleaved_aborted.log`（脚本 KeyError + 设备异常，与 kernel 无关）
- v1.4 补充数据（远端机器，尚未同步到本仓库）：`logs/abba_hbm.log`、`logs/hostbudget.log`、40 份 ABBA CSV + 21 份 profiler CSV（共 61 份）；13 MB 的 `export_only_prof_dir/` 已删

## 附录 C：版本来源与冲突取舍

| 条目 | 原"测量复盘"稿 | 原"实现细节"稿 | 融合处理 | 依据 |
|---|---|---|---|---|
| 性能结论总体 | 全部降级 | 引用 11.3% / 29% | 方向保留、幅度作废 | 4.1–4.2 的跨 run 数据与顺序混杂 |
| tile 4096 优势 | 判为噪声 | 区间不重叠，判为成立 | 方向成立，幅度不可引用 | 跨 run 区间几乎相接 |
| ub vs shared | 判为噪声 | 慢 29% | 方向成立，IR 差异确定 | 4/4 方向一致，幅度 1–29% 不稳 |
| 对齐 | 判为噪声 | 无稳定悬崖 | 无稳定惩罚 | 3/4 轮 <5%，1 次双峰离群 |
| rmsnorm 耗时 | 不可信 | 噪声淹没 | 只作精度旋钮 | 双方一致 |
| rstd 精度 | 端到端 max_abs | 探针相对误差 | 两套指标并列 | 指标不同，排序一致 |
| UB 预算 / DSL 坑 / 计划偏差 | 无 | 有实测 | 全部采纳 | 与耗时无关 |
| 下一步 | 测量 + kernel | 扫描 + 基线 | 合并并按依赖排序 | 见第 6 节 |

## 附录 D：修正对照

### D.1 v1.4 → v1.5（ABBA ×10 + 五点 host 探针之后）

| 条目 | v1.4 结论 | v1.5 实测 | 处理 |
|---|---|---|---|
| 与 torch 的差距 | 0.93–0.94x，持平或略优 | ABBA ×10 中位 185.67 vs 200.02 µs = **1.070x** | **推翻**：我们慢约 7%；0.93x 是 round 0 暖机样本 |
| 0.93x 的来源 | 两轮交替，差 <0.7% | 只有 round 0 相交，9/10 轮在 1.05–1.09 | **归因修正** |
| host 定律 | `max(≈240 µs, 0.27 µs×块数)` | 两段：1–256 平坦 216–232；→1024 0.04/块；→4096 0.26/块 | **修正**；原式在 1024 处高估（276 vs 252） |
| host 预算 | 未闭合（常数凑不出实测） | 五点探针闭合，缺项是 launcher 常数 ~105 µs | **关闭** |
| 计时语义 | 待核（device / end-to-end 混用） | 全部表格加口径标签，203 µs 是 device、231.9 µs 是墙钟 | **关闭** |
| launcher O(块数) | 待查，疑为 Python 按块循环 | `jit_npu.py:575-665` 单一 `rtKernelLaunch`，增量在 ACL | **死路，撤掉该任务** |
| 样本量 | 2 轮 | 10 轮 ABBA | **关闭** |

### D.2 v1.3 → v1.4（首次 profiler 复核）

| 条目 | 融合版原结论 | v1.3 实测 | 处理 |
|---|---|---|---|
| 与 torch 的差距 | N=1M 慢 3.7 倍 / 8 倍 | 128 MiB 交替对拍 0.93–0.94x | **推翻**"慢 8 倍"，但随后 v1.5 又推翻了"持平" |
| 绝对带宽 | 32 GB/s，不可引用 | ≥128 MiB 为 1.25–1.32 TB/s | **限定尺寸后恢复**；N=1M 仍不可引用 |
| 0.27 µs/块的归属 | 设备调度成本 | host 提交斜率 | **归因修正** |
| 截距≈0 | 成立 | 两点拟合假象 | **推翻**；v1.5 进一步改为两段式 |
| 4096 块设备耗时 | 约 800 µs，比 40 块快 30 倍（估计） | device 158 vs 27 µs，5.8x | **修正幅度** |
| `T.Pipelined` / 半带宽 | 建议改用流水 + 双缓冲 | abs 计算量≈0，流水无法超过链路 | **撤回** |
| "快 11.3% / 慢 29%" | 幅度作废 | host 主导，未在 device 域复测 | **维持作废** |
| host 假设 | 未验 | 256 块 host 239.2 vs device 12.99 µs | **验证成立，且比预想更极端** |

