# Day 3 · 手写 Vector 算子与测量口径复盘（10/28 周三）

> 融合版：合并"实现 / 正确性细节"稿与"测量方法学复盘"稿。文末附版本来源与冲突取舍说明。

### 今天学习总结

1. 完成 abs 与 RMSNorm 两个手写 npuir kernel，11 个测试全部通过（`11 passed in 473.63s`），fp32 RMSNorm 对 torch 参考达到 `max_abs=1.9e-6`。
2. 踩到 4 个真实的 DSL 坑（`N // block_N` 丢尾、条件分支作用域、`vcast` 不是恒等转换、硬件 `vrsqrt` 只有 ~3e-3 精度），并据此填出第一版旋钮表。
3. 复核性能数据时发现：N=1M 下每次调用被固定开销主导，跨 run 漂 10–30%，多组结论互相矛盾。
4. 拆出模型：耗时 ≈ 固定开销 + 块数 × 每块成本 + 数据搬运；并给出测量与 kernel 两层改法。精度结论全部保留，性能幅度结论降级。

### 一页结论

| 结论 | 证据 | 可用性 |
|---|---|---|
| 两个算子精度对齐 torch，尾块不丢 | 11/11 测试【实测】 | ✅ 可用 |
| 4 个 DSL 坑的机理 | 探针 + IR【探针】 | ✅ 可用 |
| 16M 下 block_N 越大越快（4x） | 1120→800→274 µs【实测】 | ✅ 可用（见 4.2 的保留意见） |
| tile 4096 比 1024 稍快 | 4 次 run 方向一致，幅度 5–14%【实测】 | 🟡 方向可信，幅度不可引用 |
| `alloc_ub` 比 `alloc_shared` 快 | 4 次 run 方向一致，幅度 1–29%【实测】 | 🟡 方向可信，幅度不可引用 |
| 1M 下对齐无稳定惩罚 | 3/4 次 run 差异 <5%，1 次双峰离群【实测】 | ✅ 可用 |
| rmsnorm 各配置的耗时排序 | 235–318 µs 落在噪声内【实测】 | ❌ 不可用；rstd 只作精度旋钮 |
| 绝对耗时 / 32 GB/s 这类带宽数字 | 约 75% 是固定开销【补充分析】 | ❌ 不可引用 |

---

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

耗时随块数近似线性：差 846 µs / 差 3072 块 ≈ **0.27 µs per block**，该尺寸下截距接近 0。数据量相同，只改块数就有 4 倍差距，瓶颈是每块固定成本，不是搬运带宽。这一组本身也是升序测量，存在顺序混杂；但 4 倍幅度远超观察到的 10–30% 漂移，判断成立【实测，可信度较高】。

**（b）小尺寸被常数项吞掉。** 补充扫描 N = 1M / 4M / 16M 做线性拟合【补充分析，未在本轮重跑】：截距 ≈ 178 µs，斜率 ≈ 148 GB/s（block_N=4096 口径）。N=1M 的 234–335 µs 里约 75% 是截距，真正随数据量增长的部分只有几十 µs。

**（c）计时方式放大了常数。** `benchmark()` 用 NPU Event 包住 **10 次 Python 调用的循环**，测的是设备时间轴的总跨度。如果 host 侧每次入队（Python → TileLang wrapper → ACL）比设备执行还慢，设备会在两次 kernel 之间空转，这段空隙会被计入。那 178 µs 很可能是 host 节奏，而非硬件 launch 成本。【未验：需要 msprof 的 kernel 级 duration 归属 host / device】

**（d）和 torch.abs 比不公平。** 8 MiB 输入大概率整块落在片上缓存；torch 的 kernel 高度优化，host 开销低一个量级。实测 8 MiB 约 18–30 µs（≈470 GB/s），128 MiB 时 223 µs（≈1.2 TB/s）。这个 8 倍差距不能当成"我们的 kernel 差 8 倍"。

### 4.4 结论可用性

| 结论 | 状态 | 理由 |
|---|---|---|
| 精度 / 尾块 / DSL 机理 | ✅ 成立 | 与耗时无关 |
| 16M 下 block_N 越大越快 | ✅ 成立 | 4x 幅度，远超漂移 |
| tile 4096 优于 1024 | 🟡 方向成立，幅度作废 | 顺序混杂；跨 run 区间相接 |
| ub 优于 shared | 🟡 方向成立，幅度作废 | 顺序混杂；IR 差异确定（见下） |
| 1M 无稳定对齐惩罚 | ✅ 成立 | 3/4 轮 <5%；run3 的 27% 为双峰离群、未复现 |
| rmsnorm 速度排序 | ❌ 不成立 | 235–318 µs 处于固定开销主导区；rstd 只作精度旋钮 |
| 绝对耗时 / GB/s | ❌ 不成立 | 约 75% 为固定开销 |

ub vs shared 的 **IR 差异是确定的**：Expert 是 `memref.alloc #hivm.address_space<ub> + memref.copy`；Developer 是 `tensor.empty + insert_slice/extract_slice + bufferization.materialize_in_destination`。所以"纯 Vector kernel 用 `alloc_ub`"作为设计建议成立，只是不能引用 29% 这个数。

### 4.5 改法

**测量层（先做，约 20 分钟）**

1. 用 `torch_npu.profiler` / `msprof` 取 **kernel 级 device duration**，不再用 Event 包 Python 循环。
2. 配置顺序随机化或交错（修好 `benchmark` 的 KeyError），避免暖机与参数效应同向。
3. 报告口径：≥64 MiB 的绝对耗时；或扫 N 做回归、**报斜率（GB/s）并扣除截距**。
4. 每组 ≥5 次 run 取中位，差异 < 2σ 写"无差别"。

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

### 4.6 教训

- **先验证测量口径，再调参**。同一配置跨 run 不稳定时，参数结论都是噪声。
- **单轮区间不重叠 ≠ 跨 run 可复现**：固定顺序会让暖机效应伪装成参数效应。
- 性能表必须回答"这个尺寸下这套方法有没有分辨力"；没有就写没有。
- 对比高度优化库时控制数据规模（出片上缓存），并区分 host 开销与 kernel 时间。
- 正确性结论和性能结论分开下：跑通了就是跑通了，慢不慢是另一件事。

## 5. 与计划的偏差

| 计划 | 实际 | 原因 |
|---|---|---|
| rstd 用 `T.vrsqrt` | fp32 默认 `vsqrt+vrec`，fp16 基线也改用 `vsqrt+vrec` | 硬件 vrsqrt 只有 ~3e-3，达不到 fp32 1e-4 |
| 4 个 buffer ≈128 KiB | 无条件分配 6 个，≈136 KiB | 解析器不允许在 `if` 里分配后被外部使用 |
| 对齐 vs 非对齐预期"悬崖" | 1M 下三点差异 <2%，未复现稳定悬崖；一次 27% 为双峰噪声 | 尾块占比小；需要更细的 N 扫描和 PMU |
| 计划 4.5 h | 实际约 60 min（含 11 个测试的全量回归 7 min 53 s） | 示例与测试组织可复用；主要时间花在编译（每次 ~45–50 s）和排错 |

## 6. 下一步

1. **msprof 归属**：确认那 178 µs 是 host 入队还是 device 固定成本（约 10 分钟）。
2. **修测量脚本**：随机化 / 交错配置顺序，修掉 interleaved 的 KeyError；每组 ≥5 run 附 σ。
3. **改 abs kernel**：固定核数 + 核内串行，block_N=8192 + `T.Pipelined(num_stages=2)`，在 ≥64 MiB 上重测。
4. **旋钮表 v2**：每格 5 run、附 σ、标注"可分辨 / 不可分辨"；补 block_M=16/32（先做 aliasing）和 block_N=8192/16384。
5. **对齐细扫**：N = 1048576 ± {1,2,4,8,16,32}，每点 20+ 轮，确认尾块长度 / 32 B 对齐与慢路径的关系。
6. **把 reduce 推出 UB**：N > UB 时按 N 切 tile、跨 tile 累加（参考 `examples/norm/rms_norm.py` 的 split-K）。
7. **补测试**：fp16 + `sqrt_rec` 精度锁定；M/N 都不整除的 2D 尾块组合。
8. **减少无条件分配**：fp32/fp16 各写一个 prim_func（或用 `T.macro` 共享 body），把 fp32 的 UB 占用从 136 KiB 降回 ~100 KiB。
9. **D8 / D9 沿用同一口径**：以 `sqrt_rec` 版本为手写基线，与 agent 生成结果对比正确性、IR，并用 v2 测量协议。

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

## 附录 C：版本来源与冲突取舍

| 条目 | 原"测量复盘"稿 | 原"实现细节"稿 | 融合处理 | 依据 |
|---|---|---|---|---|
| 性能结论总体 | 全部降级 | 引用 11.3% / 29% | 方向保留、幅度作废 | 4.1–4.2 的跨 run 数据与顺序混杂 |
| tile 4096 优势 | 判为噪声 | 区间不重叠，判为成立 | 🟡 方向成立，幅度不可引用 | 跨 run 区间几乎相接 |
| ub vs shared | 判为噪声 | 慢 29% | 🟡 方向成立，IR 差异确定 | 4/4 方向一致，幅度 1–29% 不稳 |
| 对齐 | 判为噪声 | 无稳定悬崖 | ✅ 无稳定惩罚 | 3/4 轮 <5%，1 次双峰离群 |
| rmsnorm 耗时 | 不可信 | 噪声淹没 | ❌ 只作精度旋钮 | 双方一致 |
| rstd 精度 | 端到端 max_abs | 探针相对误差 | 两套指标并列 | 指标不同，排序一致 |
| UB 预算 / DSL 坑 / 计划偏差 | 无 | 有实测 | 全部采纳 | 与耗时无关 |
| 下一步 | 测量 + kernel | 扫描 + 基线 | 合并并按依赖排序 | 见第 6 节 |

> 说明：原"测量复盘"稿在本仓库 git 历史中（提交 `c1ae166`）；原"实现细节"稿来自对话粘贴，未单独落盘，其数据以本文引用为准。
