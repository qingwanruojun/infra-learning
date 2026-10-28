---
name: ascend-npuir-bench-guardrails
description: 昇腾 TileLang/npuir 算子开发与性能评测防坑指南。写或修改 abs、RMSNorm、softmax 等 Vector kernel，跑 910B 基准，调 block_N/block_M、alloc_ub/alloc_shared、rstd 实现，和 torch 对拍，做 profiler/ABBA/精度探针，或要写性能结论时必须使用。强制非整除形状测试、host/device/end-to-end 三栏计时、跨 run 可复现和证据分级。
---

# 昇腾算子 infra 防坑技能（Day3 教训）

本技能把 Day3（2026-10-28）踩过的坑固化成开工前检查表。核心原则：
**先证明测量有分辨力，再调参；先过精度，再谈性能；先跨 run 复现，再写百分比。**

## 0. 触发与边界

触发：写/改 npuir kernel；设计 benchmark；比较 torch；调 tile/buffer/rstd；
出现 `T.ceildiv`、`T.vcast`、`vrsqrt`、`T.reduce`、`block_N`、`profile`、`ABBA`；
要输出“快/慢多少”的结论。

不触发：纯安装、纯文档翻译、与 NPU 性能无关的代码。

## 1. 十条铁律

| # | 规则 | Day3 的反例与后果 | 正确做法 |
|---|------|------------------|----------|
| 1 | 分块数用 `T.ceildiv`，尾长由运行时 `shape` 算 | `N // block_N` 在非整除时**静默丢尾、不报错** | `tail = T.min(block_N, shape - offset)`，并加非整除测试 |
| 2 | 分配写在 `if` 外，`if` 内只放算子 | 在 Python bool 的 `if` 里 `T.alloc_ub`，出了分支即 `Undefined variable` | 所有 buffer 无条件分配；按 dtype/开关只切 copy/cast/vmul |
| 3 | 同 dtype 用 `T.copy`，跨 dtype 用 `T.vcast` | `T.copy` 跨 dtype 直接报错；`T.vcast(f32→f32, "rint")` 是**取整**不是恒等（`-1.0321 → -1`） | 用 Python bool 分支：同 dtype copy，跨 dtype cast |
| 4 | 精度先过容差，再把它当性能旋钮 | 硬件 `vrsqrt` 相对误差 ~2.8e-3，fp32 的 1e-4 目标不达标 | fp32 默认 `vsqrt+vrec`；fp16 1e-2 可用 `vrsqrt`；每次用探针实测 |
| 5 | 每次调用分开报 host / device / end-to-end | Event 包 10 次 Python 调用，把 host 提交空隙算进“kernel 时间” | 三栏计时，表头写口径；host 用 wall，比例才看 cProfile |
| 6 | kernel 质量在 HBM 规模比（≥256 MiB 张量 / ≥512 MiB 流量） | 1M / 8 MiB 落在缓存；**128 MiB 正好压在 L2 边界，torch 自己跨进程漂 27%** | 小尺寸只用来测 host 定律；对拍只用 ≥256 MiB |
| 7 | 对比用 ABBA 交替 + 丢 warm-up + ≥10 轮取中位 | 固定顺序让暖机伪装成参数效应；“0.93x 持平”来自 round 0 | A-B-B-A 交替，丢弃前 1–2 轮，看中位与比值区间 |
| 8 | 跨 run 可复现才写百分比 | 单轮区间不重叠（快 11.3% / 慢 29%）跨 run 后效应只剩几 µs | 单轮只写“待验证假设”；百分比只在 E3/E4 证据下写 |
| 9 | 拟合至少 3 点，优先分段模型 | 两点拟合造出“截距≈0”；实际是两段式 host 定律 | 5 点以上；有平台期就用 `max` / piecewise，不硬套直线 |
| 10 | 正确性与性能分开下结论；每个数字带日志和口径 | “跑通了”和“慢 3.7 倍”被混在一段话里，后者被推翻两次 | 正确性只谈误差；性能另开表，标【实测】/【源码】/【未验】 |

## 2. 写 kernel 前检查表

### 2.1 形状矩阵（缺一不可）

对每个 kernel 至少编译/测这些形状：

- `N = k * block_N`（整除）
- `N = k * block_N + 1`（尾 1）
- `N = k * block_N + block_N - 1`（尾块接近满）
- 2D：`M = k * block_M - 1`（末块少一行）
- 可选参数消融：打开和关闭时都必须真实改变工作量

`own_ops/abs.py:135-143` 是范例：除了整体 `assert_close`，还对尾段单独断言，
防止“尾块没执行但主断言仍通过”。

### 2.2 静态自查

```bash
# 1) 尾块丢弃：分块数里出现整除
grep -nE '//\s*block_(N|M)' own_ops/*.py experiments/*.py

# 2) 作用域：查看 alloc 是否落在 if 内（人工确认）
grep -n 'alloc_' own_ops/*.py

# 3) 跨 dtype copy / 同 dtype cast（人工确认）
grep -n 'T.copy\|T.vcast' own_ops/*.py
```

### 2.3 必须照抄的写法

tail-safe 一维分块：

```python
num_blocks = T.ceildiv(N, block_N)
with T.Kernel(num_blocks, is_npu=True) as (cid, _):
    a_buf = alloc((block_N,), dtype)      # 分配在 if 外
    b_buf = alloc((block_N,), dtype)
    offset = cid * block_N
    tail = T.min(block_N, shape - offset)
    T.copy(A[offset:offset + tail], a_buf[0:tail])
    T.vabs(a_buf, b_buf)
    T.copy(b_buf[0:tail], B[offset:offset + tail])
```

dtype 分支：

```python
if dtype == accum_dtype:   # Python bool，解析期只剩一支
    T.copy(x_in, x_f32)
else:
    T.vcast(x_in, x_f32, round_mode="rint")
```

固定核数 + 核内串行（host 提交主导时的默认调度）：

```python
NCORE = 40  # 从接口查实际值，不要抄示例里的 20
tiles = T.ceildiv(N, block_N)
with T.Kernel(NCORE, is_npu=True) as (cid, _):
    for i in T.serial(T.ceildiv(tiles, NCORE)):
        t = cid + i * NCORE
        if t * block_N < shape:
            off = t * block_N
            tail = T.min(block_N, shape - off)
            T.copy(A[off:off + tail], a_buf[0:tail])
            T.vabs(a_buf, b_buf)
            T.copy(b_buf[0:tail], B[off:off + tail])
```

### 2.4 DSL 硬约束

- `T.reduce`：src/dst 同 rank，恰好一维不同且 dst 该维为 1。
  `(block_M, N) -> (block_M, 1)` 合法；其他花样先查文档。
- 标量只能放二元算子的 `src2`；`inv_n` / `eps` 在 `T.Kernel` 内、`T.Scope` 外定义。
- 广播 `(bM,N)×(bM,1)` 与 `(bM,N)×(1,N)` 都可用，后者用于 weight。
- 计算量≈0 的逐元素算子不要指望 `T.Pipelined`：read+write 共享链路，流水超不过带宽。
- UB 预算按“所有无条件分配”求和。例：fp32 RMSNorm 6 个 buffer ≈136 KiB / 192 KiB，
  余量 56 KiB；`block_M=16` 或多缓冲前先做 aliasing 或拆 prim_func。
- 消融要干净：`use_weight=False` 时 W 仍被 `T.copy` 进 UB，就不是“无 weight”消融。
  要么改签名真正省掉搬运，要么在报告里注明混杂项。

## 3. Benchmark 协议 v2

### 3.1 三栏口径

| 栏位 | 怎么测 | 用途 |
|------|--------|------|
| host | 单次调用 wall（每次 `synchronize`），或 cProfile 的 `launcher.launch` + `load_kernel_binary` | 提交成本、块数定律 |
| device | `torch_npu.profiler` / `msprof` 的 kernel duration | kernel 质量、和 torch 对拍 |
| end-to-end | NPU Event 包 N 次调用 / N | 用户可见耗时；**必须注明包含 host** |

cProfile 总和会比 wall 高 20–38 µs（walker 开销）：**总耗引用 wall，cProfile 只看比例**。

### 3.2 执行顺序

1. 先编译全部配置，再计时；编译时间一律排除。
2. 每配置 warm-up ≥3 次；正式轮次丢弃前 1–2 轮（round 0 是暖机样本）。
3. A/B 对比用 ABBA 交替；多配置扫描用随机/交错顺序，避免“后测的组”继承时钟效应。
4. 每组 ≥5 轮，结论性对比 ≥10 轮；报中位、min/max、σ。
5. 差异 < 2σ 写“无差别”，不写方向。
6. 同时记录 `host / device / e2e`；禁止把三种口径混进一张表。

### 3.3 尺寸选择

- **kernel 质量**：≥256 MiB 张量（≥512 MiB 流量），256/512 MiB 应接近线性。
- **128 MiB 张量是 L2 边界，不作对拍点**：同一 `torch.abs` 跨进程 167–214 µs（27%），
  比要追的差异还大；该点只能当"不可复现"的反例。
- **host 定律**：块数 1 / 64 / 256 / 1024 / 4096 五点；有平台期就分段拟合。
- **不要**用 1M / 8 MiB / 64 MiB 的绝对耗时评价 kernel：那段由块调度和缓存主导
  （64 MiB 张量 128 MiB 流量 26.86 µs ≈ 5 TB/s，已远超 HBM 稳态 1.28）。
- 带宽必须写清字节口径（读+写、是否含 weight）和 dtype。

### 3.4 交错实验的失败教训

Day3 的交错脚本因去重标签 `KeyError` 失败；修好后持续满负载让所有配置整体变慢，
把时钟效应变成了位置效应。若重做：先断言标签唯一，再随机顺序，并记录每轮时间戳。

## 4. 证据分级与措辞

| 等级 | 证据 | 允许的表述 |
|------|------|-----------|
| E0 | 单次观察 / 单点 | 只写“观察到”，不下结论 |
| E1 | 单轮内区间不重叠 | “待验证假设”，不写百分比 |
| E2 | 跨 run 方向一致 | “方向成立”，幅度仍不写 |
| E3 | ABBA ≥10 轮中位 + 比值区间 | 可以写百分比和倍数 |
| E4 | ≥3 点参数模型 + host 预算闭合 | 可以写定律/模型 |

禁止句：

- 用 round 0 说“和 torch 持平/略优”（真实比值只在 ≥256 MiB 成立，128 MiB 不可外推）。
- 用 8 MiB 缓存内数据说“慢 3.7 倍 / 8 倍”。
- 用 128 MiB 的 1.070x 说“HBM 尺度慢 7%”（该尺寸 torch 自己跨进程漂 27%）。
- 用两点拟合说“截距≈0”。
- 用单轮区间说“快 11.3% / 慢 29%”。
- 在没做细扫 + PMU 时说“尾块 32 B 对齐有稳定悬崖”。
- 把 rstd 实现说成速度旋钮（它首先是精度旋钮）。

所有数字旁边标：`【实测】`（有日志且已复核）/ `【源码】`（读代码确认）/
`【未验】`（转述、日志未归档）。日志没同步就写【未验】，不要升级成实测。

## 5. 报告模板

```markdown
| 配置 | host µs | device µs | e2e µs | 带宽 GB/s（字节口径） | 证据等级 | 日志 |
|------|--------:|----------:|-------:|----------------------|----------|------|
| A    |         |           |        |                      | E3       | path |
| B    |         |           |        |                      | E3       | path |
```

结论段只写三件事：方向、效应量（带区间）、证据等级。精度结论单独一节。

## 6. 已知基线（2026-10-28 远端，引用前重测）

- host 两段式：1–256 块平坦 216–232 µs（16 KiB 的 N=1024 也要 277 µs）；
  256→1024 为 0.04 µs/块；1024→4096 为 0.26 µs/块。4096 块时 host 1084.6 µs，
  device 仅 156–159 µs。机制在 ACL `rtKernelLaunch` 内，不在 Python 层。
- **device 随尺寸的阶梯**（同进程，配置见括号）：

  | 张量 | 流量 | 我们 | torch | 比值 |
  |-----:|-----:|-----:|------:|-----:|
  | 4 MiB | 8 MiB | 12.99 µs（256 块） | 3.55 | 3.66x，缓存区 |
  | 64 MiB | 128 MiB | 26.86 µs（40 块串行） | 22.13 | 1.21x，缓存区 |
  | 128 MiB | 256 MiB | 200.02 µs（40 块串行） | 185.67 | 1.070x，**不可外推** |
  | 256 MiB | 512 MiB | 422.57 µs（40 块串行） | 422.37 | **1.000x** |
  | 512 MiB | 1 GiB | 845.21 µs（40 块串行） | 838.46 | **1.008x** |

- 对外只引用最后两行：**HBM 尺度打平，双方 1.27–1.28 TB/s**（512 MiB 上 stdev
  0.1–0.3 µs）。128 MiB 的 1.070x 只在单个进程内成立（IQR 1.070–1.079），
  torch 在同一尺寸跨进程 167–214 µs，比值会到 0.96–1.09x。
- 块数也影响 device：N=16M、block_N=4096，4096 块 158 µs vs 40 块串行 27 µs
  （5.8x）；端到端 1095.4 vs 299.1 µs（3.7x）。小尺寸的差距主要是波调度
  （每波约 1.5 µs，**推断**），不是带宽。
- 64 MiB 张量上双方都在 5–6 TB/s（8 MiB 更低），这是缓存带宽，不能代表 HBM 性能。
- 1M abs 的 234–295 µs 是 host 地板区，不是 kernel 差距的度量。

## 7. 参考

- 详细反例与修复过程：`references/day3-failure-cases.md`
- Day3 笔记（当前口径）：[`day3/note/day3学习指标.md`](../../day3/note/day3学习指标.md)
- 代码索引与冗余审计（24 个探针的状态与归档日志）：[`day3/code/README.md`](../../day3/code/README.md)
- 代码范式：[`day3/code/own_ops/abs.py`](../../day3/code/own_ops/abs.py)、
  [`day3/code/own_ops/rmsnorm.py`](../../day3/code/own_ops/rmsnorm.py)
- 精度探针：[`day3/code/experiments/probe_rstd.py`](../../day3/code/experiments/probe_rstd.py)、
  [`probe_copy_cast.py`](../../day3/code/experiments/probe_copy_cast.py)
- 迁移 GPU 算子时的流程与重写映射：[`ascend-tileops-migration`](../ascend-tileops-migration/SKILL.md)
  ——本 skill 仍然管其中的测量口径与措辞。

## 8. 迁移任务的特别约定

用 conductor 框架迁移 TileOPs / GPU 算子时（见 `ascend-tileops-migration`）：

- 迁移只承诺"精度过 + 能集成"，**不写快慢**；Stage 5 report 的 bench 不满足铁律 5–7。
- 迁移的精度门禁是 Stage 3 内嵌 L0/L1，不等于本 skill §2.1 的形状矩阵——
  非整除形状、fp16、尾块仍要自己补测。
