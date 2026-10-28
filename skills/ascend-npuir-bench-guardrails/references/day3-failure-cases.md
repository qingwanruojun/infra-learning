# Day3 反例库（2026-10-28）

每条按「现象 → 根因 → 修复 → 下次怎么防」记录。数字来自 Day3 文档；
原始日志未同步到本机的一律标【未验】。

## A. kernel / DSL

### A1. `N // block_N` 静默丢尾

- 现象：示例 `examples/elementwise/vec_add_1d.py` 在 `N=block_N=4096` 时退化成 1 块，
  tail 是死代码；换到非整除形状会少算尾块且**不报错**。
- 根因：grid 用整除，尾块从未被调度。
- 修复：`T.ceildiv(N, block_N)` + 运行时 `tail = T.min(block_N, shape - offset)`，
  测试用 `N=4097`（尾 1）和 `N=4100, block_N=1024`（尾 4）。
- 防：任何 kernel 必须有非整除形状用例；`check()` 对尾段单独断言（`abs.py:135-143`）。
- 证据：`11 passed in 473.63s`【未验原始日志】。

### A2. 在 Python bool 的 `if` 里分配

- 现象：`if use_weight: w_in = T.alloc_ub(...)`，后面再 `T.copy(W, w_in)` →
  `Undefined variable: w_in`。
- 根因：TVMScript 确实在解析期求值 Python bool，但整段 `if` 包在
  `var_table.with_frame()` 里，分支内新名字出了分支即失效。
- 修复：所有 buffer 无条件分配；`if` 内只放 `T.copy`/`T.vcast`/`T.vmul`。
- 代价：fp32 RMSNorm 无条件分配 ≈136 KiB（UB 192 KiB），余量 56 KiB。
  若 `block_M=16` 需先做 aliasing 或拆 fp32/fp16 prim_func。
- 防：静态检查 `alloc_` 是否出现在 `if` 内。

### A3. `T.copy` / `T.vcast` 的 dtype 语义

- 现象 1：`T.copy` 跨 dtype 报
  `Check failed: npuirop.src->dtype == npuirop.dst->dtype`。
- 现象 2：`T.vcast(f32→f32, round_mode="rint")` 被当成恒等转换，实际把浮点
  **取整到整数**（`-1.0321 → -1`），RMSNorm 输出全变整数。
- 修复：Python bool 分支，同 dtype `T.copy`，跨 dtype `T.vcast`。
- 防：写任何 cast 前先跑 `experiments/probe_copy_cast.py` 验证语义。

### A4. 硬件 `vrsqrt` 是近似

- 现象：fp32 RMSNorm 端到端 `max_rel≈2.7e-3`，达不到 1e-4。
- 定位：逐行算 `y/ref`，行内标准差仅 4e-8、整行同一比例因子 → 误差在 rstd，
  不在 reduce 顺序或 cast。
- 探针：`vrsqrt` 2.8e-3；`vsqrt+vrec` ~1e-7；`vrsqrt`+1 次 Newton 1.2e-5。
- 修复：fp32 默认 `sqrt_rec`；fp16 的 1e-2 容差可用 `vrsqrt`。
- 防：任何近似指令先用独立探针测误差，禁止直接端到端点绿就收。

### A5. “无 weight”消融不干净

- 现象：`use_weight=False` 时 W 仍被 `T.copy(W, w_in)`，只省了最后一个 `vmul`。
- 后果：配置名不实；fp32/False/sqrt_rec 的 318 µs 比 weight=True 还慢 35%，
  被误归因成冷启动。
- 修复：改签名真正省掉搬运，或在报告里注明 4 KiB 搬运仍在。
- 防：消融必须同时检查“计算”和“搬运”是否都消失。

### A6. 不同形状比较精度

- 现象：Newton（M=64）`max_abs=5.865e-5` 与 sqrt_rec（M=1024）`1.907e-6`
  被写成“差 30 倍”。
- 根因：M 不同，不是同条件对比。
- 修复：同 M/N 下重测后再下倍数结论。
- 防：精度对比表必须固定所有非变量维度。

### A7. 探针标签把 `block_N` 写成块数

- 现象：`probe_profiler_kernel_time.py` 打印 “tilelang abs, 4096 blocks, N=1M”，
  但 `compile_abs(N1, 4096)` 里的 4096 是 **block_N**，N=1M → 实际 **256 块**。
- 后果：12.99 µs 被当成“4096 块只要 13 µs”，会推出“块数不影响设备时间”；
  而 16M 上 4096 块是 158 µs。标签错了，数字没错。
- 修复：标签写 `block_N` 与实际块数两个数；探针打印 `T.ceildiv(N, block_N)`。
- 防：任何探针的输出表头都带“块数 = N / block_N”的显式计算，不手写。

## B. 计时与统计

### B1. Event 包 Python 循环 = 端到端，不是 device

- 现象：`benchmark()` 用 NPU Event 包住 10 次 `kernel()` 调用，测的是设备时间轴总跨度。
  host 入队慢于设备时，kernel 间空转被算进去。
- 实测：256 块 host 239.2 µs / device 12.99 µs；4096 块 host 1091.8 µs /
  device 156–159 µs【未验原始 CSV】。
- 修复：device 用 profiler kernel duration；host 用 wall；e2e 单独一栏。
- 防：任何“kernel 耗时”先问口径：Event / device / wall。

### B2. 固定顺序 + round 0 暖机

- 现象：v1.3 两轮交替得出 `0.93–0.94x`（持平或略优）；ABBA×10 后是
  torch 185.67 vs 我们 200.02 µs = **1.070x（慢约 7%）**；9/10 轮在 1.05–1.09。
- 根因：只有 round 0 相交；torch 从 218.21 掉到 184–186，我们从 204 掉到 197。
  第一轮是冷启动样本。
- 修复：丢弃 warm-up 轮；A-B-B-A 交替 ≥10 轮；报中位与比值区间。
- 防：禁止用第一轮/单轮结果下“持平/略优”结论。

### B3. 单轮区间不重叠 ≠ 可复现

- 现象：final run 里 1024 的 min 287.5 > 4096 的 max 264.6，得出“4096 快 11.3%”；
  buffer 组得出“shared 慢 29%”。
- 跨 run 合并后：1024 中位 270–295 µs（min 263.5），4096 239–265 µs（max 267.8），
  区间几乎相接，效应只剩几到十几 µs。
- 根因：跨 run 漂移 10–30%，待比较差异只有 5–15%，信噪比≈1；且测量顺序固定。
- 修复：方向保留为假设，百分比作废；用 E3 协议重测。
- 防：单轮只给 E1，不给百分比。

### B4. 两点拟合造出“截距≈0”

- 现象：N=1M/4M/16M 两点式拟合得截距≈178 µs、斜率≈148 GB/s；
  另一组两点拟合得“截距≈0、0.27 µs/块”。
- 五点实测是两段式：1–256 块平坦 216–232 µs；256→1024 为 0.04 µs/块；
  1024→4096 为 0.26 µs/块。
- 修复：≥3 点，有平台期就用 `max` / piecewise。
- 防：拟合前先画点，别直接套直线。

### B5. 小尺寸缓存数据比较 kernel 质量

- 现象：“1M 下比 torch 慢 3.7 倍 / 8 倍”；8 MiB 上 torch 18–30 µs、我们 240 µs。
- 根因：小张量整块落片上缓存，torch host 开销低一个量级；我们被固定开销主导。
- 修复：128 MiB 上 ABBA 对拍，结论是慢约 7%（device 时长）。
- 防：kernel 质量只在 ≥128 MiB 评价；小尺寸只测 host 定律。

### B6. 单点孤值当基线

- 现象：torch 166.93 µs（1608 GB/s）被当成基线，得出“我们慢 1.27 倍”。
- 根因：另三次为 214–217 µs，166.93 是孤点。
- 修复：交替多次取中位。
- 防：任何单点先看重复性，偏离其余样本 >5% 先当离群值。

### B7. cProfile 总耗与 wall 不一致

- 现象：cProfile 总比 wall 高 20–38 µs。
- 根因：profiler walker 自身开销。
- 修复：总耗引用 wall，cProfile 只看各项比例。
- 防：报告里不要把 cProfile 总和当实际耗时。

### B8. 交错实验失败

- 现象：编译 12 个 kernel 后轮转计时；脚本 `KeyError`（去重标签丢失）；
  修好后持续满负载让所有配置整体变慢，跨组差异反而更大。
- 教训：先断言标签唯一；随机顺序；连续满负载会把时钟效应变成位置效应。
- 防：交错不是万能药，必须配合 warm-up 丢弃和随机化。

### B9. 128 MiB 当 HBM 对拍点

- 现象：ABBA×10 在 128 MiB 上得 1.070x（IQR 1.070–1.079），看着很稳；
  同一尺寸另一进程的 `torch.abs` 是 167.2/173.3（冷）、177.6/176.8（60 s 负载后）、
  214.4（新进程首个），我们 198–219 µs，比值 0.96–1.09x。
- 根因：128 MiB 压在 L2 边界，命中多少取决于进程之前的缓存状态；
  torch 自己跨进程漂 27%，是目标差异 7% 的 4 倍。
- 修复：对拍只用 ≥256 MiB 张量。256 MiB 1.000x、512 MiB 1.008x，
  stdev 0.1–0.3 µs（0.03%）。128 MiB 降级为“不可复现”的反例。
- 防：选尺寸先问“是否 ≥256 MiB 张量”；< 256 MiB 只报方向，不报比值。

### B10. 同一份总结里机制说法互斥

- 现象：`profiler_summary.log` 16:35 写 “working sets >=128 MiB are fully out of
  cache”，17:25 又写 “the one size where the workload straddles the L2 boundary”。
  同一尺寸，两种互斥机制。
- 根因：前一句按带宽阶梯（1.25–1.28 TB/s）推的，后一句按 ABBA 的 1.45 TB/s 推的；
  两次都不是缓存计数实测。
- 修复：只保留现象（“该尺寸比值跨进程不可复现”），机制标【推断】。
- 防：合订文档必须做一次交叉检查：同一对象（同尺寸同 kernel）的机制描述只允许一种。

## C. 结论与文档

### C1. 先验被实测推翻的案例

- 假设：尾块非 32 B 对齐会有 ≥20% 悬崖，尾 1 元素最坏。
- 实测：1M 下三点差异 <2%；run3 的 1048575 慢 27% 是双峰（min 256 / max 338），
  未复现。
- 处理：写“无稳定对齐惩罚”，并列入“需细扫 + PMU 才能定论”。
- 教训：不要为了保住先验而把一次双峰当结论。

### C2. 被连续推翻两次的说法

| 版本 | 说法 | 后续 |
|------|------|------|
| 初版 | 1M 比 torch 慢 3.7 倍 / 8 倍 | 小尺寸缓存现象，不成立 |
| v1.3 | 0.93–0.94x，持平或略优 | round 0 暖机样本，不成立 |
| v1.5 | 1.070x，慢约 7% | ABBA×10，9/10 轮 1.05–1.09 |

教训：同一结论被推翻两次，说明测量协议而不是 kernel 有问题。

### C3. 文档漂移与证据缺口

- Day3 的 `STATUS.md`、`worklog.md`、`README (1).md`、`learning-report.md`
  标题/路径写成 Day 2 或 `notes/day2/`。
- 声称的 `logs/`、`experiments/` 在本机不存在，只能标【未验】。
- 防：
  1. 文件标题与目录一致（dayN 对 dayN）；
  2. 每个数字附日志路径；
  3. 日志未归档就写【未验】，不升级成实测；
  4. 旧文档不改原文，另存修订版并加对照表。

## D. 一页速查

| 症状 | 先查 |
|------|------|
| 尾块结果不对/没报错 | `// block_N` → `ceildiv` |
| `Undefined variable` | `if` 内分配 → 移出 |
| 输出变整数 | `vcast(f32→f32)` → 改 `copy` |
| fp32 精度 1e-3 量级 | `vrsqrt` → `vsqrt+vrec` |
| “kernel 很慢” | 先看 host/device 口径与张量大小 |
| 参数结论不稳 | 固定顺序/暖机/单轮 → ABBA×10 |
| 拟合截距≈0 | 两点 → ≥3 点/分段 |
| 和 torch 差很多 | 是否在缓存内、是否单点基线 |
| 128 MiB 上“慢 7%” | 该尺寸跨进程不可复现；改用 256/512 MiB |
| 探针里的“N blocks” | `N / block_N` 实算，别信手写标签 |
| 同一份总结前后矛盾 | 同尺寸同 kernel 的机制只留一种，其余标推断 |
