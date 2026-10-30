# Day 5 学习报告：FlashAttention / MixCV

日期：2026-10-29
设备：Ascend910B3（20 AIC / 40 AIV），CANN 9.0.0，tilelang-ascend npuir 分支
代码：`examples/flash_attn_npuir.py`（expert）、`examples/flash_attn_npuir_dev.py`（developer）
数据：`day5-flashatten/code/logs/`、`day5-flashatten/note/day5学习指标.md`、`note/本周周报.md`

## 一句话

四项任务全部完成：**Developer 模式 FA 四个形状全部跑对**（`max_abs ≤ 3.66e-4`），
设备域 **27.33 µs（seq=512）→ 824.57 µs（seq=4096）**，512 时比 torch 三段式快 1.6x、
4096 时慢 3.76x；**expert 版通过 JIT 兼容层跑通 512/1024/4096**：把旧属性 `syn_instr_mode`
改写为 CANN 9.0.0 接受的 `sync_instr_mode`，并对后端偶发 SIGSEGV 自动重试。
机制四项（Cube/Vector 分工、CV 自动拆分、1:2 配比、double buffer）已能口述，
周报 v2 已出。

今天真正记住的是测量口径：**FA 512 的 kernel 只有 27.33 µs，event 却记到 486 µs，
e2e 是 420 µs**。host 入队成本把整个小尺寸区间吃掉了，
所以“我们比 torch 慢 1.40x”在设备侧其实是“快 1.6x”。这条和 Day 3、Day 4 是同一件事，
只是这次它决定了一个结论的正负号。

## 1. 任务完成情况（对照 Day 5 plan 四项）

| # | 计划任务 | 状态 | 证据 |
|---|----------|------|------|
| 1 | 跑 `examples/flash_attn_npuir.py`，记录 shape 与耗时 | 完成：expert 512/1024/4096 正确性通过；Developer 四形状有性能数据 | `logs/expert_seq*_fixed.log`、`logs/run_dev_example.log`、`logs/bench_fa_dev.log` |
| 2 | 读 mixcv skill（SKILL.md + pipeline.md + flash-attn-pattern.md + flash-attn-dev.md） | 完成 | 摘要见 `day5学习指标.md` §2 |
| 3 | 讲清机制：Cube/Vector 分工、CV 自动拆分、1:2、double buffer | 完成（自评：可口述） | `day5学习指标.md` §3，含 60 秒版 |
| 4 | 周报：链路图 v1 + 性能表 + 3 卡点 + 下周计划 | 完成 | `note/本周周报.md` |

## 2. 正确性与性能

### 2.1 正确性（Developer 模式，fp16 输入 / fp32 累加，`rtol=atol=1e-2`）

| seq_len | dim | block M/N/K | max_abs | 结果 |
|--------:|----:|-------------|--------:|------|
| 512 | 128 | 64/64/32 | 3.662e-04 | pass |
| 1024 | 128 | 64/64/32 | 2.441e-04 | pass |
| 2048 | 128 | 64/64/32 | 1.411e-04 | pass |
| 4096 | 128 | 64/64/32 | 3.662e-04 | pass |

官方例子 `flash_attn_npuir_dev.py`（seq=512）重跑也是 `All check passed.`
（`logs/run_dev_example.log`）。

Expert 模式修复后 seq=512/1024/4096 均为 `All check passed!`（`logs/expert_seq*_fixed.log`）；
这些日志只记录正确性和编译成功信号，没有 expert 设备时长，下面性能表仍全部是 Developer 模式。

### 2.2 性能（三列分开，别混）

| seq_len | host µs/call | event µs（含入队） | e2e ms | torch e2e ms | e2e 比值 | **设备 µs（op_summary）** | torch 设备 µs | 设备比值 | ours TFLOPs |
|--------:|-------------:|------------------:|-------:|-------------:|---------:|--------------------------:|--------------:|---------:|------------:|
| 512 | 358.9 | 486 | 0.420 | 0.301 | 1.40x | **27.33** | 44.34 | **0.62x** | 4.91 |
| 1024 | 388.3 | 500 | 0.452 | 0.311 | 1.45x | **56.60** | 39.97 | 1.42x | 9.49 |
| 2048 | 562.6 | 678 | 0.631 | 0.330 | 1.91x | **208.61** | 93.91 | 2.22x | 10.29 |
| 4096 | 1211.5 | 1329 | 1.275 | 0.354 | 3.60x | **824.57** | 219.48 | 3.76x | 10.42 |

读法：
- `bench_fa_dev.csv` 里的 `device_ms_med` 列其实是 **event** 计时，包含 host 入队，
  所以它≈e2e 而不是设备时间；真正的设备时间列在 `logs/profile_fa_dev*.log`。
  这是脚本命名留下的坑，README 已注明。
- 设备 TFLOPs 按 4·seq²·dim 算（QK + PV 两次 gemm）。
- torch 基线是手写三段式 `softmax((q@k.T)*scale)@v`，**不是 SDPA**。

### 2.3 交叉点

512 时我们赢（融合省掉 S、P 两个 `seq×seq` 中间矩阵的 GM 往返），
1024 起被 torch 反超，2048/4096 差距拉大。原因不是融合失效，
而是 **ours 的 TFLOPs 在 1024 后饱和在 ~10**，torch 一路涨到 39.14：
tile 固定 64×64×32，Cube 的 L1/L0 复用率低，规模越大越吃亏。

## 3. 学到的机制（口述版见 `day5学习指标.md` §3.5）

1. **分工**：Cube 做 QK^T 和 PV 两个 gemm；Vector 做 online softmax
   （行最大值、`vexp`、行和、输出累加与归一化）。两者通过 GM workspace 交接，
   `T.sync_block_set/wait(i)` 按 i 配对。
2. **CV 自动拆分**：expert 模式已由 codegen 直接按 `T.Scope`/`T.rs` 生成
   `_mix_aic` / `_mix_aiv` 两个函数（`hivm.part_of_mix, mix_mode="mix"`）；
   Developer 模式只生成一个 `func_core_type=MIX` 函数，拆给 HIVM 的
   CV Pipelining pass 做。**所以两条路生成的 IR 结构不同，不能跨模式比较 IR。**
3. **1:2 配比**：910B3 是 20 Cube : 40 Vector。每个逻辑块有两个 AIV 子块
   （`vid = blockIdx.z`，范围 `grid*2`），FA 用 `subid` 把 block_M 的行对半分，
   尾块用 `(tail_size_m % 2)` 修正。这就是“一个 Cube 配两个 Vector”在代码里的样子。
4. **double buffer**：`T.Pipelined(num_stages=2)` 管循环内的分阶段；
   HIVM 的 `--enable-auto-multi-buffer`（jit_npu.py 默认传）请求 GM workspace /
   L1 / L0C / UB 多槽。文档配置的 workspace 默认值为 2，但本轮没有用后端 IR 或 A/B 确认实际生效。

## 4. 我错在哪（结论翻转记录）

| 阶段 | 我当时的判断 | 实际 | 触发证据 |
|------|--------------|------|----------|
| 初始 | 官方 expert 例子应该直接能跑 | 编译就失败 | `run_default.log` 的 `Dialect '' not found for custom op 'syn_instr_mode'` |
| 第一次翻转 | 是 Expert 没开启或例子代码问题 | Expert 已拆出 AIC/AIV；失败在 CANN 9.0.0 与 pin 的 AscendNPU-IR 19.1.7 后端接口不匹配 | repo 编译器能解析，CANN 的 hivmc 不认 |
| 第二次翻转 | 换个编译器/workaround 能救 | 三条 workaround 全部死在后续 pass | `run_default_wrapped.log`、`run_default_repocc_hivmcwrap*.log` |
| 第三次翻转 | FA 512 我们比 torch 慢 1.41x | 设备侧其实快 1.6x，e2e 慢是 host | event 486 µs vs op_summary 27.33 µs |
| 第四次翻转 | CANN 9.0.0 下必须换整套工具链才能跑 expert | 具体差异是属性改名；加 JIT 兼容与 SIGSEGV 重试后可跑通 | `expert_seq*_fixed.log` |

另外两个小错：
- `bench_fa_dev.py` 的 argparse 写成 `args.block_M`（解析器给的是 `block_m`），
  第一次 sweep 直接 AttributeError；那次输出被下一次 `tee` 覆盖，没落盘，
  worklog 里如实记了“没有日志”，没有编造。
- `profile_fa_dev.py` 选 profiler run 目录时按字符串排序，把新 PID 排到了旧 PID 前面，
  导致 1024/2048 打出了 4096 的 torch 数据；改成按 `getmtime` 排序后重跑，
  1024/2048 的数是重跑后的（`profile_fa_dev_1024_2048.log`）。

## 5. 还没验的

- **expert 模式的性能**：512/1024/4096 正确性已经通过，但还没有设备时长，不能和 Developer 比快慢。
- **多缓冲是否真的生效**：没有做 `--enable-auto-multi-buffer=false` 的 A/B，
  也没在 IR 里确认 `hivm.multi_buffer` 标记。
- **tile sweep**：只有 64/64/32 一组，无法回答“差距有多少来自 tile”。
- **可复现正确性**：每个 shape 只生成了一组未固定 seed 的随机输入；需要固定 seed 并补多 seed / 边界输入。
- **torch SDPA 基线**：没跑，不知道 SDPA 在 910B3 上是否已融合。
- **bf16 / 因果掩码 / GQA**：都没测。

## 6. 时间

计划 4.5 h（1 + 1.5 + 1 + 1）。上午机器执行约 25 min（含 77 s 的 sweep、
两次 profiler、若干次编译尝试），折算人工约 2.5–3 h；16:00–16:12 又完成 expert 修复与回归。
排障（expert 编译链）比计划多，机制整理与周报比计划少。

## 7. 补记（16:12）：expert 路径已修复

把“工具链不配套”拆开后，落到了两个具体问题：

1. **属性改名**：CANN 9.0.0 使用 `sync_instr_mode`，TileLang expert codegen 仍打印
   `syn_instr_mode`。JIT 现在按 `bishengir-compile --version` 判断是否需要改写。
2. **偶发 SIGSEGV**：属性改名后，`ConvertHFusionToHIVM` 在 MIX 模块上仍会非确定性崩溃；
   同一输入重试可以成功，因此 JIT 对信号退出最多重试 8 次，普通非零返回仍立即报错。

验证结果：expert `--seq_len 512/1024/4096` 全部 `All check passed!`，Developer 例子回归通过。
三份 expert 日志都出现了一次 signal 11 后重试成功；这个重试是临时兼容方案，长线仍应升级
仓库 pin 到含 `sync_instr_mode` 与配套 hivmc 的 AscendNPU-IR。修复 diff 见
`code/tilelang-jit_npu.patch`。
