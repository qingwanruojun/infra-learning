# Day 5 工作日志：FlashAttention / MixCV

日期：2026-10-29
设备：Ascend910B3（20 AIC / 40 AIV），CANN 9.0.0，tilelang npuir 分支

## 时间线（UTC+8）

| 时间 | 动作 | 结果 / 证据 |
|------|------|-------------|
| 10:56 | 读 plan：Day 5 四项（跑 FA / 读 mixcv skill / 讲机制 / 周报） | — |
| 10:57 | 跑 `python examples/flash_attn_npuir.py`（默认 seq_len=4096） | ❌ CANN `bishengir-compile` 解析失败：`Dialect '' not found for custom op 'syn_instr_mode'`（`logs/run_default.log`） |
| 10:58 | 跑 `examples/flash_attn_npuir_dev.py` | ✅ `All check passed.`，seq_len=512（`logs/run_dev_example.log`） |
| 10:59 | 用 repo 自带 3rdparty `bishengir-compile`（llvm 19.1.7）重试 expert 例子 | ✅ 解析通过；❌ 调 hivmc 时 `--enable-cpu-trace-intrinsic` / `--enable-lir-compile` 未知（`logs/run_expert_repo_compiler.log`） |
| 11:00 | 写 wrapper 删掉 `syn_instr_mode` 再交给 CANN 编译器 | ✅ 能生成 `.o`；❌ 运行时验证前先遇到 compiler crash（`logs/run_default_wrapped.log`） |
| 11:01 | wrapper 过滤 hivmc 不认识的 flag | ❌ hivmc 仍不认 IR 里的 `syn_instr_mode`（`logs/run_default_repocc_hivmcwrap.log`） |
| 11:02 | 连 hivmc 输入 MLIR 一起 strip | ❌ `hivm.hir.nd2nz` 无法 legalize（去掉 `--enable-lir-compile` 后缺少 LIR lowering）（`logs/run_default_repocc_hivmcwrap2.log`） |
| 11:03 | 试 `bishengir-compile-a5` | ❌ A5 编译器选项集不同，不适用 910B3 |
| 11:03 | 跑 Developer 版 benchmark sweep 512/1024/2048/4096 | 见 `logs/bench_fa_dev.log` / `.csv` |
| 11:0x | 读 `tilelang-mixcv-skill` 4 份文件 + AscendNPU-IR `multi_buffer.md` / `cv_pipelining.md` / `auto_sync.md` | 机制笔记见 `day5学习指标.md` §3 |
| 11:0x | 写周报 v1（链路图 + 性能表 + 3 卡点 + 下周计划） | `周报.md` |

## 错误与修复

1. **`syn_instr_mode` 解析失败**
   - 现象：CANN 9.0.0 的 `bishengir-compile` 在任何 pass 之前就拒绝 expert IR。
   - 根因（源码/文档）：TileLang expert codegen 给 `hivm.hir.sync_block_set` 打印
     `syn_instr_mode = <INTRA_BLOCK_SYNCHRONIZATION>`（`src/target/codegen_npuir.cc:1026`）。
     该属性在 repo pin 的 AscendNPU-IR（llvm 19.1.7）里存在，在 CANN 9.0.0 的 HIVM dialect 里不存在。
   - 尝试：改成文档里的 `tsync_instr_mode` 同样失败；直接删除属性可以编译出 `.o`，
     但后续 pass crash；用 repo 编译器能过解析，却把 `syn_instr_mode` 留进 `module.hivm.opt.mlir`，
     CANN 的 hivmc 0.2.0 又不认。
   - 上午结论：Expert codegen 已生成 `_mix_aic/_mix_aiv`，不是“Expert 没开启”；当时确认阻塞在 **CANN 9.0.0 与仓库 pin 的 AscendNPU-IR 后端接口**。
     16:12 已用 JIT 属性兼容与编译重试跑通，最终结论见文末修复记录。

2. **repo 编译器 + CANN hivmc 的 flag 不匹配**
   - `3rdparty/AscendNPU-IR/build/bin/bishengir-compile` 会向 hivmc 传
     `--enable-cpu-trace-intrinsic=false`、`--enable-lir-compile=true`；CANN 9.0.0 的
     hivmc 0.2.0 两个都不认。
   - 过滤 flag 后，`nd2nz` 的 legalize 失败（`--enable-lir-compile` 不是可选项）。

3. **benchmark 脚本 argparse 命名**
   - `--block-m` 解析成 `args.block_m`，脚本里写成 `args.block_M` → AttributeError。
     已修正；该次失败输出被后续 `tee` 覆盖，未落盘（记录在此，不伪造日志）。

## 结论翻转序列

- 初判：expert FA 应该能直接跑（它是仓库自带例子）。
- 第一次翻转：IR 已拆出 AIC/AIV，但 CANN 9.0.0 编译器不认 `syn_instr_mode` → Expert 已开启，问题在后端工具链配套。
- 第二次翻转：repo 编译器能解析，但 hivmc 不认它的 flag → 两条路都堵。
- 上午阶段口径：**expert 后端未跑通；Developer 路径已跑通并给出性能数字。**
  16:12 修复后 expert 正确性也已跑通；周报里的 FA 性能数字仍全部标注为 Developer 模式。

## 16:00–16:12 修复：expert 路径跑通

### 根因（精确版）

1. **属性改名**：CANN 9.0.0 的 HIVM dialect 使用 `sync_instr_mode`，
   TileLang expert codegen 仍打印 `syn_instr_mode`。Expert 本身早已成功生成
   `_mix_aic` / `_mix_aiv`，失败发生在 CANN 解析阶段。
2. **CANN 编译器偶发崩溃**：属性改名后 expert IR 可以解析，但
   `ConvertHFusionToHIVM` pass 在 MIX 模块上会非确定性 SIGSEGV；同一输入重试可通过。
3. repo 自带 `bishengir-compile`（19.1.7）仍要求配套 hivmc；本机 CANN 的 hivmc
   与它不配套。修复后直接使用 CANN 9.0.0 编译器，不再走这条混搭路径。

### 改动（`tilelang/jit/jit_npu.py`）

- `_npuir_needs_sync_attr_rename()` 读取 `bishengir-compile --version` 首行：CANN 风格
  版本在编译前把 `syn_instr_mode` 改写为 `sync_instr_mode`；仓库 19.1.7 风格保留旧名。
  读不到版本时先按旧名编译，若错误信息出现旧属性名再改写重试。
- `_NPUIR_COMPILE_MAX_ATTEMPTS = 8`：编译子进程被信号杀死时重试并打印告警；
  普通非零返回仍停止并保留原诊断。

### 验证（原始日志）

| 命令 | 结果 | 日志 |
|------|------|------|
| `python examples/flash_attn_npuir.py --seq_len 512` | `All check passed!`（1 次 SIGSEGV 后重试成功） | `logs/expert_seq512_fixed.log` |
| `python examples/flash_attn_npuir.py --seq_len 1024` | `All check passed!`（1 次 SIGSEGV 后重试成功） | `logs/expert_seq1024_fixed.log` |
| `python examples/flash_attn_npuir.py --seq_len 4096` | `All check passed!`（1 次 SIGSEGV 后重试成功） | `logs/expert_seq4096_fixed.log` |
| `python examples/flash_attn_npuir_dev.py` | `All check passed.`（回归） | 控制台 |

### 仍未做

- expert 模式的设备时间/性能对比；本轮修复只验证正确性。
- 源码层彻底对齐 AscendNPU-IR：长线应升级 submodule pin 到含
  `sync_instr_mode` 和配套 hivmc 的版本；本次只在 JIT 做兼容。
