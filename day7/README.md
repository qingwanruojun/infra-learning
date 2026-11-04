# Day 7｜编译链路追踪（VecAdd）

这一天只做一件事：把**同一次** VecAdd 计算从 TileLang Python 追到设备产物，
确认每一级实际是什么形态、由谁产生，以及计划图里 assumed 的 TileLangIR
到底有没有参与。

```text
Python → TVM TIR → 初始 npuir → final npuir → HIVM → 设备 ELF → 上卡校验
```

先读 [学习笔记](note/day7学习笔记.md)，了解链路和结论；需要逐行证据时看
[详细技术报告](note/learning-report.md)；所有脚本、IR、原始日志和设备产物在
[code/](code/) 下，入口是 [code/README.md](code/README.md)。

## 结论一览

| 结论 | 结果 | 主要证据 |
|---|---|---|
| 主运行成功，设备结果正确 | exit 0，`assert_close` 通过，打印 `All check passed` | `code/logs/vecadd-dev-exit.log`、`vecadd-dev-dump.log:172` |
| 四级 IR 来自同一次运行 | TVM TIR → 初始 npuir → canonicalize/adapt → final npuir 均可切分 | `code/logs/step2-extract-ir.log`（`EXTRACT=PASS`）、`code/ir/step1-*`～`step3-*` |
| 实际只跑了两个 pass | `canonicalize`、`adapt-triton-kernel`；后者本次 no-op | `vecadd-dev-dump.log:62,97`、`code/logs/step2-canonicalize-vs-adapt.diff` |
| **没有 TileLangIR IR 阶段** | `libtilelangir.so` 被加载并充当 pass 管理器，但没有生成 TileLangIR dialect，也没有调度 CVSplit/Vectorize | `code/logs/step6-tilelangir-load-check.log`；passdump 中 TileLangIR 关键字计数 0；`code/ir/out-of-path-*` |
| vadd 的实际落点 | `T.npuir_add` → `CreateHIVMBinaryVectorOp<VAddOp>` → `hivm.hir.vadd` → `vadd_1d_float` | `codegen_npuir_dev.cc:3587-3588,2671`；`step2-initial-npuir.mlir:29`；`step5-final-hivm-lowered.mlir:44` |
| HIVM 下游完整流水线 | GM/UB 显式化；UB 偏移 0/16384；两次 load、一次 vadd、一次 store，带 flag/barrier | `code/ir/step4-hivm-infer-mem-scope.mlir`、`step5-final-hivm-lowered.mlir` |
| 原 JIT 产物与复编产物一致 | 设备 `.o` 和 host `.so` 均逐字节相同 | `code/logs/step7-jit-vs-recompiled.log`（两个 `IDENTICAL`） |

## 目录怎么读

```text
day7/
├── README.md                         本文件：入口、结论、复现与证据索引
├── note/
│   ├── day7学习笔记.md                首读笔记：实验经过、关键认识、卡点与边界
│   └── learning-report.md            详细技术版：逐行证据、源码位置、验收回答
└── code/
    ├── README.md                     逐文件索引：脚本、IR、日志、产物、口径
    ├── commands.md                   完整命令、退出码与口径
    ├── run_vecadd_dev.sh             主运行（Developer 模式 + IR dump + JIT shim）
    ├── extract_ir.py                 从同一次日志切分各阶段 IR
    ├── run_bishengir_checks.sh       下游检查、阴性对照、复编与产物比对
    ├── shims/bishengir-compile       记录 JIT argv，复制临时 kernel.o/libkernel.so
    ├── ir/                           各阶段 IR
    ├── logs/                         原始日志、diff、阴性对照、JIT 留痕
    ├── artifacts-vecadd.o            设备 ELF（2784 B）
    └── libartifacts-vecadd.so        host 侧 arm64 .so（4720 B）
```

两份 note 使用同一批证据，**以 `day7学习笔记.md` 为主读本**；
`learning-report.md` 保留更细的源码行号、pass dump 定位和验收回答，不另立结论。

## 环境和范围

| 项 | 值 | 证据 |
|---|---|---|
| 设备 | Ascend 910B3，`npu-smi` 显示 NPU 3 / Chip 0 | `code/logs/step0-env.log` |
| CANN | 9.0.0，`npu-smi 25.5.0` | 同上 |
| `bishengir-opt` / `bishengir-compile` | 1.1.0（`428ab8fdab46`），LLVM 19.1.7 | 同上 |
| TileLang 仓库 | `/mnt/workspace/learn/tilelang-ascend`，`bd7c60c`（含实验前已有的本地改动） | 同上 |
| 主例 | `examples/elementwise/vec_add_1d.py`，f32，N=4096，1 block | `code/ir/step1-tvm-ir.tir` |
| 实际构建的 AscendNPU-IR 副本 | `3rdparty/AscendNPU-IR`，`31f6903` | `code/logs/step0-env.log` |

没有安装或升级任何工具，也没有修改 TileLang / AscendNPU-IR 上游源码。
本实验只覆盖 N=4096、f32、单 block；不测性能，不做 Day 8 的 Agent 生成任务。

## 一键复现

```bash
cd infra-learning/day7/code

bash run_vecadd_dev.sh          # 主运行，约数十秒
python3 -I extract_ir.py        # 切分 IR，期望末行 EXTRACT=PASS
bash run_bishengir_checks.sh    # 下游检查，生成/覆盖 ir/ 与 logs/ 及产物
```

预期：

| 检查 | 期望 |
|---|---|
| `code/logs/vecadd-dev-exit.log` | `exit_code=0` |
| `code/logs/vecadd-dev-dump.log` 末尾 | `All check passed` |
| `code/logs/step2-extract-ir.log` 末尾 | `EXTRACT=PASS` |
| `code/logs/step4-roundtrip.err` | `roundtrip_exit=0` |
| `code/logs/step4-hivm-infer-mem-scope.err` | exit 0 |
| `code/logs/step5-bishengir-compile-exit.log` | `bishengir_compile_exit=0`，ELF 2784 B |
| `code/logs/step7-jit-vs-recompiled.log` | `JIT_VS_RECOMPILED=IDENTICAL`、`HOST_SO=IDENTICAL` |

`run_vecadd_dev.sh` 使用 `set -o pipefail` + `PIPESTATUS[0]`，避免 `tee` 掩盖
Python 的退出码；它还会把 `code/shims/` 放到 `PATH` 最前面，记录 JIT 对
`bishengir-compile` 的真实 argv，并在临时目录清理前复制 `kernel.o` /
`libkernel.so`。捕获失败只打 warning，不影响真实编译。

换机器时先改两处路径：`run_vecadd_dev.sh` 的 `REPO` 和
`/home/developer/Ascend/cann-9.0.0/set_env.sh`；`run_bishengir_checks.sh` 里
`tilelangir-opt` 的构建路径同理。

## 关键边界行号

主日志 `code/logs/vecadd-dev-dump.log`（172 行 / 14967 B）：

| 行号 | 内容 |
|---|---|
| 6 | `====== TVM IR ======` |
| 25 | `====== npuir ======` |
| 62 | `IR Dump After Canonicalizer` |
| 97 | `IR Dump After AdaptTritonKernel` |
| 132 | `====== final npuir ======` |
| 165 | `AscendNPU IR compile success:` |
| 172 | `All check passed` |

## 证据索引

### 主运行与 IR 提取

| 文件 | 内容 |
|---|---|
| `code/logs/vecadd-dev-command.log` | 命令、环境变量、cwd、Python 版本、shim 路径 |
| `code/logs/vecadd-dev-dump.log` | 同一次运行的完整 stdout+stderr |
| `code/logs/vecadd-dev-exit.log` | `exit_code=0` |
| `code/logs/step2-extract-ir.log` | 切分校验，`EXTRACT=PASS` |
| `code/ir/step1-tvm-ir.tir` | TVM TIR（17 行） |
| `code/ir/step2-initial-npuir.mlir` | codegen 刚产出的初始 npuir（35 行） |
| `code/ir/step2a-after-canonicalize.mlir` | canonicalize 后（33 行） |
| `code/ir/step2b-after-adapt-triton-kernel.mlir` | adapt 后（33 行） |
| `code/ir/step3-final-npuir.mlir` | final npuir（32 行） |
| `code/logs/step2-canonicalize-vs-adapt.diff` | 证明 adapt 阶段 IR 不变 |
| `code/logs/step3-initial-vs-final.diff` | 初始 vs final 的全部差异 |

### 下游工具链

| 文件 | 内容 |
|---|---|
| `code/logs/step0-env.log` | 设备/版本/仓库 revision |
| `code/logs/step4-help-passes.log` | `bishengir-opt` 的 pass 名核对 |
| `code/logs/step4-roundtrip.err` | 解析/打印 exit 0 |
| `code/ir/step4-hivm-infer-mem-scope.mlir` | 同一 final npuir 显式标注 GM/UB |
| `code/logs/step4-infer-mem-scope.diff` | 标注前后的差异 |
| `code/logs/step4-hivm-flatten-ops.err` | 阴性对照：exit 134（LLVM abort），保留未隐藏 |
| `code/logs/step4-hivm-plan-memory.err` | 阴性对照：单独跑不适用，IR 未变 |
| `code/logs/step5-bishengir-compile-passdump.log` | 完整逐 pass dump（5983 行、82 个 pass 名） |
| `code/logs/step5-bishengir-compile-exit.log` | exit 0、ELF 2784 B、sha256、符号 |
| `code/ir/step5-final-hivm-lowered.mlir` | `ConvertHIVMToStandard` 后的低层 HIR |
| `code/artifacts-vecadd.o` | 设备 ELF，`bae0063...`，`main` 340 B |

### 原 JIT 留痕与 TileLangIR 核实

| 文件 | 内容 |
|---|---|
| `code/logs/jit-compile-argv.log` | JIT 对 `bishengir-compile` 的两次调用（`--version` + 真实编译） |
| `code/logs/jit-kernel.npuir` | 原 JIT 输入副本（3105 B，比 final npuir 少末尾换行） |
| `code/logs/jit-kernel.o` | 原 JIT 设备产物（2784 B，与复编相同） |
| `code/logs/jit-libkernel.so` | 原 JIT host 产物（4720 B，与复编相同） |
| `code/logs/step7-jit-vs-recompiled.log` | 原 JIT vs 复编的哈希与逐字节比较 |
| `code/logs/step6-tilelangir-load-check.log` | `libtilelangir.so` 已加载；两个 pass descriptor 存在 |
| `code/ir/out-of-path-tilelangir-cv-split.mlir` | 出路径运行 CVSplit，输出与输入相同 |
| `code/ir/out-of-path-tilelangir-vectorize.mlir` | 出路径运行 Vectorize，输出与输入相同 |
| `code/logs/out-of-path-tilelangir-*.err` | 两次运行 exit 0 |

## 口径（重要）

- **同一进程**：`ir/step1-*`、`ir/step2*`、`ir/step3-*` 来自同一次 Python 运行，
  不是多次重编后拼出来的。
- **同一份 MLIR，另行编译**：`ir/step4-*`、`ir/step5-*` 和
  `artifacts-vecadd.o` 是把 `ir/step3-final-npuir.mlir` 再次喂给同一个
  `bishengir-compile` 得到的；它们是独立调用，不是同一个文件句柄。原 JIT 的
  argv 和产物已由 shim 留痕，且与复编逐字节相同。
- **设备正确性**以原 JIT 那次为准：`exit 0` + `All check passed`。
- **未做性能测试**；未做 Day 8 的 Agent 生成任务。
- 只覆盖 N=4096、f32、单 block；tail 分支、多 block、其他 dtype 未验证。
- `npu-smi` 的物理 NPU 编号与 `torch.npu.set_device(0)` 的对应关系未单独验证。
- pass dump 的行号不是逐字节稳定的；引用 pass 时以名称为准，行号只用于定位
  随附日志。

完整命令见 [`code/commands.md`](code/commands.md)，逐文件说明见
[`code/README.md`](code/README.md)。
