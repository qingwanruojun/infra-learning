# Day 7 code 索引

这个目录保存 Day 7 的全部可执行脚本、提取出的 IR、原始日志和设备产物。
主运行没有在这里重新跑；所有文件来自 2026-11-02 的原实验。阅读顺序建议：

1. 先看根目录 [`../README.md`](../README.md) 和
   [`../note/day7学习笔记.md`](../note/day7学习笔记.md)；
2. 复核数字时，从本文件的“关键日志”表进入；
3. 要重跑时，看下面的“复现”。

## 目录

```text
code/
├── README.md                    本文件：逐文件索引与复现
├── commands.md                  完整命令、退出码与口径
├── run_vecadd_dev.sh            主运行：Developer 模式 + IR dump + shim
├── extract_ir.py                从主日志切分 TVM/初始/final IR
├── run_bishengir_checks.sh      下游检查、阴性对照、复编与产物比对
├── shims/
│   └── bishengir-compile        记录 JIT argv，复制临时 kernel.o/libkernel.so
├── ir/                          各阶段 IR（切分结果 + 下游 pass 输出）
├── logs/                        原始日志、diff、阴性对照、JIT 留痕
├── artifacts-vecadd.o           复编设备 ELF（2784 B）
└── libartifacts-vecadd.so       复编 host 侧 arm64 .so（4720 B）
```

## 脚本

| 文件 | 作用 | 关键输出 |
|---|---|---|
| `run_vecadd_dev.sh` | 在 `/mnt/workspace/learn/tilelang-ascend` 跑 `vec_add_1d.py`；`TILELANG_ASCEND_MODE=Developer`、`TILELANG_DUMP_IR=TRUE`；`pipefail` + `PIPESTATUS[0]` 防止 `tee` 掩盖退出码 | `logs/vecadd-dev-{dump,command,exit}.log`、`logs/jit-*` |
| `extract_ir.py` | 按 banner 和 pass marker 切分同一次日志 | `ir/step1-*`、`ir/step2*`、`ir/step3-*`；末行 `EXTRACT=PASS` |
| `run_bishengir_checks.sh` | roundtrip、`--hivm-infer-mem-scope`、两个阴性对照、TileLangIR 出路径对照、完整 `bishengir-compile`、产物比对 | `ir/step4-*`、`ir/step5-*`、`logs/step4-*`、`logs/step5-*`、`logs/step7-*`、`artifacts-vecadd.o` |
| `shims/bishengir-compile` | PATH shim；记录 JIT 的 argv/cwd/返回码/哈希，并在临时目录删除前复制 `kernel.o`、`libkernel.so`；捕获失败只 warning，不改变真实返回码/信号 | `logs/jit-compile-argv.log`、`logs/jit-kernel.*`、`logs/jit-libkernel.so` |

`run_vecadd_dev.sh` 里的 `REPO=/mnt/workspace/learn/tilelang-ascend` 和
`set_env.sh=/home/developer/Ascend/cann-9.0.0/set_env.sh` 是原机器路径；换机器先改。

## IR

| 文件 | 阶段 | 说明 |
|---|---|---|
| `ir/step1-tvm-ir.tir` | TVM TIR | 1 个 block；`cid` 已折叠；`T.npuir_add` 保留 |
| `ir/step2-initial-npuir.mlir` | 初始 npuir | codegen 刚产出；vadd 在 `:29`；尚无 GM/UB address space |
| `ir/step2a-after-canonicalize.mlir` | canonicalize 后 | 删除未用的 `get_block_idx/trunci`，整理常量/SSA |
| `ir/step2b-after-adapt-triton-kernel.mlir` | adapt 后 | 与 step2a 逐行相同（仅差 banner） |
| `ir/step3-final-npuir.mlir` | final npuir | `pipeline.run()` 返回值；vadd 在 `:26` |
| `ir/step4-hivm-infer-mem-scope.mlir` | 显式 GM/UB | 只跑 `--hivm-infer-mem-scope`，成功 |
| `ir/step4-hivm-plan-memory.mlir` | 阴性对照 | 单独跑 `--hivm-plan-memory`；IR 未变，见下方 err |
| `ir/step4-hivm-flatten-ops.mlir` | 阴性对照 | **0 B**：命令 abort，未产生有效输出 |
| `ir/step5-final-hivm-lowered.mlir` | 低层 HIR | `ConvertHIVMToStandard` 后；两次 load、vadd、store、flag/barrier；vadd 在 `:44` |
| `ir/bishengir-opt-roundtrip.mlir` | 解析/打印 | 与 final npuir 基本一致 |
| `ir/out-of-path-tilelangir-{cv-split,vectorize}.mlir` | 出路径对照 | 与输入逐行相同；证明两个 pass 是 placeholder |

## 关键日志

| 文件 | 内容 | 关键结果 |
|---|---|---|
| `logs/vecadd-dev-dump.log` | 同一次运行的完整 stdout+stderr（172 行 / 14967 B） | 边界行 6/25/62/97/132/165/172；末行 `All check passed` |
| `logs/vecadd-dev-command.log` | 命令、环境变量、cwd、Python 版本、shim 路径 | 运行条件可复查 |
| `logs/vecadd-dev-exit.log` | Python 进程退出码 | `exit_code=0` |
| `logs/step0-env.log` | 设备、CANN、工具版本、仓库 revision | 910B3 / CANN 9.0.0 / bishengir 1.1.0 |
| `logs/step2-extract-ir.log` | 切分校验 | `EXTRACT=PASS` |
| `logs/step2-canonicalize-vs-adapt.diff` | 两个 pass 的差异 | 只有 banner 差异 |
| `logs/step3-initial-vs-final.diff` | 初始 vs final npuir | 全部差异（删除未用 op、常量/SSA 整理） |
| `logs/step4-commands.log` | 下游命令记录 | 命令可复现 |
| `logs/step4-help-passes.log` | `bishengir-opt` pass 名核对 | 确认 pass 存在 |
| `logs/step4-roundtrip.err` | 解析/打印 | `roundtrip_exit=0` |
| `logs/step4-hivm-infer-mem-scope.err` | 显式 GM/UB | exit 0 |
| `logs/step4-infer-mem-scope.diff` | 标注前后差异 | 出现 `#hivm.address_space<gm/ub>` |
| `logs/step5-bishengir-compile-passdump.log` | 完整逐 pass dump（5983 行 / 82 个 pass 名） | 无 TileLangIR pass；PlanMemory 在 `:1157/:5202/:5322` |
| `logs/step5-bishengir-compile-exit.log` | 编译退出码与 ELF 信息 | exit 0；2784 B；machine `0x1029`；`main` 340 B |
| `logs/jit-compile-argv.log` | JIT 对 `bishengir-compile` 的两次调用 | 第 1 条 `--version`，第 2 条真实编译 |
| `logs/step6-tilelangir-load-check.log` | `libtilelangir.so` 加载与 pass descriptor | 库已加载；CVSplit/Vectorize descriptor 可构造 |
| `logs/step7-jit-vs-recompiled.log` | 原 JIT 与复编产物哈希/逐字节比较 | `JIT_VS_RECOMPILED=IDENTICAL`、`HOST_SO=IDENTICAL` |

## 阴性对照与出路径材料（保留不隐藏）

| 文件 | 现象 | 为什么不删 |
|---|---|---|
| `logs/step4-hivm-flatten-ops.err` | exit 134（LLVM `SmallVector` abort） | 证明该 pass 不能单独用于这份输入；完整流水线里由编译器自行调度 |
| `ir/step4-hivm-flatten-ops.mlir` | 0 B | 上述 abort 的后果，不要当成有效 IR |
| `logs/step4-hivm-plan-memory.err` | exit 0，但 stderr 报 “Alloc buffer not at UB space / read before first write” | 证明单独跑 PlanMemory 缺前置 `InferHIVMMemScope`；完整流水线行为正常 |
| `ir/step4-hivm-plan-memory.mlir` | 与输入相同 | 同上，IR 未变 |
| `logs/out-of-path-tilelangir-*.err` | exit 0 | 证明两个 TileLangIR pass 单独跑也不改 IR |
| `ir/out-of-path-tilelangir-*.mlir` | 与输入逐行相同 | 与源码里的 placeholder 结论互相印证 |

## 产物

| 文件 | 来源 | 关键属性 |
|---|---|---|
| `artifacts-vecadd.o` | 把 `ir/step3-final-npuir.mlir` 再次喂给 `bishengir-compile` | 2784 B；sha256 `bae006312aa6e3a19dd94f998e296f8564d63ddbaab98d8fadf17c56aec3b3ea`；与原 JIT `kernel.o` 相同 |
| `libartifacts-vecadd.so` | 同上一次编译附带生成 | 4720 B；sha256 `84f8f137dbf9cc86e6547abb94ade6e68afa306e065e1633bbcfd374a62d4083`；与原 JIT `libkernel.so` 相同 |
| `logs/jit-kernel.o` | 原 JIT 临时目录副本（shim 捕获） | 2784 B；同 sha256 |
| `logs/jit-libkernel.so` | 原 JIT host 产物副本 | 4720 B；同 sha256 |
| `logs/jit-kernel.npuir` | 原 JIT 输入副本 | 3105 B；比 `ir/step3-final-npuir.mlir` 少末尾一个换行 |

## 复现

在原 NPU 机器、原路径可用的前提下：

```bash
cd infra-learning/day7/code

bash run_vecadd_dev.sh          # 主运行，约数十秒
python3 -I extract_ir.py        # 期望末行 EXTRACT=PASS
bash run_bishengir_checks.sh    # 下游检查，生成/覆盖 ir/ 与 logs/
```

说明：

- `run_vecadd_dev.sh` 会 `source` CANN 9.0.0 的 `set_env.sh`，设置 Developer 模式
  和 `TILELANG_DUMP_IR=TRUE`，并把 `shims/` 放到 `PATH` 最前面。
- `run_bishengir_checks.sh` 不下载任何东西，只使用现有 CANN 工具；它会生成或覆盖
  `ir/` 下的下游 IR 和 `logs/` 下的对应日志，并生成 `artifacts-vecadd.o`、
  `libartifacts-vecadd.so`。
- 该脚本只负责生成证据，不设总断言；成功与否以下列文件为准：
  `logs/vecadd-dev-exit.log`（`exit_code=0`）、
  `logs/step4-roundtrip.err`、`logs/step4-hivm-infer-mem-scope.err`、
  `logs/step5-bishengir-compile-exit.log`（`bishengir_compile_exit=0`）、
  `logs/step7-jit-vs-recompiled.log`（两个 `IDENTICAL`）。

完整命令、每个命令的退出码与口径见 [`commands.md`](commands.md)。

## 口径（重要）

- `ir/step1-*`、`ir/step2*`、`ir/step3-*` 来自**同一个 Python 进程**，不是多次
  重编拼出来的。
- `ir/step4-*`、`ir/step5-*`、`artifacts-vecadd.o` 是把 `ir/step3-final-npuir.mlir`
  **再次**喂给同一个 `bishengir-compile` 得到的；它们是独立调用，不是同一个文件
  句柄。原 JIT 的 argv 和产物由 shim 留痕，且与复编结果逐字节相同。
- 设备正确性以原 JIT 那次为准：exit 0 + `All check passed`。
- 只覆盖 N=4096、f32、单 block；tail 分支、多 block、其他 dtype 未验证。
- 没有测性能；没有做 Day 8 的 Agent 生成任务。
