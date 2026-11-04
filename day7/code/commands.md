# Day 7 命令清单

所有命令均在 `/mnt/workspace/learn/infra-learning/day7/code` 下执行；只使用现有
910B3 / CANN 9.0.0 环境，没有安装或升级工具，也没有修改 TileLang 或
AscendNPU-IR 上游源码。该路径是作者工作区；zip 解压后请换成你的
`.../day7/code`。脚本输出目录按脚本位置计算，但 `run_vecadd_dev.sh` 里的
`REPO` 和 `set_env.sh` 路径需要按机器修改。

## 0. 环境

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
```

环境、版本、设备、仓库 revision 见 `logs/step0-env.log`。

## 1. 主实验：同一次 Developer 模式运行

```bash
./run_vecadd_dev.sh
```

脚本等价于：

```bash
cd /mnt/workspace/learn/tilelang-ascend
export TILELANG_ASCEND_MODE=Developer
export TILELANG_DUMP_IR=TRUE
python3 examples/elementwise/vec_add_1d.py 2>&1 | tee \
  /mnt/workspace/learn/infra-learning/day7/code/logs/vecadd-dev-dump.log
```

脚本内部使用 `set -o pipefail`，并用 `PIPESTATUS[0]` 记录 Python 进程退出码，
避免 `tee` 掩盖失败。

脚本还会在 `source set_env.sh` 之后把 `code/shims` 放到 `PATH` 最前面，并设置：

```bash
REAL_BISHENGIR_COMPILE="$(command -v bishengir-compile)"
export DAY7_REAL_BISHENGIR_COMPILE="$REAL_BISHENGIR_COMPILE"
export DAY7_JIT_CAPTURE_DIR="$OUT_DIR"
export PATH="$HERE/shims:$PATH"
```

`shims/bishengir-compile` 记录 JIT 的 argv，并在 JIT 删除临时目录前复制
`kernel.o` / `libkernel.so`。它在子进程被信号杀死时以同一信号退出，避免破坏
`jit_npu.py` 的 SIGSEGV 重试判断。

产物：

- `logs/vecadd-dev-dump.log`：完整 stdout+stderr（`2>&1`，保留 IR 打印与
  Python 输出的先后顺序）；
- `logs/vecadd-dev-command.log`：命令、环境变量、cwd、Python 版本、shim 路径；
- `logs/vecadd-dev-exit.log`：`exit_code=0`；
- `logs/jit-compile-argv.log`：JIT 对 `bishengir-compile` 的两次调用
  （`--version` 和真实编译），含 argv、cwd、退出码、输入/产物 sha256；
- `logs/jit-kernel.npuir`、`logs/jit-kernel.o`、`logs/jit-libkernel.so`：
  原 JIT 临时目录里的输入和设备/host 产物副本。

## 2. 从同一次运行中切分三个 IR 阶段

```bash
python3 -I extract_ir.py
```

切分边界是 `lower.py` 打印的 banner 和 MLIR pass manager 的 dump marker：

- `====== TVM IR ======` → `====== npuir ======`：TVM TIR；
- `====== npuir ======` → `// -----// IR Dump After Canonicalizer`：初始 npuir；
- `====== final npuir ======` → `AscendNPU IR compile success:`：最终 npuir；
- 两个 `IR Dump After ...` 区间分别保存 canonicalize 后和 adapt-triton-kernel 后。

输出见 `ir/step1-*.tir`、`ir/step2*.mlir`、`ir/step3-final-npuir.mlir`；
切分校验见 `logs/step2-extract-ir.log`（末行 `EXTRACT=PASS`）。

## 3. 下游工具链检查

```bash
./run_bishengir_checks.sh
```

脚本先用 `bishengir-opt --version` 和 `--help` 核对工具版本与 pass 名，
再执行：

```bash
INPUT=ir/step3-final-npuir.mlir

# 解析/打印
bishengir-opt "$INPUT" > ir/bishengir-opt-roundtrip.mlir

# 本次实际适用：把隐式 GM/UB 标成显式 address space
bishengir-opt "$INPUT" --hivm-infer-mem-scope > ir/step4-hivm-infer-mem-scope.mlir

# 阴性对照：记录失败而不是隐藏
bishengir-opt "$INPUT" --hivm-flatten-ops   # exit 134 (LLVM abort)
bishengir-opt "$INPUT" --hivm-plan-memory   # exit 0 但 stderr 报错，IR 未变

# 出路径对照：TileLangIR 的两个 pass 本身是 placeholder
TLOPT=/mnt/workspace/learn/tilelang-ascend/build/tilelangir/tools/tilelangir-opt/tilelangir-opt
"$TLOPT" "$INPUT" --tilelangir-cv-split  > ir/out-of-path-tilelangir-cv-split.mlir
"$TLOPT" "$INPUT" --tilelangir-vectorize > ir/out-of-path-tilelangir-vectorize.mlir

# 同一份 final npuir → 设备 ELF，并抓逐 pass dump
bishengir-compile "$INPUT" \
  --enable-auto-multi-buffer=true \
  --disable-hivm-auto-inject-sync=false \
  --enable-triton-kernel-compile=true \
  --enable-hivm-compile=true \
  --mlir-print-ir-after-all \
  -o artifacts-vecadd.o > logs/step5-bishengir-compile-passdump.log 2>&1
```

关键结果：

| 命令 | 退出码 | 证据 |
|---|---|---|
| `bishengir-opt INPUT` | 0 | `logs/step4-roundtrip.err` 只有 `roundtrip_exit=0` |
| `--hivm-infer-mem-scope` | 0 | `logs/step4-hivm-infer-mem-scope.err` |
| `--hivm-flatten-ops` | 134 | `logs/step4-hivm-flatten-ops.err`（LLVM `SmallVector` abort） |
| `--hivm-plan-memory` | 0 | `logs/step4-hivm-plan-memory.err`（有 error，IR 未变；不能单独作为该输入的转换） |
| `--tilelangir-cv-split` / `--tilelangir-vectorize` | 0 / 0 | 输出除末尾空行外与输入完全相同 |
| `bishengir-compile ...` | 0 | `logs/step5-bishengir-compile-exit.log`，ELF 2784 B |

脚本最后还把原 JIT 产物与另行编译的产物做比较：

```bash
sha256sum logs/jit-kernel.npuir logs/jit-kernel.o artifacts-vecadd.o
sha256sum logs/jit-libkernel.so libartifacts-vecadd.so
cmp -s logs/jit-kernel.o artifacts-vecadd.o \
  && echo JIT_VS_RECOMPILED=IDENTICAL \
  || echo JIT_VS_RECOMPILED=DIFFERENT
cmp -s logs/jit-libkernel.so libartifacts-vecadd.so \
  && echo HOST_SO=IDENTICAL \
  || echo HOST_SO=DIFFERENT
```

实测两个 `kernel.o` 的 sha256 都是
`bae006312aa6e3a19dd94f998e296f8564d63ddbaab98d8fadf17c56aec3b3ea`，
host `.so` 都是
`84f8f137dbf9cc86e6547abb94ade6e68afa306e065e1633bbcfd374a62d4083`；
结果见 `logs/step7-jit-vs-recompiled.log`
（`JIT_VS_RECOMPILED=IDENTICAL`、`HOST_SO=IDENTICAL`）。

## 4. 常用定位命令

```bash
# 看三阶段边界
grep -n "======\|IR Dump After\|AscendNPU IR compile success\|All check passed" \
  logs/vecadd-dev-dump.log

# 看 pass 顺序
grep -oE 'IR Dump After [A-Za-z0-9_]+' logs/step5-bishengir-compile-passdump.log \
  | sed 's/IR Dump After //' | uniq

# 确认实际 dump 中没有 TileLangIR pass
grep -icE 'tilelangir|cvsplit|cv-split' logs/step5-bishengir-compile-passdump.log

# 看最终 HIVM/HIR intrinsic
grep -nE 'load_gm_to_ubuf|vadd_1d|store_ubuf_to_gm|set_flag|wait_flag|pipe_barrier' \
  logs/step5-bishengir-compile-passdump.log | tail -30
```

## 5. 口径

- 第 1 步的 TVM IR / 初始 npuir / final npuir 来自**同一个 Python 进程**。
- 原 JIT 的 argv 与临时 `kernel.o` 现在由 shim 留痕：见
  `logs/jit-compile-argv.log`、`logs/jit-kernel.o`。`artifacts-vecadd.o` 是
  **同一份 final npuir** 再次喂给 `bishengir-compile` 得到的；它与原 JIT
  `kernel.o` 逐字节相同（同 sha256），但仍是两次独立调用，不是同一个文件句柄。
- 原 JIT 输入 `logs/jit-kernel.npuir` 是 3105 B，`ir/step3-final-npuir.mlir`
  是 3106 B，两者只差末尾一个换行；产物相同。
- 没有测性能，也没有做 Day 8 的 Agent 生成任务。
