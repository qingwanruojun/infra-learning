# 03 · VecAdd 集成例子：编译、加载、执行、校验

## 文件分工

来源目录：`/mnt/workspace/learn/AscendNPU-IR/bishengir/test/Integration/HIVM/VecAdd/`（未构建的主 clone，commit `87ea6c4`）。

| 文件 | 作用 | 关键位置 |
|---|---|---|
| `add.mlir` | 设备侧 kernel IR（已经是 HIVM 层） | `add.mlir:2-9` |
| `main.cpp` | Host wrapper：初始化 runtime、注册 kernel、搬数据、发射、回读、打印 | 上游 `main.cpp:121-190`；本机修改后行号见第 3 节 |
| `CMakeLists.txt` | 定义 `bishengir-npu-hivm-vec-add`，链接 `runtime`、`ascendcl` | `CMakeLists.txt:1-21` |
| `README.md` | 两步流程：`bishengir-compile` 出 `kernel.o`，再跑 host 可执行文件 | `README.md:4,34-40` |
| `lit.local.cfg` + `main.cpp` 顶部 RUN | lit/FileCheck 自动测试入口 | `main.cpp:18-40` |

本机把这三个文件复制到了 `code/vecadd/`。`add.mlir` 未改；`main.cpp` **已修改**：加了运行时返回码检查和逐元素比较（见第 4 节），另外支持用 `argv[1]/argv[2]` 选择内核路径与符号。`hostshim/` 仍是本机 CANN 头文件布局的适配（原因见 `note/04-troubleshooting.md`）。下文的“上游行号”指 `AscendNPU-IR` 副本里的原文件，“本机行号”指 `code/vecadd/main.cpp`。

## 1. add.mlir 在说什么

```mlir
func.func @add(%arg0: memref<16xi16, #hivm.address_space<gm>>,
               %arg1: memref<16xi16, #hivm.address_space<gm>>,
               %arg2: memref<16xi16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %alloc   = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
  hivm.hir.load  ins(%arg0) outs(%alloc)
  %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
  hivm.hir.load  ins(%arg1) outs(%alloc_0)
  %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
  hivm.hir.vadd  ins(%alloc, %alloc_0) outs(%alloc_1)
  hivm.hir.store ins(%alloc_1) outs(%arg2)
  return
}
```

- `gm` = global memory（HBM），`ub` = Unified Buffer（片上）。
- 三个参数都是 `memref<16xi16>`，16 个 int16，共 32 字节。
- `hacc.entry` / `hacc.function_kind<DEVICE>` 标记设备入口（`add.mlir:2`）。
- 语义：A/B 从 GM 搬到 UB，在 UB 做向量加，结果搬回 GM。

## 2. 编译：add.mlir → kernel.o

实际命令（在 `code/vecadd/` 下，先 `source /home/developer/Ascend/cann-9.0.0/set_env.sh`）：

```bash
bishengir-compile add.mlir --enable-hivm-compile -o kernel.o
```

- 退出码：**0**（`code/logs/step6-compile-hivm.log`；命令本身无其他输出）。
- 产物：`code/vecadd/kernel.o`，2648 字节。
- `file kernel.o`：`ELF 64-bit LSB executable, *unknown arch 0x1029*, statically linked, not stripped`（`code/logs/step6-artifacts.log`）。
- `readelf -sW kernel.o`：全局符号 `add`，类型 `FUNC`，大小 212 字节；另有 `add$local`、`add__`（`code/logs/step6-artifacts.log`）。这说明 host 侧按名字 `"add"` 注册是能对上的。
- 未显式传 `--target`；按 `examples.md:32`，默认目标为 `Ascend910B1`。本次在该默认值下编译并在 910B3 上执行成功（实测），但没有单独验证其他 `--target` 取值。

### 编译过程里实际跑了哪些 pass

用同一条命令加 `--mlir-print-ir-after-all` 再编一次（输出到 `/tmp/kernel-passdump.o`）：

```bash
bishengir-compile add.mlir --enable-hivm-compile \
  --mlir-print-ir-after-all -o /tmp/kernel-passdump.o \
  > code/logs/step6-hivm-passdump.log 2>&1
```

- 退出码 **0**，IR dump 本体 2135 行（文件首行是命令、末行是 `exit=0`）。
- 本次产物与默认编译的 `kernel.o` sha256 相同（`8648e77e...7f4f`，`code/logs/step6-artifacts.log`），说明该输入下编译产物可复现。
- dump 中出现的 pass（节选）：`ConvertHFusionToHIVM`、`ConvertToHIVMOp`、`OneShotBufferize`、`InferHIVMMemScope`、`HIVMFlattenOps`、`HIVMDecomposeOp`、`PlanMemory`、`SplitMixKernel`、`ConvertHIVMToStandard` 等。

几个可对照的节点：

| 节点 | 日志位置 | 观察 |
|---|---|---|
| `ConvertHFusionToHIVM` | `step6-hivm-passdump.log:32` | 输入已是 HIVM，转换基本直通 |
| `HIVMFlattenOps` | `:1545` | 仍是 `hivm.hir.load/vadd/store` 的高层形式 |
| `PlanMemory` | `:1794` | `memref.alloc` 变成 `hivm.hir.pointer_cast(%c0_i64)` / `(%c32_i64)`，即 UB 内偏移 0 / 32 字节 |
| `HIVMDecomposeOp` | `:1824` 起 | `vadd` 等被拆成 `vadd_1d_int16_t` 等底层调用 |
| `ConvertHIVMToStandard` | `:2097` | 最终 HIR：`set_mask_norm`、`pointer_cast`、`load_gm_to_ubuf_1d_int16_t` / `vadd_1d_int16_t` / `store_ubuf_to_gm_1d_int16_t`、`set_flag`/`wait_flag`、`pipe_barrier` |

> `PlanMemory` 把 alloc 映射成 UB 偏移，是对 dump 文本的直接观察；“偏移 0/32 就是两个 16×i16 输入缓冲区”是按 32 字节大小做的推断，标记为推断而非实测断言。

## 3. Host 侧：加载与执行

本机没有启用 `-DBISHENGIR_BUILD_EXAMPLES=ON`，构建树里也没有 `bishengir-npu-hivm-vec-add` target（`ninja -t targets | grep vec-add` 无输出），所以没有重建整个项目，而是用文档给出的等价编译命令直接编 `main.cpp`：

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
g++ main.cpp -Ihostshim \
  -I${ASCEND_HOME_PATH}/include \
  -I${ASCEND_HOME_PATH}/include/experiment/msprof \
  -I${ASCEND_HOME_PATH}/pkg_inc \
  -L${ASCEND_HOME_PATH}/lib64 -lruntime -lascendcl \
  -Wl,-rpath,${ASCEND_HOME_PATH}/lib64 -o hivm-vec-add
```

- 退出码 **0**（`code/logs/step6-host-build.log`）；产物 `code/vecadd/hivm-vec-add`，26744 字节（加入校验代码后比 10-31 的 26368 字节略大）。
- `ldd` 确认 `libruntime.so`、`libascendcl.so` 解析到 `/home/developer/Ascend/cann-9.0.0/lib64/`（`code/logs/step6-artifacts.log`）。
- 编译参数与 `examples.md:196-207` 一致；`CMakeLists.txt:11-21` 同样是这些 include/lib。

### main.cpp 的执行顺序

以下行号均为本机 `code/vecadd/main.cpp`：

| 步骤 | 代码位置 | 做什么 |
|---|---|---|
| 选择内核/符号 | `:220-222` | `argv[1]` 默认 `./kernel.o`，`argv[2]` 默认 `add` |
| 初始化 | `:226-227` | `aclInit(nullptr)`，返回码检查 |
| 选设备 | `:229-230` | `aclrtSetDevice(0)`；npu-smi 显示该卡的 Chip Logic ID 为 0 |
| 建流 | `:231-233` | `aclrtCreateStream` |
| 读二进制 | `:84-114,116-124` | 读内核文件到内存；空文件会被拒绝 |
| 注册 binary | `:131-136` | `rtDevBinaryRegister`，`magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC` |
| 注册 function | `:138-144` | `rtFunctionRegister(binHandle, stubFunc, stubFunc, ...)`，失败返回 `nullptr` |
| 准备输入 | `:157-177` | 每个 case 单独 `aclrtMalloc` + `aclrtMemcpy(H2D)` |
| 发射 | `:179-183` | `rtKernelLaunch(stubFunc, blockDim=1, args, sizeof(args)=24, nullptr, stream)`，返回码由 `EXPECT_RT_OK` 检查 |
| 同步 | `:185-186` | `aclrtSynchronizeStream` |
| 回读 | `:188-193` | `aclrtMemcpy(D2H)` |
| 比较与打印 | `:195-204` | 打印每行 Expect/Result；`outHost[i] != expectedValue[i]` 时打印具体下标并累计 mismatch |
| 汇总退出 | `:274-281` | mismatch 非零则 `return 1`；全过才打印 `[success] compare output success` |
| 跑两组输入 | `:264-267` | `runCase` 分别跑 `case0-official` 与 `case1-mixed` |

## 4. 实际运行结果

```bash
./hivm-vec-add ./kernel.o add
```

- 退出码 **0**（`code/logs/step6-run-hivm.log`）。
- 两组输入各 16 行，Expect/Result 全部吻合：
  - `case0-official`：`1..16`；
  - `case1-mixed`：`-5,-6,0,10,0,0,0,0,0,0,0,-1,0,0,0,0`。
- 程序末行 `[success] compare output success: 2 cases x 16 elements`。
- 独立数值校验：把两次运行的 Result 列抽出来，与期望的 32 项序列比较，均 `NUMERIC_CHECK=PASS`（`code/logs/step6-numeric-check.log`）。

### “compare output success” 现在是真校验

上游 `main.cpp` 确实只 `printf`，最后一行的 `[success] compare output success` 是无条件打印（上游 `main.cpp:185-189`）；自动化测试靠 lit + FileCheck 匹配输出行（上游 `main.cpp:18-40`）。本地副本已改掉这个问题：

1. `EXPECT_RT_OK` 检查 `rtKernelLaunch` 等运行时调用的返回码（本机 `main.cpp:73-80,179-183`）；
2. `runCase()` 逐元素比较，打印具体下标、期望值和实际值，并累计 mismatch（`:195-204`）；
3. `main()` 在 mismatch 非零时返回 1（`:274-281`），只有全过才打印成功行。

**负例验证**：用同一条 HFusion 流水线把 `add` 换成 `mul`，生成 `code/hfusion-vecadd/bad-kernel.o`，再用同一 host 程序运行：

```bash
./hivm-vec-add ../code/hfusion-vecadd/bad-kernel.o add_single_outlined_0_0_0
```

结果报 `30 mismatched element(s)`，退出码 1（`code/logs/step6-run-negative-control.log`），脚本判定 `NEGATIVE_CONTROL=PASS`。这证明成功行不是无条件打印的。

结论现在有两层独立证据：

1. 程序自身的返回码和逐元素比较（失败会非零退出）；
2. `code/repro.sh` 里另抽 Result 列与期望序列比较的 `NUMERIC_CHECK`。

## 5. 一条命令复现

```bash
bash /mnt/workspace/learn/infra-learning/day6/code/repro.sh
```

脚本依次完成 6 步：

1. 编译 HIVM 内核 `add.mlir → kernel.o`；
2. 编译修好的 host 程序；
3. 在 NPU 上跑 HIVM 内核（两组输入）；
4. 把 HFusion 输入编成设备 ELF（`code/hfusion-vecadd/build.sh`）；
5. 在 NPU 上跑 HFusion 内核；
6. mul 负例必须被拒绝（exit 1）。

最后对两次成功运行做独立数值校验，全部通过才打印 `ALL_CHECKS=PASS`。任一步失败即非零退出。
