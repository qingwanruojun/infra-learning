# Day 6 学习报告 · MLIR 与 AscendNPU-IR 入门

日期：2026-10-31，修订 2026-11-02
范围：Day 6（不展开 Day 7 的 TVM IR → TileLangIR → HIVM 追踪）

> 2026-11-02 修订说明：本版修好了 host 校验（检查 `rtKernelLaunch` 返回值并在 C++ 里逐元素比较），把输入扩成两组（含负数/零/边界值），并补了一条“同一份 HFusion 输入一路编到 ELF、上卡、数值校验”的实验。第 2.4、2.5 节和验收/日志索引均改用本次新日志 `code/logs/step6-*.log`，不再引用 10-31 的运行证据。

## 1. 目标与验收标准

目标：

1. 用最小可运行 MLIR 例子讲清 dialect、operation、pass、SSA；
2. 读 architecture 与 quick_start，理清 HFusion/HIVM 职责与衔接；
3. 读 VecAdd 集成例子并解释编译/加载/执行/校验；
4. 亲手复跑 `add.mlir → 设备产物 → NPU 执行 → 结果验证`；
5. 能口头讲清 op → pass → HFusion/HIVM → NPU 产物的链路。

验收：不依赖 TileLang Python 示例，手工完成 VecAdd 全链路；能区分 HFusion 与 HIVM。

## 2. 实测

### 2.1 环境（实测）

- 设备：Ascend 910B3，`npu-smi` 显示 NPU ID 7、Chip Logic ID 0、Health OK、无运行进程。
- CANN：`/home/developer/Ascend/cann-9.0.0`，`npu-smi 25.5.0`。
- `bishengir-compile` / `bishengir-opt`：`/home/developer/Ascend/cann-9.0.0/bin/`，版本 `1.1.0 (428ab8fdab46 2026-04-30)`，LLVM 19.1.7。
- `mlir-opt`：`/mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR/build/bin/mlir-opt`，LLVM 19.1.7。
- 源码：`/mnt/workspace/learn/AscendNPU-IR` commit `87ea6c4`（未构建）；已构建副本 commit `31f690369`。
- 证据：`code/logs/step0-env.log`。

### 2.2 最小 MLIR 例子（实测）

文件 `code/mlir-demo/demo.mlir`。命令与结果：

```bash
MLIR_OPT=/mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR/build/bin/mlir-opt
$MLIR_OPT code/mlir-demo/demo.mlir                              # exit 0
$MLIR_OPT --pass-pipeline='builtin.module(func.func(canonicalize,cse))' \
    code/mlir-demo/demo.mlir                                    # exit 0
```

- 运行前：`code/logs/step1-before.mlir`
- 运行后：`code/logs/step1-after.mlir`
- 逐 pass：`code/logs/step1-ir-after-all.log`
- 退出码：`code/logs/step1-parse.log`、`code/logs/step1-pass.log`（均 `exit=0`）

关键变化（实测）：

- `arith.addi %c2, %c3` → `arith.constant 5`（常量折叠）；
- 两条相同的 `arith.muli %arg0, %c5` → 一条（CSE）；
- `arith.addi %0, %0` 保留，因为 canonicalize 在 CSE 之前已跑完；这说明 pass 顺序会影响中间形态。

概念落点：dialect 看 `module`（builtin）、`func.func`（func）、`arith.*`（arith）；operation 看 `demo.mlir:7` 的 `arith.addi`；SSA 看 `%sum` 定义一次、使用两次；pass 看上面的 pipeline。详见 `note/01-mlir-core-concepts.md`。

### 2.3 HFusion → HIVM（实测）

```bash
bishengir-opt -convert-hfusion-to-hivm code/mlir-demo/hfusion-elemwise.mlir \
  > code/mlir-demo/hfusion-elemwise.hivm.mlir        # exit 0
```

- 输入 `code/mlir-demo/hfusion-elemwise.mlir:5-13`：3 条 `linalg.elemwise_unary`（exp/abs/log）。
- 输出 `code/mlir-demo/hfusion-elemwise.hivm.mlir:3-5`：`hivm.hir.vexp`、`hivm.hir.vabs`、`hivm.hir.vln`。
- 证据：`code/logs/step5-hfusion-to-hivm.log`（`exit=0`）。

### 2.4 VecAdd 全链路（实测，2026-11-02 重跑）

命令（`code/vecadd/` 下，已 source CANN）：

```bash
bishengir-compile add.mlir --enable-hivm-compile -o kernel.o
g++ main.cpp -Ihostshim -I${ASCEND_HOME_PATH}/include \
  -I${ASCEND_HOME_PATH}/include/experiment/msprof -I${ASCEND_HOME_PATH}/pkg_inc \
  -L${ASCEND_HOME_PATH}/lib64 -lruntime -lascendcl \
  -Wl,-rpath,${ASCEND_HOME_PATH}/lib64 -o hivm-vec-add
./hivm-vec-add ./kernel.o            # argv[1]=kernel 路径，默认 ./kernel.o
```

结果：

| 步骤 | 退出码 | 产物/输出 | 证据 |
|---|---|---|---|
| 设备编译 | 0 | `code/vecadd/kernel.o`，2648 B，ELF machine `0x1029`，全局 `add` 符号 212 B | `code/logs/step6-compile-hivm.log`、`code/logs/step6-artifacts.log` |
| host 编译 | 0 | `code/vecadd/hivm-vec-add`，26744 B（新增校验代码后比 10-31 的 26368 B 大），链接 CANN `libruntime`/`libascendcl` | `code/logs/step6-host-build.log`、`code/logs/step6-artifacts.log` |
| NPU 执行（HIVM 内核） | 0 | 两组输入各 16 行，Expect/Result 全部相同；末行 `[success] compare output success: 2 cases x 16 elements` | `code/logs/step6-run-hivm.log` |
| 数值校验 | 0 | 两组的 32 个 Result 与期望序列完全一致，`NUMERIC_CHECK=PASS` | `code/logs/step6-numeric-check.log` |

两组输入：

- `case0-official`：`{0..15} + {1×16}` → `1..16`（官方 VecAdd 用例）；
- `case1-mixed`：含负数、零、正负抵消和 int16 边界值，且保证和仍在 int16 范围内：

```text
a        = {-10,-1,0,3,-7,12,-100,100,-1234,1234,32767,-32768,42,-42,7,0}
b        = {5,-5,0,7,7,-12,100,-100,1234,-1234,-32767,32767,-42,42,-7,0}
expected = {-5,-6,0,10,0,0,0,0,0,0,0,-1,0,0,0,0}
```

`code/logs/step6-numeric-check.log` 中两组的实测序列为
`1..16,-5,-6,0,10,0,0,0,0,0,0,0,-1,0,0,0,0`，与期望一致。

一致性：本次 HIVM 内核的 sha256 仍为 `8648e77e...7f4f`（`code/logs/step6-artifacts.log`），与 10-31 的两次编译相同；改动只发生在 host 程序，不改变设备产物。

编译过程（2026-11-02 重抓）：`--mlir-print-ir-after-all` 得到 2135 行 IR dump（`code/logs/step6-hivm-passdump.log`，文件另含 1 行命令和 1 行 `exit=0`），出现 `ConvertHFusionToHIVM`、`OneShotBufferize`、`InferHIVMMemScope`、`HIVMFlattenOps`、`HIVMDecomposeOp`、`PlanMemory`、`SplitMixKernel`、`ConvertHIVMToStandard` 等。`PlanMemory` 之后，`memref.alloc` 变成 UB 偏移 0 / 32 的 `hivm.hir.pointer_cast`（`:1794`）。最终 HIR（`:2097-2136`）出现 `set_mask_norm`、`load_gm_to_ubuf_1d_int16_t`、`vadd_1d_int16_t`、`store_ubuf_to_gm_1d_int16_t`、`set_flag/wait_flag`、`pipe_barrier`。

### 2.5 host 校验已修好，并用负例证明不是空校验（实测）

上游 `main.cpp` 的问题（`AscendNPU-IR/.../VecAdd/main.cpp`）：

- `:170` 直接调用 `rtKernelLaunch(...)`，没有接收返回值；
- `:185-187` 的循环只 `printf`，没有比较；
- `:189` 无条件打印 `[success] compare output success`。

本地 `code/vecadd/main.cpp` 的修改：

| 位置 | 修改 |
|---|---|
| `main.cpp:73-80` | 新增 `EXPECT_RT_OK` 宏：运行时调用返回非 `RT_ERROR_NONE` 就打印错误码并 `exit(1)` |
| `main.cpp:179-183` | 用 `EXPECT_RT_OK` 检查 `rtKernelLaunch` 的返回码，不再忽略 |
| `main.cpp:195-204` | 逐元素比较 `outHost[i] != expectedValue[i]`，累计 mismatch 并打印具体下标/期望/实际值 |
| `main.cpp:211-217` | 每个 case 单独报告 `all kElems match` 或 `n/kElems elements mismatch` |
| `main.cpp:274-281` | 任一 mismatch 非零则 `return 1`；只有全部通过才打印 `compare output success` |

负例（证明校验不是恒真）：用同一条 HFusion 流水线把 `linalg.elemwise_binary` 从 `add` 换成 `mul`，编出 `code/hfusion-vecadd/bad-kernel.o`，再用同一 host 程序运行。结果 `30 mismatched element(s)`，退出码 1，脚本判定 `NEGATIVE_CONTROL=PASS`（`code/logs/step6-run-negative-control.log`）。

注意：负例第一次尝试时，HFusion 层的 mul 因返回值未使用被 DCE，内核只输出全零，这只能证明“能发现错”，不能证明“能发现乘错”；因此改成保留完整 `linalg.elemwise_binary{mul}` 的负例，最终内核里确实是 `hivm.hir.vmul`（`code/hfusion-vecadd/mul-hfusion.device.hivm.mlir:25`）。

### 2.6 同一份 HFusion 输入一路编到 ELF 并上卡（实测）

输入 `code/hfusion-vecadd/add-hfusion-host.mlir`：单个 `linalg.elemwise_binary {fun = #linalg.binary_fn<add>}`，`tensor<16xi16>`，HOST 函数 `@add`。

工具链是两段拼接的，原因是 CANN 9.0.0 包内的 1.1.0 编译器在本机对任何 `--enable-hfusion-compile` 输入都 SIGSEGV（`code/logs/step6-packaged-hfusion-crash.log`：`exit=139`，崩溃栈在 `HIVMAggregatedDecomposeOp`；上游 HFusion 测试同样崩）。源码构建的 `bishengir-opt`（commit `31f690369`）能跑通 HFusion/HIVM pipeline；而源码构建的 `bishengir-compile` 在调用 CANN 旧 `hivmc` 时会转发 `hivmc` 不认识的新选项。因此：

```bash
# 1) 源码构建的 bishengir-opt：HFusion -> 优化后的 HIVM
$BUILT_OPT add-hfusion-host.mlir \
  --lower-hfusion-pipeline --convert-to-hivm-pipeline --optimize-hivm-pipeline \
  -o add-hfusion.optimized.mlir
# 2) 去掉与新 DEVICE 入口同名的 HOST 包装函数
python3 -I extract-device-kernel.py add-hfusion.optimized.mlir add-hfusion.device.hivm.mlir
# 3) CANN 包内 bishengir-compile：HIVM -> 设备 ELF
bishengir-compile add-hfusion.device.hivm.mlir --enable-hivm-compile -o kernel.o
```

实测证据：

| 阶段 | 结果 | 证据 |
|---|---|---|
| HFusion pipeline | exit 0；`linalg.elemwise_binary{add}` 经 `ConvertHFusionToHIVM` 变成 `hivm.hir.load/load/vadd/store`；最终 DEVICE 函数为 GM→UB→GM 的 memref 形式 | `code/logs/step6-hfusion-chain.log`、`code/logs/step6-hfusion-conversion.log`（完整 dump `code/logs/step6-hfusion-passdump.log`，5729 行） |
| 设备函数抽取 | 保留 DEVICE 入口，删除 HOST 函数 `@add` | `code/logs/step6-hfusion-chain.log`（`DEVICE_EXTRACT=PASS`） |
| 设备 ELF | exit 0，2752 B，machine `0x1029`，全局 `FUNC add_single_outlined_0_0_0` 244 B | `code/logs/step6-hfusion-chain.log`、`code/logs/step6-artifacts.log` |
| 上卡执行 | 用同一 host 程序、`argv[2]=add_single_outlined_0_0_0`；两组输入全部匹配 | `code/logs/step6-run-hfusion.log` |
| 独立数值校验 | 32 个 Result 与期望序列一致，`NUMERIC_CHECK=PASS` | `code/logs/step6-numeric-check.log` |

这条链路补上了 10-31 的空白：以前只验证了 HFusion→HIVM 的 conversion pass，现在同一份 HFusion 输入确实被融合/schedule/outline，编成设备 ELF，在 NPU 上执行并逐元素校验。代价是要手动拆成两段并用脚本删掉 HOST 包装函数；这是本机工具链的缺陷，不是 AscendNPU-IR 设计的正常路径（见第 5 节）。

## 3. 源码/文档依据

| 结论 | 依据 |
|---|---|
| HFusion 负责硬件相对无关优化；HIVM 负责精细化感知 NPU 硬件 | `docs/source/zh_cn/introduction/architecture.md:11` |
| HFusion 基于 Linalg 扩展、继承全部 Linalg op、处理 named ops；能力分转换层/预处理/融合处理 | `architecture.md:17-23` |
| HIVM 对计算/搬运/同步做 Tile 级抽象，优化分 CV 核映射/片上内存映射/处理单元映射 | `architecture.md:27-33` |
| `bishengir-compile` 把高层 Tile op 降到 low level op，输入输出都是 MLIR；`hivmc` 再转 LLVM IR 并生成二进制 | `architecture.md:83` |
| `convert-hfusion-to-hivm` pass 存在，summary 即 “Convert HFusion ops to HIVM ops” | `bishengir/include/bishengir/Conversion/Passes.td:104-107` |
| 编译 pipeline 顺序：HFusion pipeline → ConvertToHIVM → OptimizeHIVM | `bishengir/lib/Tools/bishengir-compile/PassPipeline.cpp:71-108` |
| HIVM 之后经 HIVM Minimal OpSet → convert-hir-to-lir → LLVM IR → binary | `bishengir/include/bishengir/Tools/hivmc/HIVMC.h:22-49` |
| 一条 `bishengir-compile` 内部会在 HIR pipeline 后调用 hivmc，所以直接产出 `kernel.o` | `bishengir/lib/Tools/bishengir-compile/BiShengIRCompileMain.cpp:300` |
| HFusion dialect 名与对 Linalg 的依赖 | `bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td:31-38` |
| HIVM dialect 名与对 memref 等的依赖；op 助记符统一 `hir.` | `bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td:37-44,69` |
| VecAdd 的编译/执行流程 | `bishengir/test/Integration/HIVM/VecAdd/README.md:4,34-40` |
| VecAdd host 侧注册/发射/回读（上游行号） | 同目录 `main.cpp:101,108,121-128,135,141-190`；本地副本已改，行号见 2.5 节 |
| VecAdd host 编译的 include/lib | 同目录 `CMakeLists.txt:8-21`；`docs/.../quick_start/examples.md:196-207` |
| HFusion 流水线可单独调用 | `bishengir-opt --lower-hfusion-pipeline --convert-to-hivm-pipeline --optimize-hivm-pipeline` 实测 exit 0（`code/logs/step6-hfusion-chain.log`），与 `PassPipeline.cpp:71-108` 的三段式一致 |
| 设备编译命令与 `kernel.o` 含义 | `examples.md:32-38` |
| AscendNPU-IR v1.1.0 配套 CANN 9.0.0、支持 Atlas A2/A3 | `docs/.../quick_start/version_compatibility.md:15` |
| 源码构建需 `-t` 模板库；CANN 包内工具可直接端到端 | `examples.md:31`（注释）、`installing_guide.md` 的 `--build-bishengir-template` 说明 |

## 4. 推断（不是直接实测）

1. **ACL 逻辑设备 0 = 物理 NPU 7**：依据是 `npu-smi` 的 Chip Logic ID 0，且 `aclrtSetDevice(0)` 后全链路执行成功；但没有单独打印物理设备 ID 做交叉验证。
2. **`PlanMemory` 后的 UB 偏移 0/32 对应两个 16×i16 输入缓冲**：偏移是 dump 直接可见的；“对应哪个输入”是按 32 字节大小和 op 顺序推断的。
3. **默认 `--target` 为 Ascend910B1**：来自 `examples.md:32` 的文档说明；本次未传 `--target`，编译+运行成功，但没有验证其他 target 字符串。
4. **`kernel.o` 是设备侧 ELF 而非 host 目标文件**：依据是 `file` 报 `unknown arch 0x1029`、`readelf` 有 `.ascend.stack.size.record`/`__CCE_KernelArgSize` 段、以及 runtime 用 `RT_DEV_BINARY_MAGIC_ELF_AIVEC` 注册成功。
5. **`add_single_outlined_0_0_0` 是单个融合区域的 outline 结果**：符号名里的 `single`、`outlined`、`0_0_0` 与 HFusion pipeline 的 `hfusion-outline`/`hfusion-wrap-host` 阶段对应，且该函数带 `hfusion.fusion_kind<PURE_ELEMWISE>`、`hacc.block_dim=1`；本输入只有一个 elemwise op，这是按名字和属性做的推断，没有下钻 outline pass 的源码。

## 5. 未验证 / 边界

1. **HFusion 全链依赖两段工具链，不是上游的单命令路径**。CANN 9.0.0 包内 1.1.0 在本机对任何 `--enable-hfusion-compile` 输入都 SIGSEGV（`code/logs/step6-packaged-hfusion-crash.log`，exit 139），所以“一条 `bishengir-compile` 从 HFusion 到 ELF”没有验证。代替方案（源码构建 `bishengir-opt` 跑 HFusion/HIVM pipeline → 脚本删 HOST 包装 → 包内 `bishengir-compile` 出 ELF）已端到端跑通（2.6 节），但它不是上游正常路径，且只覆盖一个 16 元素、单融合区域的 elemwise 输入。
2. **没有走 VecAdd 的 CMake target 构建路径**。构建树未开 `BISHENGIR_BUILD_EXAMPLES`（`code/logs/step4-targets-check.log` 无匹配 target），host 是用文档等价 `g++` 命令编的；`cmake --build` 路径在本机未验证。
3. **没有验证 `--target` 的其他取值**，也没有在 910B3 以外的硬件上跑。
4. **没有测性能**；本文所有结论都是功能/正确性，不涉及耗时、吞吐或 UB 占用优化效果。
5. **没有把源码构建的 `bishengir-compile` 用作最终 ELF 生成器**。本次源码副本 commit `31f690369` 用了 `bishengir-opt`（HFusion/HIVM pipeline）和早前的 `mlir-opt`；它的 `bishengir-compile` 会向 CANN 旧 `hivmc` 转发未知选项（`--enable-cpu-trace-intrinsic=false`、`--enable-lir-compile=true`），最终 ELF 仍由包内 1.1.0 生成。
6. **Day 7 内容未展开**：TVM IR → TileLangIR → HIVM 的追踪不在本报告范围。

## 6. 第 5 步口头讲解稿（可直接照着讲）

> 一个 MLIR op 先要属于某个 dialect。比如今天 HFusion 层的 `linalg.elemwise_unary{fun=exp}`，它是一个 named op，保留了“逐元素 exp”的高层语义。
>
> 编译器用 pass 改写它。我实际跑了 `bishengir-opt -convert-hfusion-to-hivm`，这条 pass 把它变成 `hivm.hir.vexp`。这一步就是 HFusion 到 HIVM 的边界：HFusion 尽量不碰硬件细节，做转换、预处理、融合；HIVM 开始显式表达搬运、计算、同步和地址空间。
>
> 再往下是 HIVM 自己的优化和降级。以今天 VecAdd 的 `add.mlir` 为例，输入是 `hivm.hir.load → hivm.hir.vadd → hivm.hir.store`，参数在 `gm`，中间结果在 `ub`。`bishengir-compile --enable-hivm-compile` 的 pass dump 里能看到：bufferize、推导 mem scope、flatten、decompose、`PlanMemory` 把 `memref.alloc` 变成 UB 偏移 0 和 32 的 `pointer_cast`，最后 `ConvertHIVMToStandard` 把高层 op 拆成 `load_gm_to_ubuf_1d_int16_t`、`vadd_1d_int16_t`、`store_ubuf_to_gm_1d_int16_t` 这些底层调用，并插入 `set_flag/wait_flag` 和 `pipe_barrier` 做流水同步。
>
> 之后 `hivmc` 把 low level MLIR 转成 LLVM IR，再编成设备 ELF，也就是今天的 `kernel.o`——2648 字节，machine 0x1029，里面有全局符号 `add`。
>
> 我还用一份真正的 HFusion 输入 `add-hfusion-host.mlir` 走了同样的后半程：`linalg.elemwise_binary{add}` 先被 HFusion pipeline 融合、schedule、outline 成 `add_single_outlined_0_0_0`，`ConvertHFusionToHIVM` 把它变成 `hivm.hir.load/load/vadd/store`，再编成 2752 字节的设备 ELF。因为包内 1.1.0 编译器跑 HFusion pipeline 会 SIGSEGV，这一步用了源码构建的 `bishengir-opt`，后面出 ELF 仍用包内 `bishengir-compile`。
>
> 最后 host 侧用 CANN runtime 加载：`rtDevBinaryRegister` 以 `RT_DEV_BINARY_MAGIC_ELF_AIVEC` 注册整个 binary，`rtFunctionRegister` 把符号注册成可调用函数，`rtKernelLaunch` 发射，同步后把结果从 device 拷回 host。host 程序现在会检查 `rtKernelLaunch` 的返回码，并在 C++ 里逐元素比较；今天两组输入（官方 `0..15 + 1`，以及含负数/零/边界值的混合组）共 32 个 int16 结果全部匹配。故意换成 mul 内核时，它报 30 个元素不符并以退出码 1 结束，说明这个校验不是空打印。
>
> 一句话：**op 先被 pass 从 HFusion 改写成 HIVM，HIVM 再被 pass 降到带地址、同步和 intrinsic 的 low level IR，hivmc 把它变成设备 ELF，host runtime 注册并发射这个 ELF，并在程序内逐元素校验结果。**

## 7. 验收自查

| 验收项 | 状态 | 证据 |
|---|---|---|
| 手工完成 `add.mlir → kernel.o` | 通过 | `code/logs/step6-compile-hivm.log`、`code/logs/step6-artifacts.log` |
| 手工编译 host 并上卡执行 | 通过 | `code/logs/step6-host-build.log`、`code/logs/step6-run-hivm.log` |
| 结果验证（程序内强校验） | 通过：`rtKernelLaunch` 返回码被检查，C++ 逐元素比较，两组输入 32/32 匹配 | `code/logs/step6-run-hivm.log`、`code/logs/step6-numeric-check.log` |
| 校验不是空校验 | 通过：mul 负例报 30/32 mismatch、exit 1 | `code/logs/step6-run-negative-control.log` |
| 同一份 HFusion 输入 → ELF → 上卡 | 通过（两段工具链，见 2.6/第 5 节） | `code/logs/step6-hfusion-chain.log`、`code/logs/step6-hfusion-conversion.log`、`code/logs/step6-run-hfusion.log` |
| 不依赖 TileLang Python | 通过：全程只用 `mlir-opt`/`bishengir-opt`/`bishengir-compile`/`g++`/CANN runtime | 本目录全部命令 |
| 区分 HFusion 与 HIVM | 通过：职责表 + 实测 conversion + HFusion 全链 | `note/02-hfusion-hivm.md`、`code/logs/step5-hfusion-to-hivm.log`、2.6 节 |
| 文档结论有证据 | 运行结论均对应日志；文档结论均给 file:line | 本报告第 3 节与 `README.md` 日志索引 |

## 8. 关键日志索引

见 `README.md` 的“关键日志索引”。最关键的一组（均为 2026-11-02 新跑）：

1. `code/logs/step6-artifacts.log` — 两种 `kernel.o` 的 ELF 类型、符号、sha256；
2. `code/logs/step6-run-hivm.log` — HIVM 内核上卡、两组输入 32 行输出；
3. `code/logs/step6-run-negative-control.log` — mul 负例被拒（30/32 mismatch，exit 1）；
4. `code/logs/step6-hfusion-chain.log` — HFusion 输入 → 设备 ELF 的三条命令与产物符号；
5. `code/logs/step6-hfusion-conversion.log` — `linalg.elemwise_binary{add}` → `hivm.hir.vadd` 的关键 IR（完整版 `step6-hfusion-passdump.log`）；
6. `code/logs/step6-run-hfusion.log` + `code/logs/step6-numeric-check.log` — HFusion 内核上卡与独立数值校验 PASS；
7. `code/logs/step6-hivm-passdump.log` — HIVM 内核逐 pass IR（2135 行本体），2.4 节的 `PlanMemory`/最终 HIR 行号来自这里。

## 9. 明日（Day 7）衔接，不展开

今天已经建立“HFusion/HIVM 两层 + pass pipeline + hivmc + runtime”的骨架。Day 7 若要追 TVM IR → TileLangIR → HIVM，可以直接把今天的 `ConvertHFusionToHIVM` 和 `PassPipeline.cpp:71-108` 当作入口，但本报告不提前展开。
