# Day 7 学习报告 · 编译链路追踪

> 这是详细技术版，保留逐行证据、源码位置和验收问答。面向读者、按故事顺序
> 写的首读版在 [`day7学习笔记.md`](day7学习笔记.md)；两份使用同一批证据，
> 结论以首读版为准，数字以 `code/logs/` 为准。

日期：2026-11-02
范围：同一次 VecAdd 计算从 TileLang Python 到 TVM TIR、npuir/MLIR、HIVM、设备产物的实际链路。
不测性能，不展开 Day 8 的 Agent 生成任务。

环境：Ascend 910B3（`npu-smi` 显示 NPU 3 / Chip 0，空闲）、CANN 9.0.0、
`bishengir-opt`/`bishengir-compile` 1.1.0（LLVM 19.1.7）、TileLang 仓库
`bd7c60c`。完整记录见 `code/logs/step0-env.log`。

## 1. 今天的贡献

1. **用同一次运行把四级边界钉死**：Python 源码 → TVM TIR → 初始 npuir →
   final npuir → HIVM 低层 IR → 设备 ELF。每一级都保存了实际文件、banner/marker
   行号和退出码；主运行 exit 0，设备侧 `torch.testing.assert_close` 通过并打印
   `All check passed`。
2. **纠正了计划图里的一个关键假设**：当前 npuir 路径**没有生成 TileLangIR IR，
   也没有调度 `tilelangir-cv-split` / `tilelangir-vectorize`**。`libtilelangir.so`
   确实被加载，并充当了 pass 管理器的宿主，但实际只跑了 `canonicalize` 和
   `adapt-triton-kernel`；两个 TileLangIR pass 的源码仍是 placeholder。

## 2. 结论分级

| 结论 | 级别 | 证据 |
|---|---|---|
| Developer 模式运行成功，设备结果正确 | 设备运行 | `code/logs/vecadd-dev-exit.log`（`exit_code=0`）、`vecadd-dev-dump.log:172`（`All check passed`） |
| TVM TIR 由 `lower.py` 的 Phase 1/2 产生，之后进入 npuir codegen | 源码判断 + IR 实测 | `tilelang/engine/lower.py:263-290`；`code/ir/step1-tvm-ir.tir` |
| TVM 的 `T.npuir_add` 进入 `CreateHIVMBinaryVectorOp<VAddOp>`，产出 `hivm.hir.vadd` | 源码判断 + IR 实测 | `src/target/codegen_npuir_dev.cc:3587-3588, 2671`；`code/ir/step2-initial-npuir.mlir:29` |
| 初始 npuir 只经过 `canonicalize` 和 `adapt-triton-kernel` 两个 pass | IR 实测 | `code/logs/vecadd-dev-dump.log:62,97`；`lower.py:281-286` |
| 当前路径没有 TileLangIR 中间 IR，也没有运行 TileLangIR pass | 源码判断 + IR 实测 | `lower.py:281-286`；`code/logs/step5-bishengir-compile-passdump.log` 中 TileLangIR 关键字计数 0；`code/logs/step6-tilelangir-load-check.log` |
| `CVSplit.cpp` / `Vectorize.cpp` 是占位实现 | 源码判断 + 出路径实测 | `tilelangir/lib/Transforms/CVSplit.cpp:24-27`、`Vectorize.cpp:24-27`；`code/ir/out-of-path-*.mlir` 与输入相同 |
| `--hivm-infer-mem-scope` 把 GM/UB 显式化 | IR 实测 | `code/ir/step4-hivm-infer-mem-scope.mlir:2,15-17` |
| 完整 HIVM 流水线把 `vadd` 降到 `vadd_1d_float`，并插入 `load/store` 与同步 | IR 实测 | `code/ir/step5-final-hivm-lowered.mlir:7-9,30,44,51-52` |
| 设备产物是可加载的 Ascend ELF | 设备产物 | 原 JIT `code/logs/jit-kernel.o` 与复编的 `code/artifacts-vecadd.o` 逐字节相同，均为 2784 B、machine `0x1029`、`main` 340 B；`code/logs/step7-jit-vs-recompiled.log` |
| ACL 逻辑设备 0 = 物理 NPU 3 | 未验证 | 只看到 `npu-smi` 与 `torch.npu.set_device(0)`，没有单独打印物理 ID 交叉验证 |

## 3. 同一次运行的证据与边界

### 3.1 命令与结果

```bash
cd /mnt/workspace/learn/tilelang-ascend
export TILELANG_ASCEND_MODE=Developer
export TILELANG_DUMP_IR=TRUE
python3 examples/elementwise/vec_add_1d.py
```

实际运行封装在 `code/run_vecadd_dev.sh`，使用 `set -o pipefail` 和
`PIPESTATUS[0]`，不会让 `tee` 掩盖 Python 的退出码。

- 命令/环境：`code/logs/vecadd-dev-command.log`
- 完整 stdout+stderr：`code/logs/vecadd-dev-dump.log`
- 退出码：`code/logs/vecadd-dev-exit.log` → `exit_code=0`
- 正确性：`vecadd-dev-dump.log:168-172` 打印参考结果、实际结果和
  `All check passed`；这是示例内的 NPU 张量 `assert_close`，不是只看退出码。
- 原 JIT 编译留痕：`code/logs/jit-compile-argv.log`、
  `code/logs/jit-kernel.npuir`、`code/logs/jit-kernel.o`、
  `code/logs/jit-libkernel.so`（由 `code/shims/bishengir-compile` 捕获）。

### 3.2 三段的切分与可靠性

`TILELANG_DUMP_IR` 的读取在 `lower.py:269-270`；banner 打印在
`lower.py:272-289`。同一次日志中的边界是：

| 行号 | 标记 | 含义 |
|---|---|---|
| 6 | `====== TVM IR ======` | Phase 1/2 之后的 TVM TIR |
| 25 | `====== npuir ======` | codegen 刚产出的初始 MLIR |
| 62 | `IR Dump After Canonicalizer` | 第一个 pass 之后 |
| 97 | `IR Dump After AdaptTritonKernel` | 第二个 pass 之后 |
| 132 | `====== final npuir ======` | `pipeline.run()` 的返回值 |
| 165 | `AscendNPU IR compile success:` | JIT 调 `bishengir-compile` 成功 |
| 172 | `All check passed` | 设备执行 + 校验通过 |

这三段都能可靠切开，没有“保留原始日志但拆不出”的段落。切分脚本是
`code/extract_ir.py`，它检查每段的首尾并输出 `EXTRACT=PASS`
（`code/logs/step2-extract-ir.log`）。

日志用 `2>&1` 合并 stdout/stderr，原因是 `pipeline.enable_ir_printing()`
把逐 pass IR 打到 stderr，而 banner 和 Python 输出在 stdout；合并后能保留
实际先后顺序。原始日志没有删减。

### 3.3 一个环境边界

仓库在本实验前已有本地改动：`tilelang/jit/jit_npu.py` 的 sync 属性重命名/
重试补丁，以及 `3rdparty/AscendNPU-IR` 子模块的构建副本。本实验没有修改
任何 TileLang/AscendNPU-IR 源码；该补丁不影响本次 Developer AIV 路径的结论
（本次 IR 里没有 `syn_instr_mode`/`sync_instr_mode` 属性）。

另外有两个 AscendNPU-IR checkout，commit 不一致：独立源码参考
`/mnt/workspace/learn/AscendNPU-IR` 是 `87ea6c4`，实际构建副本
`tilelang-ascend/3rdparty/AscendNPU-IR` 是 `31f6903`
（`legacy-interface-342-g31f690369`）。运行和产物来自构建副本；CANN 自带的
`bishengir-opt`/`bishengir-compile` 又报告自己的版本串
`428ab8fdab46`。本文引用底层源码行号时，以构建副本 `31f6903` 为准。

## 4. 逐阶段实际发生了什么

### 4.1 Python 前端：只有 1 个 block，没有显式循环

主例是 `examples/elementwise/vec_add_1d.py`：

- `seq_len = 4096`、`block_N = seq_len`，所以 `n_num = 1`
  （`vec_add_1d.py:13-18,45`）；
- `with T.Kernel(n_num, is_npu=True) as (cid, _)`（`:27`）只开 1 个 block；
- `A_VEC/B_VEC/C_VEC = T.alloc_ub(...)`（`:28-30`）映射为 scope `"shared"`，
  即 UB（`tilelang/language/allocate.py:108-109`）；
- 两次输入 `T.copy`、一次 `T.vadd`、一次输出 `T.copy`（`:34-38`）；
- `T.vadd` 是 `npuir_add` 的别名（`tilelang/language/__init__.py:73`），最终
  发出 `tl.npuir_add` intrinsic（`customize_npuir.py:102-116`）；
- `tail_size = min(block_N, shape - cid*block_N)`（`:31-33`）。本次
  `block_N == N`，tail 分支没有触发。

`T.Kernel` 的 1D NPU grid 在 C++ 里会创建两个 frame：`blockIdx.x` 是 cube/block
索引，`blockIdx.z` 是 vector 子块索引，后者 extent 是前者的 2 倍
（`src/ir.cc:175-190`）。这解释了 TVM TIR 里为什么明明只写了 1 个 grid 维度，
却出现 `blockIdx.z` extent 2；第二个变量名是 `_`，codegen 不绑定它
（`codegen_npuir_dev.cc:3734-3736`）。

### 4.2 TVM TIR：保留 Python 结构，但 `cid` 已被常量折叠

`code/ir/step1-tvm-ir.tir` 的关键部分：

```python
cid = T.launch_thread("blockIdx.x", 1)
_ = T.launch_thread("blockIdx.z", 2)
A_VEC = T.decl_buffer((4096,), scope="shared")
...
T.copy(T.region(A[0], 1, T.min(4096, shape)), T.region(A_VEC[0], 2, T.min(4096, shape)))
T.npuir_add(T.region(A_VEC[0], 1, 4096), T.region(B_VEC[0], 1, 4096), T.region(C_VEC[0], 2, 4096))
```

要点：

- `cid` 的 extent 是 1，`cid * block_N` 被简化成 0，所以 region 起点是 `A[0]`；
- 没有 `T.serial`/`scf.for`；这是一次性处理整块 4096 个 f32；
- `T.region` 的第二个参数是 access type：`1` 读、`2` 写；后面的 `4096` 或
  `T.min(4096, shape)` 是 extent；
- 输入/输出 buffer 默认是 GM，`A_VEC` 等是 `scope="shared"`（UB）；
- `T.npuir_add` 的输入是整块 4096，而 copy 的 extent 是 `min(4096, shape)`；
  本次二者相等。

### 4.3 初始 npuir：codegen 的实际入口与产物

`lower.py:275-280` 对 `target.kind.name == "npuir"` 调
`device_codegen`；`device_codegen` 在 `lower.py:180` 通过 FFI 取
`target.build.tilelang_npuir_dev`。这个全局函数的注册点是
`src/target/rt_mod_npuir.cc:99-100`，实现是
`BuildTileLangNPUIRMLIRDEV`（`:75-96`）：

```text
BuildTileLangNPUIRMLIRDEV
  -> CodeGenTileLangNPUIRDEV cg
  -> cg.AddFunction(gvar, f)          // codegen_npuir_dev.cc:4090
  -> cg.Finish()                      // :359，verify + print
```

`AddFunction` 先用 `InferFuncCoreType` 判断这是 AIV/AIC/MIX；本内核含 vector op、
不含 cube op，得到 AIV，并把 `hivm.module_core_type<AIV>` 写到 module 上
（`:4090-4125`）。随后逐个 `VisitStmt`/`VisitExpr`：

| TIR 节点 | codegen 函数 | MLIR 结果 |
|---|---|---|
| `T.copy(GM → UB)` | `AscendCopyCodegen`（`:1960`）→ `EmitCopyMemrefToTensor`（`:1732`） | `memref.subview` + `memref.alloc` + `memref.copy` + `bufferization.to_tensor` + `tensor.insert_slice` |
| `T.copy(UB → GM)` | `AscendCopyCodegen` → `EmitCopyTensorToMemref`（`:1820`） | `tensor.extract_slice` + `bufferization.materialize_in_destination` |
| `T.npuir_add` | `VisitExpr_(CallNode)`（`:3567`）→ `CreateHIVMBinaryVectorOp<hivm::VAddOp>`（`:2671`） | `hivm.hir.vadd ins(...) outs(...)` |
| `T.alloc_ub` | `VisitStmt_(AllocateNode)`（`:3777`） | `tensor.empty` |

对应的初始 npuir（`code/ir/step2-initial-npuir.mlir`）：

```mlir
%1 = hivm.hir.get_block_idx -> i64            // 后来被 canonicalize 删除
%2 = arith.trunci %1 : i64 to i32
%6 = arith.minsi %c4096_i32, %arg6 : i32      // tail_size
%alloc = memref.alloc() : memref<4096xf32>    // copy 的 UB 暂存
memref.copy %subview, %subview_2 : ...        // GM -> UB
%8 = bufferization.to_tensor %subview_2 restrict : ...
%inserted_slice = tensor.insert_slice %8 into %3[0] [%7] [1] : ...
%10 = hivm.hir.vadd ins(%inserted_slice, %inserted_slice_6 : tensor<4096xf32>, tensor<4096xf32>) outs(%5 : tensor<4096xf32>) -> tensor<4096xf32>
```

这里有一个容易误读的点：`tensor.empty` 是 `T.alloc_ub` 的逻辑 buffer
（`AllocateNode` 产生），而 `memref.alloc` 是 copy codegen 为 GM→UB 临时插入的
staging buffer；`to_tensor + insert_slice` 再把 staging 的内容接进逻辑 tensor。
此时 memref 还**没有** GM/UB address space 标注，address space 是后面 HIVM
阶段推断的。

### 4.4 final npuir：只有 canonicalize，adapt 是 no-op

`lower.py:281-286` 的 pipeline 只有两项：

```python
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.bishengir.adapt_triton_kernel)
mlir_str = pipeline.run(mlir_str)
```

实测 `canonicalize` 之后与 `adapt-triton-kernel` 之后的 IR 逐行相同（只差
dump banner），见 `code/logs/step2-canonicalize-vs-adapt.diff`。canonicalize
相对初始 npuir 的变化只有：

- 删除未使用的 `hivm.hir.get_block_idx` / `arith.trunci`；
- 把 `%c1_i32 + index_cast` 折成 `%c1 = arith.constant 1 : index`，并调整
  `set_ffts_base_addr` 与常量的顺序；
- SSA 重新编号。

`canonicalize` 前后没有语义变化，也没有生成任何新 dialect/op。最终文件是
`code/ir/step3-final-npuir.mlir`，其中 vadd 仍在第 26 行：

```mlir
%7 = hivm.hir.vadd ins(%inserted_slice, %inserted_slice_6 : tensor<4096xf32>, tensor<4096xf32>) outs(%2 : tensor<4096xf32>) -> tensor<4096xf32>
```

### 4.5 HIVM：同一份 final npuir 再编译，GM/UB 与低层 intrinsic 出现

JIT 在 `jit_npu.py:1244` 调 `_npuir_to_bin_enable_npu_compile()`，把 final npuir
写成临时 `kernel.npuir`，再调 `bishengir-compile`，选项包括
`--enable-triton-kernel-compile=true`、`--enable-hivm-compile=true`
（`:1537-1539`）。为了留下可复查的逐 pass 证据，我把**同一份**
`step3-final-npuir.mlir` 再喂给同一个 `bishengir-compile`，加上
`--mlir-print-ir-after-all`：

- 命令与退出码：`code/logs/step4-commands.log`、
  `code/logs/step5-bishengir-compile-exit.log`（`exit=0`）；
- 逐 pass 日志：`code/logs/step5-bishengir-compile-passdump.log`（随附文件
  5983 行、82 个不同 pass 名；pass 名稳定，行号仅供定位）；
- 最终低层 HIR：`code/ir/step5-final-hivm-lowered.mlir`。

关键 pass（以 pass 名为主；括号里的行号只对随附这份 dump 成立）：

```text
CanonicalizeModule
ConvertHFusionToHIVM
ConvertTensorToHIVM
ConvertToHIVMOp
...
InferHIVMMemScope        (随附 dump :3790 / :4801)
...
PlanMemory               (随附 dump :1157 / :5202 / :5322)
...
ConvertHIVMToStandard    (随附 dump :5923)
```

注意：比较两次重编的 pass dump，212 条 dump、82 个 pass 名和最终 ELF 都一致，
但 `BindSyncBlockLockArg` / `InsertInferSyncBlockLockNumAndInitFunc` 的交织位置
会变。因此行号不是逐字节稳定的；引用 pass 时以名称为准，行号只作为随附日志的
定位备注。

`InferHIVMMemScope` 后，GM 参数和 UB 分配显式化：

```mlir
%arg3: memref<?xf32, #hivm.address_space<gm>>
%alloc = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
memref.copy %subview, %subview_2 : memref<...gm> to memref<...ub>
```

`PlanMemory` 后，`memref.alloc` 变成 UB 偏移的 `hivm.hir.pointer_cast`。
本次三个 4096×f32（每个 16384 B）缓冲区只用了两个偏移：0 和 16384。按 load
顺序，A 先进入偏移 0，B 进入偏移 16384；输出 `C_VEC` 复用偏移 0，也就是覆盖
A 的 slot（A 在 `vadd` 之后不再被读）：

```mlir
%2 = hivm.hir.pointer_cast(%c16384_i64) : memref<4096xf32, #hivm.address_space<ub>>  // B
%5 = hivm.hir.pointer_cast(%c0_i64)     : memref<4096xf32, #hivm.address_space<ub>>  // A
%6 = hivm.hir.pointer_cast(%c0_i64)     : memref<4096xf32, #hivm.address_space<ub>>  // C，复用 A 的 slot
hivm.hir.vadd ins(%5, %2 : ...) outs(%6 : ...)
```

“A → 0、B → 16384、C 复用 0”的对应关系在最终 HIR 里可以直接读出来，不需要
按顺序推断：

| 数据流 | 证据（`code/ir/step5-final-hivm-lowered.mlir`） |
|---|---|
| 第一次 load 读 A，写 UB 偏移 0 | `%subview` 来自 `%arg3`（`:19,25`）；目标是 `%3 = pointer_cast(0)`（`:26-27`）；`call @load_gm_to_ubuf_1d_float`（`:30`） |
| 第二次 load 读 B，写 UB 偏移 16384 | `%subview_4` 来自 `%arg4`（`:21,31`）；目标是 `%0 = pointer_cast(16384)`（`:22,32`）；`call`（`:35`） |
| vadd 读 0 和 16384，结果写 0 | `call @vadd_1d_float(%cast_8=%3(0), %cast_9=%0(16384), %cast_10=%4(0), ...)`（`:37,40-44`） |
| store 从 UB 偏移 0 写 C | `%subview_12` 来自 `%4 = pointer_cast(0)`（`:46`）；目标是 `%reinterpret_cast_0 = %arg5`（`:20,47`）；`call @store_ubuf_to_gm_1d_float`（`:51`） |

vadd 实际有 4 个 memref 操作数（声明与调用一致，都是 4 个）。第 4 个是
`%5 = hivm.hir.pointer_cast(%c0_i64) : memref<0xf32, #hivm.address_space<ub>>`
（`:39`），经 `%cast_11`（`:43`）传入调用（`:44`）。它是
`ConvertHIVMToStandard` 新造出的**零长度占位**（随附 passdump `:5961`），不表示
第四个真实缓冲区，也不参与数据搬运；忽略它即可，不会影响上面 A/B/C 的对应
关系。

最后 `ConvertHIVMToStandard` 把高层 op 降成标准 `call`：

```mlir
func.func private @load_gm_to_ubuf_1d_float(... gm, ... ub, i32, f32, index)
func.func private @vadd_1d_float(... ub, ... ub, ... ub, ... ub)
func.func private @store_ubuf_to_gm_1d_float(... ub, ... gm, i32)
...
call @load_gm_to_ubuf_1d_float(...)       // 两次
hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
call @vadd_1d_float(...)
hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
call @store_ubuf_to_gm_1d_float(...)
hivm.hir.pipe_barrier[<PIPE_ALL>]
```

这说明 GM→UB 搬运、vadd、UB→GM 写回分别落到 MTE2、V、MTE3 管道，并带有
flag/barrier 同步。

### 4.6 设备产物

JIT 原运行的 `kernel.o` 位于 `TemporaryDirectory` 里，进程结束后会被删除。
为了不修改上游源码而留下原产物，主运行把 `code/shims/bishengir-compile` 放到
`PATH` 最前面：shim 记录 JIT 的实际 argv，并在 JIT 清理临时目录前复制
`kernel.o` / `libkernel.so`。捕获逻辑整体包在 `try/except` 里：复制或写日志
失败只向 stderr 打 warning，真实编译照常执行，最终仍按真实返回码/信号退出，
不会因为留痕失败而让主实验失败。实测：

- JIT 的真实编译调用（`code/logs/jit-compile-argv.log`，第 2 条记录）：
  ```text
  /tmp/<tmpdir>/kernel.npuir --enable-auto-multi-buffer=true
  --disable-hivm-auto-inject-sync=false --enable-triton-kernel-compile=true
  --enable-hivm-compile=true -o /tmp/<tmpdir>/kernel
  ```
  第 1 条记录是 `--version`，由 `_npuir_needs_sync_attr_rename` 触发；
- 原 JIT 输入被复制为 `code/logs/jit-kernel.npuir`（3105 B）；
- 原 JIT 设备产物被复制为 `code/logs/jit-kernel.o`：2784 B，sha256
  `bae006312aa6e3a19dd94f998e296f8564d63ddbaab98d8fadf17c56aec3b3ea`；
- host 侧 `code/logs/jit-libkernel.so`：4720 B，sha256
  `84f8f137dbf9cc86e6547abb94ade6e68afa306e065e1633bbcfd374a62d4083`。

另外，我把同一份 final npuir 再次喂给同一个 `bishengir-compile`，生成
`code/artifacts-vecadd.o`。它与原 JIT `kernel.o` **逐字节相同**（同 sha256），
host `.so` 也逐字节相同；证据是 `code/logs/step7-jit-vs-recompiled.log`
（`JIT_VS_RECOMPILED=IDENTICAL`）。因此“工具/选项相同”不再只是源码断言：
原 argv 已留痕，代表产物也已升级为原产物 + 复编产物的一致性验证。

原运行与设备正确性仍以 JIT 那次为准：exit 0 + `All check passed`。

## 5. 实际链路图

```text
examples/elementwise/vec_add_1d.py:27-38
  T.Kernel / T.alloc_ub / T.copy / T.vadd
        │  LowerAndLegalize + OptimizeForTarget
        │  tilelang/engine/phase.py:66,96,114-125
        │  （npuir 分支含 TVM 的 NpuLoopVectorize，不是 TileLangIR Vectorize）
        ▼
TVM TIR
  code/ir/step1-tvm-ir.tir
        │  lower.py:275-280  device_codegen
        │  lower.py:180      target.build.tilelang_npuir_dev
        ▼
target.build.tilelang_npuir_dev
  rt_mod_npuir.cc:99-100 -> BuildTileLangNPUIRMLIRDEV (:75)
        │  codegen_npuir_dev.cc:4090 AddFunction
        │  :3567 VisitExpr_(CallNode) -> :3588 CreateHIVMBinaryVectorOp<VAddOp>
        │  :1960 AscendCopyCodegen -> :1732/:1820
        ▼
初始 npuir（hivm + memref + tensor + bufferization）
  code/ir/step2-initial-npuir.mlir
        │  lower.py:282 canonicalize（tladapter Pipeline）
        ▼
canonicalize 后
  code/ir/step2a-after-canonicalize.mlir
        │  lower.py:283 adapt-triton-kernel（本次 no-op）
        ▼
final npuir
  code/ir/step3-final-npuir.mlir
        │  jit_npu.py:1495-1580 bishengir-compile
        │  --enable-triton-kernel-compile --enable-hivm-compile
        │  pass dump: code/logs/step5-bishengir-compile-passdump.log
        │  InferHIVMMemScope -> PlanMemory -> HIVMDecomposeOp -> ConvertHIVMToStandard
        ▼
低层 HIVM/HIR
  code/ir/step5-final-hivm-lowered.mlir
        │  hivmc -> LLVM IR -> 设备二进制
        ▼
artifacts-vecadd.o（2784 B，ELF machine 0x1029）
        │  CANN runtime 注册/发射（示例内由 TileLang JIT 完成）
        ▼
NPU 执行 + torch.testing.assert_close
  code/logs/vecadd-dev-dump.log:172  All check passed
```

图中 `bishengir-compile` 这一步的原 JIT 调用由 `code/shims/bishengir-compile`
留痕：真实 argv、临时 `kernel.o` 和 `libkernel.so` 都保存在 `code/logs/`
（§4.6）。

注意图中没有“TileLangIR 中间文件”这一级；原因见第 7 节。

## 6. 对照表

### 6.1 阶段、语义、变化、证据

| 阶段 | 保留的语义 | 发生的变化 | 证据路径 |
|---|---|---|---|
| Python DSL | 每个 block 取 4096 个 f32，A+B 写回 C | 无；这是用户写的源 | `examples/elementwise/vec_add_1d.py:27-38` |
| TVM TIR | 1 个 block、3 个 UB buffer、两次读 copy、一次 vadd、一次写 copy | `cid*block_N` 因 extent=1 被简化；`T.vadd` 变成 `T.npuir_add`；region 带 access type/extent | `code/ir/step1-tvm-ir.tir:9-17` |
| 初始 npuir | 同一计算，tensor 语义 + memref 视图 | `T.copy` → subview/alloc/copy/to_tensor/insert_slice；`T.npuir_add` → `hivm.hir.vadd`；`T.alloc_ub` → `tensor.empty`；出现未使用的 `get_block_idx` | `code/ir/step2-initial-npuir.mlir:9-32` |
| final npuir | 同一计算，vadd 仍是 tensor 形式 | canonicalize 删除未用的 block idx、整理常量/SSA；adapt 无变化 | `code/ir/step3-final-npuir.mlir:12-29`、`code/logs/step2-canonicalize-vs-adapt.diff` |
| HIVM（完整流水线） | 同一计算，搬到 memref/UB 语义 | bufferize；`InferHIVMMemScope` 标 gm/ub；`PlanMemory` 把 alloc 变成 UB 偏移 0/16384；`HIVMDecomposeOp` 拆 load/vadd/store | `code/ir/step4-hivm-infer-mem-scope.mlir:15-17`；passdump 中 `InferHIVMMemScope` / `PlanMemory`（随附 dump `:3790,5322`） |
| 低层 HIR | 同一计算，落到 intrinsic 调用 | `load_gm_to_ubuf_1d_float` ×2、`vadd_1d_float`、`store_ubuf_to_gm_1d_float`，加 flag/barrier | `code/ir/step5-final-hivm-lowered.mlir:7-9,30-52` |
| 设备 ELF | 同一 kernel 入口 | MLIR → hivmc/LLVM → ELF；`main` 340 B | `code/artifacts-vecadd.o`、`code/logs/step5-bishengir-compile-exit.log` |
| 设备运行/校验 | 同一计算结果 | 运行时加载 ELF、发射、回读；示例内 assert_close | `code/logs/vecadd-dev-dump.log:168-172` |

### 6.2 四个观察维度

| 维度 | Python | TVM TIR | 初始/final npuir | HIVM 低层 |
|---|---|---|---|---|
| 循环/块索引 | `T.Kernel(n_num=1)`，`cid` 实际为 0 | `blockIdx.x` extent 1 + `blockIdx.z` extent 2；`cid` 被折叠，无循环 | 初始有 `get_block_idx/trunci`（未使用），final 被 canonicalize 删除 | 没有循环；单 block；`pointer_cast` 只表达 UB 偏移 |
| buffer | `alloc_ub` → scope `shared`；A/B/C 默认 GM | `decl_buffer(scope="shared")`；region 读写标记 1/2 | `tensor.empty` 是逻辑 UB buffer；`memref.alloc` 是 copy 的 UB staging；尚无 address space | `#hivm.address_space<gm/ub>`；PlanMemory 后 UB 偏移 0/16384 |
| GM↔UB 搬运 | 4 次 `T.copy` | 4 次 `T.copy(T.region(...))` | `memref.copy` + `to_tensor` + `insert_slice`；回写 `extract_slice` + `materialize_in_destination` | `hivm.hir.load/store` → `load_gm_to_ubuf_1d_float` / `store_ubuf_to_gm_1d_float` |
| vadd | `T.vadd(A_VEC,B_VEC,C_VEC)` | `T.npuir_add(region,region,region)` | `hivm.hir.vadd ins(tensor,tensor) outs(tensor)` | `hivm.hir.vadd ins(memref ub, memref ub) outs(memref ub)` → `call @vadd_1d_float` |

## 7. TileLangIR 核实：加载了，但没有生成，也没有运行

这是今天最需要说清楚的一点。

### 7.1 文档里的“TileLang IR”和源码目录 `tilelangir/` 不是一回事

调试指南 3.2 节标题是“MLIR（Tilelang IR）”，正文给的是 `module { ... }` 的
MLIR 文本（`docs/Tilelang算子调试指南.md:78-124`）。这里说的是“codegen 产出的
MLIR 文本”，不是 `tilelangir/` 这个带 `CVSplit.cpp`/`Vectorize.cpp` 的 pass 库。
本次实际 codegen 产物的 banner 是 `====== npuir ======`（`lower.py:279`），
module 上只有 `hivm.module_core_type` 和 `memref.memref_as_ptr`
（`step2-initial-npuir.mlir:1`），没有任何 TileLangIR dialect/attribute。

### 7.2 实际加载了什么

`lower.py:20-21` 导入 `tilelang.tladapter` 的 `transforms` 和 `Pipeline`。
`tilelang/tladapter/__init__.py:12-34` 会 `ctypes.CDLL` 加载
`libtilelangir.so`，并把 `_native` 设为该 pybind 模块。实测：

```text
code/logs/step6-tilelangir-load-check.log
tladapter_native_file= /mnt/workspace/learn/tilelang-ascend/build/libtilelangir.so
```

所以 **TileLangIR 的 pass 管理器和 pass 注册表确实进了进程**。这也是为什么
`lower.py` 里的 `Pipeline` 能用。

### 7.3 实际运行了什么

`lower.py:281-286` 只 add 了两个 pass：

```python
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.bishengir.adapt_triton_kernel)
mlir_str = pipeline.run(mlir_str)
```

对应的实测 marker 只有两个：

```text
vecadd-dev-dump.log:62  IR Dump After Canonicalizer (canonicalize)
vecadd-dev-dump.log:97  IR Dump After AdaptTritonKernel (adapt-triton-kernel)
```

`transforms.tilelangir.cv_split` / `vectorize` 的 descriptor 虽然存在
（`tilelang/tladapter/transforms/tilelangir.py:7-8`），但没有被 `lower.py` 引用。
它们在 `code/logs/step6-tilelangir-load-check.log` 里能被构造出来，只是说明
“可用”，不说明“已调度”。

### 7.4 两个 pass 本身是 placeholder

`CVSplit.cpp:24-27` 和 `Vectorize.cpp:24-27` 的 `runOnOperation()` 只有：

```cpp
LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "]: placeholder, no transformation applied yet\n");
```

即使被调用，默认构建下 `LLVM_DEBUG` 也不输出，且不修改 IR。`Passes.td` 里也写着
“Implementation to be added”。我把 final npuir 直接喂给
`tilelangir-opt --tilelangir-cv-split` 和 `--tilelangir-vectorize` 做**出路径对照**：

- 两者 exit 0；
- 输出与输入逐行相同，只有末尾空行差异；
- 证据：`code/ir/out-of-path-tilelangir-*.mlir`、`code/logs/out-of-path-*.err`。

另外，`buildTileLangIRCompilePipeline` 只有声明
（`tilelangir/include/tilelangir/InitAllPasses.h:31`），没有定义，也没有调用点；
`InitAllDialects.h:21-24` 明确写着 “No TileLangIR dialects to register yet”。

### 7.5 结论

当前 npuir 路径的真实形态是：

```text
TVM TIR -> npuir/MLIR（hivm/memref/tensor/bufferization）-> HIVM -> 设备 ELF
```

不是：

```text
TVM TIR -> TileLangIR 专属 IR -> TileLangIR CVSplit/Vectorize -> HIVM
```

所以“TileLangIR 是否参与”要分两层回答：

- **作为基础设施**：参与。`libtilelangir.so` 被加载，提供了 `Pipeline`/pass 管理器；
- **作为 IR 阶段和变换**：不参与。没有生成 TileLangIR dialect，没有调度
  `tilelangir-cv-split`/`tilelangir-vectorize`，两者仍是占位。

还要区分 TVM 侧的 `NpuLoopVectorize`（`phase.py:114-118`）：它是 TVM transform，
在 codegen 之前跑，不是 TileLangIR 的 `Vectorize.cpp`。名字相近，路径完全不同。

## 8. 验收回答

假设别人给出本次 TVM IR 中的这一段（`step1-tvm-ir.tir:16`）：

```python
T.npuir_add(T.region(A_VEC[0], 1, 4096), T.region(B_VEC[0], 1, 4096), T.region(C_VEC[0], 2, 4096))
```

下一步的实际路径是：

1. `lower.py:275-280` 判断 target 是 npuir，调用 `device_codegen`；
2. `lower.py:180` 通过 FFI 取 `target.build.tilelang_npuir_dev`；
3. 这个函数的注册点是 `src/target/rt_mod_npuir.cc:99-100`，实现是
   `BuildTileLangNPUIRMLIRDEV`（`rt_mod_npuir.cc:75-96`）；
4. `CodeGenTileLangNPUIRDEV::AddFunction`（`codegen_npuir_dev.cc:4090`）建立
   AIV module 和函数；
5. `VisitExpr_(const CallNode*)`（`:3567`）在 `:3587-3588` 判断
   `tl.npuir_add`，调用 `CreateHIVMBinaryVectorOp<mlir::hivm::VAddOp>`（`:2671`）；
6. 输出是 `hivm.hir.vadd`，落在
   `code/ir/step2-initial-npuir.mlir:29`；
7. 之后 `lower.py:282-286` 的 `canonicalize` + `adapt-triton-kernel` 产出
   `code/ir/step3-final-npuir.mlir:26`；
8. 再之后 `jit_npu.py:1495-1580` 调 `bishengir-compile`，同一段最终降到
   `code/ir/step5-final-hivm-lowered.mlir:44` 的 `call @vadd_1d_float`。

TileLangIR 在这条路径里：**库被加载并充当 pass 管理器，但没有生成 TileLangIR
IR，也没有运行 CVSplit/Vectorize**。如果被问到“TVM IR 下一步进哪个实际 codegen
函数或 pass”，答案是 `device_codegen` → `target.build.tilelang_npuir_dev` →
`CreateHIVMBinaryVectorOp<VAddOp>`（对 vadd 这一段），随后是 `canonicalize` →
`adapt-triton-kernel`，不是 TileLangIR pass。

## 9. 未验证与边界

1. **物理设备映射未验证**：`npu-smi` 显示 NPU 3，示例用
   `torch.npu.set_device(0)`；没有单独打印物理 ID 做交叉验证。
2. **`--hivm-flatten-ops` 单独跑会 abort**（exit 134，LLVM `SmallVector`
   容量错误），见 `code/logs/step4-hivm-flatten-ops.err`。因此没有把它当作
   本次的可展示 pass；它在完整 `bishengir-compile` 流水线里由编译器自己调度，
   与单独调用不是一回事。
3. **`--hivm-plan-memory` 单独跑不适合这份输入**：exit 0 但 stderr 报
   “Alloc buffer not at UB space / read before first write”，IR 未变
   （`code/logs/step4-hivm-plan-memory.err`）。必须先 `InferHIVMMemScope`；
   完整流水线里的 `PlanMemory` 是在正确前置条件下运行的。
4. **原 JIT 产物已留痕**：`code/logs/jit-kernel.o` 是原 JIT 临时 `kernel.o`
   的副本，与复编的 `artifacts-vecadd.o` 逐字节相同；JIT 的实际 argv 见
   `code/logs/jit-compile-argv.log`。原 JIT 输入和提取出的 final npuir 只差
   末尾换行（3105 B vs 3106 B），产物相同。
5. **只覆盖一个形状/一个 dtype/一个 block**：N=4096、f32、`block_N=N`。tail
   分支没有触发；非整除形状、其他 dtype、多 block、2D grid 都未验证。
6. **没有测性能**：没有记录耗时、吞吐、UB 占用或流水效率；本文只讨论功能和
   IR 形态。
7. **没有做 Day 8 的 Agent 生成任务**。
8. 仓库里有实验前就存在的本地改动（`jit_npu.py` 的兼容补丁、已构建的
   `3rdparty/AscendNPU-IR` 副本）；本实验没有修改上游源码，但结论是在这个
   工作区状态下得到的。
