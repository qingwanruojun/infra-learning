# 02 · HFusion 与 HIVM 的职责、输入输出与衔接

## 一句话区分

- **HFusion**：硬件相对无关的高层优化层，基于 Linalg 方言扩展，尽量保留 `named op` 的高层语义，做转换、预处理和融合（包括生成 Device Kernel 与 Host Tiling）。
- **HIVM**：面向昇腾硬件的低层抽象，把计算/搬运/同步写成 Tile 级 op，显式处理 CV 核映射、片上内存（UB/L1/L0）和流水同步，最终降到可生成设备指令的形态。

依据：`AscendNPU-IR/docs/source/zh_cn/introduction/architecture.md:11`、`:17`、`:27`。

## 对照表

| 维度 | HFusion | HIVM |
|---|---|---|
| 全称/定位 | Hybrid Fusion；硬件相对无关优化 | Hybrid ISA Virtual Machine；精细化感知 NPU 硬件细节 |
| 方言基础 | MLIR Linalg 方言的扩展集，继承全部 Linalg op 并自行扩展；处理的 op 都是 `named operations` | 自研方言；`hivm.hir.*` 形式的 Tile 级 op（`HIVMBase.td:69` 给 op 统一加 `hir.` 前缀） |
| 典型 op | `linalg.elemwise_unary`、`hfusion.matmul_mx`、`hfusion.bitcast` 等 | `hivm.hir.load`、`hivm.hir.vadd`、`hivm.hir.store`、`hivm.hir.vexp`，以及 `hivm.hir.set_flag/wait_flag/pipe_barrier` |
| 显式硬件信息 | 基本不出现；地址空间、核类型等尚未落地 | 显式出现：`#hivm.address_space<gm/ub/ca/cb/...>`、`#hivm.func_core_type<AIV>`、`hivm.module_core_type<AIV>` |
| 主要职责 | ① 转换层：对接 Arith/Math/Torch；② 预处理：Tensor 表达式化简、BF16/Bool 合法化、复杂 op 组合；③ 融合：生成 Device Kernel 与 Host Tiling | ① CV 核映射（Mix Kernel 拆分、workspace、核间同步、CVPipeline/AutoSubTiling）；② 核内片上内存映射（空间推导、格式推导、对齐、临时空间、地址分配）；③ 核内处理单元映射（流水同步、SIMD 指令映射） |
| 输入 | 高层 named op（Linalg/HFusion，tensor 或 memref） | HFusion/Linalg 层的 op |
| 输出 | 融合后的 HFusion/Linalg 层 IR，交给 HIVM 转换 | low level 的 HIVM op，继续降到 HIVM Minimal OpSet，再交给 `hivmc` 生成 LLVM IR / 二进制 |
| 依据 | `architecture.md:17-23`；`HFusionBase.td:31-38`（dialect 名、依赖 `linalg::LinalgDialect`） | `architecture.md:27-33`；`HIVMBase.td:37-44`（dialect 名、依赖 memref/math 等） |

> 说明：表中“输入/输出”是按文档职责和编译 pipeline 顺序归纳的接口边界，不是某种强制类型签名。真正把两者接起来的是 conversion pass。

## 衔接关系（源码 + 实测）

### 1. 有专门的 conversion pass

`AscendNPU-IR/bishengir/include/bishengir/Conversion/Passes.td:104-107`：

```tablegen
def ConvertHFusionToHIVM : Pass<"convert-hfusion-to-hivm"> {
  let summary = "Convert HFusion ops to HIVM ops";
  let constructor = "mlir::createHFusionToHIVMConversionPass()";
  let dependentDialects = ["linalg::LinalgDialect", "hivm::HIVMDialect"];
```

实测（`bishengir-opt -convert-hfusion-to-hivm`，exit 0）：

- 输入 `code/mlir-demo/hfusion-elemwise.mlir:5-13`：三条 `linalg.elemwise_unary`，分别带 `exp/abs/log` 属性。
- 输出 `code/mlir-demo/hfusion-elemwise.hivm.mlir:3-5`：

```mlir
hivm.hir.vexp ins(%arg0 : memref<6x6xf32>) outs(%arg1 : memref<6x6xf32>)
hivm.hir.vabs ins(%arg0 : memref<6x6xf32>) outs(%arg1 : memref<6x6xf32>)
hivm.hir.vln  ins(%arg0 : memref<6x6xf32>) outs(%arg1 : memref<6x6xf32>)
```

这就是“HFusion/Linalg 层 named op → HIVM 层硬件操作”的最小实证。日志：`code/logs/step5-hfusion-to-hivm.log`（`exit=0`）。

### 2. 编译器 driver 里的 pipeline 顺序

`AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/PassPipeline.cpp:71-108` 的 `buildBiShengHIRPipeline`：

1. `createCanonicalizeModulePass()`；
2. `hacc::createAppendDeviceSpecPass(...)` 追加目标设备规格；
3. 若 `enable-hfusion-compile`：`hfusion::buildHFusionPipelines(...)`（`:89-92`）；
4. 若 `enable-hivm-compile`：`hivm::buildConvertToHIVMPipeline(...)`，然后 `hivm::buildOptimizeHIVMPipeline(...)`（`:95-107`）。

所以正常顺序是 **HFusion pipeline → ConvertToHIVM pipeline → OptimizeHIVM pipeline**；开关分别是 `--enable-hfusion-compile` 和 `--enable-hivm-compile`。本次 VecAdd 只传了 `--enable-hivm-compile`，但 pass dump 里仍出现了 `ConvertHFusionToHIVM`（`code/logs/step6-hivm-passdump.log:32`），说明本次 `--enable-hivm-compile` 的 pipeline 中确实执行了该 conversion；对已经是 HIVM 的输入基本是直通（这是对 dump 的观察，具体由 pipeline 哪一段加入未逐行追）。

### 3. HIVM 之后到二进制

`AscendNPU-IR/bishengir/include/bishengir/Tools/hivmc/HIVMC.h:22-49` 的注释图给出了工具与开关对应关系：

```
MLIR --(HFusion)--> enable-hfusion-compile
     --(HIVM)-----> enable-hivm-compile
     --(HIVM Minimal OpSet) --> convert-hir-to-lir
     --(LLVM IR) -----------> enable-lir-compile
     --(binary)
```

文档 `architecture.md:83` 也明确：`bishengir-compile` 输入输出都是 MLIR，负责把高层 Tile op 降到感知 NPU 硬件的 low level op；`hivmc` 再把 low level MLIR 转成 LLVM IR 并最终生成算子二进制。

看起来和“一条 `bishengir-compile ... -o kernel.o` 直接得到 ELF”矛盾，实际不矛盾：`bishengir-compile` driver 在跑完 HIR pipeline 后会调用 hivmc，源码 `bishengir/lib/Tools/bishengir-compile/BiShengIRCompileMain.cpp:300` 是 `runHIVMCCompileA3(mod, hivmcConfig)`。所以一条命令内部包了两段：HIR（HFusion/HIVM）→ hivmc（LLVM IR/binary）。

## HACC 在这个链路里的位置

`architecture.md:11` 说 HACC 负责“异构硬件抽象表达”。在真实 IR 里能看到它承担 Host/Device 边界标记：

- `code/vecadd/add.mlir:2`：函数属性 `{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}`，声明这是设备侧入口函数。
- 编译后 dump 中函数属性增加 `hivm.func_core_type = #hivm.func_core_type<AIV>`（`code/logs/step6-hivm-passdump.log:2100-2101`），说明 HIVM 阶段把核类型也落到了函数上。

## 本次 VecAdd 覆盖到哪一层

`add.mlir` **已经手写在 HIVM 层**（`hivm.hir.load/vadd/store` + `#hivm.address_space<gm/ub>`），所以 VecAdd 实测覆盖的是：

- HIVM → HIVM 优化/降级 → LLVM IR → 设备 ELF → NPU 执行；
- HFusion → HIVM 的 conversion 单独用 `linalg.elemwise_unary` 例子验证过；
- 11-02 补了同一份 HFusion 输入的全链：`code/hfusion-vecadd/add-hfusion-host.mlir`（`linalg.elemwise_binary{add}`）→ HFusion pipeline → HIVM → 设备 ELF `add_single_outlined_0_0_0` → NPU 执行 → 32/32 数值匹配。证据：`code/logs/step6-hfusion-chain.log`、`step6-hfusion-conversion.log`、`step6-run-hfusion.log`。

全链需要两段工具链（包内 1.1.0 跑 HFusion pipeline 会 SIGSEGV），不是上游的单命令路径，边界说明见 `learning-report.md` 第 5 节与 `note/04` 第 7 节。
