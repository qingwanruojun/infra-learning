# 01 · 用最小例子讲清 dialect / operation / pass / SSA

## 例子与运行方式

例子：`code/mlir-demo/demo.mlir`（13 行）

```mlir
module {
  func.func @fold_and_cse(%a: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %sum = arith.addi %c2, %c3 : i32
    %x = arith.muli %a, %sum : i32
    %y = arith.muli %a, %sum : i32
    %z = arith.addi %x, %y : i32
    return %z : i32
  }
}
```

实际使用的 `mlir-opt`：

```bash
MLIR_OPT=/mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR/build/bin/mlir-opt
$MLIR_OPT --version          # LLVM 19.1.7
$MLIR_OPT code/mlir-demo/demo.mlir > code/logs/step1-before.mlir       # 解析/打印，exit 0
$MLIR_OPT --pass-pipeline='builtin.module(func.func(canonicalize,cse))' \
    code/mlir-demo/demo.mlir > code/logs/step1-after.mlir              # 跑 pass，exit 0
```

- 运行前 IR：`code/logs/step1-before.mlir`
- 运行后 IR：`code/logs/step1-after.mlir`
- 每个 pass 后的 IR：`code/logs/step1-ir-after-all.log`
- 两条命令的退出码：`code/logs/step1-parse.log`、`code/logs/step1-pass.log`（均 `exit=0`）

## 1. dialect（方言）

dialect 是 MLIR 给“一组 operation/type/attribute”起的命名空间，决定 op 的语义和合法性检查规则。看 `demo.mlir` 里 op 的前缀即可辨认：

| 位置 | 文本 | dialect |
|---|---|---|
| `demo.mlir:3` | `module { ... }` | `builtin`（内建，打印时不带前缀） |
| `demo.mlir:4` | `func.func @fold_and_cse` | `func` |
| `demo.mlir:5-10` | `arith.constant` / `arith.addi` / `arith.muli` | `arith` |

一个 op 的完整名字是 `dialect.opname`。MLIR 解析时必须已经注册对应 dialect，否则会报 unknown operation；`mlir-opt` 内置了标准 dialect，所以这个例子直接可跑。NPU 侧同一个机制：`hivm.hir.load` 属于 `hivm` dialect，`hacc.entry` 属于 `hacc` dialect（见 `AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMBase.td:69`，op 助记符统一加 `hir.` 前缀）。

## 2. operation（操作）

operation 是 IR 的基本指令节点。以 `demo.mlir:7` 为例：

```mlir
%sum = arith.addi %c2, %c3 : i32
```

- op 名：`arith.addi`
- 操作数（operands）：`%c2`、`%c3`
- 结果（results）：一个，绑定到 SSA 名 `%sum`
- 结果类型：`i32`（冒号后）
- 属性（attributes）：本例没有；`demo.mlir:4` 的 `@fold_and_cse` 是符号名，`func.func` 还可以带 `attributes {...}`

`func.func` 本身也是 op，它带一个 region，region 里有一个 block，block 里才是算术 op。`module` 同理是包住所有内容的 op。

## 3. SSA（静态单赋值）

SSA 的规则：每个值只被定义一次，使用前必须已定义；名字只是引用，不是可变变量。

在 `demo.mlir` 中：

- `%a` 是函数入口 block 的参数，相当于“已定义”。
- `%c2`、`%c3`、`%sum`、`%x`、`%y`、`%z` 各自只出现一次定义。
- `%sum` 在 `:8`、`:9` 被使用两次，这是允许的；`%x` 只在 `:10` 使用。

一个能说明“名字不是语义”的现象：`mlir-opt` 解析再打印后，`%a` 被重命名为 `%arg0`，常量被重命名为 `%c2_i32` / `%c3_i32`，中间值变成 `%0..%3`（`code/logs/step1-before.mlir:2-8`）。SSA 的语义靠“定义-使用关系”，不靠文本名字。

## 4. pass（变换）

pass 是对 IR 做的一次结构化变换。这里跑两个标准 pass：

```bash
--pass-pipeline='builtin.module(func.func(canonicalize,cse))'
```

意思是：在 `builtin.module` 上，对其中的 `func.func` 依次跑 `canonicalize`、`cse`。

实测结果（`code/logs/step1-after.mlir`）：

```mlir
func.func @fold_and_cse(%arg0: i32) -> i32 {
  %c5_i32 = arith.constant 5 : i32
  %0 = arith.muli %arg0, %c5_i32 : i32
  %1 = arith.addi %0, %0 : i32
  return %1 : i32
}
```

逐条对应：

1. **常量折叠**：`arith.addi %c2, %c3` → `arith.constant 5`。证据是 `code/logs/step1-ir-after-all.log` 第一个 dump（Canonicalizer 之后）里已经没有 `%c2/%c3` 的加法，只剩 `%c5_i32`。
2. **CSE（公共子表达式消除）**：原来的 `%x` 和 `%y` 是两条完全相同的 `arith.muli %a, %sum`；CSE 后只保留一条 `%0`，`arith.addi` 的两个操作数都变成 `%0`。证据是同一日志第二个 dump（CSE 之后）里只剩一条 `muli`。
3. **pass 顺序有影响**：Canonicalizer 之后的中间 IR 仍有两条 `muli`（两条都已被折叠成 `%c5_i32` 形式），CSE 之后才合并。把顺序反过来跑 `cse,canonicalize` 最终结果相同，但中间形态不同；这解释了为什么真实编译器要精心编排 pipeline（NPU 侧的例子见 `PassPipeline.cpp:71-108`）。

## 5. 与 NPU 编译的联系

同一套机制换了 dialect 和 pass：

- `bishengir-opt -convert-hfusion-to-hivm` 是一个注册过的 pass，把 HFusion/linalg 层 op 改写成 `hivm.hir.*`（实测见 `note/02-hfusion-hivm.md`）。
- `bishengir-compile` 内部用 `OpPassManager` 串起 HFusion/HIVM pipeline（源码 `AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/PassPipeline.cpp:71-108`）。

所以“学 MLIR”不是学一个抽象名词，而是学会读这三样东西：op 属于哪个 dialect、它定义/使用了哪些 SSA 值、哪条 pass 把它改成了什么。
