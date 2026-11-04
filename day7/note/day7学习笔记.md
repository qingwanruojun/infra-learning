# Day 7｜顺着一次 VecAdd，看清 TileLang 到设备代码的路

1. **把同一次运行的编译链路接起来了。**从 `vec_add_1d.py` 出发，留下 TVM TIR、初始 npuir、两次 pass 后的 final npuir，再把同一份 final npuir 交给 `bishengir-compile` 看下游变化。原 JIT 的设备文件和另行编译的文件逐字节相同；主运行退出码为 0，示例的 `assert_close` 通过。
2. **弄清了 TileLangIR 在这次编译里做了什么。**我原本想在 TVM TIR 后面找一份独立的 TileLangIR，顺着日志和源码查下来，看到的是 `libtilelangir.so` 提供 pass 调度；这次实际运行的 pass 是 `canonicalize` 和 `adapt-triton-kernel`，没有走 `CVSplit` 或 `Vectorize`。因此链路图应按抓到的 IR 和实际执行的 pass 来画。

实验记录来自 11 月 2 日的 910B3 / CANN 9.0.0 环境。我根据交付包里的脚本、IR 和原始日志整理这份笔记，没有在本机重新上卡。下文的 `code/` 路径均相对 Day 7 根目录。主运行见 `code/logs/vecadd-dev-dump.log` 和 `vecadd-dev-exit.log`；完整命令在 `code/commands.md`。

## 先跑通一个能看懂的例子

这次只选了 `examples/elementwise/vec_add_1d.py`。输入是长度 4096 的 f32 向量，`block_N` 也设为 4096，因此只有一个 block：从 GM 把 A、B 搬到 UB，做一次 `vadd`，再把 C 写回 GM。例子足够小，每一级 IR 都能找到同一件事，没有被复杂调度盖住。

我打开 `TILELANG_DUMP_IR` 跑 Developer 模式，把 stdout 和 stderr 合到一份日志。里面依次有 `TVM IR`、`npuir`、`canonicalize` 和 `adapt-triton-kernel` 的逐 pass 输出，以及 `final npuir`。`code/extract_ir.py` 从**这同一次运行**切出各阶段文件，记录为 `EXTRACT=PASS`。运行末尾是 `All check passed`，退出码为 0；这里的数值正确性来自示例自身的 `torch.testing.assert_close`，不是我另外写的一套逐元素校验。

## 沿着加法往下看

Python 里写的是 `T.vadd(A_VEC, B_VEC, C_VEC)`。在 TVM TIR 里，它变成带读写 region 的 `T.npuir_add`（`code/ir/step1-tvm-ir.tir`）；进入 Developer codegen 后，初始 npuir 里能找到 `hivm.hir.vadd`（`code/ir/step2-initial-npuir.mlir`）。这一步让我真正把源代码中的一行加法和生成的 op 对上了，而不是只凭两个文件的名字判断它们有关。

接下来 `lower.py` 实际调度了 `canonicalize` 和 `adapt-triton-kernel`。对这份输入，前者清掉了未使用的 block index 等内容，后者没有进一步改变 IR；final npuir 里的 `vadd` 仍然在。差异文件是 `code/logs/step3-initial-vs-final.diff` 和 `step2-canonicalize-vs-adapt.diff`。这里不能把“运行了一个 pass”直接写成“完成了一次有效优化”。

再往下，编译器给 GM/UB 标出地址空间，安排 UB 偏移，最后把计算降成两次 `load_gm_to_ubuf_1d_float`、一次 `vadd_1d_float`、一次 `store_ubuf_to_gm_1d_float`，中间还有同步指令（`code/ir/step5-final-hivm-lowered.mlir`）。三个 4096×f32 的逻辑数据并没有各占一块 UB：A 和输出 C 复用偏移 0，B 在偏移 16384。这是根据降级后的指针和读写顺序判断的，不是单看 buffer 名称猜的。

## TileLangIR 到底在不在这条路上

这是今天最容易讲错的地方。`lower.py` 使用的 `Pipeline` 来自 `tilelang.tladapter`，运行时加载了 `libtilelangir.so`；日志也记录了加载路径。但主运行的 pass marker 只有 `canonicalize` 和 `adapt-triton-kernel`，没有 `CVSplit`、`Vectorize`。源码里这两个 pass 还是占位实现。我又把 final npuir 单独喂给它们作路径对照，命令成功退出，IR 内容没有实质变化（`code/logs/step6-tilelangir-load-check.log`、`code/ir/out-of-path-tilelangir-*.mlir`）。

所以这次实际看到的是：

```text
TileLang Python → TVM TIR → npuir/MLIR → HIVM 下游降级 → 设备 ELF → 上卡校验
```

`TileLangIR` 这个名字在这里指到了支撑 pass 调度的库，不能因此在 TVM TIR 和 npuir 之间再画一级独立 IR。TVM 侧还有个叫 `NpuLoopVectorize` 的 transform，它也不是这里的 `tilelangir/Vectorize.cpp`。

## 为了确认设备产物，补了一步比对

JIT 编译时的临时 `kernel.o` 原本会随临时目录清理。我用一个 shim 记录真实编译命令并复制原 JIT 产物；随后把提取出的 final npuir 再交给同一个 `bishengir-compile`。另行编译得到 2784 B 的设备 ELF，原 JIT 的 `.o` 与它逐字节相同，host `.so` 也相同（`code/logs/step7-jit-vs-recompiled.log`）。这样可以把“从同一份 IR 看下游 pass”和“原运行实际用了什么产物”接上，但下游逐 pass dump 仍是**另一次编译调用**，不是主运行当场吐出的日志。

中途也试过单独跑 `--hivm-flatten-ops` 和 `--hivm-plan-memory`。前者 abort，后者缺少合适的前置状态，虽然退出码为 0 却报错且 IR 未变。我保留了这些日志，没有把单独运行的结果写成完整编译流水线的结果。完整编译的 pass dump 和对应报错在 `code/logs/`。

## 还没做到的

- 只测了 N=4096、f32、单 block；tail、其他 dtype、多 block 和 2D grid 都没覆盖。
- 示例的 `assert_close` 证明本次输出与参考结果一致；这次没有像 Day 6 那样另做负例来检验校验逻辑。
- 没有测性能，也没有单独核对 `npu-smi` 所示物理 NPU 编号与 `torch.npu.set_device(0)` 的映射。

这一天最有用的收获是学会沿着**同一次输入**追每个阶段，并分清源码里“有这个 pass”、进程里“加载了这个库”和运行时“真的执行了变换”。需要逐行源码位置、pass dump 定位和完整复现步骤时，再看 [`learning-report.md`](learning-report.md) 与 [`../code/README.md`](../code/README.md)。
