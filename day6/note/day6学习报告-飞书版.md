# Day 6｜从 MLIR 到 NPU：把验证补完整

1. **补上了 HFusion 输入到 NPU 的实验。**同一份 `linalg.elemwise_binary{add}` 输入先转成 HIVM，再编成 2752 B 的设备 ELF，最后在 910B3 上运行；两组共 32 个结果全部正确。这次用的是两段工具链，中间还抽取了 DEVICE 函数，并非包内编译器一条命令跑通。
2. **把 VecAdd 的校验做实了。**host 现在检查 `rtKernelLaunch` 返回码并逐元素比较；官方输入和一组负数、零、边界值输入共 32 项全对。故意换成乘法内核后，程序报 30 处不匹配、退出码为 1。上次那句无条件打印的“成功”，终于有了真正的检查来替代。

## 从小 MLIR 例子开始

我先用 `code/mlir-demo/demo.mlir` 跑 `mlir-opt`。`2+3` 折成常量 `5`，两条相同的乘法被 CSE 合并。对着 `code/logs/step1-before.mlir` 和 `step1-after.mlir` 看，dialect、op、SSA、pass 才不再只是四个名词：值定义一次、被后续 op 使用，pass 则一步步改写 IR。这一步是编译器入门实验，没有上卡。

另外，`bishengir-opt -convert-hfusion-to-hivm` 把三条 Linalg 逐元素 op 变成 `hivm.hir.vexp/vabs/vln`（`code/logs/step5-hfusion-to-hivm.log`）。这里验证的是转换，不是 exp、abs、log 的数值正确性。

## HIVM VecAdd：这次真的比较了结果

`code/vecadd/add.mlir` 已经是 HIVM IR：两个输入从 GM 搬到 UB，在 UB 做 `vadd`，再写回 GM。编译得到 2648 B 的 `kernel.o`，符号是 `add`。上次 host 虽然打印了正确数字，末尾的“compare output success”却是无条件输出。这次我在 `code/vecadd/main.cpp` 里补了发射返回码检查和逐元素比较，出错会返回非零。

第一组是官方的 `0..15 + 1 → 1..16`；第二组放入负数、零、正负抵消和 `32767/-32768` 边界输入，每项预期和仍在 int16 范围内。两组各 16 项，程序内比较和独立日志检查都通过。为了确认检查不是空的，我又用同一个 host 运行乘法内核：IR 确实包含 `vmul`，32 项中有 30 项不符合加法预期，程序退出码为 1。

证据是 `code/logs/step6-run-hivm.log`、`step6-numeric-check.log`、`step6-run-negative-control.log`，设备产物和符号见 `step6-artifacts.log`。

## HFusion 输入：终于用同一份输入走到了上卡

10 月 31 日的转换实验和 VecAdd 上卡用的是两份不同输入。这次新写 `code/hfusion-vecadd/add-hfusion-host.mlir`，其中只有一条 `linalg.elemwise_binary{add}`。逐 pass dump 能看到它变成 HIVM 的 `load/load/vadd/store`；抽出 DEVICE 入口后，包内编译器生成 2752 B 的 ELF，host 加载 `add_single_outlined_0_0_0` 并上卡。两组输入仍是 32/32 正确，独立日志比较也通过。

这样可以说“HFusion 输入到设备执行已跑通”。但输入只有一个逐元素 op；虽然经过了 HFusion pipeline 和 outline pass，**这组实验没有验证多算子融合**。转换、设备编译和运行证据分别在 `code/logs/step6-hfusion-conversion.log`、`step6-hfusion-chain.log`、`step6-run-hfusion.log`。

## 卡点和处理方式

CANN 9.0.0 包内 `bishengir-compile 1.1.0` 对这份 HFusion 输入跑 `--enable-hfusion-compile` 时以 139 退出，崩溃栈落在 `HIVMAggregatedDecomposeOp`。所以本次改用源码构建的 `bishengir-opt` 跑 HFusion/HIVM 前半段，脚本去掉 HOST 包装函数，最后由包内编译器从 HIVM 生成 ELF。这条路径能完成本次实验，但不是上游原本的一条编译命令。日志只能支持“当前环境下已试输入会崩溃”，不能推成所有 HFusion 输入都不行。见 `code/logs/step6-packaged-hfusion-crash.log` 和 `step6-toolchain.log`。

## 还没做到的

- 只有单个 16 元素 int16 加法、两组数据、一台 910B3；没有性能、其他 shape/dtype 或多算子融合结果。
- `code/repro.sh` 的负例判定把任何非零退出都算通过。本次日志明确记录了 30 处数值不匹配，因此本次结论成立；后续应同时检查 mismatch 文本，避免注册失败或崩溃误报为通过。
- `code/hfusion-vecadd/extract-device-kernel.py` 按 IR 文本行抽取函数，目前只在这份 IR 上验证过。换复杂输入时要额外验证抽取结果。

这次最大的收获是把 **IR 能转换、设备文件能生成、输出真的正确** 分开看。现在每一步都有相应文件和日志，没测到的地方也能说清楚。复现命令和证据索引在根目录 `README.md`；更细的技术记录在 `note/learning-report.md` 和四篇笔记中。
