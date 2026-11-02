# Day 6｜MLIR 与 AscendNPU-IR

这一天做了两件主要的事：先用小 MLIR 例子看清 pass 怎样改写 IR，再分别从 HIVM 和 HFusion 输入出发，把 16 元素 int16 加法编成设备 ELF、放到 NPU 上运行并核对输出。10 月 31 日的初版只跑通 HIVM VecAdd；11 月 2 日补上了 host 逐元素校验、乘法负例，以及同一份 HFusion 输入到设备执行的实验。

先读 [学习报告](note/day6学习报告.md) 可以快速了解过程和结论；需要复核具体数字时再看下面的代码、IR 和原始日志。

## 目录怎么读

```text
day6/
├── README.md                         本文件：入口、结果、复现和证据索引
├── code/
│   ├── repro.sh                      重跑编译、上卡和校验的入口
│   ├── mlir-demo/                    最小 MLIR 例子与 HFusion 转换例子
│   ├── vecadd/                       HIVM VecAdd、改过的 host、设备与 host 产物
│   ├── hfusion-vecadd/               HFusion 加法输入、乘法负例、转换与抽取脚本
│   └── logs/                         原始命令、逐 pass IR、运行和数值校验日志
└── note/
    ├── day6学习报告.md               本次主报告，写给人读
    ├── day6学习报告-飞书版.md          同一正文的飞书排版稿
    ├── day6学习记录-20261102.md      早期整理稿，保留作修改记录
    ├── learning-report.md            原交付包的详细技术报告
    ├── worklog.md                    10 月 31 日与 11 月 2 日时间线
    └── 01～04-*.md                   MLIR、HFusion/HIVM、VecAdd、排查笔记
```

主报告与飞书版**正文相同**，飞书版只加章节分隔。`note/day6学习记录-20261102.md` 是早期整理稿；`note/learning-report.md` 和其他原始笔记保留了当时的更细记录。其中命令、日志路径应以本 README 的新目录为准。

## 结果一览

| 实验 | 已验证结果 | 主要证据 |
|---|---|---|
| 最小 MLIR | `2+3` 折成 `5`；重复乘法经 CSE 合并 | `code/logs/step1-before.mlir`、`step1-after.mlir` |
| Linalg 逐元素转换 | `exp/abs/log` 对应 `hivm.hir.vexp/vabs/vln`；只验转换 | `code/logs/step5-hfusion-to-hivm.log` |
| HIVM VecAdd | 2648 B 设备 ELF，符号 `add`；两组输入 32/32 正确 | `code/logs/step6-artifacts.log`、`step6-run-hivm.log`、`step6-numeric-check.log` |
| HFusion 输入到设备 | 单条 `linalg.elemwise_binary{add}` 经两段工具链得到 2752 B ELF；上卡 32/32 正确 | `code/logs/step6-hfusion-conversion.log`、`step6-hfusion-chain.log`、`step6-run-hfusion.log` |
| 错误内核负例 | 乘法内核产生 30/32 个不匹配；host 退出码 1 | `code/logs/step6-run-negative-control.log`、`code/hfusion-vecadd/mul-hfusion.device.hivm.mlir` |

两组输入分别是官方的 `0..15 + 1`，以及包含负数、零、正负抵消、int16 边界输入的混合组。`code/vecadd/main.cpp` 检查 `rtKernelLaunch` 返回码，并在 C++ 中逐项比较结果；`code/repro.sh` 还会从运行日志独立提取 Result 列再比较一次。

## 环境和工具链

- 设备：Ascend 910B3；本次运行日志记录的是远端机器上的执行，我整理报告时没有重新上卡。
- CANN：`/home/developer/Ascend/cann-9.0.0`；最终设备 ELF 由包内 `bishengir-compile 1.1.0` 生成。
- HFusion 前半段：`tilelang-ascend/3rdparty/AscendNPU-IR` 已构建副本的 `bishengir-opt`，commit `31f690369`。
- 小 MLIR 例子：同一已构建副本的 `mlir-opt`，LLVM 19.1.7。
- 版本和设备原始记录：`code/logs/step0-env.log`、`step6-toolchain.log`。

包内 `bishengir-compile --enable-hfusion-compile` 在本机的已试输入上 SIGSEGV（`code/logs/step6-packaged-hfusion-crash.log`）。因此 HFusion 路径采用“源码构建 `bishengir-opt` 转成优化后的 HIVM → 抽 DEVICE 入口 → 包内 `bishengir-compile` 出 ELF”。它验证了这份输入的完整运行结果，**不等于包内单命令路径已修复，也没有验证多算子融合**。

## 复现

在原远端环境、工具路径和 NPU 可用的前提下，从仓库根目录运行：

```bash
bash day6/code/repro.sh
```

脚本依次编译 HIVM 内核、构建 host、上卡运行、构建 HFusion 来源内核、上卡运行、用乘法负例检查校验逻辑，并做独立数值比较。成功时最后输出 `ALL_CHECKS=PASS`。脚本默认引用 `/home/developer/Ascend/cann-9.0.0` 和 `/mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR/build/bin/bishengir-opt`；换机器前先检查这些路径。源码构建的 `bishengir-opt` 路径可通过 `BUILT_BISHENGIR_OPT` 覆盖。

要单独看 HFusion 编译步骤，可运行：

```bash
bash day6/code/hfusion-vecadd/build.sh
```

输出位于 `day6/code/hfusion-vecadd/kernel.o` 和 `bad-kernel.o`。本仓库已保存当次设备 ELF、host 可执行文件与中间 IR，便于离线核对；这些文件依赖原设备与 CANN 环境，不能把本地文件存在当作本机运行成功。

## 证据索引

**11 月 2 日的当前结果**主要看：

| 文件 | 看什么 |
|---|---|
| `code/logs/step6-hivm-passdump.log` | HIVM VecAdd 的逐 pass IR |
| `code/logs/step6-hfusion-passdump.log` | HFusion 输入的逐 pass IR |
| `code/logs/step6-hfusion-conversion.log` | Linalg add 变成 HIVM load/vadd/store 的关键片段 |
| `code/logs/step6-artifacts.log` | ELF 类型、符号、SHA-256、host 链接库 |
| `code/logs/step6-run-hivm.log` | HIVM 内核两组输入的 32 项输出 |
| `code/logs/step6-run-hfusion.log` | HFusion 来源内核两组输入的 32 项输出 |
| `code/logs/step6-run-negative-control.log` | 乘法负例的 30 处不匹配和退出码 1 |
| `code/logs/step6-numeric-check.log` | 两次运行的独立结果比较 |
| `code/logs/step6-packaged-hfusion-crash.log` | 包内 HFusion 编译器崩溃记录 |

`code/logs/step4-*`、`step5-*` 是 10 月 31 日的历史证据。`step5-hfusion-to-hivm.log` 仍是小转换例子的有效记录，但旧的上卡输出和 host 校验请以 `step6-*` 为准。

## 还要注意的边界

本轮只测了单个 16 元素 int16 加法、两组数据和一台 910B3；没有性能、其他形状或 dtype、多算子融合的结论。`code/repro.sh` 的负例分支目前只要求程序非零退出，下一次完善时应再检查 mismatch 文本，避免无关故障被算作负例成功。`code/hfusion-vecadd/extract-device-kernel.py` 按当前 IR 的文本结构抽函数，换复杂输入时也要额外验证抽取结果。
