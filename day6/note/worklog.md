# Day 6 worklog（2026-10-31，修订 2026-11-02）

时间取自命令输出与产物 mtime；同一天内按顺序记录。

## 2026-10-31 初版

| 时间 | 动作 | 结果/证据 |
|---|---|---|
| 17:48 | 创建 `infra-learning/day6/{logs,mlir-demo,vecadd,notes}` | 目录存在 |
| 17:48 | 环境盘点：NPU、CANN、工具位置/版本、两份 AscendNPU-IR checkout | `code/logs/step0-env.log`：910B3 / NPU 7 / Chip Logic ID 0；`bishengir-* 1.1.0`；`mlir-opt LLVM 19.1.7` |
| 17:48 | 写 `code/mlir-demo/demo.mlir`（13 行，arith/func/builtin） | 文件 |
| 17:48 | `mlir-opt demo.mlir` 解析打印 | exit 0，`code/logs/step1-before.mlir` |
| 17:48 | `mlir-opt --pass-pipeline='...canonicalize,cse'` | exit 0，`code/logs/step1-after.mlir`：`2+3` 折成 `5`，两条 `muli` 合成一条 |
| 17:48 | `--mlir-print-ir-after-all` 抓逐 pass IR | `code/logs/step1-ir-after-all.log` |
| 17:49 | 读 `architecture.md`、quick_start 三篇、VecAdd 四文件 | 引用行号见 `note/02`、`note/03` |
| 17:49 | 复制 `add.mlir` / `main.cpp` 到 `code/vecadd/`；`bishengir-compile` | exit 0，`code/vecadd/kernel.o` 2648 B，ELF machine 0x1029，符号 `add` 212 B |
| 17:49 | 检查 CANN 头文件布局，发现 `rt.h` 不在 `include/experiment/...` | 见 `note/04` 第 2 节 |
| 17:49 | 加 `hostshim/`，`g++` 编 host wrapper | exit 0，`code/vecadd/hivm-vec-add` 26368 B，ldd 解析到 CANN lib64 |
| 17:49 | `./hivm-vec-add` 上卡 | exit 0，16 行 1..16 全对，`code/logs/step4-run.log` |
| 17:50 | 独立数值校验 | `NUMERIC_CHECK=PASS`，`code/logs/step4-numeric-check.log` |
| 17:50 | `--mlir-print-ir-after-all` 重编，抓 pass 链 | exit 0，`code/logs/step4-compile-passdump.log` 2135 行；与首次产物 sha256 相同 |
| 17:50 | `bishengir-opt -convert-hfusion-to-hivm` 跑 HFusion 小例 | exit 0，`linalg.elemwise_unary` → `hivm.hir.vexp/vabs/vln`，`code/mlir-demo/hfusion-elemwise.hivm.mlir` |
| 17:51 | 写笔记、复现脚本、报告 | `note/01..04`、`code/repro.sh`、`learning-report.md` |

## 2026-11-02 修订

| 时间 | 动作 | 结果/证据 |
|---|---|---|
| 08:48 | 写单 op HFusion 输入 `code/hfusion-vecadd/add-hfusion-host.mlir`，以及同结构的 `mul-hfusion-host.mlir` 负例 | 文件；均为 `linalg.elemwise_binary`，`tensor<16xi16>` |
| 08:49 | 修 `code/vecadd/main.cpp`：加 `EXPECT_RT_OK`，检查 `rtKernelLaunch` 返回码；`runCase()` 逐元素比较并累计 mismatch；`main()` 返回非零；`argv[1]/argv[2]` 选择内核与符号 | `main.cpp:73-80,179-183,195-204,220-222,274-281` |
| 08:50 | 写 `code/hfusion-vecadd/extract-device-kernel.py`、`code/hfusion-vecadd/build.sh`；把 `code/repro.sh` 重写为 6 步 | 文件 |
| 08:51 | 重编 HIVM 内核、重编 host、上卡跑两组输入（官方 + 负数/零/边界） | exit 0，32/32 匹配；`code/logs/step6-run-hivm.log`、`step6-numeric-check.log` |
| 08:51 | HFusion 输入经源码构建的 `bishengir-opt` 跑三段 pipeline → 抽 DEVICE 函数 → 包内 `bishengir-compile` 出 ELF | exit 0；2752 B ELF，符号 `add_single_outlined_0_0_0`；`code/logs/step6-hfusion-chain.log`、`step6-hfusion-conversion.log` |
| 08:51 | HFusion 内核上卡并校验 | exit 0，32/32 匹配；`code/logs/step6-run-hfusion.log` |
| 08:51 | mul 负例上卡 | 30/32 mismatch，exit 1；`NEGATIVE_CONTROL=PASS`；`code/logs/step6-run-negative-control.log` |
| 08:51 | 记录工具链版本与包内 HFusion 编译崩溃（去掉 `head` 重抓，得到真实退出码） | `code/logs/step6-toolchain.log`；`code/logs/step6-packaged-hfusion-crash.log`（exit 139） |
| 08:52 | 汇总两种内核的 ELF/符号/sha256、host 链接库 | `code/logs/step6-artifacts.log` |
| 08:52 | 按新日志更新报告、README、notes | 本文件、`learning-report.md`、`README.md`、`note/03`、`note/04` |
| 08:58 | 最终整体重跑 `code/repro.sh`（并把 artifacts 汇总纳入脚本） | `repro exit=0`、`ALL_CHECKS=PASS`；日志刷新为最终版本 |
| 08:59 | 把 HIVM 逐 pass dump 纳入 `code/repro.sh` 并重抓，替换报告里 10-31 的 dump 引用 | `code/logs/step6-hivm-passdump.log`，IR 本体 2135 行，产物 sha256 仍为 `8648e77e...7f4f` |

## 偏差

- 未按 VecAdd 的 CMake target 构建 host，改为直接 `g++`（原因：构建树未开 `BISHENGIR_BUILD_EXAMPLES`，不值得全量重建）。
- 未显式指定 `--target`，使用文档默认值 `Ascend910B1`；在 910B3 上运行成功。
- HFusion 输入到 ELF 已在 11-02 补上，但不是上游的单命令路径：包内 1.1.0 对任何 `--enable-hfusion-compile` 输入 SIGSEGV，因此拆成“源码构建 `bishengir-opt` 跑 HFusion/HIVM pipeline + 包内 `bishengir-compile` 出 ELF”，中间还要用脚本删掉 HOST 包装函数。
- 11-02 的 `step6-*` 日志是当前报告引用的运行证据；10-31 的 `step4-*`、`step5-*` 保留为历史记录，其中 `step5-hfusion-to-hivm.log` 仍是 conversion pass 的原始证据。
