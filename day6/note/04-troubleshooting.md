# 04 · 失败排查与环境适配记录

记录原则：只写实际发生过的问题；没发生的编译失败不虚构。

## 1. 两个 AscendNPU-IR 副本，工具在哪

现象/风险：如果按直觉去 `AscendNPU-IR/build/bin` 找 `mlir-opt`，会扑空——主 clone `/mnt/workspace/learn/AscendNPU-IR` 没有 `build/`。

实际布局：

| 副本 | commit | 状态 | 用途 |
|---|---|---|---|
| `/mnt/workspace/learn/AscendNPU-IR` | `87ea6c4` | 未构建 | 读文档、读测试源码（本日笔记的源码依据） |
| `/mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR` | `31f690369` | 已构建 | `build/bin/mlir-opt`、`build/bin/bishengir-opt` 等 |
| CANN 9.0.0 安装 | 打包版本 `1.1.0 (428ab8fd)` | 可直接用 | `bishengir-compile`、`bishengir-opt` |

处理：第 1 步用已构建副本里的 `mlir-opt`（LLVM 19.1.7）；编译设备二进制用 CANN 安装包里的 `bishengir-compile`（避免源码构建版本与 CANN 运行时不配套）。11-02 的 HFusion 全链又用了已构建副本的 `bishengir-opt` 跑 HFusion/HIVM pipeline（原因见第 7 节）。版本记录在 `code/logs/step0-env.log` 与 `code/logs/step6-toolchain.log`。

## 2. `experiment/runtime/runtime/rt.h` 找不到

现象：VecAdd 的 `main.cpp` 顶部写的是

```cpp
#include "experiment/runtime/runtime/rt.h"
```

但 CANN 9.0.0 的 `include/experiment/runtime/runtime/` 下只有 `base.h`、`rt_ffts.h`，没有 `rt.h`；`rt.h` 实际在：

```
/home/developer/Ascend/cann-9.0.0/pkg_inc/runtime/runtime/rt.h
```

文档 `quick_start/examples.md:49` 的等价示例用的是 `#include "runtime/runtime/rt.h"`，并加了 `-I${ASCEND_HOME_PATH}/pkg_inc`（`examples.md:207`）。

处理：不改上游 `main.cpp`，在 `code/vecadd/hostshim/experiment/runtime/runtime/rt.h` 放一行转发：

```cpp
#include "runtime/runtime/rt.h"
```

编译时 `-Ihostshim` 放在最前。host 编译 exit 0。这个 shim 是本机适配，不是例子本身的一部分。

## 3. 设备号：npu-smi 的 7 与 aclrtSetDevice(0)

现象：

- `npu-smi info` 显示 `NPU 7  910B3`，Chip Logic ID 为 `0`。
- `npu-smi info -t board -i 0` 报 `Invalid card id`——这里的 `-i` 要的是 NPU ID（7），不是 logic ID。
- `main.cpp:124` 写死 `aclrtSetDevice(0)`。

处理/验证：直接运行，`aclrtSetDevice(0)`、`aclrtCreateStream` 均成功，kernel 正常执行。说明本环境下 ACL 逻辑设备 0 对应当前可见的 910B3（npu-smi 的 Chip Logic ID 0）。这是“运行成功”反推的结论，未单独打印设备物理 ID。

## 4. 构建树里没有 VecAdd host target

现象：`ninja -C .../build -t targets | grep -i vec-add` 无输出。例子 target（`bishengir-npu-hivm-vec-add`）只有在 CMake 配置时带 `-DBISHENGIR_BUILD_EXAMPLES=ON` 才会生成（`README.md:21-33`）。

处理：不为一个 26 KB 的 host wrapper 重建整个 LLVM/MLIR 工程。改用 `README.md` / `examples.md` 记录的 `g++` 命令直接编译同一个 `main.cpp`，链接 CANN 的 `runtime`/`ascendcl`。已验证 exit 0、可执行、可上卡。

未验证项：CMake target 路径本身在本机没有跑过。

## 5. “compare output success” 曾不是逐元素断言（已修）

现象（上游 `main.cpp:185-189`）：循环只 `printf`，最后一行的 `[success] compare output success` 是无条件打印；`rtKernelLaunch` 的返回码也没有接收。风险是“打印了”被误当成“校验通过”。

处理（本地 `code/vecadd/main.cpp`）：

- 新增 `EXPECT_RT_OK`，检查 `rtKernelLaunch` 等调用（`:73-80,179-183`）；
- `runCase()` 逐元素比较并累计 mismatch（`:195-204`）；
- `main()` 有 mismatch 时返回 1（`:274-281`）。

负例：同一条 HFusion 链路把 `add` 换成 `mul`，得到 `code/hfusion-vecadd/bad-kernel.o`；同一 host 程序报 `30 mismatched element(s)`、exit 1（`code/logs/step6-run-negative-control.log`）。这证明校验会真的失败，不是无条件打印。

## 6. 编译日志为空

现象：`bishengir-compile add.mlir --enable-hivm-compile -o kernel.o` 的 stdout/stderr 都是空，命令日志只有命令行本身（`code/logs/step6-compile-hivm.log`）。

判断：不是失败。退出码 0、产物存在、ELF 合法、上卡成功。若需要过程证据，用 `--mlir-print-ir-after-all` 重新编译，得到 `code/logs/step6-hivm-passdump.log`（IR dump 本体 2135 行，exit 0）。

## 7. 包内 1.1.0 跑 HFusion pipeline 必 SIGSEGV

现象：在 CANN 9.0.0 包内 `bishengir-compile` 上使用 `--enable-hfusion-compile`，无论输入是自写的 `add-hfusion-host.mlir` 还是上游 HFusion 测试，都稳定崩溃：

```text
$ bishengir-compile add-hfusion-host.mlir --enable-hfusion-compile --block-dim=1 -o out.o
PLEASE submit a bug report ... include the crash backtrace.
...
exit=139 (SIGSEGV=139)
```

证据：`code/logs/step6-packaged-hfusion-crash.log`。崩溃发生在 `HIVMAggregatedDecomposeOp`；不是输入格式问题（上游测试同样崩）。

绕法（不是上游正常路径）：把链路拆成两段。

1. 已构建副本的 `bishengir-opt`（commit `31f690369`）跑
   `--lower-hfusion-pipeline --convert-to-hivm-pipeline --optimize-hivm-pipeline`，
   得到优化后的 HIVM；
2. `extract-device-kernel.py` 删掉与 DEVICE 入口同名的 HOST 包装函数（旧的 `hivmc` 无法合法化其中的 HIVM op）；
3. 包内 `bishengir-compile --enable-hivm-compile` 把 HIVM 编成设备 ELF。

产物与运行结果见 `code/logs/step6-hfusion-chain.log`、`code/logs/step6-run-hfusion.log`。源码构建的 `bishengir-compile` 也不行：它会向 CANN 旧 `hivmc` 转发 `--enable-cpu-trace-intrinsic=false`、`--enable-lir-compile=true` 等未知选项。

## 8. HFusion 层试过的其它函数形态（均失败）

在找到可用链路前，试过几种 HFusion 层 DEVICE 函数写法，均未成功：

- 返回 `%sum` 且参数是 tensor：后端报 `ERROR: error pointer address space cast`；
- 返回 void：未被使用的 `linalg` op 被 DCE，最终内核只做 `set_mask_norm`/`pipe_barrier`，输出全零；
- memref 级 `linalg.elemwise_binary`（带/不带 gm 地址空间）：转换出 GM 空间的 `vadd`，临时 buffer 为空，仍报 address-space cast。

最终可行的是“HFusion host 函数 + 完整 pipeline + 抽取 DEVICE 函数”。负例的第一次尝试也踩了 DCE：mul 的结果没被使用，内核只输出 0，只能证明“能发现错”，不能证明“能发现乘错”；改成保留完整 `linalg.elemwise_binary{mul}` 后才得到真正的 `hivm.hir.vmul` 内核。

## 9. 没有发生的失败

10-31 的 `add.mlir` 首次编译、host 首次编译、首次上卡均成功，没有需要绕过的问题。11-02 的问题集中在 HFusion 全链（第 7、8 节），属于工具链缺陷与写法试探；VecAdd 主链路本身仍然稳定。
