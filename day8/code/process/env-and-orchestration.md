# Day 8 环境与编排实况记录

时间：2026-11-04（UTC+8）。所有命令实际执行，退出码来自 `PIPESTATUS`/`$?`。

## 1. 实际版本与路径

| 项 | 实测值 | 证据 |
|---|---|---|
| NPU | Ascend910B3，20 AI Core / 40 AIV，62.4 GB HBM，`ASCEND_VISIBLE_DEVICES=2` → `npu:0` | `npu-smi info`；`torch_npu.npu.get_device_name(0)` |
| CANN | `/home/developer/Ascend/cann-9.0.0` | `ASCEND_HOME_PATH` |
| torch | 2.7.1+cpu | import 实测 |
| torch_npu | 2.7.1.post4 | import 实测 |
| TileLang | 0.1.2+bd7c60c312b0721885c78d1ce03c62be9225314c，源码 `/mnt/workspace/learn/tilelang-ascend` | import 实测 |
| AscendNPU-IR | `/mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR/build/install`（已编译） | 现有产物 |
| cannbot-skills | `/mnt/workspace/learn/cannbot-skills`（2026-10-26 快照） | 目录 |
| bishengir-compile | CANN 9.0.0 自带 1.1.0 | `day7/code/logs/step0-env.log` |

## 2. 采用的工作流路径

选中 `cannbot-skills/plugins-official/tilelang-op-orchestrator/workflows/Ascend910/workflow.md`
（硬件探测 `full_soc=Ascend910B3`, `npu_arch=2201` → Ascend910 分支）。
本机没有把 cannbot 安装成宿主插件，因此由主代理扮演 orchestrator，
用 Agent 子代理分别承担：

- Stage 1：`tilelang-op-analyst`（读取 `tilelang-op-design` + `tilelang-op-test-design`）
- Stage 2：`tilelang-op-developer`（读取 `tilelang-op-develop` + `tilelang-op-test-design`）
- Stage 3：未做（用户未要求调优；Stage 2 通过后按工作流应询问，本任务以精度/可运行为主）

这是“工作流文档 + 角色定义 + 同名 skill”手工编排，不是 opencode 的
`.claude-plugin` 自动路由。报告中明确区分。

## 3. 环境阻断点与处理（按发生顺序）

### 3.1 `prepare_framework.sh` 走了非 NPUIR 构建，丢失 `libtilelangir.so`

命令：
```bash
bash .../workflows/Ascend910/scripts/prepare_framework.sh /mnt/workspace/learn/tilelang-ascend
```
脚本检测到仓库存在后没有 pull/reset 源码，但 `verify_environment` 要求
`set_env.sh`，该 revision 没有此文件，于是判定环境无效并执行
`install_ascend.sh`。该脚本 `rm -rf build` 后用 `USE_NPUIR=OFF` 重建，
结果 `tilelang` import 失败：

```
RuntimeError: Cannot find libraries: libtilelangir.so
```

处理：不删除任何剩余产物，把被覆盖的 build 改名为
`build.after_prepare_nonpui_20261104` 保留，然后按 `install_npuir.sh` 的
配置手工新建 `build/`：

```bash
cp 3rdparty/tvm/cmake/config.cmake build/config.cmake
printf '\nset(USE_NPUIR ON)\nset(BISHENGIR_ROOT_PATH /mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR/build/install)\n' >> build/config.cmake
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DPython3_EXECUTABLE=/opt/buildtools/Python-3.11.4/bin/python3 \
      -DMLIR_INCLUDE_TESTS=OFF ..
make -j$(nproc)
```

- `libtilelangir.so` / `libtilelang_module.so` 构建成功；
- 仅 `tilelangir-opt` 工具因 `libpthread` 未显式链接失败（`make` 总退出码 2），
  该工具不参与 Python JIT，不影响实验。日志：`code/logs/00-rebuild-npuir.log`。
- import 复测退出码 0。

### 3.2 `tilelang-env-check` 与实际 revision 不兼容

`check_env.sh` 退出码 1，原因两类：
1. 仓库没有 `set_env.sh`（脚本第 175 行直接 source 报 No such file）；
2. 子模块 `cutlass/composable_kernel/pto-isa/shmem/catlass` 未初始化（NPUIR 路径不需要）。

`quick_verify.py` 退出码 1，报 `module 'tilelang.language' has no attribute 'barrier_all'`
——该脚本用旧 Expert API，当前 revision 已移除。这不是 NPU 故障。
证据：`code/logs/01-env-quick-verify.log`。

处理：用实际可用的 NPUIR 路径做上板 smoke test：
`examples/elementwise/vec_add_1d.py`，退出码 0，输出 `All check passed`。
证据：`code/logs/02-npu-smoke-vecadd.log`。因此把 `env_check_passed` 置 true，
但在报告“未完成/差异”里保留上述两个脚本失败事实。

### 3.3 文档与仓库的其它差异（analyst 已核实）

- 设计/开发 skill 引用的 `examples/{softmax,normalization,developer_mode,...}` 不存在；
  实际用 `examples/norm/rms_norm.py`。
- 设计模板里的 `TL_ASCEND_AUTO_CV_COMBINE` 等 pass key 在当前 revision 不存在；
  实际 key 是 `NPUIR_ENABLE_AUTO_MULTI_BUFFER` 等，本算子用默认值。
- 仓库示例 `examples/norm/rms_norm.py` 在 host 侧把 N padding 到 256，与本任务
  “禁止 host padding”冲突；设计改为直接用 N=1024。
- 没有 `set_env.sh`，所有运行显式传 `PYTHONPATH`。

## 4. 计时口径

- `e2e_us`：NPU Event 包住多次 kernel 调用，除以次数；与 Day 3 `benchmark()` 同口径。
- `host_us`：CPU 侧 `perf_counter_ns` 包住单次调用（不 sync），随后统一 sync。
- `device_us`：`torch_npu.profiler` Level1，解析 `kernel_details.csv` 的 `Duration(us)`；
  Day 3 kernel 实测 15.85 µs，而同一调用的 e2e 为 220.6 µs，确认 host 主导。
  证据：`code/logs/04-profiler-smoke.log`。
- 性能比较只在 device 口径有可靠值时进行；e2e 同时列出但标注 host 占比。

## 5. 状态文件与原始产物

- `.orchestrator_state.json`：Stage 1→2 的实际状态。
- `history_version/design_rev0_original.md`、`proto_rev0_original.yaml`：Stage 1 原件。
- Stage 2 的每次人工/代理修改由 developer 按 attempt 备份，另见 `history_version/CHANGES.md`。
