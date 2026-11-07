# Stage 2 Agent 时间线与命令留档

时间：2026-11-04（UTC+8）。Stage 2 由 `tilelang-op-developer` 子代理完成，
主代理只做门禁检查与独立验收，没有修改生成的 kernel。Stage 2 交付后，
主代理改过 `test_rmsnorm.py` 的门禁语义（L2/Boundary 改为阻塞），见第 5 节。

## 1. 尝试与修复

| 时间 | 事件 | 证据 |
|---|---|---|
| 09:23:24 | attempt1 生成（按 DESIGN 3.3 伪代码，无条件分配 x_f32/w_f32） | `history_version/rmsnorm_impl_s2_attempt1_original.py` |
| 09:23–09:24 | fp32 L0 运行时崩溃：`aivec error ... The DDR address of the MTE instruction is out of range`，retCode=0x31；静态 UB 204,928 B > 196,608 B | `history_version/CHANGES.md` 第 1 节；原始输出 `code/process/probes/01-fp32-mte-incident.log` |
| 09:24:36 | attempt2：删除 x_f32/w_f32，fp16 分支 inline cast；fp32 UB 135,296 B | `history_version/rmsnorm_impl_s2_attempt2_ub_fix.py` |
| 09:25:57 | L0 通过：max_abs=1.430511e-06，退出 0 | `logs/l0.log` |
| 09:26–09:27 | 全量 `--level all`：fp32/L2/Boundary 全过，3 个 fp16 用例编译失败（UB overflow 264,192 B），退出 1 | `logs/all_attempt1_fp16_compile_fail.log`、`.exit` |
| 09:27:29 | compile-only 探针：bm=16/12 fail，bm=10/8 OK，no-multibuffer bm=16 仍 fail | `code/process/probes/rmsnorm_fp16_variants.py` 与 `02-fp16-blockm-probe.log`；结果写入 `CHANGES.md` 第 2 节 |
| 09:28:20 | attempt3：fp16 `pick_block_m` 用 16 B/元素系数并封顶 8 行 | `history_version/rmsnorm_impl_s2_attempt3_pre_fp16_ub.py` → `_post_fp16_ub.py` |
| 09:28:52 | fp16 smoke：max_abs=9.766e-04，0/1,048,576 超差 | Agent 运行记录；最终全量日志复核 |
| 09:29:15 | L0 重跑通过，退出 0 | `logs/l0.log` |
| 09:29:22 | coverage 门禁：18 PASS / 0 MISS / 0 N/A，退出 0 | `logs/coverage.log` |
| 09:30:24 | 全量重跑通过：16×PRECISION_PASS、11×BOUNDARY_PASS、`Test Passed!`，退出 0 | `logs/all.log`、`logs/all.exit` |
| 09:30:51 | 流式重跑（`-u`）再次通过，退出 0 | `logs/all_debug.log` |
| 09:33:18 | `debug_log.md` 与 Stage Result 交付 | `debug_log.md` |

最终计数：L0 1 + L1 15 = 16 个 `[PRECISION_PASS]`；L2 6 + Boundary 5 =
11 个 `[BOUNDARY_PASS]`；0 个 FAIL/WARN。

## 2. 实际使用的命令

全量测试（Agent Stage Result 中给出的命令，从 repo 根目录运行）：

```bash
TILELANG_ASCEND_MODE=Developer \
PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:/mnt/workspace/learn/infra-learning/day8/code/agent_output/custom/rmsnorm:$PYTHONPATH \
python -u /mnt/workspace/learn/infra-learning/day8/code/agent_output/custom/rmsnorm/test_rmsnorm.py --level all
```

L0：

```bash
TILELANG_ASCEND_MODE=Developer PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:$OP:$PYTHONPATH \
python $OP/test_rmsnorm.py --level l0
```

覆盖门禁（从 `custom/` 的父目录运行）：

```bash
python /mnt/workspace/learn/cannbot-skills/ops/tilelang-op-test-design/Ascend910/scripts/coverage_check.py \
  custom/rmsnorm/test_rmsnorm.py --proto custom/rmsnorm/proto.yaml
```

fp16 探针（compile-only，非交付代码，只用于定 tiling）：

```bash
TILELANG_ASCEND_MODE=Developer PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:$PYTHONPATH \
python /mnt/workspace/learn/infra-learning/day8/code/process/probes/rmsnorm_fp16_variants.py
```

## 3. 人工 vs Agent

- **Agent 独立完成**：DESIGN.md / proto / kernel / test / 三次代码修复 /
  日志 / debug_log / coverage。
- **人工完成**：NPUIR build 修复、独立 float64 验收器、三时钟 harness、
  Day 3 baseline、静态指标脚本、验收口径（固定容差、100% 元素通过、
  负向对照）。
- **人工修改生成 kernel**：0 行。
- **人工修改生成 test**：只改门禁语义，把 L2/Boundary 的返回值接入
  `blocking_ok`；不改容差、测试数据、覆盖维度。改前/改后快照为
  `history_version/test_rmsnorm_pre_blocking_gate.py` /
  `test_rmsnorm_post_blocking_gate.py`，diff 为
  `history_version/human_blocking_gate.diff`。
- **人工放宽容差**：0 次。

## 4. 门禁结论

Stage 2 门禁通过：

1. 编译成功：`logs/l0.log`、`logs/all.log` 均有
   `AscendNPU IR compile success`；
2. 退出 0：`logs/l0.log`、`logs/all.exit`、`logs/coverage.log`；
3. 数值正确：Agent 自测全过，且独立 float64 验收三 case 全过
   （`code/reference/agent_acceptance.json`）；
4. 边界与 fp16：`[1023,1024]` 尾块、fp16 全链路均覆盖；
5. 覆盖门禁：18/18 PASS。

## 5. Stage 2 交付后的门禁加固（人工，10:0x）

Agent 原版只有 L0/L1 写回 `blocking_ok`，L2/Boundary 失败只打印 WARN，
退出码仍为 0。人工改动 `test_rmsnorm.py`：`_run_exception` /
`_run_boundary` 返回 bool，`test_rmsnorm_l2` / `test_rmsnorm_boundary`
聚合结果，`main` 中四层全部 `blocking_ok &= ...`。

验证：

- 正常全量重跑：`logs/all_blocking_gate.log`，16×PRECISION_PASS、
  11×BOUNDARY_PASS、0 WARN，`Test Passed!`，退出 0；
- 强制 Boundary 失败 → 退出 1；强制 L2 失败 → 退出 1；恢复后 → 退出 0
  （`code/logs/09-test-suite-gate-selftest.log`）。

独立验收脚本也加了同样的门禁：`all_ok()` 要求三组数值 PASS 且三组负向
对照被检出，`exit_code()` 才返回 0；自测见
`code/logs/08-acceptance-gate-selftest.log`，加门禁后的上卡重跑见
`code/logs/07-independent-acceptance-blocking.log`。
