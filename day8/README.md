# Day 8：cannbot-skills × TileLang NPUIR RMSNorm

本目录记录“让 Agent 在 Ascend 910B3 上生成并跑通 RMSNorm”的完整过程：
生成代码、测试、独立验收、原始日志、与 Day 3 手写版的对比。

建议先读 [Day 8 学习报告](note/day8学习报告.md)；逐项数字、命令和故障过程见本文件与 [详细技术记录](note/learning-report.md)。

## 结论

| case | 容差 (atol/rtol) | max_abs | 超差元素 | 独立验收 |
|---|---|---|---|---|
| fp32 1024×1024 | 1e-4 / 1e-4 | 1.353e-06 | 0 / 1,048,576 | PASS |
| fp32 1023×1024（尾块） | 1e-4 / 1e-4 | 1.149e-06 | 0 / 1,047,552 | PASS |
| fp16 1024×1024 | 1e-2 / 1e-2 | 3.901e-03 | 0 / 1,048,576 | PASS |

- 程序退出码 0，且这个 0 是门禁算出来的：三个 case 全 PASS、三个负向
  对照全被检出才返回 0。Agent 自带测试 L0:1 / L1:15 / L2:6 / Boundary:5
  全过（四层均阻塞），coverage `18 PASS / 0 MISS`。
- 负向对照：三个 case 注入 10× 局部容差后，独立检查器均报
  `pass=False, n_mismatch=1`；`gate_selftest.py` 证明失败路径会退出 1。
- device kernel：fp32 17.11 / 16.84 µs、fp16 12.06 µs；host 约 327–360 µs，
  e2e 约 337–376 µs（host 主导，run-to-run 约 ±2 µs）。

## 目录

```text
day8/
├── README.md                       # 本文件
├── note/
│   ├── day8学习报告.md              # 主报告：实验经过、结果与收获
│   └── learning-report.md          # 详细技术记录
└── code/
    ├── agent_output/custom/rmsnorm/  # Stage 1+2 产物（Agent 生成）
    │   ├── op_requirements.yaml
    │   ├── proto.yaml
    │   ├── DESIGN.md
    │   ├── rmsnorm.py              # 最终 kernel
    │   ├── test_rmsnorm.py         # L0/L1/L2/Boundary
    │   ├── debug_log.md
    │   ├── .orchestrator_state.json
    │   ├── history_version/        # 每次修复前后的快照 + CHANGES.md
    │   └── logs/                   # l0 / all / coverage / 失败留档
    ├── reference/                  # 人工验证工具（不参与生成）
    │   ├── independent_check.py    # float64 golden + 负向对照
    │   ├── agent_acceptance.py     # 独立验收入口（含退出码门禁）
    │   ├── gate_selftest.py        # 验证失败路径确实退出 1
    │   ├── bench_common.py         # host / e2e / device 三时钟
    │   ├── day3_baseline.py        # Day 3 同条件精度/性能
    │   ├── day3_three_clocks.py    # Day 3 三时钟
    │   └── *.json                  # 实测结果
    ├── process/
    │   ├── env-and-orchestration.md
    │   ├── stage2-agent-timeline.md
    │   ├── code_metrics.py
    │   ├── code_metrics.json
    │   └── probes/                 # 两份故障原始记录 + 复现脚本
    │       ├── 01-fp32-mte-incident.log
    │       ├── 02-fp16-blockm-probe.log
    │       ├── rmsnorm_compile_check.py
    │       ├── rmsnorm_fp16_variants.py
    │       └── rmsnorm_fp16_smoke.py
    └── logs/                       # 环境、smoke、baseline、验收原始日志
```

## 环境

实测：Ascend 910B3 / CANN 9.0.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 /
Python 3.11.4 / TileLang 0.1.2+bd7c60c（NPUIR 构建）。

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
export TILELANG_ASCEND_MODE=Developer
export REPO=/mnt/workspace/learn/tilelang-ascend
export OP=/mnt/workspace/learn/infra-learning/day8/code/agent_output/custom/rmsnorm
export REF=/mnt/workspace/learn/infra-learning/day8/code/reference
export PYTHONPATH=$REPO:$OP:$REF
```

NPU 可见性由宿主 `ASCEND_VISIBLE_DEVICES=2` 决定，容器内看到 `npu:0`。
如果 `import tilelang` 报 `Cannot find libraries: libtilelangir.so`，说明
NPUIR build 被覆盖；重建步骤见 `code/process/env-and-orchestration.md`
第 3.1 节，原始日志 `code/logs/00-rebuild-npuir.log`。

## 复现命令

### 1. 上板 smoke（确认环境）

```bash
cd $REPO
python examples/elementwise/vec_add_1d.py
# 期望：All check passed
```

### 2. Agent 生成代码的 L0

```bash
cd $REPO
TILELANG_ASCEND_MODE=Developer PYTHONPATH=$REPO:$OP:$PYTHONPATH \
  python $OP/test_rmsnorm.py --level l0 2>&1 | tee $OP/logs/l0.log
echo "EXIT=${PIPESTATUS[0]}" | tee -a $OP/logs/l0.log
```

### 3. 全量 L0/L1/L2/Boundary

```bash
cd $REPO
TILELANG_ASCEND_MODE=Developer PYTHONPATH=$REPO:$OP:$PYTHONPATH \
  python -u $OP/test_rmsnorm.py --level all 2>&1 | tee $OP/logs/all_blocking_gate.log
echo "EXIT=${PIPESTATUS[0]}" | tee -a $OP/logs/all_blocking_gate.log
```

L0/L1/L2/Boundary 四层都参与退出码。强制失败的验证记录在
`code/logs/09-test-suite-gate-selftest.log`（强制 Boundary 失败 → 退出 1，
强制 L2 失败 → 退出 1，恢复后 → 退出 0）。

### 4. 覆盖门禁

```bash
cd $OP/..
python /mnt/workspace/learn/cannbot-skills/ops/tilelang-op-test-design/Ascend910/scripts/coverage_check.py \
  custom/rmsnorm/test_rmsnorm.py --proto custom/rmsnorm/proto.yaml \
  2>&1 | tee custom/rmsnorm/logs/coverage.log
```

### 5. 独立验收（不采信 Agent 的 golden）

```bash
cd $REPO
TILELANG_ASCEND_MODE=Developer PYTHONPATH=$REPO:$OP:$REF \
  python $REF/agent_acceptance.py 2>&1 \
  | tee /mnt/workspace/learn/infra-learning/day8/code/logs/07-independent-acceptance-blocking.log
echo "EXIT=${PIPESTATUS[0]}" \
  | tee -a /mnt/workspace/learn/infra-learning/day8/code/logs/07-independent-acceptance-blocking.log
```

只有“三组数值 PASS + 三组负向对照 FAIL”时退出 0；任何一组不满足都会
退出 1。会重新生成 `code/reference/agent_acceptance.json`。

门禁自测（不需要 NPU）：

```bash
cd $REF && python gate_selftest.py   # 期望 GATE_SELFTEST_PASS，退出 0
python independent_check.py          # 期望 SELF_CHECK_PASS
```

### 6. Day 3 同条件对照（只读，不修改 Day 3）

```bash
cd $REPO
python /mnt/workspace/learn/infra-learning/day8/code/reference/day3_baseline.py
PYTHONPATH=$REPO:$REF python \
  /mnt/workspace/learn/infra-learning/day8/code/reference/day3_three_clocks.py
```

### 7. 静态结构指标

```bash
cd /mnt/workspace/learn/infra-learning/day8
python code/process/code_metrics.py \
  $REPO/own_ops/rmsnorm.py $OP/rmsnorm.py $OP/test_rmsnorm.py
```

## 证据索引

| 结论 | 证据 |
|---|---|
| NPUIR build 可用 | `code/logs/00-rebuild-npuir.log`、`01-env-quick-verify.log` |
| 卡上真实可跑 | `code/logs/02-npu-smoke-vecadd.log`（vec_add，退出 0） |
| Agent 设计 | `code/agent_output/custom/rmsnorm/DESIGN.md`（722 行） |
| 生成 kernel 与测试 | `.../rmsnorm.py`、`.../test_rmsnorm.py` |
| 三次修复的前后快照 | `.../history_version/`（attempt1/2/3 + `CHANGES.md`） |
| L0 编译成功 + 退出 0 | `.../logs/l0.log`（`AscendNPU IR compile success`、`EXIT_CODE=0`） |
| 全量测试通过 | `.../logs/all.log`（16×PASS、11×BOUNDARY_PASS、`Test Passed!`、`EXIT_CODE=0`） |
| fp16 第一次失败的原始现场 | `.../logs/all_attempt1_fp16_compile_fail.log`（264,192 B > 196,608 B） |
| fp32 首次 MTE 越界原始记录 | `code/process/probes/01-fp32-mte-incident.log`（从子代理 transcript 提取的输出尾段、`EXIT=134`）、`rmsnorm_compile_check.py` |
| fp16 block_m 探针六档输出 | `code/process/probes/02-fp16-blockm-probe.log`、`rmsnorm_fp16_variants.py` |
| 覆盖门禁 | `.../logs/coverage.log`（18 PASS / 0 MISS） |
| 独立数值验收 + 退出门禁 | `code/reference/agent_acceptance.json`、`code/logs/07-independent-acceptance-blocking.log` |
| 门禁失败路径自测 | `code/logs/08-acceptance-gate-selftest.log`、`code/reference/gate_selftest.py` |
| 测试四层阻塞自测 | `code/logs/09-test-suite-gate-selftest.log`、`.../logs/all_blocking_gate.log` |
| 负向对照 | `code/logs/06a-negative-control-selftest.log`、acceptance JSON 的 `negative_control` |
| 人工对生成 test 的修改 | `.../history_version/human_blocking_gate.diff`、`HUMAN_CHANGES.md` |
| 三时钟与 device 口径 | `code/reference/agent_acceptance.json`、`code/reference/day3_three_clocks.json` |
| Day 3 同条件 baseline | `code/reference/day3_baseline.json`、`code/logs/03-day3-baseline.log` |
| 环境/编排实况 | `code/process/env-and-orchestration.md` |
| 静态结构对比 | `code/process/code_metrics.json` |
| 主报告 | `note/day8学习报告.md` |
| 详细技术记录 | `note/learning-report.md` |

## 已知限制

- Stage 3 性能调优未做（用户任务未要求）；`perf_tuning/` 为空。
- fp16 的 `block_m` 从设计值 16 降到 8，以绕开编译期 UB overflow；
  shape/dtype/公式/容差未缩小。
- Agent 原版测试的 L2/Boundary 是非阻塞的（只打印 marker，不影响退出码）；
  人工已改为四层阻塞，并用强制失败用例验证过退出 1。改动只涉及控制流，
  容差、数据和覆盖维度未动。
- N=8192 只有编译期上限校验，没有实际上卡。
- `tilelang-env-check` 的 `check_env.sh` / `quick_verify.py` 与本仓库
  revision 不兼容，已用 vec_add smoke 替代。
- device 时间有约 ±2 µs 的 run-to-run 波动，性能结论只到“同一量级”。
