# Day 5 — 复现说明（TileOPs → NPUIR 自动迁移）

## 环境

```bash
export CANNBOT_GUI=1
export GPU_REPO=/mnt/workspace/learn/refs/TileOPs
export PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:/mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs
export TILELANG_ASCEND_MODE=Developer
```

## 单算子跑框架

```bash
cd /mnt/workspace/learn/refs/tilelang-mlir-ascend
opencode run --agent tilelang-op-conductor --auto \
  --command migrate-op LeakyReluFwdOp
# 中断后续跑：opencode run --session <ses_id> --agent tilelang-op-conductor --auto "继续 ..."
```

前置：`.opencode/agents/*.md` 里的 model 已从 `gateway/*` 改为 CANNBOT 实际模型名（见 `tilelang-ascend` 仓库改动），且已登录网关。

## 独立验收（不采信框架自述）

```bash
bash experiments/rerun_leakyrelu_acceptance.sh
```

它会跑 harness 自带的 `tests/ops/test_leaky_relu.py`（14 条），期望 `14 passed`。

## 产物位置

| 内容 | 路径 |
|---|---|
| 设计/评审/复盘 | `refs/tilelang-mlir-ascend/examples/<slug>/_make_<op>_kernel/{DESIGN,REVIEW,RETROSPECTIVE}.md` |
| NPUIR kernel | 同上 `_make_<op>_kernel.py` |
| 集成后代码 | `refs/tilelang-mlir-ascend/examples/TileOPs/tileops/kernels/elementwise/<op>/<op>_kernel/` |
| 验收报告 | `refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/<run_id>/`（run.json / benchmark.json / report.md / pytest/） |
| 本目录 logs | 上述关键文件的拷贝 + 时间线 + 队列日志 |

## 日志清单（本轮）

- `logs/run.json`, `logs/correctness.log`, `logs/benchmark.json`, `logs/benchmark.log`, `logs/benchmark.md`, `logs/report.md`
- `logs/leaky_relu_top_timeline.jsonl`, `logs/leaky_relu_func_timeline.jsonl`
- `logs/integration_log.md`, `logs/manual_acceptance_leakyrelu.log`, `logs/queue_ops_snapshot_2130.log`
- `experiments/spike_bitnot_diag.py`（uint8/bool 排障）、`experiments/rerun_leakyrelu_acceptance.sh`
