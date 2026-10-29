#!/bin/bash
# 独立复验 LeakyReluFwdOp：不经过框架，直接跑 harness 的 pytest 用例。
# 用法: bash rerun_leakyrelu_acceptance.sh [logfile]
set -u
LOG=${1:-/tmp/acceptance_leakyrelu.log}
cd /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs || exit 1
export PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:/mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs
export TILELANG_ASCEND_MODE=Developer
{ echo "# cmd: python -m pytest tests/ops/test_leaky_relu.py -q"
  echo "# cwd: $(pwd)"
  echo "# date: $(date '+%F %T')"
  /usr/local/bin/python -m pytest tests/ops/test_leaky_relu.py -q 2>&1
  echo "# exit=$?"
} | tee "$LOG"
