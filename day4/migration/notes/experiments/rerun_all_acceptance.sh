#!/bin/bash
# 独立复验 5 个迁移算子：不经过 TileOPs 框架 CLI，直接跑各 tests/ops 用例。
set -u
ROOT=/mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs
LOG=/mnt/workspace/learn/tilelang-ascend/notes/day5/logs/independent
cd "$ROOT" || exit 1
export PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:$ROOT
export TILELANG_ASCEND_MODE=Developer
export CANNBOT_GUI=1
mkdir -p "$LOG"
run() {
  name=$1; file=$2
  { echo "# cmd: python -m pytest $file -q"
    echo "# cwd: $(pwd)"
    echo "# date: $(date '+%F %T')"
    timeout 1800 /usr/local/bin/python -m pytest "$file" -q 2>&1
    echo "# exit=$?"
  } | tee "$LOG/acceptance_${name}.log"
}
run leaky_relu    tests/ops/test_leaky_relu.py
run elu           tests/ops/test_elu.py
run expm1_erf     tests/ops/test_unary_math.py
run bitwise_not   tests/ops/test_bitwise_not.py
echo "ALL DONE $(date '+%F %T')"
