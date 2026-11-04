#!/usr/bin/env bash
# Day 7: run upstream VecAdd example in Developer mode with full IR dump.
set -o pipefail
set -u

REPO=/mnt/workspace/learn/tilelang-ascend
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/vecadd-dev-dump.log"
CMD_LOG="$OUT_DIR/vecadd-dev-command.log"
RC_LOG="$OUT_DIR/vecadd-dev-exit.log"

source /home/developer/Ascend/cann-9.0.0/set_env.sh

# Record the real JIT invocation without touching upstream sources: prepend a
# shim that logs argv and copies the temp kernel.o before the JIT deletes it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_BISHENGIR_COMPILE="$(command -v bishengir-compile)"
export DAY7_REAL_BISHENGIR_COMPILE="$REAL_BISHENGIR_COMPILE"
export DAY7_JIT_CAPTURE_DIR="$OUT_DIR"
rm -f "$OUT_DIR/jit-compile-argv.log" "$OUT_DIR/jit-kernel.npuir" \
      "$OUT_DIR/jit-kernel.o" "$OUT_DIR/jit-libkernel.so"
export PATH="$HERE/shims:$PATH"
export TILELANG_ASCEND_MODE=Developer
export TILELANG_DUMP_IR=TRUE
export PYTHONUNBUFFERED=1

cd "$REPO"

{
  echo "date=$(date -Is)"
  echo "cwd=$PWD"
  echo "TILELANG_ASCEND_MODE=$TILELANG_ASCEND_MODE"
  echo "TILELANG_DUMP_IR=$TILELANG_DUMP_IR"
  echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
  echo "bishengir_compile_resolved=$REAL_BISHENGIR_COMPILE"
  echo "bishengir_compile_shim=$HERE/shims/bishengir-compile"
  echo "PATH=$PATH"
  echo "python=$(command -v python3) $(python3 -V 2>&1)"
  echo "command=python3 examples/elementwise/vec_add_1d.py"
} > "$CMD_LOG"

set +e
python3 examples/elementwise/vec_add_1d.py 2>&1 | tee "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e

echo "exit_code=$rc" | tee "$RC_LOG"
exit "$rc"
