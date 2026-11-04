#!/usr/bin/env bash
# Day 7: downstream checks on the exact final npuir extracted from the
# Developer-mode run. Nothing here changes upstream sources.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source /home/developer/Ascend/cann-9.0.0/set_env.sh

BOPT="$(command -v bishengir-opt)"
BCMP="$(command -v bishengir-compile)"
TLOPT=/mnt/workspace/learn/tilelang-ascend/build/tilelangir/tools/tilelangir-opt/tilelangir-opt
INPUT=ir/step3-final-npuir.mlir
LOG=logs/step4-commands.log

{
  echo "date=$(date -Is)"
  echo "bishengir-opt=$BOPT"
  echo "bishengir-compile=$BCMP"
  echo "tilelangir-opt=$TLOPT"
  echo "input=$INPUT sha256=$(sha256sum "$INPUT" | cut -d' ' -f1)"
  "$BOPT" --version
} > "$LOG" 2>&1

# 1) Version/help gate: only use pass names that this exact binary reports.
"$BOPT" --help 2>&1 | grep -E -- '--(canonicalize|hivm-infer-mem-scope|hivm-flatten-ops|hivm-plan-memory|adapt-triton-kernel|convert-hivm-to-std)\b' \
  > logs/step4-help-passes.log

# 2) Parse/print round trip.
"$BOPT" "$INPUT" > ir/bishengir-opt-roundtrip.mlir 2> logs/step4-roundtrip.err
echo "roundtrip_exit=$?" >> logs/step4-roundtrip.err

# 3) Applicable pass for this actual module.
"$BOPT" "$INPUT" --hivm-infer-mem-scope > ir/step4-hivm-infer-mem-scope.mlir 2> logs/step4-hivm-infer-mem-scope.err
echo "infer_mem_scope_exit=$?" >> logs/step4-hivm-infer-mem-scope.err

# 4) Negative controls: record failure instead of hiding it.
set +e
"$BOPT" "$INPUT" --hivm-flatten-ops > ir/step4-hivm-flatten-ops.mlir 2> logs/step4-hivm-flatten-ops.err
echo "flatten_ops_exit=$?" >> logs/step4-hivm-flatten-ops.err
"$BOPT" "$INPUT" --hivm-plan-memory > ir/step4-hivm-plan-memory.mlir 2> logs/step4-hivm-plan-memory.err
echo "plan_memory_exit=$?" >> logs/step4-hivm-plan-memory.err
set -e

# 5) Out-of-path control: run the two TileLangIR placeholder passes directly.
for p in tilelangir-cv-split tilelangir-vectorize; do
  "$TLOPT" "$INPUT" "--$p" > "ir/out-of-path-$p.mlir" 2> "logs/out-of-path-$p.err"
  echo "$p""_exit=$?" >> "logs/out-of-path-$p.err"
done

# 6) Same MLIR -> device ELF, with the same options jit_npu passes.
set -o pipefail
"$BCMP" "$INPUT" \
  --enable-auto-multi-buffer=true \
  --disable-hivm-auto-inject-sync=false \
  --enable-triton-kernel-compile=true \
  --enable-hivm-compile=true \
  --mlir-print-ir-after-all \
  -o artifacts-vecadd.o > logs/step5-bishengir-compile-passdump.log 2>&1
echo "bishengir_compile_exit=$?" > logs/step5-bishengir-compile-exit.log

sha256sum artifacts-vecadd.o >> logs/step5-bishengir-compile-exit.log
file artifacts-vecadd.o >> logs/step5-bishengir-compile-exit.log
readelf -sW artifacts-vecadd.o | grep -E ' main$| main__$' >> logs/step5-bishengir-compile-exit.log

# 7) Keep the final lowered HIR (up to compiler warnings) and a diff.
awk '/IR Dump After ConvertHIVMToStandard/{flag=1} flag && /^warning: overriding/{exit} flag{print}' \
  logs/step5-bishengir-compile-passdump.log > ir/step5-final-hivm-lowered.mlir
diff -u "$INPUT" ir/step4-hivm-infer-mem-scope.mlir > logs/step4-infer-mem-scope.diff || true
diff -u ir/step2-initial-npuir.mlir "$INPUT" > logs/step3-initial-vs-final.diff || true
diff -u ir/step2a-after-canonicalize.mlir ir/step2b-after-adapt-triton-kernel.mlir > logs/step2-canonicalize-vs-adapt.diff || true

# 8) The original JIT temp kernel.o was captured by shims/bishengir-compile
# during run_vecadd_dev.sh. Compare it with the recompiled artifact.
{
  echo "date=$(date -Is)"
  if [[ -f logs/jit-kernel.o && -f artifacts-vecadd.o ]]; then
    sha256sum logs/jit-kernel.npuir logs/jit-kernel.o artifacts-vecadd.o
    if [[ -f logs/jit-libkernel.so && -f libartifacts-vecadd.so ]]; then
      sha256sum logs/jit-libkernel.so libartifacts-vecadd.so
    fi
    file logs/jit-kernel.o
    if cmp -s logs/jit-kernel.o artifacts-vecadd.o; then
      echo "JIT_VS_RECOMPILED=IDENTICAL"
    else
      echo "JIT_VS_RECOMPILED=DIFFERENT"
    fi
    if [[ -f logs/jit-libkernel.so && -f libartifacts-vecadd.so ]]; then
      if cmp -s logs/jit-libkernel.so libartifacts-vecadd.so; then
        echo "HOST_SO=IDENTICAL"
      else
        echo "HOST_SO=DIFFERENT"
      fi
    else
      echo "HOST_SO=NO_CAPTURE"
    fi
  else
    echo "JIT_VS_RECOMPILED=NO_JIT_CAPTURE (run run_vecadd_dev.sh first)"
  fi
} > logs/step7-jit-vs-recompiled.log 2>&1

echo "CHECKS_DONE"
