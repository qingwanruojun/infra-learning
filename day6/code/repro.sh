#!/usr/bin/env bash
# Day6 end-to-end reproduction (post host-validator fix).
# Usage: bash /mnt/workspace/learn/infra-learning/day6/code/repro.sh
#
# Produces:
#   logs/step6-compile-hivm.log             HIVM-level VecAdd -> kernel.o
#   logs/step6-host-build.log               fixed host wrapper build
#   logs/step6-run-hivm.log                 run on the HIVM-level kernel
#   logs/step6-hfusion-build.log            HFusion input -> device ELF
#   logs/step6-hfusion-chain.log            details of the HFusion chain
#   logs/step6-run-hfusion.log              run on the HFusion-origin kernel
#   logs/step6-run-negative-control.log     wrong-kernel rejection
#   logs/step6-numeric-check.log            independent numeric check
set -euo pipefail

DAY6="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANN="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.0.0}"
LOGS="${DAY6}/logs"
source "${CANN}/set_env.sh"
cd "${DAY6}/vecadd"
mkdir -p "${LOGS}"

HFUSION_SYMBOL="add_single_outlined_0_0_0"

log_run() {
  local log="$1"
  shift
  {
    echo "\$ $*"
    "$@" 2>&1
  } | tee -a "${log}"
}

echo "== [1/6] compile HIVM-level VecAdd add.mlir -> kernel.o"
: > "${LOGS}/step6-compile-hivm.log"
log_run "${LOGS}/step6-compile-hivm.log" \
  bishengir-compile add.mlir --enable-hivm-compile -o kernel.o

# Same compile with the per-pass IR dump; used as the process evidence in the report.
: > "${LOGS}/step6-hivm-passdump.log"
{
  echo '$ bishengir-compile add.mlir --enable-hivm-compile --mlir-print-ir-after-all -o /tmp/kernel-step6-passdump.o'
  bishengir-compile add.mlir --enable-hivm-compile --mlir-print-ir-after-all \
    -o /tmp/kernel-step6-passdump.o 2>&1
  echo "exit=$?"
} >> "${LOGS}/step6-hivm-passdump.log"

echo "== [2/6] build fixed host wrapper"
: > "${LOGS}/step6-host-build.log"
log_run "${LOGS}/step6-host-build.log" \
  g++ main.cpp -Ihostshim \
    "-I${ASCEND_HOME_PATH}/include" \
    "-I${ASCEND_HOME_PATH}/include/experiment/msprof" \
    "-I${ASCEND_HOME_PATH}/pkg_inc" \
    "-L${ASCEND_HOME_PATH}/lib64" -lruntime -lascendcl \
    "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64" -o hivm-vec-add

echo "== [3/6] run on NPU: HIVM-level kernel"
: > "${LOGS}/step6-run-hivm.log"
./hivm-vec-add ./kernel.o 2>&1 | tee -a "${LOGS}/step6-run-hivm.log"

echo "== [4/6] build HFusion-level input -> device ELF (single source input)"
: > "${LOGS}/step6-hfusion-build.log"
bash "${DAY6}/hfusion-vecadd/build.sh" 2>&1 | tee -a "${LOGS}/step6-hfusion-build.log"

echo "== [5/6] run on NPU: HFusion-origin kernel"
: > "${LOGS}/step6-run-hfusion.log"
./hivm-vec-add "${DAY6}/hfusion-vecadd/kernel.o" "${HFUSION_SYMBOL}" \
  2>&1 | tee -a "${LOGS}/step6-run-hfusion.log"

echo "== [6/6] negative control: wrong kernel must be rejected"
: > "${LOGS}/step6-run-negative-control.log"
if ./hivm-vec-add "${DAY6}/hfusion-vecadd/bad-kernel.o" "${HFUSION_SYMBOL}" \
    > "${LOGS}/step6-run-negative-control.log" 2>&1; then
  echo "NEGATIVE_CONTROL=FAIL validator accepted a wrong kernel" \
    | tee -a "${LOGS}/step6-run-negative-control.log"
  exit 1
else
  rc=$?
  echo "NEGATIVE_CONTROL=PASS validator rejected a wrong kernel (exit ${rc})" \
    | tee -a "${LOGS}/step6-run-negative-control.log"
fi

# Artifact facts cited by the report (ELF type/symbols/hashes, host linkage).
{
  echo '$ file ./kernel.o ...'; file ./kernel.o "${DAY6}/hfusion-vecadd/kernel.o" \
    "${DAY6}/hfusion-vecadd/bad-kernel.o"
  echo '$ readelf -sW kernel.o | grep FUNC | grep -v UND'
  readelf -sW kernel.o | grep ' FUNC ' | grep -v ' UND '
  echo '$ readelf -sW hfusion-vecadd/kernel.o | grep FUNC | grep -v UND'
  readelf -sW "${DAY6}/hfusion-vecadd/kernel.o" | grep ' FUNC ' | grep -v ' UND '
  echo '$ sha256sum ...'
  sha256sum ./kernel.o "${DAY6}/hfusion-vecadd/kernel.o" \
    "${DAY6}/hfusion-vecadd/bad-kernel.o"
  echo '$ ls -l ./hivm-vec-add'
  ls -l ./hivm-vec-add
  echo '$ ldd ./hivm-vec-add | grep -E "runtime|ascendcl"'
  ldd ./hivm-vec-add | grep -E 'runtime|ascendcl'
} > "${LOGS}/step6-artifacts.log" 2>&1

# Independent check: parse the Result column, compare against the expected
# sequence for both cases (case0 1..16, then case1 mixed).
expected="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-5,-6,0,10,0,0,0,0,0,0,0,-1,0,0,0,0"
check_log() {
  local name="$1"
  local actual
  actual="$(awk -F'Result: ' '/Result:/{print $2}' "${LOGS}/${name}.log" | paste -sd, -)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "NUMERIC_CHECK=FAIL ${name} expected=${expected} actual=${actual}" \
      | tee -a "${LOGS}/step6-numeric-check.log"
    exit 1
  fi
  echo "NUMERIC_CHECK=PASS ${name} actual=${actual}" \
    | tee -a "${LOGS}/step6-numeric-check.log"
}
: > "${LOGS}/step6-numeric-check.log"
check_log step6-run-hivm
check_log step6-run-hfusion

echo "ALL_CHECKS=PASS"
