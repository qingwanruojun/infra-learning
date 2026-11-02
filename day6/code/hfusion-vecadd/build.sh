#!/usr/bin/env bash
# Day6 priority-2 experiment: carry one HFusion/Linalg-level input all the way
# to a device ELF, then let vecadd/hivm-vec-add launch it on the NPU.
#
# Why two compiler binaries:
#   - CANN 9.0.0's packaged bishengir-compile 1.1.0 SIGSEGVs in
#     HIVMAggregatedDecomposeOp for *any* --enable-hfusion-compile input on
#     this host (upstream HFusion tests included).
#   - The source-built bishengir-opt (commit 31f690369) runs the same HFusion +
#     ConvertToHIVM + OptimizeHIVM pipelines successfully, but cannot call
#     CANN 9.0.0's older hivmc (it forwards unknown options).
#   So: source-built bishengir-opt lowers HFusion -> optimized HIVM; the
#   packaged bishengir-compile turns that HIVM into the device ELF.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAY6="$(dirname "${HERE}")"
CANN="${ASCEND_HOME_PATH:-/home/developer/Ascend/cann-9.0.0}"
BUILT_OPT="${BUILT_BISHENGIR_OPT:-/mnt/workspace/learn/tilelang-ascend/3rdparty/AscendNPU-IR/build/bin/bishengir-opt}"
LOGS="${DAY6}/logs"
mkdir -p "${LOGS}"

# shellcheck disable=SC1090
source "${CANN}/set_env.sh"

build_one() {
  local src="$1" prefix="$2" elf="$3" log="${LOGS}/$4"
  {
    echo "\$ ${BUILT_OPT} ${HERE}/${src} --lower-hfusion-pipeline --convert-to-hivm-pipeline --optimize-hivm-pipeline -o ${HERE}/${prefix}.optimized.mlir"
    echo "\$ python3 -I ${HERE}/extract-device-kernel.py ${HERE}/${prefix}.optimized.mlir ${HERE}/${prefix}.device.hivm.mlir"
    echo "\$ bishengir-compile ${HERE}/${prefix}.device.hivm.mlir --enable-hivm-compile -o ${HERE}/${elf}"
  } > "${log}"

  "${BUILT_OPT}" "${HERE}/${src}" \
      --lower-hfusion-pipeline --convert-to-hivm-pipeline --optimize-hivm-pipeline \
      -o "${HERE}/${prefix}.optimized.mlir" >> "${log}" 2>&1
  echo "HFUSION_PIPELINE=PASS ${src}" >> "${log}"

  python3 -I "${HERE}/extract-device-kernel.py" \
      "${HERE}/${prefix}.optimized.mlir" "${HERE}/${prefix}.device.hivm.mlir" \
      >> "${log}" 2>&1
  echo "DEVICE_EXTRACT=PASS ${prefix}.device.hivm.mlir" >> "${log}"

  ( cd "${HERE}" && bishengir-compile "${prefix}.device.hivm.mlir" \
      --enable-hivm-compile -o "${elf}" ) >> "${log}" 2>&1
  # hivmc also emits a host-side shared object next to the device ELF; the
  # Day6 host wrapper only loads the ELF.
  rm -f "${HERE}/lib${elf%.o}.so"
  echo "DEVICE_ELF=PASS ${elf} ($(stat -c %s "${HERE}/${elf}") bytes)" >> "${log}"
  readelf -sW "${HERE}/${elf}" | grep " FUNC " | grep -v " UND " >> "${log}"
}

build_one add-hfusion-host.mlir add-hfusion kernel.o step6-hfusion-chain.log
build_one mul-hfusion-host.mlir mul-hfusion bad-kernel.o step6-hfusion-chain-negative.log

echo "KERNEL_SYMBOL=add_single_outlined_0_0_0"
echo "ADD_KERNEL=${HERE}/kernel.o"
echo "BAD_KERNEL=${HERE}/bad-kernel.o"
