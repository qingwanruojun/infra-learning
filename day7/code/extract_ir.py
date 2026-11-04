#!/usr/bin/env python3
"""Split the single VecAdd Developer-mode dump into stage files.

The boundaries are the literal banners printed by tilelang/engine/lower.py
and the MLIR pass-manager markers emitted by pipeline.enable_ir_printing().
Run with: python3 -I extract_ir.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "vecadd-dev-dump.log"
IR = HERE / "ir"
IR.mkdir(exist_ok=True)

log = LOG.read_text()

TVM_BANNER = "====== TVM IR ======"
NPUIR_BANNER = "====== npuir ======"
FINAL_BANNER = "====== final npuir ======"
CAN_MARK = "// -----// IR Dump After Canonicalizer"
ADAPT_MARK = "// -----// IR Dump After AdaptTritonKernel"
COMPILE_OK = "AscendNPU IR compile success:"

def between(start: str, end: str) -> str:
    i = log.index(start) + len(start)
    j = log.index(end, i)
    text = log[i:j].strip()
    assert text, f"empty section between {start!r} and {end!r}"
    return text + "\n"

tvm = between(TVM_BANNER, NPUIR_BANNER)
initial = between(NPUIR_BANNER, CAN_MARK)
i_can = log.index(CAN_MARK)
i_adapt = log.index(ADAPT_MARK)
i_final = log.index(FINAL_BANNER)
i_ok = log.index(COMPILE_OK, i_final)
canon_after = log[i_can:i_adapt].strip() + "\n"
adapt_after = log[i_adapt:i_final].strip() + "\n"
final = log[i_final + len(FINAL_BANNER):i_ok].strip() + "\n"

assert tvm.startswith("# from tvm.script import ir as I"), tvm[:80]
assert initial.startswith("module attributes"), initial[:80]
assert canon_after.startswith(CAN_MARK), canon_after[:80]
assert adapt_after.startswith(ADAPT_MARK), adapt_after[:80]
assert final.startswith("module attributes"), final[:80]
assert final.rstrip().endswith("}"), final[-80:]

(IR / "step1-tvm-ir.tir").write_text(tvm)
(IR / "step2-initial-npuir.mlir").write_text(initial)
(IR / "step2a-after-canonicalize.mlir").write_text(canon_after)
(IR / "step2b-after-adapt-triton-kernel.mlir").write_text(adapt_after)
(IR / "step3-final-npuir.mlir").write_text(final)

print("source_log=", LOG)
print("tvm_ir_lines=", tvm.count("\n"))
print("initial_npuir_lines=", initial.count("\n"))
print("canonicalize_after_lines=", canon_after.count("\n"))
print("adapt_after_lines=", adapt_after.count("\n"))
print("final_npuir_lines=", final.count("\n"))
print("EXTRACT=PASS")
