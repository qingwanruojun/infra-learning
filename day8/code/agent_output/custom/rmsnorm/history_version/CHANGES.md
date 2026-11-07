# rmsnorm kernel change log (Stage 2, attempt 1)

## 1. attempt1 -> attempt2: UB over-budget fix (fp32 path)

- Files: `history_version/rmsnorm_impl_s2_attempt1_original.py` (pre-fix)
         `history_version/rmsnorm_impl_s2_attempt2_ub_fix.py` (post-fix)

### Problem

DESIGN.md pseudo-code (section 3.3) allocates `x_f32` (block_m x N fp32) and
`w_f32` (N fp32) unconditionally so that the fp16 branch can keep an fp32 copy
of the input / weight.  With N=1024 and block_m=16 the fp32 path then holds:

  x_sh 65,536 + xsq 65,536 + w_sh 4,096 + acc 64 + rstd 64
  + x_f32 65,536 + w_f32 4,096 = 204,928 B  >  196,608 B (192 KiB UB)

Empirically confirmed (Ascend910B3, CANN 9.0.0): the kernel compiles but the
fp32 L0 shape dies at runtime with
`aivec error ... The DDR address of the MTE instruction is out of range`
(vector core exception, retCode=0x31) - i.e. the UB budget is silently
over-committed.

### Fix

Do not cache `x_f32` / `w_f32`.  In the fp16 branch the square and the final
multiply cast `x_sh[i, j]` / `w_sh[j]` to fp32 at the point of use
(fp16 -> fp32 is exact, so the arithmetic is identical).  `x_sh` and `w_sh`
still hold the original values because they are only overwritten by
`T.vcast(xsq, x_sh)` after the multiply.

Resulting UB budgets (block_m=16, N=1024):
  fp32: x_sh 65,536 + xsq 65,536 + w_sh 4,096 + acc 64 + rstd 64 = 135,296 B
  fp16: x_sh 32,768 + xsq 65,536 + w_sh 2,048 + acc 64 + rstd 64 = 100,480 B

Public API is unchanged: `rmsnorm(M, N, dtype="float32", eps=1e-5, block_m=None)`
still returns the compiled `kernel(x, w) -> y`.  `pick_block_m` keeps the
design's conservative fp16 budget formula so N=1024 still selects block_m=16.

## 2. attempt2 -> attempt3: fp16 tiling fix (UB overflow in BiShengHIR)

- Files: `history_version/rmsnorm_impl_s2_attempt3_pre_fp16_ub.py` (pre-fix)
         `history_version/rmsnorm_impl_s2_attempt3_post_fp16_ub.py` (post-fix)

### Problem

With block_m=16 the fp16 kernel fails to compile:

  loc("kernel.npuir":2:3): error: ub overflow, requires 2113536 bits while
  1572864 bits available! (possible reason: tiling basic block is too large ...
  multi-buffer feature is enabled and some ops need extra local buffer.)

2,113,536 bits = 264,192 B vs the 196,608 B (192 KiB) UB.  The logical
buffers only account for 100,480 B; the fp32 promotions of x_sh / w_sh in the
two T.Parallel loops add cast temporaries.  Probed requirements
(`/tmp/rmsnorm_fp16_variants.py`, compile-only, Ascend910B3 / CANN 9.0.0):

  block_m=16 default        -> FAIL, requires 264,192 B
  block_m=12 default        -> FAIL, requires 198,720 B
  block_m=10 default        -> COMPILE_OK
  block_m= 8 default        -> COMPILE_OK
  block_m=16 no-multibuffer -> FAIL, requires 264,192 B (feature not the cause)

The linear fit is ~16 B per element per row, i.e. four fp32 row buffers
(xsq, the two promotion temporaries and the fp32 result).

### Fix

`pick_block_m(N, "float16")` now uses the measured 16 B/element coefficient
and caps the tile at 8 rows: N=1024 -> block_m=8 (133,248 B, 68% of UB).
The fp32 path is unchanged (block_m=16, 135,296 B).  Public API unchanged;
`block_m` remains a compile-time parameter of `rmsnorm(...)`.
