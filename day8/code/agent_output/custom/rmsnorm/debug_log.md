# rmsnorm Stage 2 debug log

## Attempt 1 — 2026-11-04T09:33:08+08:00
- mode: first_impl
- classification: precision_pass
- fail_category: none
- test_level: all
- coverage: L0:1 L1:15 L2:6 Boundary:5
- boundary_warnings: none (all 6 L2 negative cases `[BOUNDARY_PASS]` - illegal dtype/mixed dtype/rank-3/W length/N>8192/M=0 correctly rejected with ValueError; all 5 Boundary cases `[BOUNDARY_PASS]` - sparse +-Inf, NaN row, all-zero, fp16/fp32 denormal-boundary)
- changes:
  - Created `rmsnorm.py`: `@tilelang.jit(out_idx=[2], target="npuir")` factory
    `rmsnorm(M, N, dtype="float32", eps=1e-5, block_m=None)` with a 1D grid
    (`T.Kernel(T.ceildiv(M, block_m), is_npu=True)`), full-N rows in UB,
    `T.reduce_sum(dim=1)` + `T.vsqrt`/`T.vrec` for fp32 rstd, explicit valid-row
    extents in the GM copy-in/copy-out (no host padding), plus `validate_inputs`
    metadata validation.
  - Created `test_rmsnorm.py`: independent float64 CPU golden, L0/L1/L2/Boundary
    suites, `--level {l0,l1,l2,boundary,all}`, `tilelang.disable_cache()` in
    `main()`, per-case metrics (max_abs, masked max_rel, matched_ratio,
    n_exceed/n_compared, mask_match), `[PRECISION_PASS]`/`[PRECISION_FAIL]`/
    `[BOUNDARY_PASS]`/`[BOUNDARY_WARN]` markers, `Test Passed!` + exit 0.
  - Fixed the DESIGN pseudo-code UB over-commit (unconditional `x_f32`/`w_f32`,
    204,928 B > 196,608 B): cast `x_sh[i,j]`/`w_sh[j]` inline to fp32 in the
    fp16 branches. fp32 tile is now 135,296 B. Public API unchanged.
    Diff: `history_version/rmsnorm_impl_s2_attempt1_original.py` ->
    `history_version/rmsnorm_impl_s2_attempt2_ub_fix.py` (CHANGES.md #1).
  - Fixed fp16 compile-time UB overflow (block_m=16 needs 264,192 B because of
    the fp32 promotion cast temporaries): `pick_block_m` uses the measured
    ~16 B/element/row coefficient for fp16 and caps at 8 rows (N=1024 ->
    block_m=8, 133,248 B). fp32 path unchanged at block_m=16.
    Diff: `history_version/rmsnorm_impl_s2_attempt3_pre_fp16_ub.py` ->
    `history_version/rmsnorm_impl_s2_attempt3_post_fp16_ub.py` (CHANGES.md #2).
- error_summary:
  - Design-literal fp32 kernel compiled but crashed at runtime:
    `aivec error ... The DDR address of the MTE instruction is out of range`,
    retCode=0x31 (UB over-commit), resolved by the inline-cast fix above.
  - fp16 compile failure before the tiling fix: `NPU IR opt failed ... Failed to
    run BiShengHIR pipeline`, underlying `ub overflow, requires 2113536 bits
    while 1572864 bits available` (264,192 B vs 196,608 B); probes showed
    bm=16 and bm=12 fail, bm=10/bm=8 compile, no-multi-buffer bm=16 still
    fails. Resolved by block_m=8 for fp16.
  - A background full-suite launch appeared to exit 1 with a 2-line log; the
    task actually completed later with exit 0, and a clean foreground re-run
    was executed to remove the ambiguity (see below).
- design_error_reason: none (the UB over-commit is an implementation budget
  issue, not a mathematical/semantic design error; fixed without changing the
  public API or the operator semantics)
- rollback: no
- backup_path: n/a (first_impl)
- instrumentation_cleaned: n/a
- next_hint:
  - Final artifacts: `rmsnorm.py`, `test_rmsnorm.py`; final full-suite run
    `logs/all.log` -> `Test Passed!`, EXIT_CODE=0, 16x `[PRECISION_PASS]`
    (1 L0 + 15 L1), 0x `[PRECISION_FAIL]`, 11x `[BOUNDARY_PASS]`
    (6 L2 + 5 Boundary), 0x `[BOUNDARY_WARN]`.
  - `logs/l0.log` -> `[PRECISION_PASS] l0_basic`, `Test Passed!`, EXIT_CODE=0.
  - Coverage gate `logs/coverage.log` -> 18 PASS / 0 MISS / 0 N/A, EXIT_CODE=0.
  - Worst metrics: fp32 max_abs 1.43e-06 (L0) / 3.58e-07 (L1) / 1.91e-06
    (bnd_db_fp32), all n_exceed=0; fp16 max_abs 9.77e-04, n_exceed=0.
  - Compiler note (not a pass criterion): `npu_loop_vectorize.cc:801` warns
    that `T.Parallel` cast lowering fixes round mode to `rint`; this only
    affects non-default floor/trunc expectations, which rmsnorm does not use.
  - For a performance stage: block_m is a compile-time knob; the fp16 path is
    UB-limited to 8 rows at N=1024, the fp32 path to 16 rows.
