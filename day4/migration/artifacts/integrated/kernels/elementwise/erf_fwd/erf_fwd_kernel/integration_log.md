# Integration Log -- ErfFwdOp
- meta: examples/TileOPs/tileops/kernels/elementwise/erf_fwd/.migration_meta.json
- integrated: erf_fwd_kernel -> erf_fwd_kernel/erf_fwd_kernel.py
- design_docs: erf_fwd_kernel -> erf_fwd_kernel/erf_fwd_kernel_DESIGN.md (source: examples/erf_fwd/erf_fwd_kernel/DESIGN.md; none missing)
- ascend_mode: Developer (kernel ASCEND_MODE literal "Developer"; wrapper ErfFwdKernel.ascend_mode unchanged, already "Developer")
- wrapper_import: rewritten to baseline/perf_opt selection block by integrate_kernel.py (baseline active: `from .erf_fwd_kernel import erf_fwd_kernel`; perf_opt commented placeholder). Post-rewrite residue cleanup (glue-layer only, backed up to history_version/erf_fwd.py_s5_attempt1_glue_cleanup): the script rewrite left the scaffold's old selection-block header + old `baseline (scaffold)` title comment + a duplicated perf_opt placeholder comment pair; removed the duplicate comment lines, kept one selection block. Active import unchanged by the cleanup.
- wrapper_call_surface: no adaptation needed -- scaffold `_erf_fwd_wrapped` already calls `erf_fwd_kernel(N, dtype)(block_size)(x)` (erf_fwd.py L117), matching the conductor factory contract `erf_fwd_kernel(N, dtype[, block_size]) -> kernel(bs=default_block) -> main` (K3/K11 two-stage; dtype passed as torch.dtype object via `getattr(torch, dtype_str)`, normalized inside the factory). Contrast expm1 which needed a manual call-surface fix; erf_fwd's Stage 0 scaffold was already the target shape.
- attempts: 0 (debug loop not entered; first full report run passed)
- integration script: [json] summary captured -- functions=["erf_fwd_kernel"], wrapper_rewritten=true, ascend_mode=Developer (unchanged), reexported=["erf_fwd_kernel"], smoke=pass, timestamp=2026-10-29T03:40:25+0800
- history_version/: erf_fwd.py_s5_attempt1_glue_cleanup (pre-cleanup wrapper backup; comment-residue removal only)

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261029_034150_398117_ErfFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261029_034150_398117_ErfFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261029_034150_398117_ErfFwdOp/report.html
- report status: passed (summary.correctness_passed=true, benchmark_passed=true; warnings=[]; exit_codes correctness=0 benchmark=0; invalid_profile_count=0, profiler_fallback_count=0)
- correctness: pass -- 16/16 cases in tests/ops/test_unary_math.py (ErfFwdOp 8/8: test_erf [1048576 x 3 dtypes, 4000000 x 2] + test_erf_edge[4096-dtype0] + test_erf_rejects_non_float_dtype + test_erf_eval_roofline; sibling Expm1FwdOp 8/8 also re-verified). ErfFwdOp max_abs_err: fp32 6.74e-07 (atol 1e-5), fp16 1.22e-04 (atol 1e-3), bf16 7.63e-06 (atol 1.6e-2), edge 5.96e-08; Avg Max Abs Error = 4.333e-05
- benchmark: pass -- benchmarks/ops/bench_unary_elementwise.py, 20/20 bench tests passed (exit_code 0), 7 msprof profiled ErfFwdOp cases all profiler_valid=true (0 fallback / 0 invalid / 0 kernel-filter fallback)
- benchmark numbers (from run.json performance.cases + benchmark_tests properties, tag=tileops, msprof, device Ascend910B3 / CANN 9.0.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 / tilelang 0.1.1.10+cann900):
  | label | dtype | shape | block_size | latency_us | bandwidth TB/s | roofline % |
  |---|---|---|---|---|---|---|
  | elementwise-16m | float16 | [4096,4096] | 2048 | 294.8059 | 1.75 | 31.24 |
  | elementwise-16m | bfloat16 | [4096,4096] | 2048 | 298.4460 | 1.75 | 32.08 |
  | elementwise-16m | float32 | [4096,4096] | 1024 | 442.6689 | 1.75 | 20.21 |
  | elementwise-64m | float16 | [8192,8192] | 2048 | 1176.0835 | 1.75 | 31.31 |
  | elementwise-64m | bfloat16 | [8192,8192] | 2048 | 1187.9637 | 1.75 | 32.22 |
  | elementwise-256m | float16 | [16384,16384] | 2048 | 4702.4741 | 1.75 | 31.32 |
  | elementwise-256m | bfloat16 | [16384,16384] | 2048 | 4761.3149 | 1.75 | 32.16 |
  (plus bench-only smoke-1m rows without profiled roofline cases: fp32 30.10 us / fp16 20.54 us / bf16 21.14 us)
- perf note (record-only, no fix): vs sibling Expm1FwdOp (same bench file, 128.14 us @ 16m fp16), ErfFwdOp runs 294.81 us @ 16m fp16 -- expected for the 22-op A&S 7.1.26 vector chain vs expm1's ~4-op chain (compute-bound transcendental, roofline 20-32% vs expm1 14-27%). Recorded for Stage 4 routing context only; performance is out of Stage 5 repair scope.

## 调试历史
- none (no debug attempts; integration + wrapper comment-residue cleanup + full report all passed on the first run)
