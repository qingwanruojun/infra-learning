# Integration Log -- Expm1FwdOp
- meta: examples/TileOPs/tileops/kernels/elementwise/expm1/.migration_meta.json
- integrated: expm1_fwd_kernel -> expm1_kernel/expm1_fwd_kernel.py
- design_docs: expm1_fwd_kernel -> expm1_kernel/expm1_fwd_kernel_DESIGN.md (source: examples/expm1/expm1_fwd_kernel/DESIGN.md; none missing)
- ascend_mode: Developer (kernel ASCEND_MODE literal "Developer"; wrapper Expm1FwdKernel.ascend_mode unchanged, already "Developer")
- wrapper_import: rewritten to baseline/perf_opt selection block (baseline active: `from .expm1_kernel import expm1_fwd_kernel`; perf_opt commented placeholder)
- wrapper_call_surface: adapted at expm1.py L95 -- `expm1_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)` -> `expm1_fwd_kernel(N, dtype_str, block_size=block_size)(block_size)(x)` (two-stage family standard, same shape as leaky_relu L98 / elu_fwd L99; drop trailing `, N` + curried `(block_size)` application). L191-area `expm1_fwd_kernel(self.N_total, self.dtype_str)` factory call unchanged (block_size default legal; dtype normalization done inside the kernel factory).
- attempts: 0 (debug loop not entered; first full report run passed)
- integration script: [json] summary captured -- functions=["expm1_fwd_kernel"], wrapper_rewritten=true, ascend_mode=Developer (unchanged), reexported=["expm1_fwd_kernel"], smoke=pass, timestamp=2026-10-29T01:19:12+0800
- history_version/: expm1.py_s5_attempt1.py (pre-call-surface-adaptation wrapper backup)

## 验证结果
- report_run_json: examples/TileOPs/reports/tileops/20261029_012203_068316_Expm1FwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261029_012203_068316_Expm1FwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261029_012203_068316_Expm1FwdOp/report.html
- report status: passed (summary.correctness_passed=true, benchmark_passed=true; warnings=[] errors=[])
- correctness: pass -- 8/8 cases (tests/ops/test_unary_math.py; 5 x test_expm1 [1048576 x 3 dtypes, 4000000 x 2], test_expm1_edge[4096-dtype0], test_expm1_rejects_non_float_dtype, test_expm1_eval_roofline), Avg Max Abs Error = 0.00e+00, exit_code 0
- benchmark: pass -- benchmarks/ops/bench_unary_elementwise.py, 10/10 bench tests passed (exit_code 0), 7 msprof profiled cases all profiler_valid=true (0 fallback / 0 invalid / 0 kernel-filter fallback)
- benchmark numbers (from run.json performance.cases, tag=tileops, msprof, device Ascend910B3 / CANN 9.0.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4):
  | label | dtype | shape | block_size | latency_us | bandwidth TB/s | roofline % |
  |---|---|---|---|---|---|---|
  | elementwise-16m | float16 | [4096,4096] | 2048 | 128.1226 | 1.75 | 14.97 |
  | elementwise-16m | bfloat16 | [4096,4096] | 2048 | 130.3226 | 1.75 | 14.71 |
  | elementwise-16m | float32 | [4096,4096] | 1024 | 219.7044 | 1.75 | 17.45 |
  | elementwise-64m | float16 | [8192,8192] | 2048 | 507.1902 | 1.75 | 18.90 |
  | elementwise-64m | bfloat16 | [8192,8192] | 2048 | 515.8703 | 1.75 | 18.47 |
  | elementwise-256m | float16 | [16384,16384] | 2048 | 2053.0210 | 1.75 | 27.08 |
  | elementwise-256m | bfloat16 | [16384,16384] | 2048 | 2090.6418 | 1.75 | 26.60 |
  (plus bench-only smoke-1m rows without profiled cases: fp32 16.46 us / fp16 10.26 us / bf16 10.38 us)
- pre-report smoke (ad-hoc, 3 dtypes @ N=4096 via Expm1FwdKernel.forward): pass; bf16 max_diff=0, fp32 max_diff=1.907e-06 (tol 1e-5)

## 调试历史
- none (no debug attempts; integration + wrapper call-surface adaptation + full report all passed on the first run)
