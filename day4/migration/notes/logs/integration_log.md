# Integration Log -- LeakyReluFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/leaky_relu/.migration_meta.json
- integrated: `_make_leaky_relu_kernel` -> `leaky_relu_kernel/_make_leaky_relu_kernel.py`
- design_docs: `_make_leaky_relu_kernel` -> `leaky_relu_kernel/_make_leaky_relu_kernel_DESIGN.md`（无缺失）
- attempts: 1
- ascend_mode: Developer（来自 Stage 3 产物字面量 `ASCEND_MODE = "Developer"` / `TILELANG_ASCEND_MODE` setdefault；wrapper `LeakyReluFwdKernel.ascend_mode` 已核对一致，脚本回显 unchanged）

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261028_205329_441609_LeakyReluFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261028_205329_441609_LeakyReluFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261028_205329_441609_LeakyReluFwdOp/report.html
- run status: passed（correctness exit 0, benchmark exit 0）
- correctness: pass, 14/14 用例（test_leaky_relu.py：op 5 + param 5 + inplace + rejects_non_float_dtype + rejects_nonfinite_negative_slope + eval_roofline），max_abs_err = 0.00e+00
- benchmark: passed（9/9 bench 用例；msprof 6 个 profile case 全部 profiler_valid=true，invalid_profile_count=0）
  - smoke-1m float32: 17.3803 us | float16: 11.4202 us | bfloat16: 11.1802 us
  - gan-feat [16,256,64,64] fp16: 141.4628 us / bf16: 144.8429 us
  - gan-feat-deep [16,512,32,32] fp16: 71.3614 us / bf16: 72.0814 us
  - fc-wide [2048,4096] fp16: 71.1014 us / bf16: 73.4215 us
  - roofline utilization 13.06%–13.55%，bandwidth 1.75 TB/s（数值引用自 run.json performance 段，仅记录不做修复）
- warnings: 无

## 集成产物

```
tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/
├── __init__.py                          # 聚合 __init__（re-export _make_leaky_relu_kernel）
├── _make_leaky_relu_kernel.py           # Stage 3 kernel（整文件覆盖复制）
├── _make_leaky_relu_kernel_DESIGN.md    # Stage 1 交付件快照（脚本复制）
├── integration_log.md                   # 本文件
├── integration_report.json              # {"run_json": ".../20261028_205329_441609_LeakyReluFwdOp/run.json"}
└── history_version/                     # attempt 备份
    ├── leaky_relu.py_s5_attempt1
    └── __init__.py_s5_attempt1
```

wrapper `leaky_relu.py` 的 import 已改写为 baseline/perf_opt 双 import 切换块（baseline 激活，perf_opt 注释占位）。

## 调试历史

- attempt 1:
  - 症状: `integrate_kernel.py` 首次运行，复制/init/wrapper 改写均成功，但 import smoke FAILED——`ImportError: cannot import name '_fp8_accum_dtype_str' from '...leaky_relu_kernel'`。根因：wrapper 原 import 行 `from ._leaky_relu_fwd_kernels import _fp8_accum_dtype_str, _make_leaky_relu_kernel` 携带了 GPU 侧 manually-appended fp8 helper（meta `manually_appended_helpers`），该 helper 不在 conductor kernel 产物内，脚本聚合 `__init__.py` 生成 `[warn] not found` 后 smoke 断言失败。该 helper 在 NPU wrapper 中本就未使用（`# noqa: F401`，K6：fp8 helpers are not needed，NPU builder 对 is_fp8=True 直接 ValueError）。
  - 分类: wrapper 胶水错误（import 层，非设计层——factory 契约 `_make_leaky_relu_kernel(N, dtype, negative_slope)` 与 wrapper 调用形态一致）。
  - 修复: 备份 `history_version/{leaky_relu.py,__init__.py}_s5_attempt1` 后，仅改 wrapper 胶水 import（baseline 行只导入 `_make_leaky_relu_kernel`，同步 perf_opt 注释占位与说明注释），重跑幂等 `integrate_kernel.py`（重新生成聚合 `__init__.py`，wrapper 已集成不再改写，mode 核对 unchanged），smoke pass。
  - 结果: 重跑完整单算子 report → passed（14/14 correctness + 9/9 benchmark），集成完成。
