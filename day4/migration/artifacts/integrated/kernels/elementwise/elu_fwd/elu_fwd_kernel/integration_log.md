# Integration Log -- EluFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/elu_fwd/.migration_meta.json
- integrated: `_make_elu_kernel` -> `elu_fwd_kernel/_make_elu_kernel.py`
- design_docs: `_make_elu_kernel` -> `elu_fwd_kernel/_make_elu_kernel_DESIGN.md`（无缺失；与 `examples/elu_fwd/_make_elu_kernel/DESIGN.md` diff 一致）
- attempts: 1
- ascend_mode: Developer（Stage 3 产物字面量 `ASCEND_MODE = "Developer"` + `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`；wrapper `EluFwdKernel.ascend_mode` 集成前即一致，脚本回显 unchanged）

## 集成过程（integrate_kernel.py 首跑，2026-10-28T22:42:55+0800）

- `[copy]` kernel：`examples/elu_fwd/_make_elu_kernel/_make_elu_kernel.py` -> `elu_fwd_kernel/_make_elu_kernel.py`（整文件覆盖复制，diff 一致）
- `[copy]` design doc：`examples/elu_fwd/_make_elu_kernel/DESIGN.md` -> `elu_fwd_kernel/_make_elu_kernel_DESIGN.md`（diff 一致）
- `[init]` 聚合 `__init__.py` 生成（re-export `_make_elu_kernel`，`__all__` 单元素）
- `[wrapper]` `elu_fwd.py` 改写为 baseline/perf_opt 双 import 切换块（`wrapper_rewritten=true`）
- `[mode]` Developer（unchanged）
- `[smoke]` pass（聚合包 export + `EluFwdKernel` class 存在性断言通过）
- `[json]` 摘要见 /tmp/opencode/elu_integrate_attempt1.log（`design_docs={"_make_elu_kernel": ".../_make_elu_kernel_DESIGN.md"}`，无 `[warn] no DESIGN.md`）

### attempt 1（wrapper 注释块清理，非验证失败）

- 症状：脚本只替换了 `from ._elu_fwd_kernels import _make_elu_kernel` 语句本身；Stage 0 脚手架在原 wrapper 中预置的旧版切换块注释（`# --- baseline (Stage 3) ---` 注释行、`# --- perf_opt (Stage 4 tuned) ---` 注释行、注释的 perf_opt import 行）残留，改写后 wrapper 出现两份重复的切换块注释（功能正确——baseline import 唯一激活、perf_opt 仅注释——但结构冗余，且 Stage 4 按注释交换切换时易误切到含重复行的旧块）。
- 分类：wrapper 胶水错误（import 层注释残留；非设计层，非测试/用例问题）。
- 修复：备份 `history_version/elu_fwd.py_s5_attempt1` 与 `history_version/__init__.py_s5_attempt1` 后，仅清理 wrapper 重复注释行，收敛为与 leaky_relu 先例一致的单一切换块（baseline 行 `from .elu_fwd_kernel import _make_elu_kernel` 唯一激活；perf_opt 行注释占位）。核对：wrapper 源中 `from .elu_fwd_kernel import ...` 与 `Kernel source selection:` 头各仅 1 处，`ascend_mode == "Developer"`，import 复验通过。
- 结果：该清理发生在首次 report 之前；report 首跑即全量通过（无验证失败触发的修复循环）。

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261028_224432_491740_EluFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261028_224432_491740_EluFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261028_224432_491740_EluFwdOp/report.html
- run status: passed（correctness passed + benchmark passed；CLI 回显 `status=passed`）
- correctness: pass, 14/14 用例（test_elu.py：op 5 + param 5 + inplace + rejects_non_float_dtype + rejects_nonfinite_alpha + eval_roofline），op 用例 max_abs_err = 0.00e+00（run.json correctness.cases）
- benchmark: passed（9/9 bench 用例；msprof 6 个 profile case 全部 profiler_valid=true，invalid_profile_count=0，warnings=[]，数值引用自本次 run.json performance/benchmark_tests 段，仅记录不做修复）
  - smoke-1m [1048576]: float32 18.9404 us | float16 12.3402 us | bfloat16 12.1202 us
  - mlp-hidden [2048,4096] fp16: 78.761574 us（roofline 16.30%）/ bf16: 80.561615 us（roofline 18.20%）
  - mlp-hidden-wide [2048,8192] fp16: 156.303131 us（roofline 16.40%）/ bf16: 158.703171 us（roofline 18.45%）
  - gan-feat [16,256,64,64] fp16: 155.903122 us（roofline 16.45%）/ bf16: 157.983154 us（roofline 18.53%）
  - bandwidth 1.75 TB/s（全部 6 case）
- 工具链 stamp：tilelang 0.1.2+f388fab185ec7f59d827045ccec3c9e647efe09f（dev root /mnt/workspace/learn/tilelang-ascend）/ Ascend910B3 / CANN 9.0.0 / torch_npu 2.7.1.post4

## 集成产物

```
tileops/kernels/elementwise/elu_fwd/elu_fwd_kernel/
├── __init__.py                          # 聚合 __init__（re-export _make_elu_kernel）
├── _make_elu_kernel.py                  # Stage 3 kernel（整文件覆盖复制，886 行）
├── _make_elu_kernel_DESIGN.md           # Stage 1 交付件快照（脚本复制）
├── integration_log.md                   # 本文件
├── integration_report.json              # {"run_json": ".../20261028_224432_491740_EluFwdOp/run.json"}
└── history_version/                     # attempt 备份
    ├── elu_fwd.py_s5_attempt1
    └── __init__.py_s5_attempt1
```

wrapper `elu_fwd.py` 的 import 已改写为 baseline/perf_opt 双 import 切换块（baseline 激活：`from .elu_fwd_kernel import _make_elu_kernel`；perf_opt 注释占位：`# from .elu_fwd_kernel.perf_opt._make_elu_kernel import _make_elu_kernel`）；`EluFwdKernel.ascend_mode = "Developer"` 与产物声明一致。

## 调试历史

- attempt 1: 见上文「attempt 1（wrapper 注释块清理，非验证失败）」；report 首跑 passed，无验证失败触发的修复。
