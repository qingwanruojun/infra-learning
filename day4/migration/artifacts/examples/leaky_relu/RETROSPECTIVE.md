# RETROSPECTIVE.md — leaky_relu (migration-harness, op 级)

> 追加式复盘日志（canonical schema: .agents/skills/tilelang-skill-evolution/references/retrospective-schema.md）。Stage 5 集成复盘为 op 级单份（集成问题不归属单个函数）；Stage 1/2/3 的逐函数复盘在 `examples/leaky_relu/_make_leaky_relu_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt1-completed — 2026-10-28

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | integrate_kernel.py 把 wrapper 原始 import 行里的**全部**名字（含 GPU 侧 manually-appended helper，如 `_fp8_accum_dtype_str`）抄进 baseline/perf_opt 切换块，而这些 helper 不在 conductor kernel 产物内，聚合 `__init__.py` 生成 `[warn] not found` 后 import smoke 必然失败——脚本不支持"wrapper import 名单 ⊃ 产物定义名单"的场景，也无自动剔除死 helper 的提示语 | examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/history_version/__init__.py_s5_attempt1（`# [warn] _fp8_accum_dtype_str: not found in integrated modules` 行）+ 首次运行 `[error] import smoke test FAILED` 输出 | integrate_kernel.py：对 `defining_module` 找不到的名字在改写 wrapper 前给出明确 `[warn] dropping non-product import '<name>' from baseline block`（或要求 `--wrapper-imports` 白名单），避免每次 harness 集成都手工修 wrapper | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 集成期 wrapper import 死 helper 剔除：NPU wrapper 对 GPU manually-appended helper（fp8 类）是死代码，集成时只导入 conductor 产物实际定义的函数即可通过 smoke | P | examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py（"Stage 5 integration note" 注释块 + baseline import 仅含 `_make_leaky_relu_kernel`） | 在 wrapper 携带 `manually_appended_helpers` import 的 meta 上运行 integrate_kernel.py → smoke 失败；仅保留产物定义名后重跑（幂等）→ smoke pass | tilelang 0.1.1.10+ubuntu.20.4.cann900 / Ascend910B3 / CANN 9.0.0 | pattern-library |
| LeakyReluFwdOp 集成基线实测：14/14 correctness（max_abs_err=0）+ 9/9 bench，msprof 6 case 全有效，smoke-1m fp16 11.42us / bf16 11.18us，gan-feat [16,256,64,64] fp16 141.46us（roofline 13.55%） | D | examples/TileOPs/reports/tileops/20261028_205329_441609_LeakyReluFwdOp/run.json#summary/performance | `cd examples/TileOPs && python -m tileops.reporting.cli run --op LeakyReluFwdOp --prof-mode msprof` | tilelang 0.1.1.10+ubuntu.20.4.cann900 / Ascend910B3 / CANN 9.0.0 | 待定（Stage 4 perf_opt 对比基线） |

### Transferable Lessons（可迁移教训）

- 集成前先核对 wrapper 的 `from .{extracted_module} import <names>` 名单：GPU 侧 manually-appended helper（meta `manually_appended_helpers`，如 fp8 dtype 选择器）不在 NPU conductor 产物内，集成 baseline import 块只保留产物实际定义的名字，否则 integrate_kernel.py 聚合 `__init__.py` 的 smoke 必失败（NPU wrapper 中这类 helper 本就是死代码，删除不改变行为）。
- integrate_kernel.py 幂等可重复跑：首次 smoke 失败后只改 wrapper 胶水 import，再跑一遍脚本即可（重新生成聚合 `__init__.py`，wrapper 判定 "already integrated" 不再改写，ascend_mode 重新核对）。
