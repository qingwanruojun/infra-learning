# RETROSPECTIVE.md — elu_fwd (migration-harness, op 级)

> 追加式复盘日志（canonical schema: .agents/skills/tilelang-skill-evolution/references/retrospective-schema.md）。Stage 5 集成复盘为 op 级单份（集成问题不归属单个函数）；Stage 1/2/3 的逐函数复盘在 `examples/elu_fwd/_make_elu_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt1-completed — 2026-10-28

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | integrate_kernel.py 的 wrapper 改写只替换 `from .{extracted_module} import ...` 语句本身；当 Stage 0 脚手架已在原 wrapper 预置"伪 baseline/perf_opt 切换块"（`# --- baseline (Stage 3) ---` 注释行 + `# --- perf_opt ---` 注释行 + 注释掉的 perf_opt import 行）时，这些注释行会残留并与脚本生成的新块重复，wrapper 出现两份切换块注释。功能上无害（baseline import 唯一激活），但破坏"单一切换块"的预期形态，且 Stage 4 按注释交换切换时可能误操作到含重复行的旧块 | examples/TileOPs/tileops/kernels/elementwise/elu_fwd/elu_fwd_kernel/history_version/elu_fwd.py_s5_attempt1（改写后 63-81 行出现两份 `Kernel source selection:` 头 + 两份 perf_opt 注释行）；清理后形态见同目录 elu_fwd.py（各仅 1 处） | integrate_kernel.py：`rewrite_wrapper_import` 在替换 import 语句前，先检测并移除 wrapper 中既有的"Kernel source selection"注释区/伪切换块（或输出 `[warn] stale scaffolder selection block removed`），保证改写后仅剩单一切换块 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| EluFwdOp 集成基线实测：14/14 correctness（op 用例 max_abs_err=0）+ 9/9 bench，msprof 6 case 全有效（invalid_profile_count=0），smoke-1m fp32 18.94us / fp16 12.34us / bf16 12.12us，mlp-hidden [2048,4096] fp16 78.76us（roofline 16.30%），gan-feat [16,256,64,64] fp16 155.90us（roofline 16.45%），bw 1.75 TB/s | D | examples/TileOPs/reports/tileops/20261028_224432_491740_EluFwdOp/run.json#summary/performance | `cd examples/TileOPs && python -m tileops.reporting.cli run --op EluFwdOp --prof-mode msprof` | tilelang 0.1.2+f388fab185ec7f59d827045ccec3c9e647efe09f / Ascend910B3 / CANN 9.0.0 / torch_npu 2.7.1.post4 | 待定（Stage 4 perf_opt 对比基线） |

### Transferable Lessons（可迁移教训）

- 集成前先检查 wrapper 是否被 Stage 0 脚手架预置了"伪 baseline/perf_opt 切换块"（注释头 + 注释的 perf_opt import 行）：integrate_kernel.py 只替换 import 语句、不清除周边注释，改写后会出现重复注释块；run 后人工收敛为单一切换块即可（功能无损，但保留单块形态才能保证 Stage 4 注释交换切换不出错）。
- wrapper 预置切换块里注释掉的 perf_opt import 路径（`from .{op_slug}_kernel.perf_opt.{func} import ...`）与脚本重写的判定正则不冲突（正则匹配的是未注释的 `from .{op_slug}_kernel... import`），不会因预置注释行误判 "already integrated"——首次运行仍会正常改写。
