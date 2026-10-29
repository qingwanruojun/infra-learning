# RETROSPECTIVE.md — erf_fwd (op 级)

> Stage 5 集成复盘（op 级单份，集成问题不归属单个函数；schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。逐函数复盘见 `examples/erf_fwd/erf_fwd_kernel/RETROSPECTIVE.md` Stage 1/2/3 章节。）

## Stage 5 (integrate) Retrospective — integrate — 2026-10-29

> 复盘对象：ErfFwdOp Stage 5 集成（integrate_kernel.py + 单算子 report）。verdict=INTEGRATE_COMPLETED，attempts=0（首次全量 report 即通过：correctness 16/16、benchmark 20/20、msprof 7 case 全 profiler_valid）。集成包 `examples/TileOPs/tileops/kernels/elementwise/erf_fwd/erf_fwd_kernel/`，报告 `reports/tileops/20261029_034150_398117_ErfFwdOp/`。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | integrate_kernel.py 的 wrapper 切换块改写不幂等：scaffold wrapper 已含 baseline/perf_opt 切换块注释时，改写采用「拼接新块」而非「整段替换」，残留旧块头注释（`baseline (scaffold)` 标题行 + 第一段块头）与重复的 perf_opt 占位注释行（改写后 L61–91 双块头 + 双 perf_opt 注释）；功能无影响（旧 GPU import 行已删、激活 import 正确），但需手工清理才与 expm1 先例形态一致 | integrate 输出 `[wrapper] ... rewritten` + 改写后 erf_fwd.py L61–91（`baseline (scaffold: extracted GPU reference)` 标题残留、L86/L89 重复 perf_opt 注释对）+ history_version/erf_fwd.py_s5_attempt1_glue_cleanup（清理前快照） | add-npu-op skill 的 integrate_kernel.py：改写 import 区段前先定位既有 `# Kernel source selection` 块的起止行并整段替换（或改写后断言该块只出现一次）；同类已知缺陷（[error]/[warn] 处理）一并回归 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| wrapper 调用面零适配集成先例：Stage 0 scaffold 调用式与 conductor factory K3/K11 契约对齐时（`erf_fwd_kernel(N, dtype)(block_size)(x)` vs `erf_fwd_kernel(N, dtype[, block_size]) -> kernel(bs)`），集成仅需切换 import 块、零胶水改动——dtype 双形态兼容（wrapper 传 torch.dtype 对象、factory 内 `str(dtype).split(".")[-1]` normalize）是关键；对照 expm1 需手工适配 `(x, N)` -> `(block_size)(x)` | P | erf_fwd.py（wrapper）L116–117 原调用式 + erf_fwd_kernel.py factory docstring「dtype argument accepts both torch.dtype objects and str」段 + 集成 smoke（`erf_fwd_kernel is k2` True、ascend_mode=Developer） | cd examples/TileOPs && python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/erf_fwd/.migration_meta.json && python -m tileops.reporting.cli run --op ErfFwdOp --prof-mode msprof（status=passed 一次通过） | tilelang 0.1.1.10+cann900 / Ascend910B3 / CANN 9.0.0 / 2026-10-29 | add-npu-op skill 已知修复目录（「wrapper 胶水错误」行补「Stage 0 scaffold 已契约对齐时零适配」正例） |
| erf A&S 22-op 链 TileOPs E2E 精度/性能基线：框架全量 correctness 8/8（max_abs_err fp32 6.74e-07 / fp16 1.22e-04 / bf16 7.63e-06，edge 5.96e-08，全部容差内）+ msprof 7 case 全 valid（16m fp16 294.81us / 64m fp16 1176.08us / 256m fp16 4702.47us，smoke-1m fp16 20.54us，roofline 20.2–32.2%）——transcendental 链相对同文件 4-op 链 expm1（128.14us @16m fp16）的 compute-bound 差距量化 | D | examples/TileOPs/reports/tileops/20261029_034150_398117_ErfFwdOp/run.json（summary + performance.cases + operators 条目）+ erf_fwd_kernel/integration_log.md 验证结果节 | cd examples/TileOPs && python -m tileops.reporting.cli run --op ErfFwdOp --prof-mode msprof | tilelang 0.1.1.10+cann900 / Ascend910B3 / CANN 9.0.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 / 2026-10-29 | pattern-library elementwise.md（transcendental 链上板基线行；Stage 4 调优路由参照） |

### Transferable Lessons（可迁移教训）

- integrate_kernel.py 改写 wrapper 后先机械核对 import 区段：grep 确认 `# Kernel source selection` 块头、`--- baseline` 标题、perf_opt 占位行各只出现一次——拼接式改写会残留 scaffold 旧块头与重复注释行（本任务残留双块头 + 双 perf_opt 注释对，功能无损但破坏先例形态一致性）。
- 集成前先比对 Stage 0 scaffold 的 wrapper 调用式与 conductor factory 契约（K3/K11 两段式）：两者已对齐时（如 erf_fwd）集成零胶水改动，勿照搬 expm1 的手工 call-surface 适配步骤；dtype 形态分歧（torch.dtype 对象 vs str）由 factory 内 normalize 兜底。
- wrapper 内残留清理属允许的胶水层编辑（wrapper 是 Stage 5 允许修改面），但须先备份到 history_version/ 且不触碰激活 import 语义——本次清理仅删重复注释行，激活 `from .erf_fwd_kernel import erf_fwd_kernel` 与 perf_opt 占位保持原样。
- 性能数值集成期只记录不修复：同 bench 文件内跨算子对比（erf 22-op 链 vs expm1 4-op 链）的 latency 差异是算法计算量的预期结果，不构成集成缺陷；数据留作 Stage 4 调优基线。
