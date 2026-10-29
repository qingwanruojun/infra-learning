<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version":1,"operator":"ErfFwdOp","total_duration_s":7179.0,"breakdown":[{"name":"场景路由与准备","duration_s":105.0,"ratio_percent":1.5,"detail":"02:16:05-02:17:50 场景识别+harness 探测+场景文件载入+leaky_relu 教训读取+statectl init+start 0+kb_stale_check(154 stale)"},{"name":"Stage 0 脚手架","duration_s":610.0,"ratio_percent":8.5,"detail":"02:17:50-02:28:00 @tileops-scaffolder：add-npu-op 机器模式 7 文件移植+Tier1 校验+meta"},{"name":"Stage 0 收束","duration_s":66.0,"ratio_percent":0.9,"detail":"02:28:00-02:29:06 gate0+complete0+migration init(聚合状态)"},{"name":"Stage 1 首次设计","duration_s":2317.0,"ratio_percent":32.3,"detail":"02:29:06-03:07:43 @tilelang-op-designer：M0→R→M1 全流程（T.verf 禁用双证+A&S 7.1.26 机器验证+persistent 分核三要素+verify_equiv EQUIV_PASS）"},{"name":"Stage 1 收束","duration_s":22.0,"ratio_percent":0.3,"detail":"03:07:43-03:08:05 gate1+complete1+start2"},{"name":"Stage 2 检视","duration_s":856.0,"ratio_percent":11.9,"detail":"03:08:05-03:22:21 @tilelang-design-reviewer：9 维度全 pass+机械复核（verify_equiv 重跑/design_calc_check/kb_stale_check 对账）"},{"name":"Stage 2 收束","duration_s":23.0,"ratio_percent":0.3,"detail":"03:22:21-03:22:44 complete2+start3"},{"name":"Stage 3 开发","duration_s":833.0,"ratio_percent":11.6,"detail":"03:22:44-03:36:37 @tilelang-op-developer：A&S 22-op v-prefix 链+fp32 transit+persistent，L0/L1/L2/Boundary 全过→PRECISION_PASS"},{"name":"Stage 3 收束+二次校验","duration_s":173.0,"ratio_percent":2.4,"detail":"03:36:37-03:39:30 complete3+conductor 二次校验 --level all 全过+func done+phase INTEGRATE+start5"},{"name":"Stage 5 集成","duration_s":2147.0,"ratio_percent":29.9,"detail":"03:39:30-04:15:17 @tilelang-op-integrator：integrate_kernel.py+单算子 report（correctness 16/16+msprof 20/20 bench）"},{"name":"终态收束","duration_s":27.0,"ratio_percent":0.4,"detail":"04:15:17-04:15:44 complete5+integrate pass+phase DONE+opencode export full.json+Session 耗时分析"}]}
-->

# Session 耗时分析（ErfFwdOp harness 迁移）

- **full.json 路径**：`examples/erf_fwd/full.json`
- **session id**：`ses_e7c460938ffepo2vrs8VdX9x4N`（title "ErfFwdOp 迁移至 TileLang NPUIR"，agent=`tilelang-op-conductor`，directory=/mnt/workspace/learn/refs/tilelang-mlir-ascend）
- **开始时间**：2026-10-29 02:16:05（conductor session time_created）
- **结束时间**：2026-10-29 04:15:44（conductor session time_updated）
- **总耗时**：7179 s ≈ 1.99 h
- 数据源：仅使用 `examples/erf_fwd/full.json`（opencode export）与会话库中该 session 及全部子 Agent session 的 time_created/time_updated；各阶段子会话时间由 `full.json` 中 task tool 的 metadata.sessionId 关联 `~/.local/share/opencode/opencode.db` session 表交叉验证。

## 各阶段耗时（按实际开始时间升序）

| # | 阶段 | 开始 | 结束 | 耗时(s) | 占比 | 证据摘要 |
|---|------|------|------|--------|------|---------|
| 1 | 场景路由与准备 | 02:16:05 | 02:17:50 | ~105 | 1.5% | 场景识别 + harness 四条件探测（M1/M2/M3 全满足 + manifest ErfFwdOp 键）+ Read migration.md/harness.md + leaky_relu Transferable Lessons + statectl init + kb_stale_check（154 stale）+ start 0 |
| 2 | Stage 0 脚手架 | 02:17:50 | 02:28:00 | 610 | 8.5% | scaffolder 子会话 ses_e7c446c89ffecucMjPx1Sf74mG：add-npu-op 机器模式，7 文件移植 + Tier1 四项校验 + .migration_meta.json（SCAFFOLD_COMPLETED，extracted_functions=[erf_fwd_kernel]） |
| 3 | Stage 0 收束 | 02:28:00 | 02:29:06 | ~66 | 0.9% | gate0（pass）+ complete0 + migration init（聚合状态 DEV_LOOP） |
| 4 | Stage 1 首次设计 | 02:29:06 | 03:07:43 | 2317 | 32.3% | designer 子会话 ses_e7c3a1c3bffeTSqEQ6vWpo5a4L：Phase M0→R→M1，T.verf 泰勒本质禁用双证、A&S 7.1.26 等价替代、persistent 分核三要素（910B3 核数实查 20→40）、verify_equiv.py EQUIV_PASS（fp32 max\|err\| 5.868e-07） |
| 5 | Stage 1 收束 | 03:07:43 | 03:08:05 | ~22 | 0.3% | gate1（pass）+ complete1 + start2 |
| 6 | Stage 2 检视 | 03:08:05 | 03:22:21 | 856 | 11.9% | reviewer 子会话 ses_e7c166b44ffeq89yZWlLVlp9lF：9 维度全 pass（0 阻塞）+ 机械复核段（verify_equiv 重跑逐位一致 / design_calc_check 4 checks 3 skip 人工补位 / kb_stale_check 154 条与设计 §9 R4 声明吻合） |
| 7 | Stage 2 收束 | 03:22:21 | 03:22:44 | ~23 | 0.3% | complete2 + start3 |
| 8 | Stage 3 开发 | 03:22:44 | 03:36:37 | 833 | 11.6% | developer 子会话 ses_e7c09018affejSrDK9KjeK2hNz：A&S 22-op v-prefix 链 + fp32 transit + persistent 分核，L0/L1/L2/Boundary 全过 → PRECISION_PASS（fp32 max_diff 8.028e-07） |
| 9 | Stage 3 收束+二次校验 | 03:36:37 | 03:39:30 | 173 | 2.4% | complete3 + conductor 二次校验 `--level all` 全量重跑（L0/L1/L2/Boundary 全 PASS，exit 0）+ migration func done + phase INTEGRATE + start5 |
| 10 | Stage 5 集成 | 03:39:30 | 04:15:17 | 2147 | 29.9% | integrator 子会话 ses_e7bf9a877ffeEs4DUDmwy5cT1o：integrate_kernel.py 一次成功（wrapper 切换块零适配）+ `tileops.reporting.cli run --op ErfFwdOp --prof-mode msprof`：correctness 16/16、msprof 20/20 bench 全 profiler_valid、report passed |
| 11 | 终态收束 | 04:15:17 | 04:15:44 | ~27 | 0.4% | complete5（gate5 pass）+ migration integrate pass + migration phase DONE + opencode export full.json + 本 Session 耗时分析（与 Stage 5 尾端衔接） |

> 阶段 4（Stage 1 设计）占 32.3%，为最大单项——含迁移分析（源码亲读、T.verf 禁用论证、A&S 7.1.26 复核）与设计期机器验证（verify_equiv.py 6M 随机 + 23 角点），属迁移任务的设计高投入区，合理。阶段 10（Stage 5 集成）占 29.9%——含 msprof 20/20 benchmark 采集（7 个 ErfFwdOp profiled case），上板 profiling 为耗时主体，属硬件实测必要成本。

## 减少耗时方案

1. **Stage 5 msprof 采集并行化**：当前 20 个 bench 用例（含 ErfFwdOp 7 个 profiled case + 同文件 Expm1FwdOp 回归）串行 msprof，占 2147s 主体。可按 `tool-perf-collect` 的 `--batch` 批量并行模式拆分 profiled case，预期可将 Stage 5 压缩至 ~1200–1500s（节省 ~10–15% 总时长）。
2. **Stage 1 与 Stage 0 探测重叠**：harness 四条件探测与 leaky_relu 教训读取可在 stage_plan 组装期并行（当前串行 ~105s，节省有限但零成本）。
3. **二次校验复用 developer 测试缓存**：Stage 3 收束 173s 中二次校验 `--level all` 重跑占主体；其与 developer 的 L0/L1/L2/Boundary 为同一测试套件，可在 developer 返回 `[PRECISION_PASS]` 时以 `--level all` 直接作为二次校验依据（同批结果），免去独立重跑，预期节省 ~90s。前提：conductor 侧保留「独立复跑」作为门禁兜底，仅在时间受限时启用。
4. **Stage 0/5 的 gate 与聚合状态操作批量化为单 CLI 调用**：当前 gate0→complete0→migration init、complete5→integrate→phase DONE 各为 2–3 次串行 statectl 调用，可合并为一次原子迁移（各阶段节省 ~20–40s，共 ~60–100s）。
