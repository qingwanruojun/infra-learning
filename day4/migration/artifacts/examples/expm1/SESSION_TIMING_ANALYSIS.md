# SESSION_TIMING_ANALYSIS.md — Expm1FwdOp 迁移 TileLang NPUIR

> 由 tilelang-op-conductor 按 harness.md §7.1 在 Stage 5 终态后生成；数据源仅本次导出的 `examples/expm1/full.json`。

<!-- TILEOPS_SESSION_TIMING_V1
{"schema_version":1,"operator":"Expm1FwdOp","total_duration_s":9541.6,"breakdown":[{"name":"SCAFFOLD (Stage 0)","duration_s":738.4,"ratio_percent":7.7,"detail":"23:05:41→23:18:00 tileops-scaffolder 完成 7 文件脚手架 + .migration_meta.json + 迁移 prompt（日志：task 会话 ses_e7cf45728..."},{"name":"DESIGN first (Stage 1)","duration_s":2169.0,"ratio_percent":22.7,"detail":"23:19:50→23:55:58 tilelang-op-designer first_design 生成 DESIGN.md + verify_equiv.py（54/54 行精度机器验证）"},{"name":"REVIEW first (Stage 2)","duration_s":1414.9,"ratio_percent":14.8,"detail":"23:56:46→00:20:21 tilelang-design-reviewer 首轮检视：结论不通过（3 阻塞：内嵌表不可复现 / wrapper 契约断言错误 / UB 口径误用）"},{"name":"DESIGN revision (Stage 1)","duration_s":1738.3,"ratio_percent":18.2,"detail":"00:21:10→00:50:08 designer mode=revision 修复 3 阻塞 + 4 建议（表格重填 / 两段式契约 / 平台值口径+guard 分 dtype）"},{"name":"REVIEW re-review (Stage 2)","duration_s":812.4,"ratio_percent":8.5,"detail":"00:50:28→01:04:01 复审：结论通过（0 阻塞，修订闭环逐项验证到决策行）"},{"name":"DEVELOP (Stage 3)","duration_s":719.3,"ratio_percent":7.5,"detail":"01:04:28→01:16:27 tilelang-op-developer first_impl：[PRECISION_PASS]，L0/L1/L2/Boundary 全过（含二次校验重跑全量）"},{"name":"INTEGRATE (Stage 5)","duration_s":1437.0,"ratio_percent":15.1,"detail":"01:17:28→01:41:25 tilelang-op-integrator 集成 + wrapper 两段式适配 + 单算子 report（correctness 8/8、benchmark 7/7 有效）"},{"name":"conductor 编排（Primary 内联）","duration_s":512.4,"ratio_percent":5.4,"detail":"23:03:06→01:42:08 场景路由/状态机/门禁/知识预注入/耗时分析等内联工具调用与推理，含各阶段间编排间隙"}]}
-->

## 1. 基本信息

- **session id**: `ses_e7cf6db23ffeg5DIjyKH2YMjIT`
- **full.json 路径**: `examples/expm1/full.json`（`opencode export` 导出，739 KB，63 messages，77 个带时间戳的 tool 调用）
- **开始时间**: 2026-10-28 23:03:06（首个 tool 事件 start）
- **结束时间**: 2026-10-29 01:42:08（末个 tool 事件 end）
- **总耗时**: 9541.6 s ≈ 159.0 min ≈ 2 h 39 m
- **算子/场景**: Expm1FwdOp（op_slug=expm1，family=elementwise）/ migration-harness，stage_plan=[0,(1→2→3)×1函数,5]

## 2. 阶段耗时细分（按实际开始时间升序）

| 序号 | 阶段 | 开始时间 | 结束时间 | 耗时 (s) | 占比 | 证据摘要 |
|---|---|---|---|---|---|---|
| 1 | SCAFFOLD (Stage 0) | 23:05:41 | 23:18:00 | 738.4 | 7.7% | tileops-scaffolder 子会话 `ses_e7cf45728...`，7 文件脚手架 + Tier1 校验 + meta |
| 2 | DESIGN first (Stage 1) | 23:19:50 | 23:55:58 | 2169.0 | 22.7% | tilelang-op-designer 首轮设计，DESIGN.md(826 行) + verify_equiv.py，54/54 行精度收口 |
| 3 | REVIEW first (Stage 2) | 23:56:46 | 00:20:21 | 1414.9 | 14.8% | tilelang-design-reviewer 首轮检视 → 不通过（3 阻塞） |
| 4 | DESIGN revision (Stage 1) | 00:21:10 | 00:50:08 | 1738.3 | 18.2% | designer revision 1，修复 3 阻塞 + 4 建议，DESIGN.md → 857 行 |
| 5 | REVIEW re-review (Stage 2) | 00:50:28 | 01:04:01 | 812.4 | 8.5% | 复审 → 通过（0 阻塞） |
| 6 | DEVELOP (Stage 3) | 01:04:28 | 01:16:27 | 719.3 | 7.5% | tilelang-op-developer first_impl → [PRECISION_PASS]，L0/L1/L2/Boundary 全过 |
| 7 | INTEGRATE (Stage 5) | 01:17:28 | 01:41:25 | 1437.0 | 15.1% | tilelang-op-integrator：集成 + wrapper 两段式适配 + 单算子 report（correctness 8/8 + bench 7/7） |
| 8 | conductor 编排（Primary 内联） | 23:03:06 | 01:42:08 | 512.4 | 5.4% | 场景路由/状态机/门禁/知识预注入/耗时分析等内联工具与推理，含阶段间编排间隙 |

> 注：阶段 8 与阶段 1–7 在时间上存在交叠（编排动作穿插于子 agent 调度之间），表中保持实际开始时间顺序并显式标注；`(1→2→3)` 设计-检视-修订循环合计 6134.6 s（64.3%）。

## 3. 总耗时与占比分析

1. **设计-检视-修订循环占大头（64.3%）**：首轮 DESIGN(22.7%) + 首轮 REVIEW(14.8%) + 修订 DESIGN(18.2%) + 复审(8.5%)。其中**修订循环额外消耗 2550.7 s（26.7%）**，根因是首轮设计的三处阻塞问题（内嵌表不可复现 / wrapper 契约断言错误 / UB 口径误用）全部属「设计自述与实跑/落地侧现状不一致」类，非算法难度本身。
2. **子 agent 总执行 9029.3 s（94.6%）**，conductor 编排仅 512.4 s（5.4%）——编排开销健康，时间几乎全部花在子 agent 工作。
3. **INTEGRATE 15.1%（1437.0 s）**：单算子 report（correctness + msprof benchmark 7 case）本身耗时较长（上板 profiling），且含 wrapper 两处适配的落实与验证。
4. **SCAFFOLD 7.7%（738.4 s）**：7 文件移植 + Tier1 校验，单函数算子属正常下限。

## 4. 减少耗时的方案

1. **在设计阶段引入「可复现性自检」门禁（可省 ~26.7% 修订循环）**：三个阻塞问题均可在首轮设计内部拦截——① verify_equiv.py 内嵌表必须由脚本输出直接回填（禁止手工誊写/星号标注），② wrapper 契约断言以「亲读 wrapper + 家族先例」为准而非外推，③ UB 口径必须按 pattern-library 平台值（非 ×系数）并逐档复算。若在 designer 的 Phase 7 自检中增加这三条机械检查，修订循环 2550.7 s 可大部分避免。
2. **知识预注入提前命中关键条目**：本次 kb_search 已返回 PL-1.17（transit）、CASE-elementwise-elu-migration 等，但 TRAP-UB-multibuffer-inflation 的「平台值口径」未在设计首轮被正确引用（问题 3 根因）。将「UB 多缓冲论证必须引用平台值」作为 designer 的固定检查项，可消除首轮阻塞。
3. **并行化空间有限**：harness 单函数串行依赖（SCAFFOLD→DESIGN→REVIEW→DEVELOP→INTEGRATE）无法并行；但若 op 含多函数（N≥2），SCAFFOLD 与首个函数 DESIGN 可部分重叠，且多函数 DESIGN 可并行（本任务仅 1 函数不适用）。
4. **上板 profiling 时间优化**：INTEGRATE 的 msprof 采集（7 case × 多轮）是硬性上板成本，可通过减少冗余轮次或复用预热会话压缩，但收益有限且以数据有效性为代价（本次 7/7 全有效，不建议压缩）。
