# Day 5 STATUS

**一句话**：5 个 TileOPs GPU 算子已全部迁到 Ascend 910B3 NPUIR，框架侧 5/5 `phase=DONE`、集成 report 全 `passed`；独立复验见 `logs/independent/`，交付包见 `tilelang-ascend/day5/`。

- 已完成：LeakyRelu / Elu / Expm1 / Erf / BitwiseNot，正确性 14/14、14/14、8/8、8/8（共享用例 16/16）、9/9；benchmark 9/9、9/9、10/10、20/20（含 Expm1 10 条）、8/8。
- 队列：v2 串行跑完 Elu → Expm1 → Erf → BitwiseNot，2026-10-28 21:13 → 10-29 06:07（UTC+8），约 8h54m；v1 队列在 Elu DONE 后已清理。
- 独立复验：5 个算子的 pytest 不经框架 CLI 直接复跑，结果与日志在 `logs/independent/`（详见 `COMPLETION-REPORT.md`）。
- 产物：每算子 `examples/<slug>/` 下有 DESIGN / REVIEW / RETROSPECTIVE / kernel / timeline；报告在 `examples/TileOPs/reports/tileops/`；框架仓库已有 5 条 evolution 提交。
- 已知缺口：Stage 4（性能调优）按要求跳过；Erf 报告混入 10 条 Expm1 benchmark 记录；`/tmp` 队列级日志因会话重启丢失（算子级证据完整）；迁移产物本身未 git commit。
- 状态探测口径：顶层 `.stage_state.json` 在 DEV_LOOP 期间停在 `INTEGRATE`，只有 Stage 5 结束才写 `DONE`；中途进度看 `_make_<op>_kernel/.stage_state.json`。
