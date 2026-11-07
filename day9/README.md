# Day 9｜Amin 任务 · Agent 性能调优

当前状态：**Agent 阶段评测已归档，人工调优进行中**。包内 `final`、`终态` 等措辞均指 Agent 本轮收束版本，不代表人工优化已经完成。

后续 `why-not-60` 人工复核材料归入 [Day 10 算子开发阶段记录](../day10/note/算子开发-阶段记录.md)，作为下一天多算子汇总中的 Amin 部分。

先读 [Day 9 学习报告](note/day9学习报告.md)。当前正式评测：Amin 正确性 14/14，四 workload 等权平均 Ratio 42.73%→51.97%；尚无人工调优后的结果。

```text
day9/
├── README.md
├── note/day9学习报告.md
└── code/agent-snapshot/       原始交付包，按原目录保留
    ├── PACKAGE.md
    ├── notes/                原始学习记录、估算、口径分析
    ├── kernel/               Agent 阶段采用版本、候选、A/B 与回归
    ├── official-report/      正式评测、pytest、msprof 原始产物
    ├── raw/                  benchmark 快照与标定记录
    ├── logs/
    └── gaps/
```

主要证据：

- [正式报告](code/agent-snapshot/official-report/report.md) · [benchmark JSON](code/agent-snapshot/official-report/benchmark.json) · [运行元数据](code/agent-snapshot/official-report/run.json)
- [优化过程](code/agent-snapshot/kernel/opt_log.md) · [候选与设备测量记录](code/agent-snapshot/kernel/perf_records.jsonl)
- [Agent 阶段采用代码](code/agent-snapshot/kernel/final/) · [原始带宽分析](code/agent-snapshot/notes/note/roofline-calibration.md)

本文通过离线核对交付包整理，没有重新上卡。原包的性能上限和编译器机制分析仍需结合证据审慎使用；更新结论优先看主报告。复现路径依赖原机器的 `refs/tilelang-mlir-ascend/examples/TileOPs` 工程，本包并不包含完整框架、wrapper 与测试环境。
