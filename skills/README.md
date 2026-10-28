# Skills · 昇腾算子开发与评测

四个可被 Claude Code / 兼容 harness 直接加载的 skill。每个 skill 是一个目录：

```
skills/<name>/SKILL.md            # frontmatter: name / description，正文是检查表
skills/<name>/references/*.md     # 可选：失败案例、细节
```

`description` 决定什么时候触发——模型在开始任务前读它，命中就加载正文。
所以写法是**触发条件 + 硬规则**，不是教程。

## 清单

| Skill | 何时触发 | 核心内容 |
|---|---|---|
| [`ascend-npuir-bench-guardrails`](ascend-npuir-bench-guardrails/SKILL.md) | 写/改 npuir kernel、调 tile/buffer/rstd、要写"快/慢多少"的结论 | 10 条铁律、形状矩阵、三栏计时、ABBA 协议、证据分级（E0–E4）、禁止句清单；[`references/day3-failure-cases.md`](ascend-npuir-bench-guardrails/references/day3-failure-cases.md) 收录 7 个 A 类 + 10 个 B 类真实失败案例 |
| [`ascend-npu-benchmark`](ascend-npu-benchmark/SKILL.md) | 要证明"比 torch/CANN 快还是慢"，或探针数字明显不对 | host / device / e2e 三时钟分离，`op_summary` 取 device 时长，10 轮 ABBA，尺寸 ≥256 MiB，IQR/MAD |
| [`tilelang-npuir`](tilelang-npuir/SKILL.md) | 在 910B 上新增或修改 TileLang 算子 | UB 预算与多缓冲、Expert/Developer 模式、动态 shape 尾部、环境与缓存陷阱、launcher/NPUGraph 限制 |
| [`study-notes-daily`](study-notes-daily/SKILL.md) | 一天收尾、写汇报、结论被推翻要同步多处文档 | `notes/dayN` 的 logs/worklog/report 三层结构，笔记 v1.x 版本化 + 修正对照表 |

## 安装

项目级（推荐，随仓库走）：

```bash
mkdir -p .claude/skills
cp -r skills/* .claude/skills/
```

用户级（所有项目可用）：

```bash
mkdir -p ~/.claude/skills
cp -r skills/* ~/.claude/skills/
```

## 来源

这四个 skill 来自 2026-10-28 的 Day3 实验：手写 abs / RMSNorm 两个 npuir kernel，
12 个精度用例，24 个运行时探针，43 份日志（全部在 [`../day3/code/`](../day3/code/)）。
每条规则背后都有一次真实返工，案例编号在
[`ascend-npuir-bench-guardrails/references/day3-failure-cases.md`](ascend-npuir-bench-guardrails/references/day3-failure-cases.md)。

配套笔记：[`day3/note/day3学习指标.md`](../day3/note/day3学习指标.md)
（第 7 节列出全部产出与证据）。

## 约定

- 一次只做一件事：`guardrails` 管"别踩坑"，`benchmark` 管"怎么量"，`tilelang-npuir` 管"怎么写"。
  有冲突时以 `guardrails` 为准（它是 Day3 事后复盘的最终版）。
- 规则写"可执行的动作"，不写"要注意"。例：不写"注意尾块"，写
  `num_blocks = T.ceildiv(N, block_N)` + `tail = T.min(block_N, shape - offset)`。
- 新坑进来先加到 `references/day3-failure-cases.md` 的案例表，再把能自动检查的部分
  （grep / 断言）提升到 SKILL.md 的检查表。
