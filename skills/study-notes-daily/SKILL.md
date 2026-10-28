---
name: study-notes-daily
description: 按日产出可复现的学习/实验交付物：notes/dayN 下的 logs + worklog + learning-report，权威笔记按 v1.x 版本化并维护"修正对照表"。当一天的工作要收尾、要写汇报、或某个结论被推翻需要同步更新多处文档时使用。
---

# 每日实验笔记与汇报

## 1. 目录结构（一天一套）

```
notes/dayN/
├── worklog.md            # 时间线 + 错误清单 + 翻转序列
├── learning-report.md    # 给人读的结论 + 依据
├── STATUS.md             # 一句话状态、当前结论、待办
├── optimization-knobs.md # 权威技术文档（按 v1.x 版本化）
├── README.md             # 怎么复现
├── logs/                 # 每个探针一份 .log；原始 CSV 放 logs/profiler_raw/ + 清单
└── experiments/          # probe_*.py，日志里写清运行命令
```

## 2. 每个数字都要能指到一份日志

- 结论句后面直接写日志名（`abba_hbm_256MiB.log`）。写不出来 ⇒ 这数还没测，别写。
- **跑废的日志不删**：改名加 `_invalid`，文件头写清为什么废（例：`ladder_bidir_v1_invalid.log` —— torch 作用在 512 MiB 全量张量上、我们复用单一二进制）。
- 每个探针脚本入库到 `experiments/`，日志顶部写运行命令，别人能一键重跑。
- 收尾时报计数（N 个探针 / N 份日志 / N 份 CSV），便于自查有没有漏。

## 3. 权威笔记版本化

- 标题带版本：`# Day 2 优化旋钮表 v1.5（…测量口径复核后）`。
- 顶部 changelog block 只写**这次翻了什么**：`> **v1.5 变更**：按尺寸重测…7% 只在 128 MiB 成立…`
- 维护 `§0.2 修正对照表`：旧结论 / 新结论 / 依据日志。**只增不删**，让读者能看到演化。
- 一次结论翻转，要同一轮改完：权威文档 → `STATUS.md` → `learning-report.md` 顶部一句话 → `worklog.md` 的 flip sequence → 记忆文件。
- 两份文档内容重叠时，**指定一份为权威**，另一份只留指针；否则必然漂移。

## 4. 汇报结构（固定四段）

1. **一句话结论**（带数字和口径）。
2. **表格**：每列标明是什么时钟 / 什么尺寸，不混。
3. **我错在哪**：本轮被推翻的旧结论 + 依据。
4. **还没验的**：明确列出 open items，不要用"基本没问题"糊过去。

## 5. 结论可信度清单（写进文档前过一遍）

- [ ] 口径（host / device / e2e）是否标明？
- [ ] 尺寸是否在有效区间（见 `ascend-npu-benchmark` 技能）？
- [ ] 统计是否 median + IQR/MAD，是否丢了预热轮？
- [ ] 是否只有一个进程/一次运行？
- [ ] 归因是否超出证据？"慢 7%" 是因为 kernel 差，还是因为测量尺寸选错？
- [ ] 有没有把"未解释的现象"包装成"已解释的规律"？

## 6. 翻转序列示例（Day 2 真实记录）

```
v1.2 慢 1.27 倍 → v1.3 打平略快 → v1.4 慢约 7%（尺寸选错）→ v1.5 ≥256 MiB 打平
```

每次翻转都保留旧版本段落和对应日志，最后在 `worklog.md` 里汇总这条序列——这是判断"这人在做实验还是在编数字"的最快证据。

## 7. 打包交接

```bash
tar -czf dayN.tar.gz --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='export_only_prof_dir' notes/dayN own_ops dayN/note
```

排除：CANN 原始导出（几十 MB，可由探针重跑）、`__pycache__`、构建产物（`build/`、`*.so`、`install.log`）。
