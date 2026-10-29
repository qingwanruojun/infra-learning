# Day 5 Worklog — TileOPs GPU → Ascend 910B NPUIR 迁移（OpenCode 框架自动跑）

日期：2026-10-28 傍晚起（本日 Day 4 交付包已于 20:02 打包，本文件记录其后的迁移工作）
机器：Ascend 910B3 x1，CANN 9.0.0，tilelang 0.1.1.10+ubuntu.20.4.cann900，torch 2.7.1+cpu / torch_npu 2.7.1.post4

## 0. 本轮要做什么

用 `refs/tilelang-mlir-ascend/.opencode/agents` 的 conductor+subagent 框架，把 `refs/TileOPs` 里 5 个 GPU 算子迁到 NPUIR：

1. LeakyReluFwdOp ✅（本轮完成）
2. EluFwdOp ⏳
3. Expm1FwdOp ⏳
4. ErfFwdOp ⏳
5. BitwiseNotFwdOp ⏳

流程固定为 `Stage 0 (scaffold) → 每函数 (Stage 1 设计 → Stage 2 设计评审 → Stage 3 开发) → Stage 5 集成验收`，Stage 4（性能调优）按要求跳过。
时间线原始数据：`logs/leaky_relu_top_timeline.jsonl`、`logs/leaky_relu_func_timeline.jsonl`；队列日志：`logs/queue_ops_snapshot_2130.log`。

## 1. LeakyReluFwdOp 时间线（本地时间 UTC+8）

| 时刻 | 事件 | 耗时 |
|---|---|---:|
| 18:33:09 | conductor 启动，Stage 0 init | — |
| 18:34:02 → 18:46:59 | Stage 0 scaffolder：抽 GPU 代码、建 `examples/leaky_relu/`、manifest 探测 | 777 s |
| 18:47:16 → 19:18:21 | Stage 1 设计（第 1 次）：**机械门禁 gate1 失败**（S1-SECT-CONCL×3 + S1-REF-PATHS×8） | 1865 s |
| 19:18:24 → 19:26:10 | Stage 1 设计（第 2 次）：补齐结论段与引用路径 | 466 s |
| 19:26:10 → 19:46:46 | Stage 2 评审（第 1 次）：**design_revision**，2 个阻塞问题 | 1236 s |
| 19:46:48 → 20:08:27 | Stage 1 设计修订：修 factory 契约 + fp32 路径写回 | 1299 s |
| 20:08:27 → 20:24:43 | Stage 2 评审（第 2 次）：通过 | 976 s |
| 20:24:44 → 20:48:37 | Stage 3 开发：写 NPUIR kernel 并自测通过（中途 20:31 被外层 `timeout 7200` 杀掉，20:31:17 用 `--session` 续跑） | 1433 s |
| 20:49:10 → 21:10:43 | Stage 5 集成验收：pytest 14/14 + msprof benchmark 9/9 | 1293 s |
| 21:10:44 | `migration-phase → DONE`，队列自动接手 EluFwdOp | — |

总计约 **2 h 37 min**（18:33 → 21:10），其中设计+评审占 1 h 38 min（含一次返工和一个评审否决），开发 24 min，验收 22 min。

## 2. 错误清单（含当时怎么判、怎么修）

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| E1 | Stage 1 机械门禁 `gate1` 失败：S1-SECT-CONCL×3、S1-REF-PATHS×8 | DESIGN.md 缺 3 个结论小节、8 条引用路径不合规范 | designer 按 gate 提示补写，466 s 通过（未改设计内容） |
| E2 | Stage 2 阻塞问题 1：factory 契约写成 `kernel(threads, npt)` | NPU 侧 wrapper 是单参 `kernel(block_size)`，契约与调用方冲突 | 设计改为单参 NPU factory |
| E3 | Stage 2 阻塞问题 2：fp32 分支仍从 `x_ub` 拷出，而 `vselect` 写进了 `sx32` | 输出会等于输入（负半轴 slope 失效） | 改为 `T.vselect(cond, x_ub, sx32, x_ub)`，即 Out=A 原地写 |
| E4 | `vbrc` 标量绑定写在工厂作用域 → `input vector and output vector must have same rank` | npuir 要求标量绑定在 `@T.prim_func` 体内（`with T.Kernel` 块内） | 移入 prim_func；已写入下个算子的 prompt 事实清单 |
| E5 | 测试极值手写 `3.4028235e38` 溢出 fp32 数据 | 十进制字面量在 dtype 转换时溢出 | 统一用 `torch.finfo(torch.<dtype>).max` |
| E6 | 外层 `timeout 7200` 在 20:31:03 杀掉 conductor（EXIT=124） | 我用前台 timeout 限时，估算过短 | 写 detached watchdog（PPID=1、独立 session），`opencode run --session` 原会话续跑；后续改用 11 h 队列 |
| E7 | 关掉 Claude 页面会连带杀死迁移进程 | 进程树是 `opencode → bash → claude` | 改用 `setsid`/`nohup` 脱离，watchdog 复查 phase，未完成则续跑 |

### 迁移前离线排障（spike，见 `experiments/spike_bitnot_diag.py`）

| # | 现象 | 结论 |
|---|---|---|
| S1 | uint8 `vnot` 结果 MISMATCH | 位运算字节正确，是 `out_idx=[1]` 把输出张量分配成了 float32（dtype 不匹配）。规避：`out_idx=[]` + 调用方 `torch.empty_like`，或 uint8 视图成 int8 |
| S2 | 原生 `dtype="bool"` 编译通过但执行挂死（200 s 超时） | bool 禁用；bool 用 int8 视图 + `T.fill(ones,1)` + `T.vxor`（`vnot` 对 bool 不成立：~0=0xFF 仍为真） |
| S3 | `T.verf` 数值诡异 | 不是硬件 erf，是 7 阶泰勒 `(2/√π)(x−x³/3+x⁵/10−x⁷/42)`：\|x\|=1 误差 4.5e-3、\|x\|=3 误差 39。全范围改用 A&S 7.1.26，离线 2M 点最大绝对误差 5.8e-7 |

## 3. 结论翻转序列（写进权威文档前先看这条）

```
阶段一：以为 T.verf 可用 → 实测是 7 阶泰勒，|x|≥1 完全不可用 → 弃用，改 A&S 7.1.26
阶段二：以为 uint8 的 vnot 算错了 → spike 逐字节核对，计算正确、是输出 dtype 分配错 → 保留 vnot，改分配方式
阶段三：以为 bool 只是难写 → 实测原生 bool 直接挂死 → 禁止原生 bool，改 int8 视图 + vxor
阶段四：LeakyRelu 设计一次过（自评）→ Stage 2 评审否决 2 条（契约、写回路径）→ 修订后才过
```

## 4. 队列（21:10:48 起）

串行跑 Elu → Expm1 → Erf → BitwiseNot，每算子独立 `opencode run --agent tilelang-op-conductor --auto --command migrate-op <Op>`，单算子 `timeout 9000`，快速失败（<1200 s）重试一次，11 h 硬 deadline。日志：`/tmp/migrate_<slug>.log`，队列日志 `logs/queue_ops_snapshot_2130.log`。

> 注意：队列里 `phase_of` 用 slug=`elu` 读状态文件，但框架实际建的是 `examples/elu_fwd/`，所以队列自己的 Done 判断读不到（只影响日志汇总与快速失败重试判定，不影响是否启动下一个算子）。收尾时需按实际目录手工核对。

## 5. 22:32 队列 v2 接管（为什么动它）

- 发现 v1 单算子 `timeout 9000`（2.5 h）后**不恢复**：Elu 之外的 Expm1/Erf 很可能超过 2.5 h（LeakyRelu 实测 9449 s），一旦超时整段工作沉没，违反「不浪费机器时间」。
- 做法：`kill -STOP` v1 队列（PID 226658，冻结在 `timeout 9000 opencode` 上，正在跑的 conductor 不受影响），另起 `/tmp/queue_v2.sh`（PID 246447，独立 session）：
  - 接管 Elu：轮询顶层 `examples/elu_fwd/.stage_state.json` 的 `phase`，DONE 即放行；若 conductor 进程消失而 phase≠DONE，用日志里最后一个 `session.id` + `--session` 续跑（该机制在 LeakyRelu 上已验证）。
  - 之后串行跑 Expm1/Erf/BitwiseNot，同样「进程消失就续跑」，单次 14400 s，最多 6 次，全局 deadline 约 09:00。
  - 每个算子的 slug 用「start 之后最新出现的 `examples/*/.migration_state.json`」自动发现（不再硬编码）。
- v1 队列在 Elu DONE 后被 SIGKILL，避免它再启动一遍。
- 状态探测经验：顶层 `.stage_state.json` 在 DEV_LOOP 期间停在 `INTEGRATE`，只有 Stage 5 结束才变 `DONE`；中途进度看 `_make_<op>_kernel/.stage_state.json`。

## 6. 队列最终结果（2026-10-29 06:07 收尾）

v2 队列 21:13 接手 Elu，按 Elu → Expm1 → Erf → BitwiseNot 串行跑完，06:07 全部 DONE，耗时约 8 h 54 min。5 个算子最终状态：

| 算子 | 完成时刻（UTC+8） | 正确性 | benchmark | 墙钟 |
|---|---|---|---:|---:|
| LeakyReluFwdOp | 10-28 21:10 | 14/14 | 9 条 | 2 h 37 min |
| EluFwdOp | 10-28 23:02 | 14/14 | 9 条 | 1 h 49 min |
| Expm1FwdOp | 10-29 01:41 | 8/8 | 10 条 | 2 h 36 min |
| ErfFwdOp | 10-29 04:15 | 8/8（共享文件 16/16） | 10 条（共享文件 20 条，含 Expm1 10 条） | 1 h 58 min |
| BitwiseNotFwdOp | 10-29 06:06 | 9/9 | 8 条 | 1 h 20 min |

- 全部 `.stage_state.json` `phase=DONE`，`.migration_state.json` `integration=pass`，5 份 `run.json` `status=passed`。
- 框架仓库新增 5 条 evolution 提交；迁移代码本身仍为 untracked。
- 队列级日志（`/tmp/queue_v2.log`、`/tmp/migrate_*.log`）因 10-29 08:16 左右会话/机器重启丢失；算子级 timeline、report、RETROSPECTIVE 完整。
- 独立复验与最终报告见 `COMPLETION-REPORT.md`（本包根目录）与 `logs/independent/`。
