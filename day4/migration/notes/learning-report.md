# Day 5 Learning Report — 第一个自动迁移算子 LeakyReluFwdOp 验收

## 1. 一句话结论

LeakyReluFwdOp 已从 GPU TileLang 迁到 Ascend 910B3 NPUIR，**14/14 正确性用例全过、最大绝对误差 0.00e+00**，msprof 实测单 kernel 时延 **11.2–144.8 µs**（1M–16.8M 元素，fp16/bf16/fp32）；端到端（Stage 0→Stage 5）耗时 **2 h 37 min**，其中真正写 kernel 只有 24 min，其余是设计、评审与验收。依据：`logs/run.json`、`logs/benchmark.json`、`logs/leaky_relu_func_timeline.jsonl`。

## 2. 数据表（口径逐列标明）

### 2.1 正确性（pytest，`logs/correctness.log`、`logs/run.json`）

| 用例 | 形状 | dtype | max_abs_err | 结果 |
|---|---|---|---|---|
| test_leaky_relu_op[1048576-dtype0] | 1,048,576 | float32 | 0.00e+00 | pass |
| test_leaky_relu_op[1048576-dtype1] | 1,048,576 | float16 | 0.00e+00 | pass |
| test_leaky_relu_op[1048576-dtype2] | 1,048,576 | bfloat16 | 0.00e+00 | pass |
| test_leaky_relu_op[4000000-dtype3] | 4,000,000 | float16 | 0.00e+00 | pass |
| test_leaky_relu_op[4000000-dtype4] | 4,000,000 | bfloat16 | 0.00e+00 | pass |
| 其余 9 条（参数化 / inplace / dtype 拒绝 / nonfinite slope / roofline） | — | — | — | pass |

**合计 14 passed / 0 failed**，耗时 101 s。误差为 0 是因为逐元素只做比较与一次乘法，没有超越函数。

### 2.2 性能（msprof device kernel 时间，`logs/benchmark.json`）

| label | dtype | 元素数 | kernel 时延 | 实测吞吐（=2·N·bytes/latency） | 框架 Ratio |
|---|---|---:|---:|---:|---:|
| smoke-1m | float32 | 1,048,576 | 17.38 µs | 482.6 GB/s | 13.79% |
| smoke-1m | float16 | 1,048,576 | 11.42 µs | 367.3 GB/s | 10.49% |
| smoke-1m | bfloat16 | 1,048,576 | 11.18 µs | 375.2 GB/s | 10.72% |
| gan-feat | float16 | 16,777,216 | 141.46 µs | 474.4 GB/s | 13.55% |
| gan-feat | bfloat16 | 16,777,216 | 144.84 µs | 463.3 GB/s | 13.24% |
| gan-feat-deep | float16 | 8,388,608 | 71.36 µs | 470.2 GB/s | 13.43% |
| gan-feat-deep | bfloat16 | 8,388,608 | 72.08 µs | 465.5 GB/s | 13.30% |
| fc-wide | float16 | 8,388,608 | 71.10 µs | 471.9 GB/s | 13.48% |
| fc-wide | bfloat16 | 8,388,608 | 73.42 µs | 457.0 GB/s | 13.06% |

口径说明：
- 时延是 **msprof 采集的 device kernel 时间**，不是 host 侧 wall time（本项目 Day 2 已确认 host 开销 250–450 µs，会淹没这种小算子）。
- 「实测吞吐」是我按 2·N·元素字节/时延 算的，属于**有效带宽**；框架 JSON 里的 `Bandwidth(TB/s)=1.75` 在 9 条记录里是常数，看起来是 roofline 参数而非实测量，**不要当实测值引用**。
- 框架 Ratio 的分母对不上 1.75 TB/s（按 482.6 GB/s ÷ 13.79% 反推约 3.5 TB/s），口径未确认，列为 open item。

## 3. 验收判据（哪些算过）

框架给的三道门，全部机器判定，不靠 agent 自评：

| 阶段 | 判据 | 结果 | 证据 |
|---|---|---|---|
| Stage 2 | design-reviewer 给 pass（否则 design_revision） | 第 2 轮 pass | `logs/leaky_relu_func_timeline.jsonl` |
| Stage 3 | 开发者自测（精度 + 运行）通过，且 review 通过 | pass | 同上 |
| Stage 5 | integrator 跑 `tileops.reporting.cli run --op LeakyReluFwdOp --prof-mode msprof`：pytest 全过 + benchmark 产出 run.json/report | pass（1293 s） | `logs/run.json`、`logs/report.md`、`logs/integration_log.md` |

我自己另外独立复跑了一遍 `tests/ops/test_leaky_relu.py`（不经框架 CLI）：**14 passed in 101.09 s**，并 grep 确认集成目录内是 NPU 实现（`target="npuir"` ×2、`alloc_ub` ×11），GPU 源仓库 `git status --short` 为空。见 `logs/manual_acceptance_leakyrelu.log`。

## 4. 我错在哪 / 被推翻的结论

1. **「T.verf 可以直接用」** → 实测是 7 阶泰勒多项式，|x|=1 误差 4.5e-3、|x|=3 误差 39；全范围 erf 必须换 A&S 7.1.26。依据：`experiments/spike_bitnot_diag.py` 的数值对比（erf 部分离线 2M 点最大误差 5.8e-7）。
2. **「uint8 的 vnot 算错了」** → 逐字节复核后计算正确，是 `out_idx` 把输出分配成 float32；改预分配即可。同上 spike。
3. **「bool 只是写起来麻烦」** → 原生 bool 编译通过但执行挂死；必须 int8 视图 + vxor。
4. **「设计一次过」** → Stage 2 打回两条阻塞问题：factory 契约两参 vs NPU 单参；fp32 分支从 stale `x_ub` 拷出。都是我（在 prompt 里给的设计约束）没覆盖到的。
5. **进程管理** → 用前台 `timeout 7200` 限时，20:31 把 conductor 杀掉，白等约 13 min 才被 watchdog 续上；关掉 Claude 页面也会连带杀死迁移。已改成 detached 队列 + session 续跑。

## 5. 还没验的（open items）

- [ ] 框架 Ratio 的分母（反推约 3.5 TB/s）与 `Bandwidth=1.75` 字段口径未确认，本次不引用 Ratio 作为性能结论。
- [ ] `SESSION_TIMING_ANALYSIS.md` 被判 invalid：`breakdown[10].duration_s must be a number`；报告里 Session Timing 显示 N/A。属框架产物格式问题，不影响验收，但耗时拆解只能以 `.task_timeline.jsonl` 为准。
- [ ] Stage 4（性能调优）按要求跳过：当前 ~470 GB/s 是否已接近该尺寸上限，未做对照实验（Day 2 结论是 ≥256 MiB 才能稳定测 HBM，本次最大 16.8M 元素 = 67 MB 流量，偏小）。
- [ ] 4M 元素（fp16/bf16）只在正确性里覆盖，未进 benchmark 列表。
- [ ] Elu / Expm1 / Erf / BitwiseNot 尚在队列中，结果未出。
- [ ] 打包与 memory 更新：本文件写于队列运行中，收尾时需补 4 个算子的结果与队列计数。

## 6. 收尾补充（2026-10-29）

队列已于 06:07 跑完，4 个算子的结果：

| 算子 | 正确性 | benchmark | 最大绝对误差 | 墙钟 | 返工 |
|---|---|---|---:|---:|---|
| EluFwdOp | 14/14 | 9 条 / 6 shape | 0 | 1 h 49 min | Stage 1 runtime 失败 1 次 |
| Expm1FwdOp | 8/8 | 10 条 / 7 shape | 0 | 2 h 36 min | 评审打回 1 次 |
| ErfFwdOp | 8/8（共享文件 16/16） | 10 条 / 7 shape（共享文件 20 条含 Expm1） | fp32 6.74e-7 / fp16 1.22e-4 / bf16 7.63e-6 | 1 h 58 min | 一次通过 |
| BitwiseNotFwdOp | 9/9 | 8 条 / 5 shape | 0（位级精确） | 1 h 20 min | Stage 1 runtime 失败 1 次 |

5 个算子合计墙钟约 10.3 h，其中 Stage 3 真正写 kernel 仅 77 min（12.5%），设计+评审（含返工）约占 56.6%。完整结果、独立复验、性能基线与遗留问题见 `COMPLETION-REPORT.md`。

第 5 节 open items 的更新：
- [x] Elu / Expm1 / Erf / BitwiseNot 结果已出。
- [x] 已打交付包（`tilelang-ascend/day5/`），memory 待更新。
- [ ] `Ratio` / `Bandwidth` 口径仍未确认，报告中不引用为性能结论。
- [ ] Stage 4 仍未做，性能为未调优基线。
- [ ] Erf report 混入 Expm1 benchmark 记录，引用时需按 `operator` 字段过滤。
