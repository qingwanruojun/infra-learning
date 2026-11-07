# 排队算子预备信息（2026-11-05）

来源：GPU 源仓 `/mnt/workspace/learn/refs/TileOPs/tileops/manifest/*.yaml`（迁移时由 scaffolder 复制进 harness manifest）。
用途：排期、workload 裁剪决策、ratio 分组口径。

## MaxPool1dIndicesFwdOp（pool.yaml，纯 vector，3 workloads）

| label | shape | dtype | k/s/p/d | ceil | 备注 |
|---|---|---|---|---|---|
| sincnet-speaker-local | [32, 80, 4096] | fp16 | 3/3/0/1 | false | L_out 1365 |
| textcnn-global | [64, 256, 128] | fp16 | 128/1/0/1 | false | 每行只输出 1 个值 → 归约型 |
| ecg-cnn-dilated | [16, 128, 2048] | bf16 | 5/2/2/2 | true | **bf16 必须 fp32 transit（G1）**；步长 2 + 膨胀 2 → strided 取样（G5） |

- 输出含 int64 indices（[N,C,L_out]），需 arg-reduce 语义（G3）；tie-break 取 first-valid，padding 区不参与。
- 先验（外部调优日志）：改核↔任务映射顺序 1 行 → −5.9%；双缓冲/静态展开/3 槽/删 pass 均无效。
- 预期 ratio 口径：纯 vector 组，目标 ≥60%。

## FFTC2COp（sequence_modeling.yaml，纯 vector，3 workloads）

| label | shape | dtype | 状态 |
|---|---|---|---|
| fft-4k-c64-unbatched | [4096] | complex64 | 可迁移（fp32 旋转因子 + 级联，误差 ~2.5e-5） |
| fft-4k-c64-b64 | [64, 4096] | complex64 | 可迁移 |
| fft-4k-c128-b64 | [64, 4096] | complex128 | **受 G2（无 fp64 向量路径）阻塞**，atol/rtol 1e-8 不可达 |

- 迁移策略：先做 c64 两条；c128 在报告中明确标“不支持/跳过”，必要时用 two-sum/two-product 模拟（4–6× 计算）。
- bit-reversal 重排需运行时索引 gather（G4）→ 现阶段用常量索引表按 shape 编译。
- **注意**：若 c128 计入平均，会拉低整组；报告必须分 dtype 报。

## GatedDeltaNetPrefillFwdOp（linear_attention.yaml，mix cube+vector，16 workloads）

- 任务空间 `(batch=1, seq, head, dim=128)`，chunk_size=64，dtype fp16/bf16。
- 4 个 seq（4k/32k/64k/128k）× 4 个 head（16/32/48/64），label 前缀：fallback（4k h16）+ qwen35。
- 单 workload 显存：q/k/v 各 `seq*head*128*elem`。
  - 32k×h16 ≈ 128 MiB/张量；128k×h64 ≈ **2.0 GiB/张量，q+k+v ≈ 6 GiB**，单次 bench 极慢。
- **裁剪建议**：保留 4k-h16（smoke）+ 32k-h16/h32/h64（覆盖 seq 与 head 维度）× fp16/bf16；64k/128k 标 “excluded-by-cost” 并在报告中说明。
- 缺口：G6（3D 任务空间需压 1D grid）、G7（变长/尾块）、G8/G9（2D Parallel、multitile+bm>1）。
- 预期 ratio 口径：mix 组，目标 ≥40%。

## FusedAddRMSNormFwdOp（normalization.yaml，纯 vector，7 workloads）

smoke-llama [64,4096]；llama-3.1-8b-prefill [2048,4096]；decode [1,4096]；70b [2048,8192]/[1,8192]；405b [2048,16384]/[1,16384]。fp16+bf16。
- 3 条 M=1 decode：kernel 几十 µs，host launch 250–450 µs（G12），msprof kernel 口径可用。
- roofline：flops `5*M*N`，bytes `(4*M*N + N) * elem_bytes`（读 x/residual/w，写 y/residual_out）。
- 报告需按 prefill / decode 分组。

## ratio 记录口径

- 全部来自 `reports/tileops/*/benchmark.json`（msprof kernel 时间），`/tmp/npu-work/ratio_table.py` 汇总。
- 现有向量类基线（2026-11-05 早）：LeakyRelu 12.8%、Elu 16.2%、Expm1 17.6%、Erf 28.6%、BitwiseNot 9.1% —— 与 60% 目标差距大，主因是 G13/G14（搬运不重叠 + 尾核），不是“缺算子”。
