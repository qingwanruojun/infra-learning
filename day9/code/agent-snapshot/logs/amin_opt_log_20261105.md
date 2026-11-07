# Stage 4 调优日志 — _simple_reduce_kernel / _simple_reduce_kernel_tiled（amin / AminFwdOp）

- project: TileOPs ; operator: AminFwdOp（reduction/amin_fwd）；kernel_id: `_simple_reduce_kernel`（single-tile, N≤32512）+ `_simple_reduce_kernel_tiled`（tiled, N>32512）
- mode: full（scenario=optimize，TileOPs 集成算子）
- 工具链戳：tilelang 0.1.2（dev root build /mnt/workspace/learn/tilelang-ascend/build，2026-11-05 实查）+ CANN 9.0.0 + **Ascend910B3**（npu-smi NPU 3；20 AI Core / **40 Vector Core**，`NPUUtils.get()` 实查 20/40）+ torch 2.7.1
- 性能口径：**`msprof op` Task Duration(us)**（唯一 kernel 时延口径）；launch-count=20 / warm-up=5，取 20 次 captured launch 的中位数。kernel-name=`main`（wrapper 未声明 msprof_kernel_name，`@T.prim_func` 名解析）
- 性能目标（用户规格）：roofline_ratio；纯 vector 类**平均 ≥60%，挑战 80%**；Ratio = 实测 GM 读带宽 / 1.75 TB/s；baseline = 迁移版均值 42.73%
- benchmark 来源：`examples/TileOPs/benchmarks/ops/bench_amin_fwd.py::test_amin_bench`（`workloads_to_params("AminFwdOp", include_extra=True)` 展开 7 案例：smoke×3 + tune×4）。采集走 `perf_opt/bench_driver.py`（重建 op 层 `_prepare_input` 后的精确输入 + 直接调工厂，不接 Stage-5 wrapper benchmark；3d workload 的 permute+contiguous GM 布局与正式数据流一致）
- baseline 文件（只读）：`amin_fwd_kernel/_simple_reduce_kernel.py`、`amin_fwd_kernel/_simple_reduce_kernel_tiled.py`；wrapper `amin_fwd.py` 的 default_config 启发式实查：hidden bm=4 / 3d bm=1 / long-seq bm=1 tile_n=16384（tn_eff=16384，2 full tiles 零尾）——**wrapper 传参不可改，tuned kernel 须在其实际选择下最优或不回退**
- 知识预注入消费：CASE-reduction-amin-migration（本 op 迁移档案）、TRAP-reduce-mixed-dtype-silent-corrupt（同 dtype 纪律）、TRAP-reduce-clear-default-overwrite（T.reduce 默认 clear=True 覆写——本 kernel 每任务整体覆写 row_min/acc，不依赖累积语义 ✓）、VP-2026-0188（pending，仅方向参考不引用为结论）。stale 清单 161 项（reduction/vector 子集 57 项）——本轮引用的 pattern-library 条目（CONST-reduce-dims0-skinny / TRAP-parallel2d-bm-ge2-scalarize / PL-1.19 / CONST-mte2-degradation）均在 stale 清单内，按 E-5 以「待重验」方式引用，其中被用作决策依据的条目以本轮 profile 直接证据为准，不直接沿用旧数字。

## Workload Inventory（benchmark 展开，详见 workload_inventory.json）

| workload_id | kernel (M,N) dtype | wrapper cfg | 路径 | kind |
|---|---|---|---|---|
| smoke-amin-float16 / bfloat16 / float32 | (32,256) ×3 | bm=8 tn=0 | single-tile | smoke（仅合并/最终精度回归） |
| hidden-state-reduce-float16 | (2048,4096) fp16 | bm=4 tn=0 | single-tile，512 逻辑块 → 40 核 × 13 任务 | tune |
| hidden-state-reduce-bfloat16 | (2048,4096) bf16 | bm=4 tn=0 | single-tile，同上 | tune |
| long-seq-reduce-bfloat16 | (64,32768) bf16 | bm=1 tn=16384 | tiled（tn_eff=16384，2 full+0 tail），64 逻辑块 → 40 核 × 2 任务 | tune |
| 3d-multidim-reduce-float16 | (4,128,4096) fp16 dim=[0,2] | bm=1 tn=0 | single-tile (128,16384)，128 逻辑块 → 40 核 × 4 任务（8 核 4 任务/32 核 3 任务） | tune |

3d 数据布局核实：op 层 `flatten_for_multidim` 做 `permute([1,0,2]).contiguous()` 后 kernel 收到 **(128,16384) 行连续 32KB/行** 的 GM 张量（conductor 头寸线索中「4×8KB 跨 1MB 跳读」描述的是 flatten 前的逻辑视图；实测 profile 读侧 128KB/核精确无膨胀，见下表——读侧不是瓶颈）。

## Performance Test Data — Baseline（round 0, phase=baseline）

baseline 文件 = Stage 3 验收版（L0/L1 已过），baseline 采集跳过重复 L0（perf_records 中 `l0_pass=false` 表示本次 run 未重跑 L0，非精度失败）。

| workload_id | candidate | Task Duration(us) median [min,max] | aiv_time | vec_ratio | mte2_ratio/bw | mte3_time/instr | scalar_ratio | GM 流量（per-core 对账） | profile |
|---|---|---|---:|---|---|---|---|---|---|
| hidden-state-reduce-float16 | baseline | **15.09** [14.28,15.86] | 10.72 | 0.301 | 0.617 / 60.0GB/s | 4.72µs / 14 | 0.295 | 416KB/核 ×40 = 16.65MB ✓ | profiles/baseline/round0/baseline_*/hidden-state-reduce-float16_* |
| hidden-state-reduce-bfloat16 | baseline | **19.33** [19.04,19.84] | 17.26 | 0.492（全 fp32） | 0.337 / 68.2GB/s | 4.87µs / 14 | 0.247 | 416KB/核 ✓ | 同上 bfloat16 |
| long-seq-reduce-bfloat16 | baseline_tiled | **7.48** [7.24,7.72] | 5.35 | 0.504 | 0.340 / 67.2GB/s | 0.42µs / 3 | 0.086 | 128KB/核 ×40 = 5.0MB（64 行 × 64KB，24 核 2 行/16 核 1 行）✓ | profiles/baseline/round0/baseline_tiled_*/long-seq-* |
| 3d-multidim-reduce-float16 | baseline | **8.52** [8.30,9.00] | 5.82 | 0.172 | 0.357 / 58.8GB/s（2.08µs） | **4.93µs / 5 条**（写 0.5KB） | 0.050（wait 3.14µs） | 128KB/核 ✓ 无读膨胀 | profiles/baseline/round0/baseline_*/3d-* |

### Roofline Ratio 换算（口径：min 分支，GM 读 / 1.75 TB/s）

| workload_id | baseline µs | GM 读 MB | 实测 TB/s | Ratio | 60% 需 | 80% 需 |
|---|---:|---:|---:|---:|---:|---:|
| hidden-state fp16 | 15.09 | 16.78 | 1.112 | **63.5%** | ≤15.98 | ≤11.98 |
| hidden-state bf16 | 19.33 | 16.78 | 0.868 | **49.6%** | ≤15.98 | ≤11.98 |
| long-seq bf16 | 7.48 | 4.19 | 0.561 | **32.1%** | ≤3.99 | ≤2.99 |
| 3d fp16 | 8.52 | 4.19 | 0.492 | **28.1%** | ≤3.99 | ≤2.99 |
| **均值** | | | | **43.3%**（conductor 基线 42.73% ✓ 口径一致） | **≥60%** | 挑战 80% |

### 设计估算 vs 实测偏差行（D-2 回填，对照 DESIGN.md §1.6.0）

| workload | 设计估算下界 | 实测 baseline | 偏差 | 失准项 |
|---|---:|---:|---:|---|
| hidden-state fp16 | 13.9µs（流量项） | 15.09µs | 1.09× | 无（估算准确；余量在 mte2/vec 未重叠） |
| 3d fp16 | 8µs（发射项主导） | 8.52µs | 1.07× | 无（发射项模型准确——但发射项的构成是 MTE3 tiny-store 4.9µs + VEC 1.0µs） |
| smoke | 2.5µs | —（smoke 不采性能） | — | — |

### Baseline 现象结论（Phase 1 排序依据）

1. **3d-multidim（ratio 28.1%，头寸最大）**：发射项主导实锤——单核 aiv_time 5.82µs 中 **MTE3 管道 4.93µs / 仅 5 条指令（写 0.5KB，0.097 GB/s）**；每任务 1 条 (1,) 2B tiny store，4 任务/核 → 4-5 条。MTE2 读 128KB 仅 2.08µs。VEC 1.0µs。**store 发射粒度（2B/条 × 每核 5 条 ≈ 1µs/条固定成本）是首要敌人**，其次是任务数（4/核，copy 32KB→frag 32KB→reduce→store 全串行）。L2 read hit 0%（randn 首读，正常）。
2. **long-seq bf16（ratio 32.1%）**：每核 2 任务 × 2 tiles 串行；vec 2.70µs（vcast fp32 域 + reduce）与 mte2 1.82µs 未重叠；aiv 5.35µs + Task-aiv gap 2.1µs。单套 UB（x_ub 32KB + x_work 64KB + tail）双缓冲会到 192KB 超 184KB 预算——流水未铺满与 R-D5 预算公式相关。
3. **hidden bf16（ratio 49.6%）**：vec 8.49µs 全 fp32 域（vcast 416KB→832KB + fp32 reduce 832KB 读 + cast back）+ mte2 5.81µs + scalar 4.26µs 串行链；vec_wait 0.36 + scalar_wait 13.8µs。单套 UB（x_ub 32KB bf16 + x_work 64KB fp32 = 96KB）双缓冲 192KB 超预算——auto-multi-buffer 无法铺开。
4. **hidden fp16（ratio 63.5%，控制组不得回退）**：mte2 6.61µs 与 vec 3.23µs 未重叠（vec_wait 0.678——VEC 等 copy→frag copy 串行链）；x_ub+x_frag 64KB 单套，理论可双缓冲（128KB）但 PlanMemory 未展开。
5. 调优顺序：**3d fp16 → long-seq bf16 → hidden bf16 → hidden fp16（控制组，最后验证非回退）**。理由：3d 头寸最大且机制清晰（tiny-store/任务数）；long-seq 结构独立（tiled）；hidden 双 dtype 共享 single-tile 结构，先 bf16（差距大）后 fp16（只验证非回退，避免控制组在迭代中先被改）。

### 关键假设与约束（本轮调优边界）

- 工厂签名不变：`_simple_reduce_kernel(M,N,op_kind,dtype) -> (block_m)` / `_simple_reduce_kernel_tiled(M,N,op_kind,dtype,tile_n) -> (block_m)`；`@T.prim_func` 名保持 `main`。
- wrapper default_config（bm=4/1、tn=16384）不可改；tuned kernel 内部的 effective block_m/双缓冲/sub-tile 结构以模块级常量暴露，且必须验证 wrapper 实际传参下最优或不回退。
- 已知工具链约束（baseline 文件头 + DESIGN §9.2，版本戳 2026-11-05 dev root）：reduce src/dst 同 dtype；bf16 不支持 reduce（vcast(rint)→fp32）；T.vcast f32→f32 非恒等；T.copy 配对显式 slice；T.Kernel 1-D；tiled block_m 钳 1（CG-2026-0014）。
- reduce-on-shared（省 frag copy/UB 减半）是 DESIGN §1.6.3 明示的 Stage 4 lever（T.reduce.md §2.4 文档先例；VP-2026-0121 竞态前提是多消费者，本 kernel shared 为单消费者）——按 lever 协议探针验证后启用。
- TRAP-parallel2d-bm-ge2-scalarize（stale 待重验）：本 kernel 无 2D T.Parallel 融合循环（epilogue 为 1D (bm,) 收缩），eff_bm 上调预期不触发；若 scalar_ratio 异常升高即回滚验证。

## Iteration Log

### Iteration 1（round 1）：3d 发射项——reduce-on-shared（r1a）与 eff_bm 上调（r1b）

- base：baseline（single-tile）/ baseline_tiled（tiled）
- 现象（base profile）：3d 的 MTE3 管道 4.93µs/5 条（写 0.5KB，每任务 1 条 (1,) 2B tiny store ≈1µs 固定成本）+ 任务数 4/核；hidden bf16 的 scalar 4.26µs；hidden fp16/bf16 的 mte2 与 vec 全串行。
- 候选优化点：

| opt_id | 目标现象 | 优化点 | 判断依据 | 具体改法 | 验证指标 | 状态 |
|---|---|---|---|---|---|---|
| OP1 | 3d vec 链多 1 pass + UB 加倍 | reduce-on-shared（省 frag copy） | T.reduce.md §2.4 文档先例；VP-2026-0121 竞态前提（多消费者）不存在 | fp16/fp32 分支 T.copy(x_ub,x_frag)→T.copy 删除，reduce_min(x_ub,...) | Task Duration / L0 | candidate |
| OP2 | 3d 任务 4/核、store 2B/条 | eff_bm 上调（工厂内放大 wrapper block_m） | BP_per_block_fixed_overhead + MTE3 tiny-store 固定成本 | _uplift_block_m 公式（模块常量 _EFF_BM_*） | Task Duration / L0 | candidate |

- 实验分支与结果（**本 op 存在 ±12% run-state 双态（3d 同 kernel 8.52 vs 7.59），单 run 不可靠——本 op 所有 <5% 裁决及全部 winner 均以 ab_test 交错 3-pair 协议为准**）：

| branch | workload | baseline/当前 best (us) | branch (us) | L0 | 结果 |
|---|---|---:|---:|---|---|
| r1a_reshared | 3d fp16 | 8.52（首测慢态） | 7.90 | PASS | 首测 -7.3% → ab_test **tie**（b_rel +1.3%，p=0.25） |
| r1a_reshared | hidden fp16 | 15.09 | 14.52 | PASS | 首测 -3.8% → ab_test **tie**（-0.6%） |
| r1a_reshared | hidden bf16 | 19.33 | 20.05 | PASS | 首测 +3.7% → ab_test **tie**（+0.26%，p=1.0） |
| r1b_bmup | L0-5 (2048,4096) bf16 eff=6 | — | — | **FAIL** | **blocked**：BiShengIR ub overflow，需要 1573120 bits / 可用 1572864 bits（**仅差 32B**）——auto-multi-buffer 对 x_ub 双缓冲未计入预算公式 |

- 结论：r1a 全 tie（无回退、UB 减半，保留为后续组合前提但不作为 winner 合并）；r1b blocked（**教训：eff_bm 预算必须按双缓冲口径**）。本轮未产生 winner，current best 不变。

### Iteration 2（round 2）：eff_bm 修正预算（r2a）与叠加（r2b）

- 现象：r1b 的溢出根因 = bf16 resident 按 6B/elem（单套）计，实际 x_ub 被双缓冲。
- 优化点：r2a = eff_bm uplift + 双缓冲口径 resident 常量（fp16 4B、bf16 8B、fp32 8B）；r2b = r2a + r1a（叠加，消融 r1a 贡献）。
- 实验分支与结果：

| branch | workload | ab 对比（A=baseline, B=branch） | L0 | 结果 |
|---|---|---|---|---|
| r2a_bmup | 3d fp16 | 7.44 → **5.69（-23.5%）**，3 pairs 方向一致 | PASS | **improved（B_better）** |
| r2a_bmup | hidden fp16 | 14.57 → 14.59（+0.14%） | PASS | tie |
| r2a_bmup | hidden bf16 | —（eff 公式钳回 4，代码不变） | PASS | tie（对照 ✓） |
| r2a_bmup | 3d profile | mte3 4.93µs/5 条 → **1.11µs/3 条**；任务 4→2/核 | — | 机制证实（store 合并 + 任务减半） |
| r2b_bmreshare | hidden fp16 | 14.26 → **15.50（+8.7%）** | PASS | **A_better（回退）**——组合（eff=8 + 无 frag）劣化 |
| r2b_bmreshare | 3d fp16 | 首测 6.01（r2a 5.69） | PASS | 无增益 |

- 结论：**r2a 成为 3d workload winner（-23.5%），fp16/bf16 非回退**；r2b 弃用（reshared 在高 UB 占用下负收益；r1a 单点 tie + r2b 回退 → reduce-on-shared 全线弃用）。fp16 3d ratio 28.1%→42.1%。
- 合并检查：3d 未收束（42.1% < 60%），r2a 作为 current best 候选继续迭代（合并复测延后至 final 全量）。

### Iteration 3（round 3）：tiled tile 级流水（long-seq，ratio 32.1%）

- 现象（long-seq baseline）：每核 2 任务 × 2 tiles 串行；vec 2.70µs（vcast fp32 域 + reduce）与 mte2 1.82µs 未重叠；Task-aiv gap 2.13µs；单套 UB（x_ub 32KB + x_work 64KB）双缓冲 192KB 超预算。
- 优化点：r3a = _fit_tile_n 预算谓词 ×2（_UB_PIPELINE_FACTOR=2），迫使 tn_eff 下调（16384→8192，2→4 tiles）让双缓冲集合放得下。
- 结果：

| branch | workload | baseline (us) | branch (us) | L0 | 结果 |
|---|---|---:|---:|---|---|
| r3a_pipe | long-seq bf16 | 7.48 | **8.84（+18%）** | PASS | **config_no_gain**——profile：mte2 11→19 条（burst 减半拆条）、mte2_active_bw 67→42 GB/s（小 burst 退化）、aiv 5.35→6.58µs。发射开销与小 burst 带宽退化 > 重叠收益 |

- 结论：tn=8192 形态否定；不否定 tile 流水整类（继续 r3b 直线形态）。

### Iteration 4（round 3 续）：直线 2-tile 交错（r3b）——**blocked + 新编译器陷阱实证**

- 优化点：num_full==2 且 tail==0 时工厂期生成直线代码（x_ub_a/x_ub_b 双 staging + 共用 x_work），copy(t1) 与 cast/reduce(t0) 跨管道无依赖可重叠；直线代码无循环 → 规避 auto-multi-buffer 膨胀；保持 tn=16384 大 burst。
- L0 结果：**FAIL（(64,32768) bf16 32/64 mismatch）**。
- 最小 probe 定位（`logs/probe_r3b_variants.py`，3 变体对照）：

| probe 变体 | mismatch | 结论 |
|---|---|---|
| v0 both(a,b)（先 a 后 b） | 33/64 | 输出 = min(t0)——**只有第一次 vcast+reduce 生效** |
| v1 tile0 only | 33/64 | 输出 = min(t0)（该变体的预期行为，证明 copy(a)+cast+reduce(a) 链本身正确） |
| v2 tile1 only | 29/64 | 输出 = min(t1)（同上，b 链正确） |
| v3 both(b,a)（次序交换） | 29/64 | 输出 = min(t1)——**先算者赢** |

- 机制：直线代码中**同 dst 的两次 T.vcast 被编译器折叠/CSE 成一次**（两次 reduce 均读同一次 cast 结果；先出现者的数据胜出）。T.serial 循环版同样语句序列正确——直线展开语义不保真。
- UB 复核：双独立 x_work（192KB）超 UB，绕法不可行；fp16 分支同构风险。
- 结论：**r3b blocked（正确性地雷 + UB 无绕法）**；该发现回写 pattern-library（traps-compiler.md）与 capability-gaps.md。

### Iteration 5（round 4）：bf16 epilogue 向量化下转（r4a）⭐

- 现象：hidden bf16 的 scalar 4.26µs（single 0.84 + dual 1.41）——epilogue `T.Parallel(block_m): out_ub[i] = T.cast(row_min[i,0], dtype)` 每任务 4 次标量 fp32→bf16 转换 × 13 任务。
- 优化点：`T.vcast(row_min, row_min_bf, round_mode="rint")`（(4,1) 整 buffer 单发向量 op）+ 纯移动 epilogue。rint 语义与 T.cast 一致（T.vcast.md f32→bf16=rint）。
- 结果：

| branch | workload | ab 对比 | L0 | 结果 |
|---|---|---|---|---|
| r4a_bfcast | hidden bf16 | 19.58 → **14.99（-23.4%）**，3 pairs 方向一致 | PASS | **improved（B_better）** |
| r4a_bfcast | hidden fp16 | 代码未涉及（fp16 分支 epilogue 本就无 cast） | PASS | 对照不适用 |

- 结论：r4a 成为 hidden bf16 winner。bf16 ratio 49.6% → **64.1%（过 60% 线）**。

### Iteration 6（round 5）：合并 m1 = r2a + r4a（single-tile current best 合并）

- 两改动域不相交（r2a=eff_bm 机制；r4a=bf16 epilogue），且各自单点 ab 验证有效——合并为 m1 并全 workload ab 复核：

| workload | A=baseline | B=m1 | b_rel | verdict |
|---|---:|---:|---:|---|
| 3d fp16 | 7.96 | **5.84** | -26.6% | B_better |
| hidden bf16 | 19.55 | **14.97** | -23.4% | B_better |
| hidden fp16 | 17.33 | **14.77** | -14.8% | B_better（vs r2a 单测 tie 水平 14.26-14.6，同代码；A 慢态膨胀） |

- smoke 精度：m1 `--level L0` PASS（12 cases 含 (32,256)×3 dtype smoke，max_diff=0）。
- **m1 成为 single-tile 已合并 current best**。

### Iteration 7（round 6）：均衡分核（r6a/r6b）——tie

- 现象：64 逻辑块（long-seq；3d eff=2 后）在 40 核下两波不均衡（40+24）——BP_task_concurrency_sweet_spot/HBM 波形分析预期波 2 满载可省 mte2 段 ~8-14%。
- 优化点：`_balanced_core_count`（num_kernels = ceil(L / ceil(L/cores))：64→32 核 × 2 满波；hidden 512 块公式不变 40 核）。
- 结果：r6a（tiled long-seq）7.34 → 7.27（-0.95%）**tie**；r6b（m1 3d）5.81 → 5.74（-1.2%）**tie**；hidden 双 dtype tie（公式不变对照）。**不采纳**（current best 保持 m1/baseline_tiled）——HBM 波形模型高估（mte2 读非纯 HBM 串行波）。

### Iteration 8（round 7）：连续任务映射 + 每核合并 store（r7a/r7b）——**A_better 回退 ⭐ 负结果**

- 现象：m1 的 MTE3 tiny store 依旧（bf16 hidden 4.57µs/12 条、fp16 2.21µs/8 条）——stride 映射（s*40+cid）下每核输出行分散无法合并。
- 优化点：block_id = cid*num_tasks+s（每核连续行块）+ out_ub 扩为 (num_tasks*bm,) + 循环后单条 MTE3 store。预期 mte3 4.57→0.4µs（bf16 -28%）。
- 结果（ab 裁决）：

| branch | workload | A=m1 | B=r7 | verdict |
|---|---|---:|---:|---|
| r7a_rowblock | hidden bf16 | 15.23 | 18.05（**+18.5%**） | **A_better** |
| r7a_rowblock | hidden fp16 | 15.49 | 16.35（+5.5%） | **A_better** |
| r7b_rowblock | long-seq | 首测 7.62 vs 7.48 | — | 弃用（同机制） |

- **机制（负向 P 类价值点）**：stride 映射下 40 核第 s 波读相邻行块（波内 320KB 地址聚簇 → L2/DDR 局部性好）；连续映射让 40 核同时读 40 个分散大区间——**读局部性损失 > store 合并收益**（bf16 store 收益最大 4.5µs，读损失 >6µs）。

### Iteration 9（round 8）：bf16 任务循环 T.Pipelined（r8a）——tie

- 现象：bf16 hidden 的 mte2 5.81µs 与 vec 8.49µs 串行；UB 余量（x_ub 双缓冲 2×32KB + x_work 64KB = 128KB < 184KB）未被 auto-multi-buffer 利用。
- 优化点：bf16 分支任务循环 `T.Pipelined(num_tasks, num_stages=2)`（文档合法形态：T.Kernel.md L72 + fused_add_rms_norm 先例）。
- 结果：L0 PASS；ab 15.27 → 15.44（+1.1%）**tie**。auto-multi-buffer 已覆盖该重叠机会或 stage 化无净收益。**不采纳**。

### Plateau 判定（T-1）

- R6（tie）/ R7（回退）/ R8（tie）——**连续 3 轮无 valid 分支 >3% 提升**；主要结构候选已充分验证：eff_bm（improved）、reduce-on-shared（tie→组合回退）、均衡分核（tie）、连续映射+合并 store（回退）、T.Pipelined（tie）、tiled tn 下调（config_no_gain）、直线 2-tile 交错（blocked：编译器折叠 + UB）、N-split（数学否决：128 单元/40 核关键路径 4 units 与现状等长 + merge kernel 开销）、单 tile tn=32768（UB 否决：bf16 vcast fp32 frag 4B/elem 使 192KB 溢出）。
- **参照结构 diff 检查（morph ladder 轻量版）**：同族已知最优 = argmax `perf_opt/_argreduce_kernel.py` r4b 形态（opt_log 记录 hidden-state fp16 38.4µs）——本算子 m1 同 shape 14.77µs，**快 2.6×**（argmax 的链式/deinterleave 形态为索引归约与窄行 N=4 特化，不适用于值归约宽行）；pattern-library cases.md 参考实现集无更接近的值归约参照。**结论：m1 已是同族参照可比范围内最快形态，未发现可移植结构差异** → plateau 成立，进入收束。
- [DESIGN_LIMIT] 判定：**不触发**——①剩余差距（long-seq 33%、3d 41%）归因于工具链能力（bf16 reduce API 缺失 [T.reduce.md §2.2.1 文档条款]、直线同 dst 双 vcast 折叠 [r3b probe 实证]、auto-multi-buffer 在本域不展开），非 DESIGN.md 假设错误（设计估算 vs 实测偏差仅 1.07-1.09×，见 Baseline 段 D-2 表）；②结构性加速估计 long-seq 1.7×（bf16 原域 reduce 假设下）/3d 1.3×（gap 消除假设下）均 <2x。**两门槛不同时满足 → 按标准禁止产出 perf_feedback.md**；工具链能力发现走 capability-gaps.md 记录路径。
## Final Performance Test Data

同一最终候选 `final_merged`（merged m1 single-tile + baseline-form tiled），最终文件 `perf_opt/_simple_reduce_kernel.py`（sha256 15c3e7e962753eb2…，phase=final 记录的 artifact_sha256 与该文件一致）；tiled 路径的 final 记录亦在该合并模块上实测（gate 4 单工件约束）。时延口径：msprof op Task Duration，launch-count=20/warm-up=5，取 20 次 captured launch 中位数；baseline 与 final 同口径（round 0 baseline 慢态首测与本表 final 的比值偏保守——各 winner 的 ab_test 交错中位对比见 Iteration Log）。

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| _simple_reduce_kernel | hidden-state-reduce-float16 | 15.09 | 14.37 | final_merged |
| _simple_reduce_kernel | hidden-state-reduce-bfloat16 | 19.33 | 14.93 | final_merged |
| _simple_reduce_kernel | 3d-multidim-reduce-float16 | 8.52 | 5.84 | final_merged |
| _simple_reduce_kernel_tiled | long-seq-reduce-bfloat16 | 7.48 | 7.27 | final_merged |

逐 workload 补充（不进上表的口径信息）：hidden fp16 提升 -4.8% / final Ratio 66.7%；hidden bf16 -22.8% / 64.2%；3d fp16 -31.5% / 41.0%；long-seq bf16 -2.8%（噪声内）/ 33.0%。raw profile：`profiles/final/round9/final_merged_*/{workload}_*/`。

- **Ratio 均值：43.3% → 51.2%**（口径：GM 读字节 / 1.75 TB/s；hidden 双 workload 16.78MB 存在 L2 驻留效应——40 核聚合 mte2 活跃带宽 ~2.4TB/s 超 HBM，声明：bench 20 次同输入 launch 下部分读命中 L2，冷读口径 ratio 略低）。
- smoke/skip 排除：smoke-amin ×3（(32,256) fp16/bf16/fp32）不采性能；最终精度 = 合并模块 `--level all`：single-tile L0 12 + L1 12 + L2 5（record-only）+ Boundary 全 PASS，tiled L0 10 + L1 10 + L2 5 + Boundary 全 PASS，`max_diff=0.00e+00`（`logs/final_alllevels.log`，rc=0）；shim `--level all`（tiled 套件转发）rc=0（`logs/shim_alllevels.log`）。无 skipped 案例。
- 关键有效优化点（按收益排序）：
  1. **r4a bf16 向量化 epilogue 下转**（`T.vcast` 替代逐行标量 `T.cast`）：hidden bf16 -23.4%（ab B_better）——标量管道 4.26µs 归零；
  2. **r2a eff_bm 上调**（wrapper block_m 为下限、UB 双缓冲口径预算内工厂放大）：3d -26.6%（ab B_better）——MTE3 tiny-store 4.93µs/5 条 → 1.11µs/3 条 + 任务 4→2/核；
  3. hidden fp16 -4.8%（eff_bm=8 任务 13→7 的收益被读侧局部性抵消，净小幅正）。
- 未达 60% 均值目标的 workload 与证据链：
  - **long-seq bf16（33.0%）**：VEC 流量地板 2.7µs（bf16 reduce 不支持 → vcast fp32 中转，T.reduce.md §2.2.1）+ Task-aiv gap 2.1µs；重叠手段全部受挫（r3a 小 burst 退化 +18%、r3b 直线折叠 bug + UB、r6a 均衡 tie、r7b 映射回退）；N-split 数学否决（关键路径等长 + merge kernel 开销）；单 tile UB 否决。
  - **3d fp16（41.0%）**：eff=4 单波被 UB 卡死（fp16 frag 2B + staging 2B × (4,16384) 双缓冲 256KB）；gap 2.0µs（copy-only probe 形态死锁无法进一步分解——`logs/probe_gap.py` 记录）；r2b/r6/r7 均无增益。

## Final Summary

- 最终候选：`perf_opt/_simple_reduce_kernel.py`（合并模块：tuned single-tile 工厂 `_simple_reduce_kernel` = m1（r2a eff_bm uplift + r4a bf16 vcast epilogue，模块常量 `_EFF_BM_*` 暴露 tuned 默认）+ tiled 工厂 `_simple_reduce_kernel_tiled`（baseline 形态原样）；`perf_opt/_simple_reduce_kernel_tiled.py` 为薄 shim（相对导入 + 直跑 fallback 双路径，`--level all` 转发 tiled 套件）。工厂签名与 `@T.prim_func` 名 `main` 均与 baseline 一致——wrapper 翻转注释即可采纳，`default_config` 传参（hidden bm=4 / 3d bm=1 / long-seq bm=1 tn=16384）下全部非回退（ab 验证）。
- wrapper 采纳影响说明：3d workload 的 wrapper 传 bm=1，tuned 工厂内部放大到 eff=2（UB 预算上限）——已实测为最优形态；若未来 wrapper 调大 block_m，`_uplift_block_m` 以传入值为下限仍取 UB 允许最大值，行为封闭。
- 中止原因：**plateau**（R6/R7/R8 连续 3 轮无 >3% valid 提升；主要结构候选充分验证；参照 diff 检查确认无未试结构差异；预算实际使用 8 轮 / ~22 个实验分支 < 10 轮 / 30 上限，非预算耗尽）。**性能目标部分达成**：2/4 workload 过 60% 线（fp16 66.7%、bf16 64.2%），均值 51.2% < 60%（long-seq/3d 的工具链级瓶颈证据链完整，见上）。
- 遗留问题：(1) long-seq 的 bf16 原域归约缺失（API 级）与直线双 vcast 折叠（编译器级）——已记 capability-gaps.md 与 pattern-library，待工具链演进后重验；(2) 3d/long-seq 的 Task-aiv gap ~2µs 未归因到底（copy-only probe 形态死锁）；(3) run-state 双态 ±12%（3d）在本 op 显著——所有 <5% 裁决已走 ab_test 协议，但单 run 数据（含本表 baseline 列）仍可能偏慢态。

## Skill Retrospective

- 流程问题：无阻塞级。run_experiments.py 的 T-4 批处理 runner 本任务早期已建（bench_driver + --exp 语法），双 kernel 的 kernel_id 按 workload 解析顺畅；ab_test.py 的交错协议在本 op 的 ±12% 双态环境下成为**必需品**而非可选项（3 次首测误导被协议拦截：r1a 3d "-7.3%"、r2b fp16 "-8.7%" 首测提升实为回退/噪声、r7a fp16 首测 -6% 实为回退）。
- 证伪/更正记录：
  1. r1b → r2a：eff_bm 预算从单套口径改为双缓冲口径（+32B ub overflow 实证）；
  2. r3b：直线代码同 dst 双 vcast 折叠（T.serial 循环版正确）——"直线展开等价于循环展开"假设被证伪（probe 4 变体）；
  3. r7：**"连续任务映射对读侧更友好"的设计直觉被证伪**——stride-40 波次读的地址聚簇（L2/DDR 局部性）价值大于合并 store；
  4. r6：HBM 波形均衡模型高估收益（mte2 读非纯 HBM 串行波，tie）。
- value point proposals（vp_type + 证据三件套；回写位置见括号）：
  - **D：MTE3 tiny-store 固定成本 ≈0.4-1.0µs/条**（2B store 1.0µs/条 @ 3d baseline 4.93µs/5 条；8B store 0.38µs/条 @ bf16 hidden 4.57µs/12 条；store 合并/任务减半直接线性缩减）→ pattern-library/constants.md（CONST-mte3-tiny-store-cost；repro：repro/repro_mte3_tiny_store.py 骨架）。
  - **D：3d/hidden bf16 run-state 双态 ±12%（同 kernel 同 config）**→ BP_run_state_bimodality 第三证（mish/lerp 后 reduction 族首证；单 run 中位数跨 run 8.52 vs 7.59µs）→ bottleneck-patterns.md 该条目追加 origin 实例（回写于 opt_log 本节，pattern-library 不重复记 BP——BP 目录属 skill 文档，仅 proposal）。
  - **P：bf16/fp16 收缩 epilogue 的标量 cast 消除模式**（(bm,1) fragment 全宽 vcast + 纯移动收缩，-23.4%）→ pattern-library/elementwise.md（PL-1.23-shrink-epilogue-vcast；repro：repro/repro_shrink_epilogue_vcast.py delta 骨架）。
  - **P（负向）：多核 stride 波次读聚簇 > 连续块读 + 合并 store**（连续映射 +18.5% 回退；适用判定：40 核级 persistent 行归约的输出/输入映射不宜为 store 合并改连续——读局部性优先）→ pattern-library/elementwise.md（PL-1.24-strided-wave-read-cluster；repro：repro/repro_strided_wave_cluster.py 骨架）。
  - **D（陷阱实证）：直线代码同 dst 双 vcast 折叠（先算者赢）**——T.serial 循环版同语句序列正确 → pattern-library/traps-compiler.md（TRAP-straightline-same-dst-vcast-fold；probe：perf_opt/logs/probe_r3b_variants.py，4 变体对照）。
  - **D：eff_bm/双缓冲预算口径**——任何工厂期 UB 预算谓词必须按 auto-multi-buffer 双缓冲计入 staging（r1b 差 32B 溢出实证）→ pattern-library/constants.md（CONST-capacity-910B2C update 注记）。
  - **C：CASE-reduction-amin-stage4**（本档案：双 kernel 单工件 gate 布局 + 双态环境下的 ab_test 协议纪律 + 三类负结果证据链）→ pattern-library/cases.md。
  - **R（proposal，不改流程文档）**：建议 tilelang-op-optimize skill 的 profile-collection.md 增加「双 kernel 单 final 工件」布局模板（本任务的合并模块 + shim 形态可复用）；run_experiments_template.py 的 EXPERIMENTS 注册表建议支持 `FILE:OPT:PARENT:KERNELS` 四元组（双 kernel op 直接可用）。
- capability-gaps.md 记录（AGENTS.md 强制）：① bf16 reduce 原域不支持（T.reduce.md §2.2.1 文档条款已确认，本任务实测其代价 = bf16 workload VEC 流量 ×1.5 + UB 预算 ×2，long-seq 1.7x 结构性收益被锁）；② 直线代码同 dst 双 vcast 折叠（编译器 CSE/DCE 层，r3b probe 四变体实证）——两条均首次出现 1 次（non-recurring），按 schema 登记 occurrences=1。
