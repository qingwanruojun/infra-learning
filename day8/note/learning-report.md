# Day 8 学习报告：让 Agent 在 Ascend 910B3 上生成一个能跑的 RMSNorm

> 日期：2026-11-04（UTC+8）
> 环境：Ascend 910B3 / CANN 9.0.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 /
> TileLang 0.1.2+bd7c60c（`/mnt/workspace/learn/tilelang-ascend`，NPUIR 分支）
> 目标算子：`Y = X * rsqrt(mean(X², dim=-1, keepdim=True) + 1e-5) * W`
> L0：X `[1024,1024]` fp32，W `[1024]` fp32，Y `[1024,1024]` fp32；另测
> `[1023,1024]` 尾块与 fp16 全链路。

## 0. 今天实际完成了什么

1. **Agent 生成的 RMSNorm 真的在 910B3 上编译并跑通了**。三种验收输入
   （fp32 1024×1024、fp32 1023×1024 尾块、fp16 1024×1024）都产出了
   NPU 结果，程序退出码 0；独立 float64 参考逐元素检查全部通过
   （max_abs 分别为 1.353e-06、1.149e-06、3.901e-03，0 个超差元素），
   并用负向对照证明检查器能检出错误。这个 0 由门禁判定：三组数值全 PASS
   且三组负向对照全被检出才返回 0，失败路径另有自测
   （第 4 节、`code/logs/07-*.log`、`08-*.log`）。
2. **定位并修掉了两类 UB 预算错误，并把验证口径固定下来**。设计伪代码把
   fp32 输入/权重副本无条件分配，静态占用 204,928 B > 192 KiB，编译能过
   但运行时 MTE 越界；fp16 又被编译器插入的 cast 临时缓冲撑到 264,192 B，
   编译期报 `ub overflow`。Agent 用 inline cast 修掉第一类，又用编译探针
   实测出 fp16 的 16 B/元素/行系数、把 `block_m` 从 16 降到 8，修掉第二类。
   两次故障的原始输出已从子代理 transcript 提取到
   `code/process/probes/01-*.log`、`02-*.log`。
   同时用 host / e2e / device 三时钟确认：host 230–360 µs、device 仅
   12–17 µs；Agent 版 host 比手写版高约 90–120 µs。`out_idx` 每次
   分配输出可能贡献了这段差异，但尚无复用输出的 A/B 实验。

## 1. 环境与工作流：先确认“本机到底有什么”

### 1.1 实测版本与路径

| 项 | 实测值 | 证据 |
|---|---|---|
| NPU | Ascend910B3，20 AI Core / 40 AIV，62.4 GB HBM | `npu-smi info`、`torch_npu.npu.get_device_name(0)` |
| CANN | `/home/developer/Ascend/cann-9.0.0` | `ASCEND_HOME_PATH` |
| torch / torch_npu | 2.7.1+cpu / 2.7.1.post4 | import 实测 |
| TileLang | 0.1.2+bd7c60c…，源码 `/mnt/workspace/learn/tilelang-ascend` | import 实测 |
| AscendNPU-IR | `3rdparty/AscendNPU-IR/build/install` | 现有产物 |
| cannbot-skills | `/mnt/workspace/learn/cannbot-skills`（2026-10-26 快照） | 目录 |
| bishengir-compile | CANN 9.0.0 自带 1.1.0 | `day7/code/logs/step0-env.log` |

硬件探测结果 `full_soc=Ascend910B3`、`npu_arch=2201`，对应 cannbot 的
`Ascend910` 工作流分支。

### 1.2 采用的编排路径（以及为什么不是全自动）

本机没有把 cannbot-skills 安装成宿主的 opencode 插件，因此没有
`.orchestrator_state.json` 自动状态机可用。实际采用：

- 主代理扮演 orchestrator，读
  `plugins-official/tilelang-op-orchestrator/workflows/Ascend910/workflow.md`；
- Stage 1 用 `tilelang-op-analyst` 子代理，读 `tilelang-op-design` 与
  `tilelang-op-test-design`；
- Stage 2 用 `tilelang-op-developer` 子代理，读 `tilelang-op-develop`，
  产物写到 `custom/rmsnorm/`；
- Stage 3（perf tuning）没有做：用户任务以“能跑 + 数值正确”为主，
  Stage 2 通过后按工作流本应询问是否调优，本报告在“未完成”里如实列出。

这是“工作流文档 + 角色定义 + 同名 skill”的手工编排，不是插件自动路由。
`.orchestrator_state.json` 由主代理按工作流字段维护。

### 1.3 与当前仓库不兼容的地方（实测，不是推断）

1. `prepare_framework.sh` 要求 `set_env.sh`，该 revision 没有这个文件，
   于是脚本判定环境无效并执行 `install_ascend.sh`；后者 `rm -rf build`
   后用 `USE_NPUIR=OFF` 重建，导致 `import tilelang` 报
   `Cannot find libraries: libtilelangir.so`。
   处理：保留被覆盖的 build（改名 `build.after_prepare_nonpui_20261104`），
   按 NPUIR 配置新建 `build/` 重编，`libtilelangir.so` 恢复，import 退出 0。
   原始日志：`code/logs/00-rebuild-npuir.log`。
2. `tilelang-env-check` 的 `check_env.sh` 退出 1（缺 `set_env.sh`、缺可选
   子模块）；`quick_verify.py` 退出 1（脚本还在用已移除的
   `T.barrier_all`）。这两个是 **skill 与仓库 revision 不匹配**，不是 NPU 故障。
   改用仓库自带 `examples/elementwise/vec_add_1d.py` 做上板 smoke，
   退出 0、输出 `All check passed`（`code/logs/02-npu-smoke-vecadd.log`）。
3. 设计/开发 skill 引用的 `examples/{softmax,normalization,...}` 不存在，
   实际参考 `examples/norm/rms_norm.py`；该示例在 host 侧把 N padding 到
   256，与本任务“不缩小规格、不做 host padding”的要求冲突，设计改为
   直接 N=1024。
4. 设计模板里的 `TL_ASCEND_AUTO_CV_COMBINE` 等 pass key 在当前 revision
   不存在；实际 key 是 `NPUIR_ENABLE_AUTO_MULTI_BUFFER` 等。

## 2. Agent 怎样生成

### 2.1 Stage 1：设计与测试计划

`tilelang-op-analyst` 产出：

- `custom/rmsnorm/DESIGN.md`（722 行，44,601 B）；
- `custom/rmsnorm/proto.yaml`（`rmsnorm(Tensor x, Tensor w, float eps=1e-5) -> Tensor y`）；
- `custom/rmsnorm/op_requirements.yaml`（用户规格、boundary/fp16 必测项、容差）。

设计的关键决策：

| 决策 | 选择 | 理由 |
|---|---|---|
| 编程模式 | Developer（`alloc_shared` + `T.Parallel` + `reduce_sum`） | 用户指定；本 revision 的 Developer 路径可用 |
| grid | 1D `T.Kernel(ceildiv(M, block_m), is_npu=True)` | 本仓库 2D grid 会静默串行化第二维（Day 7 已踩过） |
| 行切分 | 一个 block 拥有 `block_m` 个整行，N 完整放 UB | 行内归约一次完成，不需要跨 tile 累加 |
| fp32 tiling | `block_m=16`（N=1024） | 静态 UB 135,296 B |
| fp16 tiling | 设计写 `block_m=16`，实测后改为 8 | cast 临时缓冲导致 16 行编译期 UB overflow |
| rstd | `vsqrt` + `vrec` | 硬件 `vrsqrt` 相对误差约 3e-3，过不了 fp32 1e-4；fp16 也保留高精度路径 |
| 尾块 | `rows = T.min(block_m, M - m0)`，只 copy 有效行 | 不做 host padding、不丢行 |
| 输出 | `@tilelang.jit(out_idx=[2])`，kernel 内分配 Y | 调用形式 `kernel(x, w) -> y` |

Stage 1 原件已保存为 `history_version/design_rev0_original.md` 与
`history_version/proto_rev0_original.yaml`。

设计文档自己也承认了一处矛盾：3.3 节伪代码无条件分配 `x_f32` / `w_f32`，
按 4.5 节预算 fp32 路径会到 204,928 B，超过 192 KiB。这个矛盾没有被
Stage 1 自己修掉，成了 Stage 2 的第一个运行时故障源。

### 2.2 Stage 2：代码与测试

`tilelang-op-developer` 产出：

- `custom/rmsnorm/rmsnorm.py`：最终 5,834 B，4 个函数
  （`pick_block_m` / `validate_inputs` / `rmsnorm` / kernel `main`），
  5 个 `T.alloc_*`，5 个 `T.Parallel`，3 个 v-prefix 调用
  （`vsqrt` / `vrec` / `vcast`）；
- `custom/rmsnorm/test_rmsnorm.py`：486 行（包含 black 格式化后的换行），L0/L1/L2/Boundary 四层，
  `--level {l0,l1,l2,boundary,all}`；人工把 L2/Boundary 也改为阻塞门禁后，
  四层全部写回 `blocking_ok`，任一层失败退出码非 0；
- `logs/`：`l0.log`、`all.log`、`coverage.log`（另有第一次 fp16 失败的全量日志
  `all_attempt1_fp16_compile_fail.log` 与流式重跑 `all_debug.log`）；
- `debug_log.md`：Stage 2 的尝试、分类、修复摘要；
- `history_version/`：每次修改前后的完整快照与 `CHANGES.md`。

kernel 的计算序列（最终版）：

```text
x_sh  = copy(X 的有效行)                 # block_m x N，dtype
w_sh  = copy(W)                          # N
xsq   = cast(x_sh)^2 或 x_sh^2           # fp32 fragment
acc   = reduce_sum(xsq, dim=1)           # (block_m,1)
acc   = acc * (1/N) + eps
rstd  = 1 / sqrt(acc)                    # vsqrt + vrec
y     = x_sh * rstd * w_sh               # fp32 计算
copy(y 的有效行 -> Y)                     # fp16 时先 vcast(rint)
```

和设计伪代码相比，最终实现没有分配 `x_f32` / `w_f32`：fp16 分支在
`T.Parallel` 里就地 `T.cast(x_sh[i,j], "float32")`，权重也在使用点 cast。
fp16→fp32 是精确转换，数值上等价，但省掉了两块常驻 UB。

## 3. 哪里出错、怎样修好

### 3.1 错误一：设计预算漏算，fp32 运行时 MTE 越界

- 现象：`block_m=16`、fp32、N=1024，编译成功，运行 L0 时 vector core
  异常：`The DDR address of the MTE instruction is out of range`，
  `retCode=0x31`。
- 原因：设计伪代码无条件保留 fp32 的 `x_f32`（65,536 B）和
  `w_f32`（4,096 B）。静态合计
  `65,536 + 65,536 + 4,096 + 64 + 64 + 65,536 + 4,096 = 204,928 B`，
  超过 910B 单核 192 KiB（196,608 B）。编译器不报错，运行时才炸。
- 修法（Agent 自主完成）：删除这两块缓存，fp16 分支在乘加时 inline cast；
  UB 降到 fp32 135,296 B / fp16 100,480 B。
- 留证：`history_version/rmsnorm_impl_s2_attempt1_original.py` →
  `attempt2_ub_fix.py`，`history_version/CHANGES.md` 第 1 节。原始失败输出
  已从 Stage 2 子代理 transcript 中提取为
  `code/process/probes/01-fp32-mte-incident.log`（含原命令、输出尾段、`EXIT=134`、
  首次 `MTE instruction is out of range` 行），复现脚本同目录
  `rmsnorm_compile_check.py`。

### 3.2 错误二：fp16 编译期 UB overflow，静态预算算不出来

- 现象：fp16、`block_m=16`、N=1024，BiShengHIR 直接失败：
  `ub overflow, requires 2113536 bits while 1572864 bits available`
  （264,192 B > 196,608 B）。fp32 同配置能过。
- 原因：fp16 路径有两次 fp32 promotion 和一次 fp32 结果，编译器为
  `vcast` / `vmul` 插入临时缓冲；逻辑缓冲只有 100,480 B，但实际需要
  264,192 B。`NPUIR_ENABLE_AUTO_MULTI_BUFFER=False` 也压不下去，说明不是
  单纯多缓冲造成的。
- 诊断（Agent 自主完成）：写了一个 compile-only 探针，扫 `block_m` 与
  pass config：

  | 配置 | 结果 |
  |---|---|
  | fp16 bm=16 default | FAIL，264,192 B |
  | fp16 bm=12 default | FAIL，198,720 B |
  | fp16 bm=10 default | COMPILE_OK |
  | fp16 bm=8 default | COMPILE_OK |
  | fp16 bm=16 no-multibuffer | FAIL，264,192 B |

  线性拟合约 16 B/元素/行（`xsq`、两个 promotion 临时、fp32 结果四块）。
- 修法：`pick_block_m(N,"float16")` 用 16 B/元素系数并封顶 8 行；
  N=1024 时 `block_m=8`（约 133 KiB，68% UB）。fp32 仍是 16。
- 留证：`attempt3_pre_fp16_ub.py` → `attempt3_post_fp16_ub.py`，
  `CHANGES.md` 第 2 节；修后 fp16 smoke `max_abs=9.766e-04`，
  `n_exceed=0/1048576`。探针脚本与六档原始编译输出已提取为
  `code/process/probes/02-fp16-blockm-probe.log` 和
  `rmsnorm_fp16_variants.py`；该日志是 compile-only，`EXIT=0` 只表示六个
  探针都执行完，不代表六个都编译成功。

### 3.3 错误三（人工侧）：prepare 脚本把 NPUIR 构建冲掉

这是环境问题，不是 Agent 代码问题，但直接阻断实验。处理见 1.3 第 1 条。
没有 reset 源码、没有升级任何包，只重建了 `build/`。

### 3.4 修改链一览

| 版本 | 文件 | 改动 | 触发 |
|---|---|---|---|
| attempt1 | `rmsnorm_impl_s2_attempt1_original.py` | 初版，按设计伪代码分配 `x_f32/w_f32` | Stage 2 首次生成 |
| attempt2 | `rmsnorm_impl_s2_attempt2_ub_fix.py` | 去掉两块 fp32 缓存，inline cast | fp32 L0 运行时 MTE 越界 |
| attempt3-pre | `rmsnorm_impl_s2_attempt3_pre_fp16_ub.py` | 与 attempt2 相同（留档） | fp16 编译失败前 |
| attempt3-post | `rmsnorm_impl_s2_attempt3_post_fp16_ub.py` | fp16 `block_m` 16→8，预算系数改 16 B | fp16 编译期 UB overflow |

> 人工没有改过 `rmsnorm.py` 的任何一行；kernel 的三次修改都是 Agent
> 自己完成的。人工动过的是 `test_rmsnorm.py` 的**门禁语义**：把原本只打印
> `BOUNDARY_WARN` 的 L2/Boundary 改为返回 bool 并参与退出码（见
> `history_version/human_blocking_gate.diff` 与 `HUMAN_CHANGES.md`）。
> 容差、测试数据和覆盖维度未改。另外人工做了环境修复、验证工具和验收口径。

## 4. 上卡结果（独立验收）

### 4.1 编译、退出码、数值分别给证据

- **编译成功**：`logs/l0.log` 中 `AscendNPU IR compile success:`；
  `logs/all.log` 每个 case 前都有该行（fp16 修复后重跑）。
- **程序退出 0**：`logs/l0.log` 末尾 `EXIT_CODE=0`；
  `logs/all.exit` 为 `0`；`logs/coverage.log` 为
  `COVERAGE: 18 PASS / 0 MISS / 0 N/A -> PASS`。
- **数值正确**：不采信 Agent 自带的 `test_rmsnorm.py` 作为唯一证据，
  另跑 `code/reference/agent_acceptance.py`：CPU float64 参考、
  逐元素 `|a-g| <= atol + rtol*|g|`、100% 元素必须通过，
  容差固定 fp32 1e-4/1e-4、fp16 1e-2/1e-2，不允许放宽。

（独立验收的具体数字见下表；加门禁后的重跑原始输出在
`code/logs/07-independent-acceptance-blocking.log`，加门禁前的第一次运行在
`code/logs/06-independent-acceptance.log`。）

| case | 容差 (atol/rtol) | max_abs | max_rel | 超差元素 | 结论 |
|---|---|---|---|---|---|
| fp32 1024×1024 | 1e-4 / 1e-4 | 1.353e-06 | 2.221e-07 | 0 / 1,048,576 | PASS |
| fp32 1023×1024（尾块） | 1e-4 / 1e-4 | 1.149e-06 | 2.355e-07 | 0 / 1,047,552 | PASS |
| fp16 1024×1024 | 1e-2 / 1e-2 | 3.901e-03 | 2.113e-03 | 0 / 1,048,576 | PASS |

负向对照：三个 case 都把输出中绝对值最大的元素增加
`10 × (atol + rtol·|golden|)`，独立检查器三次都返回
`pass=False, n_mismatch=1`。

退出码不是“跑完就 0”。`agent_acceptance.py` 的 `all_ok()` 要求
**三个 case 的 `metrics.pass` 全为 True，且三个负向对照的 `pass` 全为
False**，`exit_code()` 才返回 0；否则返回 1，`main()` 通过 `sys.exit()`
把它带出去。重跑日志
`code/logs/07-independent-acceptance-blocking.log` 末尾为
`ACCEPTANCE_RESULT: PASS (3/3 cases pass, 3/3 negative controls caught)`、
`EXIT_CODE=0`。

门禁本身也做了自测：`gate_selftest.py` 构造“全好 / 一个精度失败 /
一个负向对照漏检”三组假结果，期望退出码分别为 0/1/1，输出
`code/logs/08-acceptance-gate-selftest.log`（`GATE_SELFTEST_PASS`）。
这证明失败路径确实能让程序非 0 退出，而不是只改了打印文案。

说明：fp32 的 max_abs 比 Agent 自带测试（3.576e-07）略大，因为独立验收用
`randn` 生成输入，而 Agent 的 L1 用 `uniform[-1,1]`；两者都远低于 1e-4。
fp16 的 3.901e-03 与 Day 3 手写版的 3.906e-03 基本一致，主要来自 fp16
输出量化，而不是 rstd 计算。

### 4.2 能检出错误的校验对照

`code/reference/independent_check.py` 自带负向对照：把输出中绝对值最大的
一个元素加上 `10 × (atol + rtol·|golden|)`，检查器必须报 fail。自测输出：

```text
identity: pass=True ...
negative control: pass=False, n_mismatch=1, max_abs=...
SELF_CHECK_PASS
```

这说明“数值正确”不是检查器永远返回 True 的假阳性。

这里也踩过一个小坑：第一版负向对照用固定 `10*max(atol,1e-3)` 扰动，
fp32 能检出，但 fp16 的最大元素 |golden|≈10，rtol 项让局部容差约 0.11，
0.1 的扰动落在容差内，对照反而 `pass=True`。改成
`10 × (atol + rtol·|golden|)` 后，三个 case 都稳定
`pass=False, n_mismatch=1`（`code/logs/06a-negative-control-selftest.log`）。
“负向对照本身也要验证”是这次的一个教训。

### 4.3 三种时钟

下表是补门禁前的一轮数据；补门禁后的重跑结果见第 5.3 节和
`code/reference/agent_acceptance.json`。

| case | host µs | e2e µs | device µs |
|---|---|---|---|
| fp32 1024×1024 | 351.79 | 352.73 | 15.31 |
| fp32 1023×1024 | 351.42 | 364.26 | 15.31 |
| fp16 1024×1024 | 356.90 | 354.62 | 13.10 |

Agent 版的 host 比 Day 3 高约 90–120 µs。`@tilelang.jit(out_idx=[2])`
每次调用都会在 host 侧分配输出，而 Day 3 的 `kernel(x, w, y)` 复用
预分配的 `y`；这是差距的一个可能来源。device 时间两版基本同一量级
（见 5.3），但没有复用输出的 A/B 实验，不能确定各因素的贡献。

`e2e` 与 Day 3 的 `benchmark()` 同口径（NPU Event 包住多次调用）；
`host` 是 CPU 侧入队时间；`device` 是 `torch_npu.profiler` 的
`kernel_details.csv` 中的单 kernel `Duration(us)`。
Day 3 的 device 已经实测为 15.85–17.31 µs，而 e2e 约 220–267 µs，
host 占绝对主导。因此第 5 节的性能比较只认真实 device 口径。

## 5. 与 Day 3 手写版的对比

### 5.1 代码结构（静态统计）

| 指标 | Day 3 手写 | Day 8 Agent |
|---|---|---|
| kernel 文件行数 | 410（含测试/benchmark） | 145（只含 kernel 与校验） |
| 测试文件 | 同文件，7 个 `test_*` | 独立 486 行，4 个层级函数 |
| 函数数 | 17 | 4（kernel 文件）+ 15（测试文件） |
| UB 分配数 | 10（全部无条件分配） | 5 |
| `T.Parallel` | 0（Expert，逐 op 调用） | 5 |
| v-prefix 调用 | 17 | 3（+ `reduce_sum`） |
| 手动同步 | 0 | 0 |
| host padding | 无 | 无 |
| 可调 `block_M` | 是（参数） | 是（`block_m` 参数，fp16 封顶 8） |
| rstd 模式 | 3 种（sqrt_rec/rsqrt/newton） | 1 种（sqrt_rec） |
| 权重开关 | 有（`use_weight` ablation） | 无（规格固定带 W） |

### 5.2 精度

| case | Day 3 手写 max_abs | Day 8 Agent max_abs | 容差 |
|---|---|---|---|
| fp32 1024×1024 | 1.907e-06 | 1.353e-06 | 1e-4 |
| fp32 1023×1024 | 1.907e-06 | 1.149e-06 | 1e-4 |
| fp16 1024×1024 | 3.906e-03 | 3.901e-03 | 1e-2 |

两版都远低于容差；Agent 版 fp32 的误差更小，主要是它用 float64 参考
（Day 3 用 float32 参考）以及 tiling 不同，不能单独归因于 kernel 更好。

### 5.3 性能（同 shape、同 dtype、同公式、同 device 口径）

| case | Day 3 device µs | Day 8 device µs | Day 3 e2e µs | Day 8 e2e µs |
|---|---|---|---|---|
| fp32 1024×1024 | 17.04 | 17.11 | 236.97 | 336.58 |
| fp32 1023×1024 | 17.31 | 16.84 | 267.31 | 375.65 |
| fp16 1024×1024 | 12.82 | 12.06 | 251.01 | 351.11 |

（Day 8 一列取自加门禁后的第 07 次重跑，与 `agent_acceptance.json` 一致；
更早两轮的 device 为 17.10/17.01/12.08 与 15.31/15.31/13.10。）

只有在 profiler 确实解析到 `main` kernel 行时才填 device 数字；
e2e 受 host 调度影响，仅作参考。两版 block 配置不同（Day 3 `block_M=8`，
Agent fp32 16 / fp16 8），这本身就是同条件下的实现差异，不做“谁更快”
的强结论。

device 的三次独立验收分别是 17.10/17.01/12.08 µs、15.31/15.31/13.10 µs
和上表的 17.11/16.84/12.06 µs，fp32 波动约 ±2 µs（10% 以上），所以两版
只能判断为“同一量级”。e2e 的差距（约 +90–120 µs）稳定存在，且 host 与
e2e 几乎相等，说明瓶颈在 host 入队/分配，而不是 device。

### 5.4 可维护性

- Agent 版把 kernel 与测试彻底分开，测试有 L0/L1/L2/Boundary 分层、
  coverage manifest 和 `--level` 入口，**测试工程化明显更好**；
- Agent 版的 fp32/fp16 分支写在同一个 kernel 里，靠 Python `if dtype`
  在编译期分派，读起来比 Day 3 的十块 buffer 逐 op 流水更短；
- Day 3 版保留了 `use_weight`、`rstd_mode` 等 ablation 开关，适合做
  性能/精度实验；Agent 版面向单一规格，参数更少；
- Agent 版把 UB 预算写成注释和 `pick_block_m` 公式，但第二类错误说明
  **公式仍然不可靠**，真正的约束要编译探针确认；
- 两版都没有手动同步，说明当前 NPUIR 的自动同步覆盖了这些模式。

### 5.5 与 Day 4 OpenCode conductor 五算子迁移的关系

Day 4 是不同算子，性能数字不可直接对拍。可比较的是流程：

- Day 4 有 conductor / designer / design-reviewer / developer / optimizer /
  integrator / skill-evolver 的角色链；本次因为插件未安装，只手工复用了
  analyst + developer 两个角色，没有 design-reviewer 的独立评审；
- Day 4 的验收靠 orchestration 状态机推进；本次靠主代理读 workflow 文档
  手工推进，并把每次失败写进 `history_version/CHANGES.md`；
- 共同点：都要求“设计→实现→测试→修复”的闭环，且都把编译/运行/精度
  分开验收。

## 6. 人工实际改过什么，Agent 最容易错在哪

### 6.1 人工介入清单

| 事项 | 谁做的 | 说明 |
|---|---|---|
| 修复 NPUIR build | 人工 | prepare 脚本 `rm -rf build` 后用非 NPUIR 配置重建，人工改名保留并重编 |
| 选择工作流与角色 | 人工 | 读 workflow/AGENTS.md，决定 analyst + developer 手工编排 |
| 写独立验收器 | 人工 | `independent_check.py` + `agent_acceptance.py` + `bench_common.py` |
| 修 `code_metrics.py` 的统计口径 | 人工 | 初版漏计 `alloc = T.alloc_ub` 间接调用，改为同时匹配 `alloc(` |
| 改 Agent 生成的 kernel | **没有** | 三次 kernel 修改均由 Agent 完成 |
| 改 Agent 生成的 test | 人工 | L2/Boundary 从非阻塞改为阻塞门禁；只改控制流，不改容差/数据 |
| 给独立验收加退出门禁 | 人工 | `all_ok()` / `exit_code()`，并用 `gate_selftest.py` 验证失败路径 |
| 放宽容差 | **没有** | 独立验收固定 1e-4/1e-4 与 1e-2/1e-2 |

### 6.2 最容易出错的一步

**UB 预算 / tiling**，而且错了两次，形态不同：

1. 设计阶段按“逻辑 buffer 大小”做预算，漏掉无条件分配，fp32 运行时越界；
2. fp16 按逻辑 buffer 算只有 100,480 B，但编译器插入的 cast 临时缓冲
   把需求推到 264,192 B，编译期才报错。

两次都说明：在这条 NPUIR 路径上，**静态 buffer 求和不能替代编译探针**。
Agent 第二次的处理方式（写 compile-only 探针扫 `block_m` 和 pass config，
再用线性系数固化到 `pick_block_m`）是这次最值得复用的工程动作。

## 7. 没完成 / 与原始规格的差异（如实列出）

1. **Stage 3 性能调优未做**。用户任务没有要求调优，Stage 2 通过后即停止；
   `perf_tuning/` 为空。
2. **cannbot 插件未安装**，没有自动 orchestrator 状态机；本次是手工编排，
   `.orchestrator_state.json` 由主代理维护。
3. **`tilelang-env-check` 未通过**（`check_env.sh` 与 `quick_verify.py` 均
   退出 1），原因是 skill 与仓库 revision 不匹配；已用仓库自带 vec_add
   示例替代上板 smoke，但这个 skill 本身在本机不可用。
4. **fp16 `block_m` 从设计的 16 降到 8**。这是为绕开编译期 UB overflow
   做的实现调整，shape/dtype/公式/容差没有缩小；代价是 fp16 每块行数减半，
   可能的性能影响见第 5.3 节（device 口径）。
5. **L2/Boundary 的非阻塞问题已由人工修掉，不算遗留缺口**：Agent 原版
   只打印 `[BOUNDARY_PASS]/[BOUNDARY_WARN]`，不写回 `blocking_ok`；人工
   改成四层全部参与退出码，并用强制失败用例验证过退出 1
   （`code/logs/09-test-suite-gate-selftest.log`）。改后全量重跑
   `logs/all_blocking_gate.log` 为 16×PRECISION_PASS、11×BOUNDARY_PASS、
   0 WARN，退出 0。
6. **N=8192 只做了编译期上限校验**，没有实际上卡跑；设计里的
   `_MAX_N=8192` 是防御性上限，不是已验证范围。
7. **Stage 2 的“attempt”粒度与工作流不完全一致**：Agent 在一次 attempt
   内自修了三次，`stage_retry_count[2]` 仍为 0；工作流原意是每次
   `[PRECISION_FAIL]/runtime_fail` 都走 `fail_stage` 累加。本次把三次
   修复都记在 `history_version/CHANGES.md` 和 `debug_log.md`，状态文件
   没有虚增重试次数。

## 8. 学到的

- **“编译成功”不等于“能跑”**。fp32 的 UB 超限是运行时 MTE 越界，
  如果只看编译日志会误判。
- **“逻辑 buffer 能放下”不等于“编译器能放下”**。fp16 的 cast 临时缓冲
  是隐形成本，必须用 compile-only 探针量，而不是拍公式。
- **验收要独立**。Agent 自带测试的 golden、容差和“通过”判定都在它自己
  手里；只有换一套 float64 参考、固定更严的容差、再加负向对照，才能把
  “它说过了”变成“我能证明它过了”。
- **“打印 PASS”不等于“门禁拦住”**。脚本跑完就算成功、`WARN` 不影响
  退出码，都会让失败静默通过。必须把所有判定收敛到一个显式 `exit_code`，
  再用构造的失败结果验证它真的返回非 0——门禁自己也要有测试。
- **host 开销会掩盖 kernel 差异**。这个 shape 下 e2e 在 230–350 µs、
  device 只有十几 µs；不比 device 口径，性能结论基本没有意义。
- **工作流文档和仓库 revision 会漂移**。env-check、示例路径、pass key
  都对不上；先跑通一个最小上板示例，再谈 Agent 编排，顺序不能反。
