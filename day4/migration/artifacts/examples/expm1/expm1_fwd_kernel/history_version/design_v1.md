# expm1_fwd_kernel 算子设计文档（迁移：TileOPs GPU → TileLang-NPUIR）

| 项 | 值 |
|----|----|
| project / op | `expm1` / `expm1_fwd_kernel`（严格沿用源 `@tilelang.jit` 装饰的 builder 函数名；`@T.prim_func` 参数名与顺序保持 `(x, y)`） |
| scenario / mode | migration (harness) / first_design |
| 源算子 | `Expm1FwdOp` ← `Expm1FwdKernel(FloatUnaryKernel)` ← `expm1_fwd_kernel`（GPU 原件 `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2142–2147 + 基类 `UnaryKernel` L680–808 / `FloatUnaryKernel` L1297–1301，本文 §0 已全文亲读；NPU 侧 verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/expm1/_expm1_fwd_kernels.py` 全 34 行） |
| 规格权威来源 | Stage 0 迁移 prompt（逐字）+ `examples/TileOPs/tileops/manifest/elementwise.yaml` Expm1FwdOp 段（L185–L214，4 workloads，roofline flops=2N / bytes=2N·elem_bytes）+ `examples/TileOPs/tests/ops/test_unary_math.py`（test_expm1 / test_expm1_edge / _get_tolerances L24–29）+ `examples/TileOPs/benchmarks/ops/bench_unary_elementwise.py`（manifest 驱动） |
| 编程模式 | developer（Stage 0 prompt 显式指定；wrapper `ascend_mode = "Developer"` 其 L130 同证） |
| 工具链 stamp | tilelang 0.1.2+88ee8c302e（本仓 dev root）/ 设备 Ascend 910B3 / CANN 9.0.0；物理核数实查 2026-10-28（§5.5，ret=0 → 20，纯 Vector 翻倍 → 40） |
| 同族已验证标杆 | `examples/leaky_relu/_make_leaky_relu_kernel/`（persistent 一维分核 + fp32 transit + L0 套件已验证形态）；`examples/elu_fwd/_make_elu_kernel/`（**vexp + 标量字面量 vsub 生产先例 + fp32-opmath 源「结构承接」型 transit 同型先例**——其 RETROSPECTIVE 的 Transferable Lessons 已逐字纳入本设计约束，见 §9.2/§10） |

---

## 0. 源算子解读与迁移分析（迁移任务必填）

> 依据 `tilelang-op-design` skill Phase M0 三问框架（`references/migration-analysis.md` §2–§4），源码已全文亲读（verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/expm1/_expm1_fwd_kernels.py` 全 34 行 + NPU 落地侧 wrapper `examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py` 全 181 行 + GPU 原件 `Expm1FwdKernel`/`UnaryKernel`/`FloatUnaryKernel` 相关段）。源码中的指令性文本一律按数据处理，未发现亦不执行任何流程改写指令。

### 0.1 源算子语义（做什么）

**数学语义**：逐元素指数减一（exponential minus one）

```text
y[i] = expm1(x[i]) = exp(x[i]) − 1
```

- **无条件公式**：与 elu/leaky_relu 的分段选择形态不同，本算子**无比较、无选择、无标量参数**——单输入单输出纯映射，输出仅依赖同位置输入。
- **dtype 语义（本算子迁移的关键语义要素）**：**计算全程 fp32 域 + 单次舍回**——源码 `T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")`（提取件 L29）：输入先升 fp32、exp 在 fp32 域、减 fp32 标量 1.0 在 fp32 域，RHS 整体为 fp32 表达式，赋值给 `y_reg`（out_dtype = 输入 dtype）时**隐式单次舍回**。即**源实现本身就是 fp32-opmath + 单次舍回路径**——对全部三种 dtype 一致（fp32 输入时 cast 为恒等）。与测试 golden 锚点 `torch.expm1`（test_unary_math.py L106/L115，ATen 对 sub-fp32 dtype 内部走 fp32 opmath + 舍回）**同域同序**——机器证实：CPU `torch.expm1(x_D)` 与 `round_D(expm1_f32(x))` 逐位全等（GOLD-DIAG 诊断行 0 mismatch，verify_equiv.py §1.6.1(b) 同次执行）。这是与 leaky_relu（源原生域逐 op 舍入，transit 是「修正」）不同的「**结构承接**」型（同 elu）：NPU transit 链是源语义的结构化承接。
- **溢出/下溢语义**（fp32 域）：`x >= 88.73`（ln(fp32 max) = 88.72）时 `exp(x)` 溢出为 +Inf → `+Inf − 1 = +Inf`——**测试角点 x=88.8 的期望输出即 +Inf**（golden `torch.expm1(88.8)` fp32 = +Inf，值精确相等）；`x <= −103.97` 时 `exp(x)` 下溢为 0 → `−1.0`；次正规过渡带 `x ∈ [−103.97, −87.3]`：`exp(x)` 为次正规 → `exp(x) − 1` 舍入后仍为 `−1.0`（|exp(x)| < 1.7e-38 ≪ ulp(1.0)/2 = 5.96e-8，**FTZ（flush-to-zero）与否不改变输出**）；`x = −88.8` 角点输出 `−1.0`（golden 同为 −1.0——D-1 edge 行机器证实二者差 5.96e-8 ≪ atol，值判定通过，见 §1.6.1）。
- **±0 语义（零符号差值等价点）**：`exp(±0) = 1` → `1 − 1 = +0.0`；golden `torch.expm1(−0.0) = −0.0`（ATen 符号保持）——**零符号差异（+0.0 vs −0.0），IEEE 值相等**，容差断言通过；L0-5 断言规则按此调整（§8.3/§9.1#3，与 elu 同款豁免逻辑：零符号差由「减法产生 +0」与「ATen 符号保持」的实现差异决定，非条件方向差异）。
- **NaN 传播**：`exp(NaN) = NaN` → `NaN − 1 = NaN`，与 golden 逐值一致。
- **规约语义**：无规约、无累加——纯逐元素映射，不存在累加顺序问题。
- **边界语义**：非整除尾部——源按整块 fragment 覆盖（`block_size = threads × npt`，依赖 grid 收尾；`T.Parallel(threads, npt)` 展开后无逐元素 guard）；manifest workloads 的 N（2²⁰ / 2²⁴ / 2²⁶ / 2²⁸）均被默认 block（1024/2048）整除，源实现实际从未执行 partial tail 路径。**源无 `N == 0` 显式拦截**（Op 层校验为 `numel != N_total`，空张量可穿透）——NPU 侧以 builder 防御性 `N <= 0 → ValueError` 明确拒绝（§9.3，适配项）。
- **host 侧语义**：输入任意 shape → 展平 `(N,)` 连续 → kernel → 还原 shape（Op 层）；无 inplace 参数（expm1 无该契约）；dtype 白名单 `(float16, bfloat16, float32)`——NPU wrapper `SUPPORTED_DTYPES`（其 L133）在 builder 之前即 ValueError 拒绝越界 dtype（其 L142–L146，**fp8 在此被拒**，与 GPU `FloatUnaryKernel.SUPPORTED_DTYPES = _FLOAT_DTYPES`（GPU 原件 L1300）同语义）。
- **完成标准核验**：golden 函数可直接写出（§8.1）——`golden_expm1(x) = torch.expm1(x)`，与源实现数值一致（D-1 机器验证 54/54 行 0 容差违反，最窄裕度 67×，§1.6.1）。

### 0.2 源算子输入输出

| 项 | 值 | 依据 |
|----|----|------|
| 输入 `x` | 展平 `(N,)` 一维连续张量；NPU 契约 dtype **float16 \| bfloat16 \| float32**（manifest `input.dtype`）；host 支持任意 ND，展平后 N = product(shape) | manifest elementwise.yaml L193 + Op 层 |
| 输出 `y` | `(N,)`，**dtype same_as(input)**（`output_dtype=None → out_dtype == dtype`，提取件 L17）；`output.shape == input.shape`（shape_rules） | 提取件 L14/L17 + manifest L195/L197 |
| 标量参数 | **无**——与 elu（alpha）/leaky_relu（negative_slope）不同，本算子无任何构造期标量；`output_dtype` / `threads` / `num_per_thread` 为后端相关参数（§0.5 处置），`threads=256, num_per_thread=8`（签名默认，GPU 基类实际 fp32 npt=4），`block_size = threads×npt`（提取件 L16） | 提取件 L14–L16 |
| 动态轴 | N（展平后元素总数）；builder 按 N 工厂特化（N 为闭包编译期常量） | manifest workloads |
| 不变量 | `y[i] = f(x[i])`，1 读 1 写，无形状变换、无 layout 变换（连续 1-D） | §0.3 数据流 |
| roofline 契约 | flops = **2·N**（exp(1) + sub(1)）；bytes = **2·N·elem_bytes**（1 读 1 写） | manifest L212–L214 + test_unary_math.py test_expm1_eval_roofline L128–137（roofline 断言） |

### 0.3 实现算法解读（怎么算）

**结构化阅读协议 7 步**（migration-analysis §3.1）：

1. **入口签名**：`expm1_fwd_kernel(N, dtype, output_dtype=None, threads=256, num_per_thread=8)`，**提取件自带 `@functools.lru_cache(maxsize=32)`**（提取件 L13——与 elu 提取件不同，elu 提取时 lru_cache 被剥离；本件保留，NPU 侧同款沿用）；返回 `@tilelang.jit(out_idx=[1])` 工厂 `kernel(threads_arg, npt_arg) -> main`——两参调用式是 GPU 基类 `UnaryKernel` 的源侧契约（GPU 原件 `_build_kernel` L787–L808：register_copy 策略传 `threads=cfg["threads"], num_per_thread=cfg["num_per_thread"]`）。**该两参依赖不进入 NPU 设计决策**：NPU harness wrapper 已按 K3/K11 适配为 block_size 形态（§0.7 契约权威，其 L81/L155 亲读）。
2. **Kernel 结构**：`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg)`——**一维 grid**，`block_size = threads * num_per_thread` 为 builder 闭包编译期常量（提取件 L16）；单一 prim_func，**无 fp8 分支**（FloatUnaryKernel 族在 dtype 白名单层已拒 fp8，提取件无双 prim_func 结构——比 elu 源更简）。
3. **计算原语**（逐语句归步）：`T.copy(x[bx·bs:(bx+1)·bs], x_reg)` 整块载入 fragment → `for i, j in T.Parallel(threads_arg, npt_arg)` 二级展开，逐元素 `y_reg[i·npt+j] = T.exp(T.cast(x_reg[i·npt+j], 'float32')) - T.cast(1.0, 'float32')`（提取件 L28–L29）→ `T.copy(y_reg, y[...])` 写回。**计算全程 fp32 域（cast→exp→sub）+ 赋值单次舍回**（§0.1 dtype 语义）。
4. **内存分配**：`x_reg`/`y_reg` = `T.alloc_fragment((block_size,), dtype)`——GPU **寄存器 fragment** 驻留（每元素恰 1 次 GM 读 + 1 次 GM 写，无 SMEM 中转）。
5. **循环与调度**：仅 `T.Parallel(threads_arg, npt_arg)` 一层元素展开（SIMT block×thread 映射，每线程 npt 元素 ILP）；无 serial、无 Pipelined、无 persistent。
6. **host 侧逻辑**：GPU 基类链——`UnaryKernel.__init__`（dtype 白名单校验 → strategy 解析（默认 `register_copy`，GPU 原件 L700 + H200 benchmark 注 L698–699）→ `_build_kernel("register_copy")` → `_make_unary_regcopy(N, dtype_str, op_func, output_dtype, threads, npt)`（GPU 原件 L801–L806）→ `init_config`）；GPU 默认 config threads=256 / npt=4（fp32）/ 8（fp16/bf16）（GPU 原件 `_strategy_npt` L181–L195）。**NPU 落地侧 wrapper（全文亲读）**：K11 折叠常量 `_BLOCK_SIZE_FP32=1024` / `_BLOCK_SIZE_NON_FP32=2048`（其 L65–L66）；custom_op `npub::expm1_fwd`（其 L74–L86）；Kernel 类 `Expm1FwdKernel`（其 L95–L181：dtype 白名单、`ascend_mode="Developer"` 其 L130、`default_config` 按 dtype 折叠 block_size 其 L159–L168、forward 走 custom_op 其 L170–L181）。
7. **编译配置**：无 pass_configs / 环境变量（GPU 默认管线）。

**计算步骤分解表**：

| 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|------|------|------|------|---------|
| S1 | 整块载入 `T.copy(x_tile, x_reg)` | GM x | fragment x_reg | 读 x |
| S2 | 逐元素 `exp(cast(x, f32)) − cast(1.0, f32)`，赋值隐式单次舍回至 out_dtype | x_reg + fp32 常量 1.0 | y_reg | 公式本体（§0.1 全部传播语义） |
| S3 | 整块写回 `T.copy(y_reg, y_tile)` | y_reg | GM y | 写 y |

**数据流图**（源硬件视角，覆盖全部 buffer）：

```text
GM x ──T.copy──> x_reg(GPU register fragment) ──T.Parallel 逐元素 fp32 域 exp−1（赋值单次舍回）──> y_reg(fragment) ──T.copy──> GM y
常量：1.0 以 fp32 标量内联（编译期 cast）；无其他 buffer、无中间 GM 往返
```

### 0.4 优化手段解读（为什么快）

逐项识别（清单法，migration-analysis §4.1；"识别不出来 ≠ 不存在"已对照——源码中确无 SMEM tiling / mma / warp shuffle / 异步流水 / swizzle / online / persistent / fp8 分支结构）：

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---------|---------|------|----------------|------|
| O1 | register_copy fragment 驻留 | `alloc_fragment` + 整块 `T.copy` 进出 | 每元素恰 1 次 GM 读 + 1 次 GM 写，中间全寄存器 | GPU 寄存器堆 | 硬件强相关（NPU 等价片上层 = UB） |
| O2 | coalesced 连续访问 | 连续 tile 载入/写回（`bx·bs` 连续窗） | 合并访存事务、吃满带宽 | memory coalescing | 意图可移植（访存组织需重排） |
| O3 | block×thread 二级并行 | `T.Parallel(threads_arg, npt_arg)` | 每线程 npt 元素 ILP + SIMT 占用 | CUDA SIMT 层级 | 硬件强相关，需重设计 |
| O4 | **fp32 中间精度 + 标量常量 fp32 化** | `T.cast(x, "float32")` / `T.cast(1.0, "float32")`、赋值单次舍回 | 数值路径与 golden（fp32 opmath）一致；消除原生域逐 op 舍入损失——**本算子唯一的算法层性质决策**（对 exp(x)−1 的相消误差而言，fp32 域单次舍回是源给出的精度基线） | —（纯算法层 dtype 路径决策） | **可移植**（NPU transit 结构化承接，§0.6 R2） |
| O5 | 编译期常量折叠 + host 编译缓存 | `block_size = threads*npt` 闭包常量、grid 表达式特化、`@lru_cache(32)`（提取件 L13 自带） | host 期折叠、消除重复编译 | host 层 | 保留 |

**算法层 vs 实现层区分**：本源算法为"单步逐元素映射"，无 online / 无分块累加等算法层结构优化；**O4 是源码中唯一算法层性质的决策**（dtype 域选择：无损升域保序 + fp32 域计算 + 单次舍回），它同时是 §0.6 R2 的直接输入与 D-1 验证的基线——**迁移设计中不得静默丢弃**（调度 prompt 明示要求显式承接）。

### 0.5 硬件耦合性分析与 NPU 适配决策

三层判定模型 + 四态处置（migration-analysis §5；判定输入含 Phase R 调研结论 §1.6.0——调研选定 = 基线公式（`exp(x)−1` fp32 域形态，R1 七族候选评估后无算法族替换）+ persistent/UB/transit 落地形态，源算法公式未被替换，仅实现结构重设计）：

| # | 条目 | 层 | 处置 | 依据 |
|---|------|----|------|------|
| 1 | 语义层全集（公式 exp(x)−1 / fp32 域单次舍回 / NaN·Inf·±0·下溢溢出传播 / `(N,)` 契约 / dtype same_as / host 展平 / 无标量参数） | 语义层 | **保留** | 语义层不可妥协（映射表前置规则） |
| 2 | O1 register_copy fragment | 优化层 | **等价替换** | fragment → UB：`T.alloc_ub`（映射表「shared memory → L1/UB」「register 累加 → UB 中间缓冲」；同族先例 leaky_relu/elu/lerp/mish kernel 全 UB 链） |
| 3 | O2 coalesced 连续访问 | 优化层 | **保留**（意图承接） | NPU 侧 UB 连续零起点切片 + persistent grid-stride 聚集窗口（pattern-library elementwise.md PL-1.6 实测：grid-stride 优于每核连续分块 +3.2%@256M；kb_stale_check 2026-10-28 复核非 stale——front-matter 含 2026-10-28 结构重验戳） |
| 4 | O3 block×thread 二级并行 | 优化层 | **重新设计** | 映射表「block × thread 两级并行 → 重新设计（本项目 `T.Kernel` 只接受一维 block 数）」；元素级并行改由 v-prefix 全 buffer 向量 op 承接（§0.6 R1） |
| 5 | O4 fp32 中间精度 + 单次舍回 + 常量 fp32 化 | 算法层（dtype 路径） | **保留**（transit 结构化承接） | 源路径 = golden 路径（§0.1，GOLD-DIAG 机器证实）；NPU fp16/bf16 走 vcast(rint) 升 fp32 → fp32 域 vexp/vsub → vcast(rint) 单次舍回 = 源舍入路径结构复刻（PL-1.17 第四例 elu 同「承接」型先例）；1.0 常量以 vsub 标量字面量（fp32 值参与）内联 |
| 6 | O5 编译期常量折叠 + lru_cache | host 层 | **保留** | 同款：N/block_size 闭包常量 + `@lru_cache(maxsize=32)`（提取件 L13 自带，直接沿用） |
| 7 | `output_dtype` 参数 | 接口契约 | **保留**（签名 + 校验） | 源签名参数；manifest same_as(input) → NPU 侧校验 `output_dtype ∈ {None, dtype}` 否则 ValueError（防御性承接） |
| 8 | `threads` / `num_per_thread` 参数 | 接口契约（后端相关） | **等价替换** | K11 折叠：`threads×npt` 积 → 单参 `block_size`（迁移规则「后端相关参数可适配」类目）；参数保留于工厂签名但不消费（源签名保真 + lru_cache 键空间一致，elu §10.3 同款），NPU 侧 block_size 语义见 §3.5.2 |

**处置统计：保留 4（#1/#3/#5/#6）/ 等价替换 2（#2/#8）/ 重新设计 1（#4）/ 舍弃 0**。无存疑项（全部条目有迁移映射表条目佐证，API 细节佐证为 `docs/Tilelang.language/` 各 API 文档与 §3.5.3 所列 examples 同类实现文件）。**本算子无「舍弃」项**——与 elu 不同（elu 舍弃 fp8 分支），本源无 fp8 死代码、无 inplace、无标量参数，接口面最简。

### 0.6 NPU 算法重设计

对 §0.5「重新设计」项 + 调研选定算法族（§1.6.0）落地为 NPU 侧算法。重设计输入：§0.5 #4（并行结构）+ §0.4 全部优化意图 + Phase R 调研结论（§1.6.0：基线公式四口径并列最优、Taylor/fp64 等候选经机器验证评估后不进基线、无算法族替换，选定 = 基线公式 + NPU 重设计落地形态）。

**R1 并行结构重设计**（承接 §0.5 #4，模式库「并行结构重设计」）

- 源方案：`T.Kernel(ceil(N/bs))` × `T.Parallel(threads, npt)` 两级 SIMT 展开，逐元素 fp32 域 exp−1。
- NPU 新算法：**一维 `T.Kernel(num_kernels)` persistent 分核**（`num_kernels = min(ceil(N/block_size), vector_cores)`）+ 核内 `T.serial(num_local_tasks)` 静态边界 grid-stride（`block_id = i·num_kernels + cid`，`if block_id < num_logical` 屏蔽）+ **v-prefix 全 buffer 向量链**（`vexp` → `vsub(标量 1.0)`，fp32 域）替代逐元素展开。**与 elu 相比链更短**：无 vcmp/vselect（无条件公式）、无 vbrc（无 α/阈值常量——唯一标量 1.0 由 vsub 标量字面量操作数承载，零缓冲）。
- **语义保持论证**：① 逐元素映射无累加顺序，lane 重排不改结果；每元素恰计算同一公式一次。② 边界：尾块 `tail = T.min(block_size, N−t0)` 切片 + 零起点 UB 拷出覆盖任意 N（含 N < block_size、非 32B 对齐尾、N=1），行为与源整块覆盖语义一致（源依赖 block_size 整除，NPU tail 方案覆盖非整除——严格更强）。③ NaN/Inf/±0/下溢溢出：`vexp` 逐 lane 独立计算（NaN→NaN、±0→1、x≥88.73→+Inf、x≤−103.97→0），`vsub(·, 1.0)` 逐 lane IEEE 减法——传播语义与源逐元素表达式一致（§0.1 全表）；§1.6.1 机器验证 0 容差违反（54/54 行）。

**R2 dtype 路径重设计**（算法替代·dtype 路径；承接 §0.5 #5 / §0.4 O4——**源 fp32-opmath 语义的结构化承接而非修正**，「承接」型同 PL-1.17 第四例 elu；采纳须与 §1.6.0 调研结论一致，见 §1.6.0 R4 dtype 路径行）

- 源方案：全程 fp32 域（cast→exp→sub）+ 赋值单次舍回（§0.1）。
- NPU 新算法：**fp16/bf16 走 fp32 中转链**——`T.vcast(x_ub, x32, round_mode="rint")` 无损升 fp32 → fp32 域 `vexp` → `vsub(w32, 1.0, w32)`（标量字面量，fp32 值参与）→ `T.vcast(w32→x_ub, round_mode="rint")` **单次舍回**；fp32 走原生域（无 transit，`vsub(w32, 1.0, x_ub)` 跨缓冲直接产出结果驻留 x_ub）。与源的「升域→fp32 域计算→单次舍回」结构**逐位同构**（升域无损 → fp32 域同序计算 → 唯一一次 D 域舍入）。
- 采纳依据（双触发，PL-1.17 四例实证 pattern + 本任务第五例）：① **bf16 无 v-prefix 算术/超越 op**——`T.vexp` dtype 矩阵 bf16 ×（docs/Tilelang.language/数学操作/T.vexp.md §2.2.1，亲读）、`T.vsub` bf16 ×（docs/Tilelang.language/数学操作/T.vsub.md §2.2.1，亲读）——bf16 原生域链无计算路径，vcast 升 fp32 是唯一路径；② **fp16 golden/source 对齐**——fp16 原生域链（vexp fp16 原生可用，T.vexp.md §2.2.1 fp16 √）逐 op 舍入偏离源 fp32 域单次舍回路径（与 golden 的对齐以 transit 为先例，PL-1.17：leaky_relu/lerp 原生域 1–3 ulp 残差 vs transit 0 ulp）。
- **语义保持论证**：fp16/bf16 → fp32 升变换**数学无损**（值域子集）且保序；fp32 域 `exp→sub` 与源表达式同序同域逐 op 单次舍入，中间量逐位一致；末端单次舍回 = 源赋值舍回同一对象；fp32 dtype 无 transit（原生域 = 源域）。机器验证 A1：全 dtype 0 容差违反（§1.6.1(b)）。**与 elu 的差异注记**：elu 的 golden（F.elu fp32 opmath）与 transit 链逐位一致（0 ulp）因其 golden 也走 exp 子计算；本算子 golden（torch.expm1）在中间量域与候选式同为 fp32，但 **expm1 与 exp(x)−1 是不同舍入路径**——cand vs golden 存在固有 ≤2 ulp 级位型差异（fp32 randn 判定集 47% lane ≤2 ulp），**验收基准是容差包络（viol=0，最窄裕度 67×）而非 bitwise**（§8.2，与 leaky_relu/elu 的 bitwise 主张差异如实声明）。

**R3 尾块重设计**（承接 §0.5 #8→尾块语义）：tile 级 `tail = T.min(block_size, N−t0)` + 零起点切片，无逐元素分支（源本无逐元素 guard，NPU 形态与其语义一致且覆盖非整除）。

**源优化意图承接说明**（algorithm-research §4：不得静默丢弃）：

| 源手段 | 意图 | NPU 承接 |
|--------|------|---------|
| O1 fragment 驻留（1 读 1 写 GM） | 消中间 GM 往返 | UB 驻留整 tile，单遍 load→compute→store，GM 流量同为 2N·B 下界（§1.6.0 R3） |
| O2 coalesced | 满带宽连续访问 | UB 连续零起点切片 + persistent grid-stride 聚集窗口（PL-1.6 实测最优访问形态） |
| O3 二级并行 ILP | 元素级并行度 | v-prefix 全 buffer 向量 op（Vector 单元全宽度）+ Developer 自动多缓冲（PL-1.6 实证 lerp 7-pass 链完全隐藏于 MTE2 窗口——本算子 2-op/4-op 链更短，同判加强） |
| O4 fp32 中间精度 + 单次舍回 | 数值路径与 golden 一致 | R2 transit 链结构承接（D-1 0 容差违反）；1.0 常量以 vsub 标量字面量（fp32 值）内联（消标量访存意图同承） |
| O5 编译期折叠 + lru_cache | host 期特化 | 同款保留（提取件 L13 自带 lru_cache 直接沿用） |

### 0.7 标杆实现

- **源算子 verbatim 提取件**：`examples/TileOPs/tileops/kernels/elementwise/expm1/_expm1_fwd_kernels.py`（Pattern-A 提取，自带 lru_cache，自包含无外部 helper，全 34 行已亲读）。
- **GPU 原件**：`/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2142–2147（`Expm1FwdKernel.op_func`）+ L680–L808（`UnaryKernel` 基类：strategy/build 链）+ L1297–1301（`FloatUnaryKernel` dtype 白名单）。
- **NPU harness wrapper（集成契约权威，全文亲读）**：`examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py`。**两处调用面 verbatim**：
  - custom_op 内（其 L81）：`return expm1_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)`——工厂带 `block_size=` 关键字构建 + 返回 callable 以 `(x, N)` 两实参调用；
  - Kernel 类 `__init__` 内（其 L155）：`self.kernel = expm1_fwd_kernel(self.N_total, self.dtype_str)`——工厂两位置参构建（block_size 缺省）。
  - 其余契约面：K11 折叠常量（L65–L66）、K6 fp8 死代码注记（L29–L31）、`ascend_mode="Developer"`（L130）、`SUPPORTED_DTYPES`（L133）、dtype 白名单 ValueError（L142–L146）、`default_config`（L159–L168）、forward（L170–L181）。
  - **调用面契约分析 → 最终工厂签名建议与 Stage 5 适配清单：见 §3.5.2（本设计的关键差异化决策点）**。GPU 基类两参调用式（`self.kernel(cfg["threads"], cfg["num_per_thread"])`）仅为源侧事实，永不进入 NPU 设计决策行（leaky_relu Stage 2 检视教训）。
- **同族已迁移标杆（本仓已验证）**：`examples/leaky_relu/_make_leaky_relu_kernel/`（DESIGN.md + 实现 + verify_equiv.py + RETROSPECTIVE.md——persistent 分核 + fp32 transit + L0 六项套件，**结构最接近模板**）；`examples/elu_fwd/_make_elu_kernel/`（**vexp 生产链 + 标量字面量 vsub + fp32-opmath 承接型先例**，与本算子计算核同族）；`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/`（E1/E2 单参工厂契约同形）。
- **精度/性能契约**：`examples/TileOPs/tests/ops/test_unary_math.py`（test_expm1 / test_expm1_edge / _get_tolerances）、`examples/TileOPs/benchmarks/ops/bench_unary_elementwise.py`（manifest 驱动）。
- **manifest**：`examples/TileOPs/tileops/manifest/elementwise.yaml` Expm1FwdOp 段（L185–L214）。

---

## 1. 概述

### 1.1 算子名称

`expm1_fwd_kernel`（`expm1` 项目；@tilelang.jit builder 函数名严格沿用源签名）。

### 1.2 功能描述

NPU 侧逐元素指数减一前向：对展平 `(N,)` 连续输入按 `y = expm1(x) = exp(x) − 1` 做无条件映射，输出同 shape 同 dtype。支撑 `Expm1FwdOp` 的 4 个 manifest workloads（smoke-1m [1048576] fp32|fp16|bf16 / elementwise-16m [4096,4096] fp16|bf16|fp32 / elementwise-64m [8192,8192] fp16|bf16 / elementwise-256m [16384,16384] fp16|bf16）。本节描述的是 **§0.6 迁移决策后的 NPU 侧算法**（与源算法的结构差异：两级 SIMT 展开 → persistent 一维分核 + UB 向量链；源 fp32-opmath 语义由 transit 链结构承接；`kernel(threads_arg, npt_arg)` 两参 → block_size 形态 K11 契约，§3.5.2）。

### 1.3 数学公式

```text
y[i] = expm1(x[i]) = exp(x[i]) − 1     （fp32 域计算，单次舍回；无比较、无选择、无标量参数）
```

FLOPs 口径（manifest roofline L212–L214）：exp(1) + sub(1) = **2 FLOPs/元素**；访存 = 2·N·elem_bytes（1 读 1 写）。算术强度 2/(2B) = 1/B flop/B：fp16 = 0.5、fp32 = 0.25——远低于 Vector 单元平衡点，**带宽受限算子**（比 elu 的 2/B 更低，访存主导更强）。

### 1.4 算法描述

分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的算法（与源算法非直译关系）：

1. **分核**：persistent 一维分核——`num_logical = ceildiv(N, block_size)` 个逻辑 tile，固定启动 `num_kernels = min(num_logical, vector_cores)` 个物理核（纯 Vector 翻倍核数，§5.5 实查 20→40），核内 `T.serial(num_local_tasks)` 静态边界 grid-stride。
2. **单遍 tile 流水**（每 tile）：GM→UB 同 dtype 载入（`tail` 切片）→ [fp16/bf16: `vcast(rint)` 升 fp32，域输入 = x32] → fp32 域 `vexp` → `vsub`（标量字面量 1.0）→ 结果落位（fp32：`vsub(w32, 1.0, x_ub)` 跨缓冲写入载入 buffer；transit：`vsub(w32, 1.0, w32)` 原地后 `vcast(rint)` 降域回写 x_ub）→ UB→GM 拷出 `x_ub[0:tail]`（**两路线统一拷出源 = x_ub**，§1.5/§3.1/§3.3/§4.3/§4.4 五处交叉一致）。
3. **常量预备**：**无**——无 vcmp/无 α/无阈值常量；唯一标量 1.0 以 `T.vsub` 标量字面量操作数承载（fp32 值参与，零缓冲，mish L78 `T.vadd(buf, 1.0, buf2)` 同形先例）。
4. dtype 路径：fp32 原生域；fp16/bf16 fp32 中转（§0.6 R2——源 fp32-opmath 语义的结构化承接，PL-1.17 第五例）。

### 1.5 数据流图

```text
每 tile（persistent 串行，两路线结果落点分列）：
GM x[t0:t0+tail] ──T.copy──> x_ub(UB, 输入 dtype)

[fp16/bf16 路线 use_transit=True]
   x_ub ──vcast(rint)──> x32(UB, fp32，无损升域)
   x32 ──vexp──> w32(UB, fp32)                      # 全 buffer 计算；x≥88.73 lane 的
                                                    # exp 溢出 Inf = 期望输出语义（§1.6.3）
   w32 ──vsub(w32, 1.0, w32)──> w32                 # 标量字面量 1.0，原地
   w32 ──vcast(rint)──> x_ub（⑤ 单次舍回，复用载入 buffer）
   x_ub ──T.copy[0:tail]──> GM y

[fp32 路线 use_transit=False（无独立 x32，域即 x_ub，无 vcast）]
   x_ub ──vexp──> w32(UB, fp32)
   w32 ──vsub(w32, 1.0, x_ub)──> x_ub               # 标量字面量 1.0，跨缓冲写入：
                                                    # x_ub 原值已于 vexp 消费后死亡（RAW/WAR
                                                    # 由 Developer 自动栅栏保证），结果驻留 x_ub
   x_ub ──T.copy[0:tail]──> GM y
```

> 拷出源一致性（leaky_relu Stage 2 检视问题 2 教训，逐路线字面推演）：两路线的 UB→GM 拷出源**统一为 x_ub**——transit 路线经 ⑤ 降域回写 x_ub、fp32 路线经 vsub 跨缓冲写入 x_ub（**注意：fp32 路线的 vsub 不是原地形态，dst=x_ub ≠ src0=w32，是 vsub.md §2.4 示例一的标准三缓冲形态**）；不存在「结果驻留 buffer ≠ 拷出源」的组合（transit 结果驻留 w32 → ⑤ 回写 x_ub → ⑥ 拷 x_ub；fp32 结果直接驻留 x_ub → ⑥ 拷 x_ub）。**与 elu 的关键差异**：elu fp32 路线靠 vselect Out=A 原地写回实现统一；本算子无 select，fp32 路线以 vsub 的 dst 指向载入 buffer 实现统一——若误写为 `vsub(w32, 1.0, w32)` 原地 + 拷出仍用 x_ub，则 fp32 路线输出恒等输入（载入 buffer 未被更新，恒等映射错误）——§9.2 防呆条款。§1.4/§3.1/§3.3/§4.3/§4.4 五处交叉一致。

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

调研对象：同一数学语义 `y = expm1(x) = e^x − 1` 的算法族。算子特征判定（M0/Phase 1）：**单步逐元素**（无规约/窗口/矩阵）→ **轻量调研**（algorithm-research.md §2），四问逐问回答并给依据。

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1，此处只做初判；对照 algorithm-candidates.md ALG-elementwise 条目 R1 手段清单——公共子表达式消除/除法转乘倒数/cast 链合并/恒等变形——逐项对照 + 恒等变形自查 + 同类 examples 公式）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法/输入公式）** | `expm1(x) := exp(x) − 1`（fp32 域）：fp32 域 vexp+vsub = 2 op（manifest 口径 2 FLOPs/元素） | — | — | ✅（基线） |
| 1 | **Taylor 分段**（小 \|x\| 多项式直算 expm1） | `\|x\| < T` 时 Horner `x·(1 + x/2·(1 + x/3·(…)))`（T=0.5 需 k=2..9 共 8 项，余项 ≤ T¹⁰/10! = 2.7e-11 ≪ 2⁻²⁴），否则基线式；分支选择需 vabs+vcmp+vselect | **容差内等价**（数学恒等；舍入路径偏离 golden 的 expm1 直算——机器证据：fp32 randn 判定集 mism 496251→306906，viol 同为 0） | 小 \|x\| 相对误差 0.192@1e-7 → 5e-8（机器证据 §1.6.3.1 B 行） | ⚠️ 纳入对比；**评估后不进基线设计**：绝对误差已 ≪ atol（§1.6.1 裕度量化，viol 0→0 无测试可观测收益），代价 +19 op/元素（vabs+vcmp+vselect+16 mul/add）+ 阈值/系数缓冲——copy-floor ROI 判据否决（CONST-copy-floor-method：计算链 ≥16M 被 MTE2 窗口完全隐藏，op 数增加零墙钟收益、纯增 UB 与编译复杂度）；**保留为精度兜底杠杆**（若下游出现超容差相对精度需求时启用，机器证据已留存） |
| 2 | **fp64 中间量** | `round_D(expm1_f64(x))`（升 fp64 计算 + 单次降域） | 数学等价（理想精度——D-1 C 行近零 mismatch） | 消除相消误差 | ❌ 双重否决：① **npuir 无 fp64 向量计算/转换路径**——`T.vexp` dtype 矩阵 fp64 ×（docs/Tilelang.language/数学操作/T.vexp.md §2.2.1 亲读）、`T.vcast` 规格表无任何 f64 行（docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1 亲读）、`T.vbrc` dtype 列无 fp64（docs/Tilelang.language/shape操作/T.vbrc.md §2.2.1 亲读）；② manifest 契约 dtype 仅 fp16\|bf16\|fp32，fp64 输入在 wrapper 白名单层即被拒（§0.1 host 语义）。机器证据（C 行）仅作精度上界参照留存 |
| 3 | exp2 降代换 | `e^x = vexp2(x·log2e)`（T.vexp2 存在，docs/Tilelang.language/数学操作/ 目录亲核） | 数学恒等（浮点舍入另计） | 无 | ❌ +1 vmul（log2e 预乘）换 1 个 vexp，op 数不减反增；且 `x·log2e` 预舍入引入额外误差、偏离 golden 舍入路径——弃选 |
| 4 | 高精度补偿式（two-sum / Dekker 修复） | `r = exp(x)`，以补偿量恢复 `exp(x)−1` 的丢失低位：`expm1 ≈ (r_hi − 1) + r_lo` | 数学等价（需 FMA 或二次 vexp 求值提取残差） | 消除相消误差 | ❌ **无 FMA 类 v-op**——docs/Tilelang.language/数学操作/ 目录全 19 文件枚举亲核（vabs/vadd/vcos/vdiv/verf/vexp/vexp2/vln/vlog2/vmul/vpow/vrelu/vrsqrt 见 T.rsqrt/vsin/vsqrt/vsigmoid 见 T.sigmoid/vsub/vtanh/ceildiv——无 vfma/vmac 残差提取原语），二次 vexp 求值则 op 数 ≥ Taylor 候选且更复杂——弃选 |
| 5 | CSE 公共子表达式消除 | — | 结构上无重复子式（单 exp、单 sub、无常量重用） | 无 | ❌ 无 CSE 空间 |
| 6 | fp16 原生域链 | fp16 域 vexp(原生 √)/vsub 逐 op 舍入 | **容差内等价但偏离源路径**——源计算与舍回均在 fp32 域单次完成（§0.1），原生域逐 op 舍入 vs golden 的分歧形态同 PL-1.17 实证（leaky_relu/lerp 原生域 1–3 ulp 残差 vs transit 0 残差） | 省 2×vcast | ❌ golden/source 对齐优先（PL-1.17 先例）；bf16 原生域更不可行（vexp/vsub bf16 ×） |

R1 结论：**基线公式即公式级最简形态**（候选 1/4 精度改善但零测试可观测收益 + op 数大增、候选 2/4 受 API 缺失硬约束、候选 3 弃选有据、候选 5 无空间、候选 6 偏离源路径——全部候选评估后无进基线设计项）；cast 链已由 §0.6 R2 定为最小单次舍回（2×vcast，bf16 路线本就必需）。

**R2 在线算法**：**天然单遍，无需在线变体**。结构依据（algorithm-research §3 R2 判据）：输出 `y[i]` 仅依赖 `x[i]`，无 running 统计量（和/max/计数）、无跨元素依赖——逐元素映射本身即单遍流式；源算法亦为单遍（§0.4 无 online 手段），一致。收益口径不适用（无两遍/三遍结构可化）。

**R3 算法复杂度对比**（带宽受限算子——访存为主导项；口径：FLOPs 含每元素全部向量 op（manifest 2/元素口径）；Bytes 为 GM 流量含输入读+输出写；缓冲为 UB 峰值驻留）：

| 算法候选 | FLOPs（op/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------|---------------|---------|-------------|---------------------|
| 基线 `exp(x)−1`（+fp32 transit 落地形态） | manifest 口径 2（fp32 域 2 op 链；fp16/bf16 另加 2×vcast 属 dtype 路径） | **2N·B**（1 读 + 1 写，无中间 GM 往返——tile 驻留 UB）= **I/O 下界** | 1（每元素每张量恰读一次写一次） | O(block_size)：fp32 8B/elem、fp16/bf16 10B/elem（§4.5） | N/block_size 个完全独立 tile；persistent 零跨核同步 |
| 候选 1 Taylor 分段 | 21（+19：vabs+vcmp+vselect+8 项 Horner 16 mul/add） | 2N·B（同） | 1 | +1B/elem（cond bool）+ 系数/阈值缓冲 | 同上 |
| 候选 2 fp64 中间（假设 API 存在） | 2（域翻倍） | 4N·B 假设（fp64 流量翻倍，未实测假设） | 1 | ≥16B/elem | 同上 |

四口径结论：基线在 FLOPs 最少（候选 1 +19 op 属计算链优化——**PL-1.6 copy-floor 实证 ≥16M 档计算链被 MTE2 窗口完全隐藏（ROI≈0）**，op 增加零墙钟收益纯增 UB）、访存已达 I/O 下界、单遍、缓冲最小档。**基线唯一最优，无算法族替换**（与 leaky_relu/elu 调研同构结论）。

**R4 硬件亲和性评估**（逐候选对照检查清单，algorithm-research §3 R4；评估对象 = 基线 + transit 落地形态）：

| 检查项 | 基线（+transit 落地形态） | 结论 |
|--------|--------------------------|------|
| 计算单元匹配 | vexp/vsub/vcast 全为 Vector 单元 op（超越/算术/转换族）；无 MAC 密集段，无 Cube 需求 | ✅ 亲和 |
| 片上容量 | 工作集 = 8–10B/elem × block_size ≤ 8192 → ≤81.9KB < UB 192KB（§4.5）——**比 elu（19B）与 leaky_relu（19B）更松** | ✅ 有界 |
| 对齐/整除 | UB 切片零起点（官方 vec_add_1d 对齐模式）；尾轴对齐由 GM 侧连续 1-D + [0:tail] 拷出消化；v-prefix 全宽度 lane 并行（lane 宽度/lane 组大小为编译器实现细节，不作设计依赖），尾 tile 陈旧 lane 由拷出切片屏蔽、浪费 O(1)/tile | ✅ |
| 静态边界 | N 为 builder 工厂特化编译期常量 → num_logical/num_local_tasks/tail 上界全部 trace 期折叠（leaky_relu/elu 同款 host 常量折叠） | ✅ |
| 流水可融合性 | 单阶段 copy→compute→copy；Developer 自动多缓冲（§6.3） | ✅ |
| 跨核结构 | 无跨核归并（逐元素独立 tile），persistent 零核间同步 | ✅ |
| dtype 路径 | **bf16 无 vexp/vsub（T.vexp.md/T.vsub.md §2.2.1 bf16 ×，逐一亲读）→ fp32 transit 唯一路径；fp16 transit 为 golden/source 对齐（PL-1.17 四例先例）**；vexp 约束 src/dst 同形状同 dtype（T.vexp.md §2.1）→ 独立 w32 输出缓冲；vsub 约束操作数 buffer 在 UB（T.vsub.md §2.3.1）+ 标量仅 src1（§2.3.4）→ `vsub(w32, 1.0, dst)` 形态合法 | ✅（transit 形态） |

**调研结论（选定算法族 + 依据）**：

1. **选定**：基线公式 `expm1(x) = exp(x) − 1`（fp32 域）+ §0.6 重设计落地形态（persistent 一维分核 + UB tile 驻留 + fp32 域 vexp/vsub 链 + fp16/bf16 fp32 中转）。依据：R3 四口径基线唯一最优（访存达 2N·B 下界、FLOPs 最少）；R4 全项亲和；候选 1（Taylor）精度收益零测试可观测 + ROI 负、候选 2/4 API 缺失、候选 3/5/6 弃选有据。**结构差异（vs 源算法）**：算法族不变（同公式同 dtype 路径），实现结构由两级 SIMT 展开 + 寄存器 fragment → persistent 一维分核 + UB 向量链 + tile 级尾块。
2. **调研范围声明**（「无更优替代」举证）：① 参考表 algorithm-candidates.md ALG-elementwise 条目命中行全部评估（R1 手段清单见 R1 表）；② 恒等变形自查覆盖 Taylor 分段/fp64 中间量/exp2 降代换/高精度补偿式（two-sum）/CSE/fp16 原生域六族；③ 本仓同类实现 leaky_relu（persistent+transit 已验证形态）、elu（vexp+vsub 标量字面量生产先例 + 承接型 transit）、lerp_tensor（E1/E2 契约）、mish（vexp 生产 + 全 dtype fp32 域策略）、vec_add_1d（官方尾块模式）；④ pattern-library elementwise.md PL-1.6/PL-1.17（copy-floor 与 transit 实证，前者 STALE 已标注、后者 2026-10-28 重验存活）+ constants.md 带宽常数（STALE 条目按设计期估算处理）；⑤ R2 结构判据（逐元素天然单遍）；⑥ docs/Tilelang.language/ 数学操作/ 目录全 19 文件枚举（expm1 家族 API 本体即本算子实现目标、无更简超越原语；vexp2 存在但劣于 vexp——候选 3）。本地源已覆盖该族全部已知算法形态；未启用互联网检索（简单算子轻量调研，algorithm-research §6：通常无需联网）。
3. **设计期估算下界（D-2，协议 `_shared/standards/hardware-cost-model.md` §2）**：估算下界 = max(流量项, 发射项, 容量项)。
   - 流量项（主导）：引用 pattern-library/constants.md `CONST-mte2-degradation` 实测条目（**版本戳：tilelang 0.1.2+ed787bb / Ascend 910B2C / CANN 8.5.0 / 48 vector core；kb_stale_check 判定 STALE——按「设计期估算 + Stage 4 实测校准」口径引用**）——1M 档 33.9 GB/s/core、16M 档 30.2 GB/s/core、64M 档 20.9、256M 档 18.4。本设备 910B3 / 40 vector core（§5.5 实查），**跨设备缩放按核数比例折算，标注为设计期假设**（40/48 ≈ 0.83 → 1M ≈ 28.2、16M ≈ 25.2、64M ≈ 17.4、256M ≈ 15.3 GB/s/core）。
   - 发射项：2 op/元素（fp32）/ 4 op（fp16/bf16 transit 链：2 vcast + vexp + vsub）——lerp 7-pass 链在 ≥16M 已被 MTE2 窗口完全隐藏（`CONST-copy-floor-method` 实测 delta ≤1.1µs@≥16M；同条 STALE，同口径引用），2–4 op 链更短，同判隐藏，发射项 < 流量项。
   - 容量项：UB 驻留 ≤81.9KB < 192KB（§4.5），非约束（轻量调研·单步逐元素豁免，hardware-cost-model.md §2）。
   - **估算下界行**（设计期，待 Stage 4 实测校准；同族非 stale 实测锚点 = pattern-library/constants.md `PL-1.21-910b3-elementwise-baseline`（版本戳 tilelang 0.1.1.10+f388fab / CANN 9.0.0 / Ascend 910B3 / torch_npu 2.7.1.post4 / 2026-10-28，当前工具链非 stale））：

| workload | N | dtype | GM 流量 | 估算下界 | 同族实测锚（PL-1.21，leaky/elu） |
|----------|---|-------|---------|---------|-------------------------------|
| smoke-1m | 2²⁰ | fp32 | 8.4 MB | ≈ 7.4 µs | 17.38 / 18.94 µs |
| smoke-1m | 2²⁰ | fp16/bf16 | 4.2 MB | ≈ 3.7 µs | 11.42–12.34 / 11.18–12.12 µs |
| elementwise-16m [4096,4096] | 2²⁴ | fp16/bf16 | 67.1 MB | ≈ 55–67 µs | —（elu mlp-hidden 2²³ fp16 78.76µs 同量级折算 ≈157µs@2²⁴） |
| elementwise-64m [8192,8192] | 2²⁶ | fp16/bf16 | 268.4 MB | ≈ 385 µs | — |
| elementwise-256m [16384,16384] | 2²⁸ | fp16/bf16 | 1073.7 MB | ≈ 1758 µs | — |

   常数未覆盖量：910B3 的 MTE2 曲线（跨设备缩放假设）、1:1 R/W 混合（PL-1.6 为 lerp 3:2 混合）、L2 驻留档（bench harness 同输入复用可能命中 L2 口径，估算将偏保守）——**均为未实测假设，估算依据 = PL-1.6 曲线 × 核数比例；Stage 4 以 msprof 实测校准并回写**。
   - **canonical 估算下界行**（hardware-cost-model.md §2 格式；代表 workload = smoke-1m fp32，N=2²⁰）：**估算下界: ≈7.4 µs（流量 ≈7.4 µs = 8.4 MB ÷ ~1.13 TB/s〔1M 档 33.9 GB/s/core × 40/48 核数缩放，设计期假设〕 + 发射 ≈0〔2-op 链被 MTE2 窗口完全隐藏，CONST-copy-floor-method 实测 ≥16M delta ≤1.1 µs，STALE 条目按设计期估算口径引用〕 + 容量 0〔轻量调研·单步逐元素豁免，hardware-cost-model.md §2；UB 驻留 ≤65.5KB < 192KB〕，引用 CONST-mte2-degradation〔STALE〕；版本戳 tilelang 0.1.2+ed787bb / CANN 8.5.0 / Ascend 910B2C / 48c，当前工具链 88ee8c302e 需 Stage 4 重验；同族非 stale 锚点 PL-1.21 实测 17.38µs〔含 harness 开销〕——估算下界为纯流量地板，两者差异属口径差非矛盾）**。其余 workload 行见上表（同口径拆分：流量项主导、发射项被隐藏、容量项豁免）。

#### 1.6.1 数学等价优化（公式级）

**（a）公式级降算优化：采纳 0 项进基线设计；dtype 路径采纳 1 项（A1，实为源语义的结构承接）。**

依据：§1.6.0 R1 候选表 7 行逐一评估——候选 1（Taylor 分段）机器验证 0 容差违反但**与基线同为 0 违反**（viol 0→0，无测试可观测收益），绝对误差已 ≪ atol（§下方裕度量化），而代价 +19 op/元素 + cond/系数缓冲——copy-floor ROI 判据（CONST-copy-floor-method，pattern-library/constants.md L104：计算链被 MTE2 窗口完全隐藏）否决进基线，**机器证据留存、保留为精度兜底杠杆**（§9.1#2）；候选 2（fp64）API 缺失（T.vexp.md §2.2.1 fp64 × + T.vcast.md §2.2.1 无 f64 行 + T.vbrc.md §2.2.1 无 fp64 列，三处亲读）+ 契约 dtype 外，弃选；候选 3（exp2 降代换）op 数不降反增 + 额外预舍入，弃选；候选 4（高精度补偿式）无 FMA 类 v-op（docs/Tilelang.language/数学操作/ 目录全 19 文件枚举亲核）且二次求值代价 ≥ Taylor，弃选；候选 5 无 CSE 空间；候选 6（fp16 原生域）偏离源 fp32 域单次舍回路径，弃选（PL-1.17 先例）。基线访存已达 I/O 下界 2N·B、FLOPs 已是 manifest 口径最小（2/元素）、计算链在 ≥16M 档被 MTE2 窗口完全隐藏（PL-1.6 实测）——**公式级无进基线设计的降算/降访存空间**，按原式（源 fp32 域形态）设计。**机器证据归属**（gate 规范 `.agents/skills/_shared/standards/gate-and-retry.md` §1）：弃选/评估后不采纳候选（候选 1 Taylor / 候选 2 fp64）的机器证据行归档于 §1.6.3.1，本节仅保留采纳项 A1 的结果表（(b)）。

**核心精度风险点量化收口（调度 prompt 指定必答项）**：朴素 `exp(x)−1` 在 |x|→0 的相消误差——机器验证（verify_equiv.py [1] 段 band 表，11 个 \|x\| 区间 × 每区间 2 万样本双符号）实测：**|x| ≤ 0.1 域绝对误差被 ulp(1.0)/2 ≈ 6.0e-8 封顶**（减法由 Sterbenz 律在 exp(x) ∈ [0.5, 2] 精确，误差全部来自 exp 的舍入），相对误差虽发散至 1.0（x→0）但容差包络在小区间由 atol 项主导——**fp32 atol=1e-5 的最窄裕度 67×**（band (0.1, 1.0]，abs err 1.5e-7）；|x| ≥ 1e-4 域相对误差 ≤ 1.16e-7，**rtol=1e-5 裕度 ≥ 86×**。测试角点逐值判定：x=1e-7 → abs err 1.92e-8 ≪ atol（rel err 19.2% 被 atol 吸收）；x=88.8 → 双侧 +Inf 精确相等；x=−88.8 → cand −0.99999994 vs golden −1.0，差 5.96e-8 ≪ atol。**结论：fp32 1e-5 容差风险收口为 0 违反（54/54 行机器验证），残余风险仅 vexp 硬件 ULP（§9.1#1，裕度 ≥43×），风险等级由「核心风险」降为低。**

**（b）dtype 路径等价性（§0.6 R2 采纳项 A1，四要素 + 机器验证）**：

- **原式**（源基线，逐语句）：`v32 = f32(x)；y32 = exp(v32) − f32(1.0)；y = cast_D(y32)`（fp32 域计算 + 赋值单次舍回——D ∈ {fp16, bf16}；fp32 时全程原生域）。
- **优化后公式**（NPU transit 候选）：`x32 = rint(f32·x)（无损升域）；w32 = exp(x32)；w32 = w32 − 1.0；y = rint_D(w32)`（单次舍回）。fp32 路线无 transit（`w32 = exp(x_ub)；x_ub = w32 − 1.0`）。
- **等价性论证**：① f16/bf16 ⊂ f32，升变换逐值无损且保序；② fp32 域 `exp→sub` 与源表达式同序同域逐 op 单次舍入，中间量逐位一致；③ 末端 `rint_D(w32)` = 源赋值隐式舍回同一对象（唯一一次 D 域舍入）；④ fp32 dtype 无 transit（原生域 = 源域）。**验收口径声明**：cand 与 golden（torch.expm1，expm1 直算）是**不同舍入路径**（exp(x)−1 的相消 vs expm1 的保护位直算）——机器验证存在位型差异（fp32 randn 判定集 496251/2²⁰ lane ≤2 ulp）但**全部落在容差包络内**（viol=0）——本算子验收基准 = 容差包络 + 裕度量化，**不主张 bitwise**（与 leaky_relu/elu 的 0-ulp 主张差异如实声明，§8.2）。
- **收益量化**：非 FLOPs 收益——**正确性/精度收益**：bf16 路线唯一可行路径（vexp/vsub bf16 ×）；fp16 路线消除原生域逐 op 舍入分歧（PL-1.17 实证 transit 残差 0 vs 原生域 1–3 ulp），对齐源 fp32 域单次舍回语义消除 Stage 3 精度门禁返工风险（lerp 实证此类分歧 Stage 3 才暴露损失 2485s）；代价 +2 vcast/tile（fp16/bf16），bf16 路线本就必需。
- **机器验证（D-1，`verify_equiv.py` 已执行，torch CPU；判定集 = randn 2²⁰ + ×0.001/×0.01/×10 缩放集 + uniform[−89,89] + 角点集（±0/±88.8/±1e-7/±Inf/NaN/±max-finite/±min-subnormal/±min-normal/±2⁻²⁴/±1e-4/±0.5/±16/±88/±89，极值取自 torch.finfo）；容差判定含 `out == ref` 精确相等分支（Inf/Inf 防假违反）+ 双 NaN 相等；每 case 同批数据判定，模块级种子固定可复现）**：

| 采纳项 | dtype | 判定集 | cand vs golden mismatches | cand vs golden 容差违反 | max ulp（非相等有限 lane） | cand vs base（同式） | 结论 |
|--------|-------|--------|---------------------------|--------------------------|---------------------------|---------------------|------|
| A1 fp32-transit | float16 | randn | 440 | 0 | 1 | 逐位一致（同式恒等） | **EQUIV_PASS** |
| A1 fp32-transit | float16 | ×0.001（相消区） | 125745 | 0 | 32768* | 同上 | **EQUIV_PASS** |
| A1 fp32-transit | float16 | 角点 | 1 | 0 | 32768* | 同上 | **EQUIV_PASS** |
| A1 fp32-transit | bfloat16 | randn | 7 | 0 | 4 | 同上 | **EQUIV_PASS** |
| A1 fp32-transit | bfloat16 | ×0.001（相消区） | 6351 | 0 | 20054* | 同上 | **EQUIV_PASS** |
| A1 fp32-transit | bfloat16 | 角点 | 5 | 0 | 32768* | 同上 | **EQUIV_PASS** |
| A1 原生域 | float32 | randn | 496251 | 0 | 2 | 同上 | **EQUIV_PASS** |
| A1 原生域 | float32 | ×0.001（相消区） | 1047656 | 0 | 1314243271* | 同上 | **EQUIV_PASS** |
| A1 原生域 | float32 | 角点 | 10 | 0 | 2147483647* | 同上 | **EQUIV_PASS** |

  \* 带星号的 max ulp 为**跨零位型距离**（有序位空间中 +0.0 与负次正规/小值分居符号边界两侧的位距伪影）或相消区小值 lane 的位距——对应绝对差均 ≤ 6.1e-5（fp16）/ 1.6e-5（bf16）/ 6.1e-8（fp32），全部 ≪ 各自 atol；判定以 mismatches/容差违反为准（viol 全 0）。**汇总：9/9 代表行 EQUIV_PASS（脚本全量 54/54 行 0 违反，含 ×0.01/×10/ulin 判定集），无任何 FAIL 项。** GOLD-DIAG 诊断行（g_opmath vs f32-opmath 舍入路径）全 dtype 0 mismatch——证实 CPU golden `torch.expm1(x_D)` 即 fp32 opmath + 单次舍回路径（§0.1 语义解读的机器佐证）。

**优化结论**：采纳 A1——fp16/bf16 走 fp32 transit 单次舍回（机器验证 9/9 代表行 EQUIV_PASS、全量 54/54 行 0 违反，脚本 `verify_equiv.py`，结果表见上 (b)；候选 1 Taylor 精度兜底杠杆证据归档 §1.6.3.1）；公式级无进一步进基线设计的优化空间（R1 全部候选弃选有机器证据/API 负向证据/ROI 判据 + 基线访存已达 I/O 下界 2N·B + FLOPs 已最小），最终公式 = 源式 fp32 域形态 `expm1(x) = exp(x) − 1` + A1 dtype 路径，与 §1.4/§3.1/§4.4 一致。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案全部循环与标量计算点盘点（逐项判定）：

| # | 计算点 | 类型 | 替代方案 | API 佐证 | 结论 |
|---|--------|------|---------|---------|------|
| V1 | 逐元素 `exp(cast(x,f32)) − cast(1.0,f32)`（源算法核心计算） | 逐元素标量超越+算术 | `vexp` + `vsub(w32, 1.0, dst)` 全 buffer 向量链（2 op） | docs/Tilelang.language/数学操作/T.vexp.md（fp16/fp32 √）、数学操作/T.vsub.md（§2.1 src1 标量 / §2.3.3 原地 / §2.4 示例一三缓冲形态）；先例 mish L77–L78（vexp 生产链 + 标量字面量操作数）、elu 设计（vexp+vsub 同链） | **向量化 ✅** |
| V2 | `T.Parallel(threads_arg, npt_arg)` 二级展开 + `i·npt+j` fragment 索引算术（源并行结构） | 逐元素标量索引/展开 | v-prefix 全 buffer op 隐式 lane 并行（persistent + 无显式元素循环） | leaky_relu/elu/lerp 已验证形态；Vector 单元全宽度 | **向量化 ✅（消除）** |
| V3 | 标量常量 1.0（源 fp32 cast 常量） | 标量广播 | `T.vsub(w32, 1.0, dst)` 标量字面量操作数——**免常量缓冲**（无 vbrc 需求：本算子无比较/无 α，常量集合 = {1.0} 一个） | docs/Tilelang.language/数学操作/T.vsub.md §2.1（src1 tensor 或标量）+ mish L78 `T.vadd(buf1_ub, 1.0, buf2_ub)` 生产先例 | **向量化 ✅** |
| V4 | fp16/bf16→fp32 域转换（源 v32 cast 的向量形态）+ 赋值单次舍回 | 逐元素 cast | `T.vcast(..., round_mode="rint")` 全 buffer（升域 + 降域各一） | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 rint / bf16→f32 rint / f32→f16·bf16 rint） | **向量化 ✅** |
| V5 | persistent 核内 `for i in T.serial(num_local_tasks)`（tile 调度循环） | tile 级标量循环体 | 不可向量化——**理由**：任务映射循环（每迭代处理一个不相交 tile，循环携带 block_id/t0/tail 的 block 级索引算术，O(1)/tile 非逐元素热点）；tile 间无逐元素计算可合并 | — | **保留（理由：block 级索引/任务映射，可接受类别）** |
| V6 | `tail = T.min(block_size, N − t0)` / `block_id = i·num_kernels + cid` / `if block_id < num_logical` | 标量算术/分支 | 不可向量化——**理由**：block 级索引与动态边界处理（依赖 N 的尾块语义，无法静态化为全 tile——N 本身工厂特化常量但 tail 随 tile 变化；零起点切片方案已将逐元素分支静态化，此标量仅 O(1)/tile） | — | **保留（理由：动态边界 + block 级索引）** |
| V7 | host 侧 N 校验、block_size UB guard、output_dtype 校验、展平 reshape、lru_cache | host 标量 | 不在 kernel 内（host 元数据/契约校验/编译缓存） | — | **保留（理由：host 侧元数据）** |

**向量化结论**：7 个循环/标量计算点中 4 处以 v-prefix 向量操作等价替代（V1 逐元素超越+算术 → vexp+vsub 标量字面量、V2 二级展开 → v-prefix 全 buffer 消除、V3 标量常量 → vsub 标量操作数零缓冲、V4 dtype 域转换 → vcast，API 佐证见上表），3 处保留（V5 tile 调度循环——任务映射类 / V6 block 级索引与动态边界 / V7 host 元数据，逐项理由见上表，无"实现简单"式保留）；本结论与 §6 循环结构一致（tile 内无显式逐元素循环，v-prefix 全 buffer 形态）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

- **I/O layout（契约，非设计变量）**：输入/输出均为展平连续 1-D `(N,)`（host `contiguous().reshape(-1)`，Op 层契约；manifest 任意 ND 输入经 host 展平）。
- **候选枚举**：本算子核内仅一条数据轴（展平元素轴），**无第二非 batch 维**——属"纯 1-D 逐元素"单候选轴情形（豁免多轴枚举，理由：无广播轴、无归约轴、无窗口轴；ND→1D 展平在 host 视图完成，核内无重排自由度）。候选：①（唯一主选）**连续元素轴 lane 映射**——UB buffer `(block_size,)` 一维连续，v-prefix op 全宽度 lane 并行（lane 宽度/lane 组大小为编译器实现细节，本设计不依赖具体数值）；②（枚举即弃）核内重排布局——无收益场景（无跨轴访问、无 bank 冲突问题域），且引入 repack 代价，弃选（量化理由：重排仅为负收益，pattern-library/layout.md 转置链 µs 级开销纯增）。
- **向量化宽度与尾浪费**：v-prefix op 全 buffer 全宽度 lane 并行（设计依赖的仅是「整 buffer 计算 + `[0:tail]` 拷出」的屏蔽机制，官方 vec_add_1d 同款，不依赖具体 lane 宽度数值）；尾 tile `[tail:]` 陈旧 lane 由拷出切片屏蔽（正确性无碍，尾浪费 O(1)/tile）。**溢出 lane 语义附加说明**：vexp 对 x ≥ 88.73 的 lane 产生 +Inf——本算子**无 vselect 屏蔽**，该 Inf 就是期望输出（`expm1(88.8) = +Inf`，测试角点期望值即 Inf，golden 同为 +Inf）；尾 tile 陈旧 lane 的 vexp 溢出值/未初始化值同样被 `[0:tail]` 拷出切片屏蔽，无 GM 落点（elu「溢出 lane + 未初始化 lane 两类覆盖」教训同承）；L0-5 角点覆盖。
- **与 §3.3/§4/§6 一致性**：主选轴 = 展平连续轴，UB 驻留缓冲一维 `(block_size,)`，计算链无跨步系数——§3.3 伪代码、§4 缓冲表、§6 循环结构均按此布局（三方一致）。
- **实验裁决模式**：不触发——判定不依赖未实证常数（单候选轴；带宽常数引用 PL-1.6 已实测条目——STALE 条目按设计期估算处理并留 Stage 4 校准条款，非判定裕度依赖）。

**布局决策结论**：数据布局 = GM → UB（一维连续 `(block_size,)`，零起点切片）→ Vector 单元（v-prefix 全宽度 lane）→ UB（**结果统一落位载入 buffer x_ub：fp16/bf16 经 ⑤ 降域复用写回、fp32 经 vsub 跨缓冲写入**）→ GM；主选轴 = 展平连续元素轴（单候选轴、一维分块），NZ 分形布局不适用（纯逐元素无矩阵语义、无跨轴访问自由度）；尾 tile 陈旧 lane 浪费 O(1)/tile 由 `[0:tail]` 拷出屏蔽（lane 宽度为编译器实现细节）——与 §3.3 伪代码、§4 缓冲表、§6 循环结构三方一致。

#### 1.6.3.1 公式级弃选/评估后不采纳候选机器证据（非采纳项归档）

> **归属说明**：按 gate 规范（`.agents/skills/_shared/standards/gate-and-retry.md` §1），§1.6.1 仅保留采纳优化项的等价性验证结果表；弃选/评估后不采纳候选的机器证据行归档本小节。全部行由脚本 `examples/expm1/expm1_fwd_kernel/verify_equiv.py` 产出（torch CPU、模块级种子固定可复现，2026-10-28 执行，与 §1.6.1(b) A1 表为同一次执行）。

| 候选 | dtype | 机器证据摘要（verify_equiv.py） | 容差违反 | 佐证路径（仓库相对路径）或「未文档化」标注 | 处置 |
|------|-------|-------------------------------|---------|------------------------------------------|------|
| 候选 1 Taylor 分段（T=0.5，k=2..9） | float16 / bfloat16 / float32 | 全 dtype 全判定集 0 容差违反（viol 与基线同为 0）；小 \|x\| 相对误差 0.192@1e-7 → 5e-8（脚本 [3] 段量化表）；mismatches 较基线减少 40%（fp32 randn 496251→306906）但位型差异仍在（中段 \|x\| ∈ [0.5, 17] 的 exp 舍入差异不因 Taylor 消除） | 0（同基线） | `examples/expm1/expm1_fwd_kernel/verify_equiv.py`（cand_b + [3] 段）+ ROI 判据 pattern-library/constants.md `CONST-copy-floor-method`（L104，STALE 条目按设计期估算口径引用）+ pattern-library/elementwise.md PL-1.6（计算链隐藏实测） | **评估后不采纳**——0 测试可观测收益（viol 0→0）+ 19 op/元素代价；保留为精度兜底杠杆（§1.6.1(a)/§9.1#2） |
| 候选 2 fp64 中间量 | float16 / bfloat16 / float32 | 近零位型差异（fp32 randn mism 2737、max 1 ulp——精度上界参照）；全判定集 0 容差违反 | 0 | **API 负向证据（三处亲读）**：docs/Tilelang.language/数学操作/T.vexp.md §2.2.1（fp64 ×）、docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（规格表无任何 f64 行）、docs/Tilelang.language/shape操作/T.vbrc.md §2.2.1（dtype 列无 fp64）；契约约束 manifest elementwise.yaml L193（dtype 仅 fp16\|bf16\|fp32） | **弃选（API 缺失 + 契约外）**——机器证据仅作精度上界参照留存 |

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（Stage 0 prompt 显式指定；落地侧 wrapper `ascend_mode = "Developer"`（其 L130）同证）。

### 2.2 选型理由

1. 调度 prompt 契约 + wrapper `ascend_mode` 契约（最高优先级，双源一致）。
2. 算子形态匹配：单阶段 copy→compute→copy 逐 tile 流水，无跨核协作、无手动流水插桩需求——Developer 自动同步/自动多缓冲已覆盖（同族先例 leaky_relu/elu/lerp/mish 同判）。
3. v-prefix API（vcast/vexp/vsub）均为 Developer Op 形态（docs/Tilelang.language/ 各 §1 标注 `[Developer Op]`——T.vcast.md L8 明示双形态、T.vexp.md/T.vsub.md 示例即 Developer 形态）。

### 2.3 模式影响

- 同步：编译器自动插入 UB 读写栅栏，无手动 `set_flag/wait_flag`（§7）。
- 多缓冲：auto multi-buffer（Stage 2 编译期自动 stage 化），PL-1.6 实证可将 7-op 计算链完全隐藏于 MTE2 窗口——本算子 2-op/4-op 链更短，同判加强。
- 模块头 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`（leaky_relu/elu 同款，防 Expert/Developer 混跑进程互相污染——kernel_base CG-2026-0010 同源问题域）。

## 3. API 映射设计

### 3.1 公式拆解

优化后公式（= §1.6.1(b) A1 采纳式 + §1.6.0 选定落地形态）拆解为逐 tile 原语序列：

```text
D = dtype（fp16|bf16|fp32）；域输入 = x32（transit）或 x_ub（fp32，x32 ≡ x_ub）
① 载入    x_ub[0:tail]      ← GM x[t0:t0+tail]
② 升域    x32[0:tail]       ← rint(fp32 ← x_ub)            （仅 fp16/bf16）
③ 指数    w32[0:tail]       ← exp(域输入)                   （全 buffer；x≥88.73 lane 溢出 Inf = 期望输出语义）
④ 减一    [fp32]    x_ub[0:tail] ← w32 − 1.0（标量字面量，跨缓冲写入——vsub.md §2.4 示例一三缓冲形态）
          [transit] w32[0:tail] ← w32 − 1.0（标量字面量，原地——vsub.md §2.3.3）
⑤ 降域    x_ub[0:tail]      ← rint(D ← w32)                 （仅 fp16/bf16；复用载入 buffer）
⑥ 写出    GM y[t0:t0+tail]  ← x_ub[0:tail]                  （两路线统一拷出源 = x_ub，§1.5/§3.3/§4.3/§4.4 一致）
预备：无常量广播步骤（无 vcmp/无 α/无阈值——唯一标量 1.0 由 ④ 的 vsub 标量字面量承载，零缓冲）
```

FLOPs：vexp+vsub = 2/元素（manifest 口径）；fp16/bf16 另 +2 vcast（dtype 路径代价，非算法 FLOPs）。

### 3.2 TileLang API 映射

| 语义步骤 | API（v 前缀优先） | 关键参数 | 文档佐证（亲读） |
|---------|------------------|---------|-----------------|
| ① 载入/⑥ 写出 | `T.copy` | `T.copy(x[t0:t0+tail], x_ub[0:tail])` 零起点切片 | docs/Tilelang.language/内存操作/T.copy.md；官方 vec_add_1d / leaky_relu / elu 先例 |
| ②⑤ 域转换 | `T.vcast` | `round_mode="rint"`（f16→f32 / bf16→f32 仅 rint；f32→f16/bf16 rint） | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1 |
| ③ 指数 | `T.vexp` | `T.vexp(x32, w32)`（fp32 路线 `T.vexp(x_ub, w32)`）——src/dst 同形状同 dtype（§2.1 约束 → 独立 w32）；fp16 √/fp32 √/bf16 ×/fp64 ×（bf16 经 transit 后仅 fp32 域使用） | docs/Tilelang.language/数学操作/T.vexp.md §2.1/§2.2.1 |
| ④ 减一 | `T.vsub` | `T.vsub(w32, 1.0, x_ub)`（fp32，跨缓冲）/ `T.vsub(w32, 1.0, w32)`（transit，原地）——src1 标量字面量（仅 src1 可标量，§2.3.4）；dst 可为任意 UB buffer（§2.4 示例一）或 src0（原地，§2.3.3）；标量操作数须在 T.Kernel 内定义（§2.3.2） | docs/Tilelang.language/数学操作/T.vsub.md §2.1/§2.2.2/§2.3/§2.4；mish L78 标量字面量生产先例 |
| 分配 | `T.alloc_ub` | `(block_size,)` 各缓冲；**无 bool cond 缓冲**（无 vcmp） | docs/Tilelang.language/内存操作/T.alloc_ub.md |
| 核/循环 | `T.Kernel(num_kernels, is_npu=True)` / `T.serial` / `T.ceildiv` / `T.min` | persistent 一维分核 | leaky_relu / elu / lerp_tensor 先例；docs/Tilelang.language/数学操作/T.ceildiv.md |

### 3.3 计算伪代码

```python
def expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=None,
                     threads=256, num_per_thread=8):
    # 最终工厂签名（§3.5.2 建议）：(N, dtype, output_dtype) 源接口保留；
    # block_size = K11 折叠新参（None → dtype 默认，wrapper L81 现状调用面 1:1 兼容）；
    # threads/num_per_thread 保留于签名但 NPU 侧不消费（源签名保真 + lru_cache 键空间，elu 同款）。
    # host 校验（builder 期；ValueError 非 assert，python -O 安全）：
    #   N <= 0            -> ValueError（§9.3 防御，源无此项）
    #   output_dtype 非 None 且 != dtype -> ValueError（manifest same_as(input)）
    #   dtype 归一化后不在 {float16,bfloat16,float32} -> ValueError
    # dtype 入参归一化（leaky_relu RETROSPECTIVE 教训）：wrapper custom_op 直传 dtype_str（其 L81）、
    #   Kernel 类直传 self.dtype_str（其 L155——str 形态），直调可传 torch.dtype——
    #   torch 2.7.1 的 torch.dtype 无 .name 属性，str(torch.float16) == "torch.float16"：
    dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]
    use_transit   = dtype_key in ("float16", "bfloat16")       # §0.6 R2
    default_block = block_size or (1024 if dtype_key == "float32" else 2048)
    vector_cores  = NPUUtils.get().get_aicore_num() * 2        # 实查，纯 Vector 翻倍（§5.5）

    # host 编译缓存：@functools.lru_cache(maxsize=32) 施加于本外层 builder 函数
    # （提取件 L13 自带，直接沿用；键 = (N, dtype, output_dtype, block_size, threads, num_per_thread)）

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(bs=default_block):                              # E2 单参契约（K11；leaky_relu L194 同形；
                                                               # 工厂级 block_size 传入时成为此默认值——wrapper L81 兼容）
        # host guard（kernel 期）：
        #   bs <= 0 或 > 8192 -> ValueError（§4.5 UB 预算）
        num_logical     = ceildiv(N, bs)                       # 编译期常量
        num_kernels     = min(num_logical, vector_cores)       # ≤ 物理核
        num_local_tasks = ceildiv(num_logical, num_kernels)    # 静态边界

        if use_transit:                                        # 工厂期 Python if（leaky_relu L218 同款，规避 C11）
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((bs,), dtype_key)        # 载入 + ⑤ 降域回写目标
                    x32  = T.alloc_ub((bs,), "float32")        # 升域输入
                    w32  = T.alloc_ub((bs,), "float32")        # exp→sub 工作缓冲
                    for i in T.serial(num_local_tasks):        # 静态边界 grid-stride
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0   = block_id * bs
                            tail = T.min(bs, N - t0)           # 尾块（静态上界）
                            T.copy(x[t0:t0+tail], x_ub[0:tail])   # ①
                            T.vcast(x_ub, x32, round_mode="rint") # ② 无损升域
                            T.vexp(x32, w32)                      # ③ 全 buffer
                            T.vsub(w32, 1.0, w32)                 # ④ 标量字面量，原地
                            T.vcast(w32, x_ub, round_mode="rint") # ⑤ 单次舍回复用载入 buffer
                            T.copy(x_ub[0:tail], y[t0:t0+tail])   # ⑥ 拷出源 = x_ub
        else:                                                  # fp32 原生域（无 vcast，域即 x_ub）
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    x_ub = T.alloc_ub((bs,), dtype_key)        # 载入 + ④ 结果落位 buffer
                    w32  = T.alloc_ub((bs,), "float32")
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0   = block_id * bs
                            tail = T.min(bs, N - t0)
                            T.copy(x[t0:t0+tail], x_ub[0:tail])   # ①
                            T.vexp(x_ub, w32)                     # ③（域输入 = x_ub）
                            T.vsub(w32, 1.0, x_ub)                # ④ 跨缓冲写入 x_ub（三缓冲形态，
                                                                  #    vsub.md §2.4 示例一；x_ub 原值死于 ③）
                            T.copy(x_ub[0:tail], y[t0:t0+tail])   # ⑥ 拷出源 = x_ub（两路线统一）
        return main

    return kernel
```

注：① 两 dtype 路线在工厂期以 Python if 静态选择（`use_transit` 为闭包常量，整支 prim_func 分置——leaky_relu L218–L309 同款结构，规避 TVM-script if 块作用域限制 C11；全链无 in-body Python if）。② **fp32 路线不分配独立 x32**——vexp 以载入 buffer x_ub 为域输入，vsub 结果跨缓冲写回 x_ub（标准三缓冲形态，非原地——与 elu 的 vselect Out=A 原地写回不同，本路线 dst=x_ub ≠ src0=w32，无原地别名问题），⑥ 拷出源与 transit 路线统一为 x_ub。③ transit 路线 ④ 为原地形态（Out≡src0=w32），结果仅同迭代被 ⑤ 消费——跨迭代携带禁令见 §9.2（无 loop-carried 状态，每 tile 由 ① 全量重载、③ 全量重算）。④ **无 vbrc/vcmp/vselect**——无条件公式 + 标量字面量承载常量（比 elu 少 3 op + 3 buffer）。⑤ 标量 1.0 为 Python float 字面量（fp32 精确可表示），mish L78 同形先例；不使用工厂作用域变量绑定（vbrc rank 陷阱不适用于字面量，但统一用字面量最简形态规避该问题域）。

### 3.4 API 可行性确认

| API | 存在性证据 | 形态先例（examples/） |
|-----|-----------|----------------------|
| T.vexp（fp32，src/dst 同形状同 dtype） | docs/Tilelang.language/数学操作/T.vexp.md §2.1/§2.2.1（fp32 √，fp16 √，bf16 ×，fp64 ×） | mish L77 `T.vexp(x_f32_ub, buf1_ub)`（生产）+ elu 设计同链 |
| T.vsub（标量 src1 + 三缓冲/原地双形态） | docs/Tilelang.language/数学操作/T.vsub.md §2.1（src1 标量）/§2.2.2（标量广播）/§2.3.3（原地）/§2.4（示例一三缓冲 + 示例二原地） | mish L78 `T.vadd(buf1_ub, 1.0, buf2_ub)`（标量字面量先例）；T.vsub.md §2.4 示例一（三缓冲）/示例二（原地） |
| T.vcast rint | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 rint / bf16→f32 rint / f32→f16·bf16 rint） | leaky_relu/lerp/mish/elu transit 链（四任务实证 PL-1.17） |
| T.alloc_ub | docs/Tilelang.language/内存操作/T.alloc_ub.md | 同族全 UB 链先例 |
| T.copy 零起点切片 | docs/Tilelang.language/内存操作/T.copy.md | vec_add_1d（官方尾块）、leaky_relu/elu |
| T.Kernel(is_npu=True) + T.serial persistent | leaky_relu / elu / lerp_tensor | persistent 分核已验证形态 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（ascend-constraints.md 强制检测项）

| 检测项 | 结果 |
|--------|------|
| 三维 Kernel | ✅ 不涉及——一维 `T.Kernel(num_kernels)` |
| L0C 容量 | ✅ 不涉及——纯 Vector，无 Cube/L0C 路径 |
| GEMM 非整除 | ✅ 不涉及——无 GEMM；逐元素尾块另有 tail 方案（§6.4） |
| L1 分形限制 | ✅ 不涉及 |
| UB 容量 | ✅ 有界——10B/elem × 8192 = 81.9KB < 192KB（§4.5）；host guard 拒绝超限 block_size |
| bool/i1 写 GM | ✅ 不涉及——无 cond buffer（无 vcmp）；vexp/vsub/vcast 输出均为 fp32/原生 dtype |
| bf16 v-算术/超越缺失 | ✅ 绕开——transit 后全部计算在 fp32 域（§0.6 R2；T.vexp.md/T.vsub.md bf16 × 已逐一亲读） |
| fp64 缺失 | ✅ 不涉及——契约 dtype 仅 fp16/bf16/fp32；fp64 候选已弃选（§1.6.0 R1#2 三重负向证据） |
| 一维 Kernel 核数超发 | ✅ persistent 分核（§5.5） |
| in-place 参数 | ✅ 不涉及——expm1 无 inplace 契约；kernel 始终写独立 y |

#### 3.5.2 参考实现差异说明（含 wrapper 调用面契约分析——本设计关键差异化决策点）

**wrapper 调用面现状（契约权威，逐字引用）**：

- `examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py` L81（custom_op `npub::expm1_fwd` 内）：
  `return expm1_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)`
- 同文件 L155（Kernel 类 `__init__` 内）：
  `self.kernel = expm1_fwd_kernel(self.N_total, self.dtype_str)`

**契约分析**：

1. **`(x, N)` 调用式与 jit callable 契约不兼容**：`@tilelang.jit(out_idx=[1])` 下返回 callable 的形参 = prim_func 非输出参数 = `(x)`（main 签名 `(x, y)`、y 为 out_idx）——`(x, N)` 的第二实参无法绑定（TypeError）。同族已验证调用面均为 `(x)`：leaky_relu（`_make_leaky_relu_kernel(N, dtype, negative_slope)(block_size)(x)`，其 L98）、elu（其 L100）、lerp_tensor。
2. **工厂级 `block_size=` 关键字**：wrapper L81 将 block_size 传入工厂（而 leaky_relu/elu 传入 curried 层）——此调用面本身可兼容（工厂接受 block_size 参数并使其成为返回 kernel 的默认 bs）。
3. **`dtype_str` 直传工厂**：wrapper L81/L155 直传 str（而 leaky_relu/elu custom_op 先 `getattr(torch, dtype_str)` 转 torch.dtype）——工厂内做 str/torch.dtype 双形态归一即可兼容，wrapper 无需改。
4. **L155 两参构建**：工厂 block_size 缺省（None → dtype 默认）时合法——无需改。

**建议的最终 kernel 工厂签名**：

```python
@functools.lru_cache(maxsize=32)
def expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=None,
                     threads=256, num_per_thread=8):
    # -> kernel(bs=default_block) -> main(x)   （main(x, y)，out_idx=[1]）
```

- `(N, dtype, output_dtype)`：源接口三参保留（迁移规则「不要改变接口的参数」）。
- `block_size=None`：K11 折叠新参（替代 GPU `threads×npt` 积——后端相关参数适配，迁移规则允许）；None → dtype 默认（fp32 1024 / fp16·bf16 2048 = wrapper `_BLOCK_SIZE_*`）；**传入时成为返回 kernel 的默认 bs → wrapper L81 现状调用面 1:1 兼容**。
- `threads=256, num_per_thread=8`：保留于签名但 NPU 侧不消费（源签名保真 + lru_cache 键空间一致，elu §10.3 同款先例——K11 已折叠为 block_size）。
- 返回值：curried `kernel(bs=default_block)` 单参形态（E2 家族契约，支持 L0 双配置复用：同一工厂先后以两个 bs 调用）。

**wrapper Stage 5 适配清单**（标注：**需要适配，共 1 处必需 + 0 处可选**）：

| # | 位置 | 现状（逐字） | 适配后 | 必要性 |
|---|------|-------------|--------|--------|
| 1 | expm1.py L81 | `expm1_fwd_kernel(N, dtype_str, block_size=block_size)(x, N)` | `expm1_fwd_kernel(N, dtype_str, block_size=block_size)(x)` | **必需**——`(x, N)` 在 jit callable 契约（out_idx=[1] + main(x,y)）下 TypeError；仅删除尾部 `, N` 一个 token，工厂调用面（含 block_size 关键字与 dtype_str 直传）保持不变 |
| 2 | expm1.py L155 | `expm1_fwd_kernel(self.N_total, self.dtype_str)` | 不变 | 无需——工厂 block_size 缺省合法 |
| 3 | dtype 传递形态 | custom_op 直传 `dtype_str` | 不变 | 无需——工厂内 str/torch.dtype 双形态归一（§3.3） |

**参考实现差异表**：

| 差异点 | 源（GPU） | NPU 设计 | 理由 |
|--------|----------|---------|------|
| 并行组织 | `T.Parallel(threads, npt)` 两级 SIMT | persistent 一维分核 + v-prefix 全 buffer | §0.5 #4（一维 Kernel 限制） |
| 片上驻留 | register fragment | UB | §0.5 #2 |
| dtype 域 | 全程 fp32 域 + 赋值单次舍回 | **fp16/bf16 fp32 中转单次舍回（5 步链）**——逐位同构承接源路径；fp32 原生域（3 步链） | §0.6 R2（bf16 vexp/vsub 缺失 + golden/source 对齐，D-1 0 违反） |
| 尾块 | 整块 fragment（依赖 block_size 整除 N 的 grid 收尾） | tile 级 tail 切片（覆盖非整除——严格强于源） | §0.5 #8（尾块语义行） |
| E2 工厂签名 | `kernel(threads_arg, npt_arg)`（GPU 基类 `init_config` 两参依赖——**源侧事实**） | **`kernel(bs=default_block)` 单参 curried + 工厂级 block_size 兼容**（K11 折叠；默认 fp32 1024 / fp16·bf16 2048 = wrapper `_BLOCK_SIZE_*`；wrapper L81 现状工厂调用面 1:1 兼容） | NPU harness wrapper 现状为契约权威（L81/L155 亲读）；`(x, N)` 尾参在 jit callable 契约下非法 → Stage 5 适配 1 token（见上清单）；threads/npt 属迁移规则「后端相关参数可适配」类目；leaky_relu/elu/lerp 已验证件同形 |
| dtype 入参形态 | GPU 基类经 builder args 传 dtype_str | **str 与 torch.dtype 双形态接受并归一**：`dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]`（torch 2.7.1 无 `.name`），transit 分支/guard/默认 block 一律基于 `dtype_key` | wrapper L81/L155 直传 str；torch 对象与 str 元组直接 `in` 比较恒 False，会误走 fp32 路线（leaky_relu Stage 2 检视问题 1④ + Stage 3 复核教训） |
| 标量参数 | 无 | 无（1.0 由 vsub 标量字面量承载） | 源契约即无标量参 |
| host 编译缓存 | 提取件自带 `@lru_cache(32)`（L13） | 同款沿用 | 提取件事实（与 elu 提取件不同，本件 lru_cache 未被剥离） |

#### 3.5.3 本项目同类实现参考

1. `examples/leaky_relu/_make_leaky_relu_kernel/_make_leaky_relu_kernel.py` —— 同族（单输入逐元素）已验证迁移：persistent 分核、fp32 transit、UB guard、L0 六项套件、E2 单参工厂契约（**结构最接近模板**；本设计在其结构上增加 vexp 计算级、去掉 vcmp/vselect/常量广播三级）。
2. `examples/elu_fwd/_make_elu_kernel/` —— **最近同族先例**：vexp 生产链 + 标量字面量 vsub 操作数 + fp32-opmath 源「结构承接」型 transit（与本算子同型）+ 承接型 D-1 验证纪律；其 RETROSPECTIVE Transferable Lessons 已逐字纳入 §9.2。
3. `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` —— **vexp 生产先例**（L77）+ **标量字面量 v-op 操作数先例**（L78 `T.vadd(buf1_ub, 1.0, buf2_ub)`）+ 全 dtype fp32 域计算策略 + 尾块切片。
4. `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` —— E1/E2 单参工厂契约 `_make_X(N, dtype) -> kernel(block_size=default_block) -> main` 同形模板；`perf_opt/` 子目录为 persistent nc 调优标杆（Stage 4 参考）。
5. `examples/elementwise/vec_add_1d.py` —— 官方一维逐元素 + 尾块零起点切片模式。
6. `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` —— NPUUtils 物理核数查询与 per-factory 重查先例。
7. verbatim 源件：`examples/TileOPs/tileops/kernels/elementwise/expm1/_expm1_fwd_kernels.py`。

## 4. 数据规格与内存规划

### 4.1 输入张量

| 项 | 值 |
|----|----|
| 符号/形状 | `x: (N,)`，N = product(原始 shape)，host 展平连续视图 |
| dtype | float16 \| bfloat16 \| float32（manifest；越界 ValueError） |
| 布局 | 连续 1-D（`contiguous()` 保证），32B 对齐由 GM 分配保证 |

### 4.2 输出张量

| 项 | 值 |
|----|----|
| 符号/形状 | `y: (N,)`，shape == input.shape（shape_rules），dtype same_as(input) |
| 生成方式 | kernel 内新 buffer（`out_idx=[1]`） |

### 4.3 中间缓冲区（全部 UB，`(block_size,)` 一维）

| 缓冲 | dtype | 字节/元素 | 角色 | 复用 |
|------|-------|----------|------|------|
| `x_ub` | 输入 dtype | 2（fp16/bf16）/ 4（fp32） | 载入 + 降域回写目标（transit）/ **④ 减一结果落位 buffer（fp32 路线，vsub 跨缓冲写入）** | 载入后原值死于 ③ vexp 消费；两路线 ⑥ 拷出源均为 x_ub |
| `x32` | fp32 | 4 | 升域输入（仅 transit 路线分配；fp32 路线无独立 x32，域即 x_ub） | — |
| `w32` | fp32 | 4 | exp→sub 工作缓冲（③ vexp 输出受 §2.1 同 dtype 约束需独立缓冲；④ 原地推进，结果仅同迭代被 ⑤ 消费） | vsub 原地写回（迭代内消费，§9.2） |
| `cond`/常量缓冲 | — | — | **无**——无 vcmp（无条件公式）、无 vbrc（1.0 由 vsub 标量字面量承载） | 比 elu 少 3 buffer（cond/zero32/alpha32） |

### 4.4 内存搬运路径

```text
[fp16/bf16] GM x ──MTE2──> UB(x_ub) ──vcast──> UB(x32) ──vexp──> UB(w32) ──vsub(1.0, 原地)──> UB(w32) ──vcast──> UB(x_ub) ──MTE2──> GM y
[fp32]      GM x ──MTE2──> UB(x_ub) ──vexp──> UB(w32) ──vsub(1.0, dst=x_ub 跨缓冲)──> UB(x_ub) ──────────────────────────────> GM y
```

纯 Vector 路径：GM↔UB 直连（MTE2/MTE3），无 L1/L0 介入；计算全在 Vector 单元 + UB。

### 4.5 UB 内存预算

| dtype | 缓冲字节/元素 | block_size 上界（196608B UB） | 保守 cap | 默认 block（E2 契约） |
|-------|--------------|------------------------------|---------|----------------------|
| float32 | 4+4 = **8B**（fp32 路线无独立 x32） | 24576 | **8192** | K11 折叠默认 = **1024**（wrapper `_BLOCK_SIZE_FP32`） |
| float16/bfloat16 | 2+4+4 = **10B** | 19660 | **8192** | K11 折叠默认 = **2048**（wrapper `_BLOCK_SIZE_NON_FP32`） |

- host guard：`bs > 8192 → ValueError`（leaky_relu 同款 guard 形态，`python -O` 安全用 ValueError 非 assert）。
- auto multi-buffer（Developer 默认）的 UB 占用口径：**本算子在一切已知口径下安全**——朴素 ×2 模型（最悲观）：fp32 2×8×8192 = 131KB < 192KB ✓；fp16/bf16 2×10×8192 = 163.8KB < 192KB ✓；TRAP-UB-multibuffer-inflation 平台口径（pattern-library traps-compiler.md，status=verified：footprint 与 staging 数解耦的结构类平台值 + 系数结构依赖，基础分配偏差系数 1.10–1.12——**引用 pattern-library 平台值口径，不自设 ×2 线性模型**，同族 RETROSPECTIVE 教训）：fp32 8×8192×1.12 ≈ 73.4KB、fp16/bf16 10×8192×1.12 ≈ 91.8KB，均 ≪ 192KB ✓。默认 block（1024/2048）双缓冲峰值 16.4KB / 41KB 更安全。
- **guard 8192 保留**（同族同形 guard 先例；本算子字节预算远松于 leaky_relu/elu 的 19B——8192 档余量充足）；Stage 4 如需上调 block（如 16384），fp32 路线朴素 ×2 = 262KB 超限，必须先以编译探针/BishengIR `ub overflow requires` 报文复算并连带复验 num_kernels（风险点 §9.1#4）。

### 4.6 动态轴定义

- 唯一动轴：N（展平元素总数）。处理方式：**builder 工厂按 N 特化**（N 闭包常量 + `lru_cache`；非 kernel 内动态 shape）——N 为编译期常量，num_logical/num_local_tasks/tail 上界全部 trace 期折叠；任意 N 的尾部由 §6.4 tail 方案覆盖。与源 Op 契约一致（`N_total` constructor 参数、每 N 一 kernel）。

### 4.7 JIT 配置

- `@tilelang.jit(out_idx=[1], target="npuir")`；E2 工厂 `kernel(bs=default_block)`（**单参 curried**，K11 折叠；default_block 按 dtype fp32 1024 / fp16·bf16 2048 = wrapper `_BLOCK_SIZE_*`，工厂级 block_size 传入时覆盖——wrapper L81 兼容，§3.5.2）。bs 经工厂闭包在 trace 期折叠为 grid / UB 尺寸编译期常量。
- `TILELANG_ASCEND_MODE=Developer`（模块头 setdefault）。
- pass_configs：默认（无特殊项；Stage 4 可评估 auto multi-buffer stage 数）。

## 5. Tiling 策略

### 5.1 计算类型

纯 Vector（逐元素超越/算术/转换），无 Cube 参与；仅需 UB，无 L1/L0A/L0B/L0C。无 CV 融合（§8.2 说明）。

### 5.2 Block 划分

- **block_size（tile 内元素数）= E2 工厂单参注入**（K11：GPU threads×npt 积折叠为单参 block_size，§3.5.2；不再由 threads_arg×npt_arg 计算；工厂级 block_size 传入时成为默认值——wrapper L81 兼容形态）。
- 默认：fp32 1024、fp16/bf16 2048（= NPU wrapper `_BLOCK_SIZE_FP32`/`_BLOCK_SIZE_NON_FP32`，同源于源 default_config threads=256, npt=4/8 的折叠积）；UB cap 8192（§4.5）。
- 逻辑 tile 网格：1-D，`num_logical = ceildiv(N, block_size)`。

### 5.3 约束分析

- 非 GEMM：无非整除 M/N 分形问题；尾部处理统一走 tile 级 tail（§6.4）。
- UB 预算约束 block_size ≤ 8192（§4.5 表）。
- 核数约束：num_kernels ≤ 物理核（§5.5）。

### 5.4 注意事项

- block_size 为 32B/elem 组（fp32 8 元素、fp16/bf16 16 元素）非整除时，尾 tile 由 [0:tail] 拷出消化——block_size 本身建议保持 2 的幂（默认 1024/2048 满足）。
- manifest workload 与默认 block 的任务数见 §5.5 表。

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1 三要素表（依据 docs/开发指南.md §3.3）。

**要素② 物理核数依据（实查记录，2026-10-28 本设计独立复跑）**：

- 目标设备 **Ascend 910B3**。本设计环境 `import tilelang` 因 dev root 私有 TVM 包缺失不可用，无法直接调用 `NPUUtils.get().get_aicore_num()`；采用**等价实查路径**：ctypes 直调 CANN runtime `libruntime.so` 的 `rtGetAiCoreCount`——与 `NPUUtils` 内部实现**同一运行时调用**（`tilelang/utils/npu_utils.cpp` `getAiCoreNum()` 即封装 `rtGetAiCoreCount`，elu 任务亲读核对）。查询代码与返回值：

  ```python
  import ctypes
  lib = ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so')
  cnt = ctypes.c_uint32()
  ret = lib.rtGetAiCoreCount(ctypes.byref(cnt))   # ret = 0 (RT_ERROR_NONE)
  # → cnt.value = 20
  ```

- **实查结果：aicore_num = 20**；纯 Vector 算子核数翻倍 → **vector_cores = 40**（core-split-strategy.md §1 要素②翻倍规则）。
- 参照注：同族记录 ada_layer_norm 注释与本机 910B2C 上曾测 24/48——**核数随设备变化，kernel 工厂期按 `NPUUtils.get().get_aicore_num() * 2` 现查现用（部署自适应），本表数值仅对本设备有效**。

**要素① 逻辑核数计算**：`num_logical = ceildiv(N, block_size)`，按 manifest workload（默认 block：fp32 1024 / fp16·bf16 2048）：

| workload | N | dtype（block） | num_logical |
|----------|---|---------------|-------------|
| smoke-1m [1048576] | 2²⁰ | fp32（1024） | 1024 |
| smoke-1m [1048576] | 2²⁰ | fp16/bf16（2048） | 512 |
| elementwise-16m [4096,4096] | 2²⁴ | fp32（1024） | 16384 |
| elementwise-16m [4096,4096] | 2²⁴ | fp16/bf16（2048） | 8192 |
| elementwise-64m [8192,8192] | 2²⁶ | fp16/bf16（2048） | 32768 |
| elementwise-256m [16384,16384] | 2²⁸ | fp16/bf16（2048） | 131072 |

**要素③ 规模判定与分核方案**：**极大规模**——全部 manifest workload 的 num_logical（512–131072）均 > 物理 Vector 核数 40，且无法通过增大 block_size 缩减到 ≤40（UB cap 8192 → smoke-1m fp16 已是 512 个 tile，2²⁸ 需 32768 个，均 ≫ 40）。方案（core-split-strategy.md §1 要素③ 极大规模档）：

- 固定启动 `num_kernels = min(num_logical, 40)`（编译期常量；小 N 自动退化为 num_logical ≤ 40，同模板全覆盖）。
- 核内串行：`for i in T.serial(num_local_tasks)`，`num_local_tasks = ceildiv(num_logical, num_kernels)`——**静态边界**（N/block_size/40 皆 trace 期常量）。
- 任务映射：`block_id = i × num_kernels + cid`（grid-stride），`if block_id < num_logical` 屏蔽越界。
- 负载分布：1M fp16 → 40 核 × ≤13 任务；2²⁴ → 40 × ≤205；2²⁸ → 40 × ≤3277——核启动 40 次（对比超发 131072 次串行调度），负载不均 ≤1 任务/核（≤0.5%，可接受）。
- **失败信号**（供 Stage 3/4 对照）：① num_local_tasks 依赖运行时值（非静态）→ 违反标准；② msprof 核串行排队/启动开销占比异常 → 检查 num_kernels 是否误用逻辑核数；③ 末核任务数 2 倍 → 核对 min() 与 ceildiv 上界。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 循环 | 形态 | 边界 | 选择理由 |
|------|------|------|---------|
| 核间 | `T.Kernel(num_kernels, is_npu=True)`（隐式 block 并行） | `min(num_logical, 40)` 编译期常量 | persistent 分核（极大规模档，§5.5） |
| 核内任务 | `T.serial(num_local_tasks)` | 静态（ceildiv(num_logical, num_kernels)，trace 期常量） | grid-stride 任务映射，摊薄核启动开销；静态边界满足 core-split 标准 |
| tile 内计算 | 无显式循环——v-prefix 全 buffer op 隐式向量并行（§1.6.2 V1–V4） | block_size | Vector 单元全宽度 lane 并行 |

无 `T.Pipelined`（单阶段 tile 流水，Developer 自动多缓冲承担 stage 化）；无 `T.Parallel`（v-prefix op 已是全 buffer 向量形态，无需再展开）；无 `T.Persistent`（persistent 语义由 `T.Kernel(num_kernels)` + 核内 `T.serial` 显式表达——同族 leaky_relu/elu 已验证形态）。

### 6.2 循环伪代码

见 §3.3（`for i in T.serial(num_local_tasks)` + `if block_id < num_logical` + tail 切片即完整循环结构）。

### 6.3 流水线优化

- **Developer 自动多缓冲**：copy→compute→copy 每 tile 单阶段，编译器自动 stage 化 GM→UB 载入与 UB→GM 写出（双缓冲），PL-1.6 实证该形态下计算链（lerp 7-pass）在 ≥16M 档完全隐藏于 MTE2 窗口——本算子 2-op（fp32）/4-op（transit）链更短，同判加强。
- **grid-stride 聚集窗口**：40 核聚集于移动窗口访问 GM（PL-1.6 实测优于每核连续分块 +3.2%@256M）。
- **Stage 4 可选方向**（不在本设计展开，按 ROI 排序）：① block_size 调优（UB 倍数复算 + 编译探针，§4.5 条款；本算子 8–10B/elem 预算松，16384 档 fp16/bf16 可探）；② stage 数评估；③ mte2_ratio 判定标量削减（PL-1.6 判定式：mte2_ratio ≥0.95 时无收益）；④ **候选 1 Taylor 分段启用**（仅当下游出现超容差相对精度需求时——机器证据已留存 §1.6.3.1，启用代价 +19 op 需先实测确认计算链未被 MTE2 完全隐藏）。

### 6.4 尾块处理

- **方案**：tile 级动态尾块 + 零起点切片——`tail = T.min(block_size, N − t0)`；载入 `T.copy(x[t0:t0+tail], x_ub[0:tail])`；v-prefix 整 buffer 计算覆盖 `[tail:]` 陈旧 lane（无污染：vexp/vsub/vcast 均 lane 独立；陈旧 lane 的 vexp 溢出值/未初始化值同被屏蔽——elu「溢出 lane + 未初始化 lane 两类覆盖」教训同承）；**拷出仅 `[0:tail]`**（官方 vec_add_1d 模式）。
- 覆盖场景：任意 N（含 N < block_size、尾 = 1、非 32B 对齐尾）；`tail` 为 trace 期可折叠表达式上界（N 编译期常量），`if block_id < num_logical` 保证最后核不越界。
- 无 padding+crop（无需 host 侧补内存——切片方案无 GM 越界风险，优于 padding 方案的额外流量）。

## 7. 同步策略

### 7.1 同步模式

**Developer 自动同步**（编程模式匹配，§2）：编译器在 GM↔UB 搬运与 Vector 计算之间自动插入栅栏；无手动 `set_flag/wait_flag`/`pipe_barrier`。

### 7.2 同步点说明

| 位置 | 同步机制 | 说明 |
|------|---------|------|
| GM→UB 载入 → vcast/vexp 链 | 自动（UB WAR/RAW 栅栏） | Developer 编译器插入 |
| vexp → vsub 原地/跨缓冲链（w32/x_ub）→ [fp16/bf16] vcast 降域 / [fp32] UB→GM 拷出 | 自动（RAW） | UB 内依赖（结果 buffer w32/x_ub 的消费；fp32 路线 vsub 的 WAR 于 x_ub 由自动栅栏保证——x_ub 原值死于 vexp 消费） |
| vcast → UB→GM 拷出 | 自动（RAW） | — |
| persistent 串行迭代间 | 自动（tile 间无跨迭代数据依赖；buffer 全量复用由自动栅栏保证） | — |
| 跨核 | 无（tile 两两不相交，无共享输出；无 workspace 往返） | 零核间同步 |

### 7.3 pass_configs 配置

默认（无手动 pass_configs）；`TILELANG_ASCEND_MODE=Developer` 模块级固定（§2.3）。Stage 4 如需显式 stage 数再评估。

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现，独立于 NPU 算法）

```python
import torch

def golden_expm1(x: torch.Tensor) -> torch.Tensor:
    """expm1 参考：源仓测试锚点（test_unary_math.py ref_fn = torch.expm1）。

    以 §0.1 语义为唯一依据（y = exp(x) − 1，fp32 opmath + 单次舍回路径——
    ATen 对 sub-fp32 dtype 内部 fp32 opmath，GOLD-DIAG 机器证实与 fp32 域
    计算逐位一致），**不复刻 NPU 算法**（无 vcast transit / vexp+vsub 链 /
    persistent 结构知识）。CPU 可跑，不依赖 NPU。
    """
    return torch.expm1(x.detach().cpu()).to(x.dtype)
```

- 独立性核验：golden 直接取 `torch.expm1` 语义（§0.1），与 NPU 侧 transit/向量链实现零共享；无标量参数（与 elu 的 alpha 参数化不同，golden 单参）。

### 8.2 精度标准与 CV 融合说明

| dtype | atol | rtol | 来源 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | test_unary_math.py `_get_tolerances` L29 |
| float16 | 1e-3 | 1e-3 | 同上 L26 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 同上 L28 |

预期（**与 leaky_relu/elu 的 bitwise 主张差异如实声明**）：本算子 cand（exp(x)−1）与 golden（expm1 直算）是**不同舍入路径**——机器验证存在位型差异（fp32 randn 判定集 47% lane ≤2 ulp）但**全部落在容差包络内**（viol=0，最窄裕度 67×，§1.6.1）；验收基准 = 容差断言（assert_close，atol/rtol 如上）+ 特殊值规则；**不主张逐位一致**。特殊值断言规则：`out == ref` 精确相等分支先行（Inf==Inf——x=88.8 角点双侧 +Inf；NaN 双方相等；±0 值相等——x=−0.0 时 golden −0.0 vs NPU +0.0 零符号差豁免，§0.1/§9.1#3）。

**CV 融合设计**：不适用——纯 Vector 算子，无 Cube 段（§5.1），无 `Scope("Cube")/Scope("Vector")` 协同、无 sync_block_set/wait 需求（本节即 design-template §8.2 要求的显式结论）。

### 8.3 L0 门槛测试计划（六项 + 附加；完整 L1/L2/Boundary 由 tilelang-op-develop 展开）

| # | 项 | 用例 |
|---|----|------|
| L0-1 | smoke 全 dtype | N=2²⁰，fp16/bf16/fp32，randn |
| L0-2 | 大规模抽查 | N=2²⁴，fp16/bf16（205 任务/核多迭代——原地/跨缓冲 vsub 无 loop-carried 纪律兜底） |
| L0-3 | 非整除尾块 | N ∈ {1000, 4097, 5000}（单 partial tile / 尾=1 / 非 32B 对齐尾），fp16+fp32 |
| L0-4 | 微型 shape | N ∈ {1, 7, 32}（单非满 tile / num_logical=1 退化），fp16/bf16/fp32 |
| L0-5 | 角点与特殊值 | ±Inf / NaN / ±0（**值断言 + NaN/Inf 逐位检查 + ±0 零符号豁免**——x=−0.0：golden −0.0 vs NPU +0.0，IEEE 值相等，§0.1）/ 次正规（finfo.min×0.5）/ max-finite（torch.finfo(d).max，勿手写字面量）/ **x=88.8（期望输出 +Inf——expm1 溢出语义本身，双侧精确相等）** / x=−88.8（期望 −1.0，cand −0.99999994 差 5.96e-8 ≪ atol）/ exp 下溢带（−104）/ 溢出边界带（±89）/ 小值消除区（±1e-8、±1e-7、±2⁻²⁴）；断言规则：**容差断言（assert_close equal_nan=True）+ 特殊值精确相等分支**（本算子无 bitwise 主张，§8.2——与 leaky_relu 的 sign-bit strict、elu 的 ±0 豁免值断言同族但口径独立定制） |
| L0-6 | persistent 分核边界 | num_logical ∈ {39, 40, 41}（fp16 block 2048 → N = 39·2048 / 40·2048 / 40·2048+1）：不满核 / 恰好满核 / 掩蔽+不均任务激活 |
| 附加 | E2 双配置复用 | 同一工厂先后以 bs=2048 与 1024 单参调用（kernel(2048)(x) / kernel(1024)(x)）均编译运行通过（E2 单参契约，jit wrapper 按 bs 缓存编译产物；leaky_relu L0-extra 同款）；**另验工厂级 block_size 兼容面**：`expm1_fwd_kernel(N, dtype_str, block_size=2048)(x)`（wrapper L81 适配后调用面）编译运行通过 |
| 附加 | UB guard + N/output_dtype 防御 | block_size=8193 → ValueError（fp16/bf16/fp32 三档）；N=0 / N<0 → ValueError；output_dtype="float32"（dtype=fp16 时）→ ValueError（builder 防御，§9.3） |

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 风险 | 等级 | 缓解 |
|---|------|------|------|
| 1 | **vexp 硬件 ULP 与 golden 的 exp 实现差异**：D-1 的 0-违反主张限于公式/舍入路径（torch CPU 侧模拟）；NPU vexp（hivm.hir.vexp）与 golden 侧 expm1 的 exp 实现（ATen opmath）可能在 fp32 域相差 1–2 ulp——经 expm1 的相消结构，小 \|x\| 域输出绝对误差增量 ≤ ~1.2e-7（总 ≤1.8e-7，atol 裕度 ≥55×）、大 \|x\| 域相对误差增量 ≤ ~1.2e-7（总 ≤2.3e-7，rtol 裕度 ≥43×）；fp16/bf16 经舍回多偶发 1 ulp（容差 1e-3/1.6e-2 覆盖） | 中低 | 测试容差覆盖（fp16 1e-3 / bf16 1.6e-2 / fp32 1e-5）；L0-1/L0-5 全 dtype 容差断言 + 角点；本设计不声明 vexp 自身的 bitwise 等价（elu 同款如实标注）；elu 上板实证 fp32 max_diff=5.96e-08 = vexp 硬件 ULP 上限（同族量级参照） |
| 2 | 小 \|x\| 相对精度天然受限（相消）：cand 在 \|x\|→0 相对误差 →1.0（机器验证 x=1e-7 rel err 19.2%）——测试包络由 atol 吸收（abs err 1.92e-8 ≪ 1e-5），但**若下游语义需要小值相对精度**则不足 | 低 | 候选 1 Taylor 分段机器证据留存（§1.6.3.1：rel err 0.192→5e-8，viol 0→0），作为精度兜底杠杆登记 Stage 4（§6.3④）；默认不启用（0 测试可观测收益 + 19 op 代价） |
| 3 | ±0 零符号值等价点：NPU 减法给 +0.0、golden ATen 符号保持给 −0.0（x=−0.0 时）——IEEE 值相等 | 低 | L0-5 断言规则调整（容差断言 + NaN/Inf 逐位、±0 豁免，§8.3）；不作为失败 |
| 4 | auto multi-buffer UB 占用：本算子 8–10B/elem，**朴素 ×2 模型（最悲观）下 8192 档 131KB / 163.8KB 均 < 192KB——一切已知口径安全** | 低 | guard 8192 保留（同族同形先例）；Stage 4 上调 block（16384）前必须以编译探针/BishengIR `ub overflow requires` 报文复算（fp32 朴素 ×2 = 262KB 超限）+ 连带复验 num_kernels（§4.5） |
| 5 | 物理核数跨设备差异（本设备 20/40；同族记录他设备 24/48） | 低 | 工厂期 `NPUUtils.get().get_aicore_num()*2` 现查现用（ada_layer_norm 同款），部署自适应 |
| 6 | D-2 估算常数跨设备外推 + **CONST-mte2-degradation / CONST-copy-floor-method 为 STALE 条目**（stamp ed787bb/CANN 8.5.0/910B2C，当前 88ee8c302e/CANN 9.0.0/910B3） | 低 | §1.6.0 已按「设计期估算 + Stage 4 实测校准」口径引用并标注；非 stale 主锚点 = PL-1.21（910B3/CANN 9.0.0/2026-10-28 实测）；PL-1.17/TRAP-UB 各条目 front-matter 含 2026-10-28 重验戳；Stage 4 msprof 实测校准回写 |
| 7 | **wrapper Stage 5 适配遗漏**：L81 `(x, N)` 尾参未删 → jit callable TypeError（运行时失败，非编译失败——构建期不暴露） | 中 | §3.5.2 适配清单明示（仅 1 token）；Stage 5 集成检查单必核项；L0 附加用例「工厂级 block_size 兼容面」直接以适配后调用面验证（§8.3） |
| 8 | vsub 标量字面量操作数形态（`T.vsub(w32, 1.0, dst)`）在本工具链的编译兼容性：文档 §2.1/§2.2.2 支持 + mish L78 vadd 标量先例 + T.vsub.md §2.4 示例一二种 dst 形态，但 vsub+标量+跨缓冲（fp32 路线 dst=x_ub ≠ src0）组合无逐字同形生产先例 | 低 | **回退方案**：若 Stage 3 编译报错，改 one32 缓冲形态（`T.vbrc(1.0, one32)` + `T.vsub(w32, one32, x_ub)`）——+4B/elem（fp16/bf16 10→14B、fp32 8→12B），8192 档朴素 ×2 后 229KB/196.6KB 临界（届时 guard 同步降至 4096 并留痕）；transit 路线回退为 one32 + 原地 vsub；不阻塞设计 |
| 9 | exp 次正规输出 FTZ 语义：x ∈ [−103.97, −87.3] 时 fp32 exp(x) 为次正规——若 vexp 硬件 FTZ 则 exp→0 | 低 | 输出不受影响：|exp(x)| < 1.7e-38 ≪ ulp(1.0)/2 → `exp(x) − 1` 无论 FTZ 与否均为 −1.0，golden 同为 −1.0；L0-5 −88.8/−104 角点覆盖 |
| 10 | 溢出 lane 输出语义：x ≥ 88.73 的 lane vexp → +Inf → 输出 +Inf（无屏蔽——本算子 Inf 即正确输出）；尾 tile 陈旧 lane 的溢出值由 [0:tail] 拷出屏蔽 | 低 | 与 golden 语义一致（torch.expm1(88.8) = +Inf）；L0-5 x=88.8 角点覆盖（双侧 +Inf 精确相等分支）；randn workload 不触及溢出带 |
| 11 | `output_dtype` / `threads` / `num_per_thread` 死参数误用：传入非 None output_dtype 或非默认 threads 不应改变行为 | 低 | output_dtype 校验 same_as(input) 否则 ValueError；threads/npt 签名保留但 builder 不消费（文档化，§3.5.2）；L0 附加防御用例 |
| 12 | bf16 无 vexp/vsub（docs 两处 §2.2.1 bf16 ×，逐一亲读） | 低 | transit 路线唯一可行（已采纳，§0.6 R2）；负向断言证据：T.vexp.md/T.vsub.md dtype 矩阵单元格 |

### 9.2 常见错误（Stage 3 防呆，含同族 RETROSPECTIVE 教训逐字承接）

- 不要把 `num_kernels` 写成 num_logical（超发 → 串行调度开销，core-split 失败信号②）。
- 不要用 `@T.prim_func` 内 Python if 引用循环变量（TVM-script if 块作用域限制，C11/lerp 先例——dtype 分支放工厂期；本设计全链无 in-body if）。
- **fp32 路线 vsub 必须写 `T.vsub(w32, 1.0, x_ub)`（dst=载入 buffer）而非 `T.vsub(w32, 1.0, w32)` 原地 + 拷出 x_ub**——后者 fp32 路线输出恒等输入（x_ub 未被更新，恒等映射错误）；transit 路线 ⑤ 必须回写 x_ub 后拷出（多 dtype 路线冻结前逐路线推演「① 载入 → … → ⑥ 拷出」字面 buffer 数据流，结果落位 buffer 必须与拷出源一致——leaky_relu Stage 2 检视问题 2 教训；本设计 §1.5 已按路线分列推演）。
- 原地/跨缓冲 vsub（transit ④ 原地 w32 / fp32 ④ 跨缓冲 x_ub）仅限「同迭代消费」：输出必须在同一 `T.serial` 迭代内被 ⑤/⑥ 消费，禁止把输出当作跨迭代携带状态（TRAP-vselect-inplace-carried-state 同源问题域；诊断指纹 = 输出停留 tile-0 局部值）。
- dtype 入参一律先归一化再分支：`dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]`（§3.3；torch 2.7.1 无 `.name`）；禁止把 torch.dtype 对象直接与 str 元组做 `in` 比较（恒 False，会误走 fp32 路线、fp16/bf16 输出错误升域结果）。
- E2 契约：NPU harness wrapper 以 block_size 形态调用工厂（L81 现状 + §3.5.2 适配后）——**禁止**按 GPU 基类两参形态 `self.kernel(cfg["threads"], cfg["num_per_thread"])` 实现——该依赖是源侧专用，NPU wrapper 不存在（接口契约核对必须查「落地侧现状」，leaky_relu Stage 2 检视问题 1 教训）。
- 测试数据 dtype 极值一律从 `torch.finfo(torch.<dtype>).max` 取，不要手写十进制字面量（fp32 max 十进制近似赋 fp32 张量会 overflow 抛 RuntimeError——RETROSPECTIVE 教训）。
- 同一 kernel 内多条 dtype 分支共享 API 约束时，逐分支单独审计约束遵守情况（不要凭「一条分支修好了」推断另一条也正确——RETROSPECTIVE 教训）。
- 涉及 torch.dtype 的 builder 归一化用 `str(dtype).split(".")[-1]`，不用 `.name`（torch 2.7.1 无该属性）。
- 本算子无 vbrc 需求（无常量广播）——**不要画蛇添足添加 zero32/cond 等缓冲**（无条件公式，elu 的 vcmp/vselect/vbrc 三件套在本算子均不存在；链越短越接近 I/O 下界）。

### 9.3 特殊场景处理

- **N=0/负 N**：源无显式拦截（Op 层校验为 `numel != N_total`，空张量可穿透）——NPU builder 防御性 `N <= 0 → ValueError`（适配项，早于编译失败）；L0 附加用例覆盖（§8.3）。
- **超大 N（2²⁸ = 256M workload）**：persistent 分核 40 核 × ≤3277 任务/核，静态边界 trace 期折叠；无额外处理。
- **output_dtype ≠ dtype**：builder ValueError（manifest same_as(input) 契约防御，§0.5 #7）。

## 10. 交付清单

### 10.1 目录结构

```text
examples/expm1/expm1_fwd_kernel/
├── DESIGN.md            # 本文档
├── verify_equiv.py      # D-1 等价性机器验证（已执行，§1.6.1 结果表）
├── RETROSPECTIVE.md     # Stage 1 复盘（Phase 8）
└── expm1_fwd_kernel.py  # Stage 3 产出：kernel + golden + 内嵌 L0 套件
```

### 10.2 文件清单（Stage 3 输入契约）

| 文件 | 内容 | 状态 |
|------|------|------|
| DESIGN.md | 全部设计决策（§0–§10） | 本文档 |
| verify_equiv.py | A1 采纳项结果表（§1.6.1(b)，9 代表行 / 全量 54 行 0 违反）+ 候选 1/2 弃选证据行（归档 §1.6.3.1）+ band 误差分布表 + Taylor 量化表 + GOLD-DIAG 诊断行；torch CPU 独立可跑（模块级种子固定可复现） | ✅ 已产出并执行（2026-10-28，全行 0 容差违反） |
| RETROSPECTIVE.md | Stage 1 复盘 | 随本文档产出 |
| expm1_fwd_kernel.py | `expm1_fwd_kernel` builder + `golden_expm1` + L0 套件 | 待 Stage 3 |
| （Stage 5）wrapper 适配 | `examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py` L81 `(x, N)` → `(x)` | 待 Stage 5（§3.5.2 适配清单） |

### 10.3 命名规范

- builder 函数名/参数名：`expm1_fwd_kernel(N, dtype, output_dtype=None, block_size=None, threads=256, num_per_thread=8)`（源三参保留 + K11 新参 block_size + threads/npt 保留不消费——§3.5.2 建议形态；`@functools.lru_cache(maxsize=32)` 施加于该外层 builder，提取件 L13 自带沿用）。
- E2 工厂：`kernel(bs=default_block)`（default_block 按 dtype：fp32 1024 / fp16·bf16 2048，工厂级 block_size 传入时覆盖——wrapper L81 兼容；leaky_relu L194 同形）；prim_func 名 `main`，参数 `(x, y)` 名与顺序保持源形态（迁移规则）。
- golden：`golden_expm1`。

### 10.4 实现顺序（Stage 3 建议）

1. builder 骨架 + host 契约校验（§3.3）→ 2. persistent 分核 + fp32 路线（3 步链：copy→vexp→vsub 跨缓冲，先通 fp32 全 L0——**⑥ 拷出源 = x_ub 数据流字面推演**）→ 3. transit 分支（fp16/bf16，5 步链：+vcast 升域/降域）→ 4. L0-5 角点（88.8→+Inf 期望 / ±0 豁免 / 1e-7 消除区 / max-finfo）+ L0-6 分核边界 → 5. E2 复用（含工厂级 block_size 兼容面）+ UB guard + N/output_dtype 防御用例。
