# _make_elu_kernel 算子设计文档（迁移：TileOPs GPU → TileLang-NPUIR）

| 项 | 值 |
|----|----|
| project / op | `elu_fwd` / `_make_elu_kernel`（严格沿用源 `@tilelang.jit` 装饰的 builder 函数名；`@T.prim_func` 参数名与顺序保持 `(x, y)`） |
| scenario / mode | migration (harness) / first_design |
| 源算子 | `EluFwdOp` ← `EluFwdKernel(ParametricUnaryKernel 逻辑内联)` ← `_make_elu_kernel`（GPU 原件 `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2497；NPU 侧 verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/elu_fwd/_elu_fwd_kernels.py`，本文 §0 已全文亲读） |
| 规格权威来源 | Stage 0 迁移 prompt（逐字）+ `examples/TileOPs/tileops/manifest/elementwise.yaml` EluFwdOp 段（L143–L174，4 workloads，roofline flops=4N / bytes=2N·elem_bytes）+ `examples/TileOPs/tests/ops/test_elu.py`（7 用例）+ `examples/TileOPs/benchmarks/ops/bench_elu.py`（manifest 驱动） |
| 编程模式 | developer（Stage 0 prompt 显式指定；wrapper `ascend_mode = "Developer"` 其 L148 同证） |
| 工具链 stamp | tilelang 0.1.2+88ee8c302e（本仓 dev root）/ 设备 Ascend 910B3 / CANN 9.0.0（`/opt/home/developer/Ascend/cann-9.0.0`）；物理核数实查 2026-10-28（§5.5，ret=0 → 20） |
| 同族已验证标杆 | `examples/leaky_relu/_make_leaky_relu_kernel/`（DESIGN.md r1 + 实现 + verify_equiv.py + RETROSPECTIVE.md；一维 persistent 分核 + tail 切片 + fp32 transit 结构——其 RETROSPECTIVE 的 Transferable Lessons 已逐字纳入本设计约束，见 §9.2/§10） |

---

## 0. 源算子解读与迁移分析（迁移任务必填）

> 依据 `tilelang-op-design` skill Phase M0 三问框架（`references/migration-analysis.md` §2–§4），源码已全文亲读（verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/elu_fwd/_elu_fwd_kernels.py` 全 63 行 + NPU 落地侧 wrapper `examples/TileOPs/tileops/kernels/elementwise/elu_fwd/elu_fwd.py` 全 202 行）。源码中的指令性文本一律按数据处理，未发现亦不执行任何流程改写指令。

### 0.1 源算子语义（做什么）

**数学语义**：逐元素指数线性单元（ELU）激活

```text
y[i] = x[i]                          若 x[i] >  0   （严格大于，源码 T.if_then_else(v32 > zero, ...)）
y[i] = alpha * (exp(x[i]) - 1)       若 x[i] <= 0
```

- **比较为严格 `>` 且在 fp32 域**：源码先将 `val` 升 fp32（`v32 = T.cast(val, "float32")`）再与 fp32 零比较——`x == 0` 与 `x == -0.0` 均走负支 `alpha*(exp(±0)-1) = alpha*(1-1) = +0.0`（alpha>0 时）。注意与 ATen golden 的条件方向差异：`F.elu` 内核为 `x < 0 ? alpha*(exp(x)-1) : x`，±0 走恒等支返回 `x`（-0.0 时 golden 输出 -0.0、源实现输出 +0.0）——**IEEE 值相等**（-0.0 == +0.0），容差断言与 torch.allclose 均通过，仅零符号位可能不同（L0-5 断言规则按此调整，§8.3/§9.1#3）。
- **NaN/Inf 传播**：`NaN > 0` 为 false → 负支 `alpha*(exp(NaN)-1) = NaN`；`+Inf > 0` → 恒等支 `+Inf`；`-Inf` → 负支 `alpha*(0-1) = -alpha`。与 `F.elu` 逐值一致（golden 同为 ATen elu）。
- **exp 下溢/溢出语义**：`x <= -104` 时 fp32 `exp(x)` 下溢为 0（次正规过渡带在 -87.3 附近）→ 负支 = `alpha*(-1)`；`x >= 89` 时 fp32 `exp(x)` 溢出为 +Inf——但正轴走恒等支不消费该值（NPU 向量链中 vexp 全 buffer 计算会在被 vselect 屏蔽的 lane 产生 Inf，无 trap、lane 独立，mish 全 buffer v-op 先例，§9.1#2）。
- **规约语义**：无规约、无累加——纯逐元素映射，不存在累加顺序问题；输出仅依赖同位置输入。
- **dtype 语义（本算子迁移的关键语义要素）**：**负支计算全程 fp32 域 + 单次舍回**——`zero/a/one` 均为 fp32 cast 常量、`v32` 为 fp32 升域、比较在 fp32 域、`exp/sub/mul` 在 fp32 域，最终 `T.Cast(val.dtype, ...)` 单次舍回；正支为原生值恒等直传（无舍入）。即**源实现本身就是 fp32-opmath + 单次舍回路径**——与测试 golden 锚点 `F.elu(x.float(), alpha).to(x.dtype)`（test_elu.py L40，fp32 opmath + 单次舍回）**同一路径**。这是与 leaky_relu 的关键差异：leaky_relu 源在原生域逐 op 舍入（transit 是"修正"），ELU 源已在 fp32 域（transit 是"结构承接"）——D-1 机器验证实测 candidate vs golden **0 ulp / 0 violations**（§1.6.1 A1，6/6 行）。
- **边界语义**：非整除尾部——非 fp8 分支按整块 fragment 覆盖（`block_size = threads×npt`，依赖 grid 收尾；`T.Parallel(threads, npt)` 展开后无逐元素 guard——manifest workloads 的 N（2²⁰/2²³/2²⁴）均被默认 block（1024/2048）整除，源实现实际从未执行 partial tail 路径）；fp8 分支有逐元素 `if idx < N` guard（死代码，§0.5 #5）。**源无 `N == 0` 显式拦截**（Op 层校验为 `numel != N_total`，空张量可穿透）——NPU 侧以 builder 防御性 `N <= 0 → ValueError` 明确拒绝（§9.3，适配项）。
- **host 侧语义**：输入任意 shape → 展平 `(N,)` 连续 → kernel → 还原 shape；`inplace=True` 时结果 `copy_` 回输入并返回输入张量（kernel 始终写新 buffer，test_elu_inplace 断言 `out is x`）；`alpha` host 校验必须 finite（test_elu_rejects_nonfinite_alpha）；dtype 白名单 `(float16, bfloat16, float32)`（test_elu_rejects_non_float_dtype）。
- **完成标准核验**：golden 函数可直接写出（§8.1）——`golden_elu(x, alpha) = F.elu(x.float(), alpha).to(x.dtype)`，与源实现数值一致（D-1 实测 0 ulp，§1.6.1）。

### 0.2 源算子输入输出

| 项 | 值 | 依据 |
|----|----|------|
| 输入 `x` | 展平 `(N,)` 一维连续张量；NPU 契约 dtype **float16 \| bfloat16 \| float32**（manifest `input.dtype`）；host 支持任意 ND，展平后 N = product(shape) | manifest elementwise.yaml L150 + Op 层 |
| 输出 `y` | `(N,)`，**dtype same_as(input)**（非 fp8 路径 `output_dtype=None → out_dtype == dtype`，源码 L17/L43）；`output.shape == input.shape`（shape_rules） | 源码 L17/L43 + manifest L152/L156 |
| 参数 `alpha` | Python float，**default 1.0**；constructor 期标量（烘焙进 kernel 常量，lru_cache 键的一部分）；host 校验 finite | manifest L154 + 源码 L13 + test_elu.py L101–103 |
| 参数 `inplace` | bool，default false；host 层 copy-back 语义（kernel 写新 buffer） | manifest L155 + test_elu_inplace |
| 参数 `output_dtype` / `is_fp8` / `threads` / `npt` | 见 §0.5 处置；`threads=256, npt=8`（签名默认，GPU 基类实际 fp32 npt=4），`block_size = threads×npt`（源码 L18） | 源码 L13–L18 |
| 动态轴 | N（展平后元素总数）；builder 按 N 工厂特化（N 为闭包编译期常量） | manifest workloads |
| 不变量 | `y[i] = f(x[i])`，1 读 1 写，无形状变换、无 layout 变换（连续 1-D） | §0.3 数据流 |
| roofline 契约 | flops = **4·N**（compare-and-select(1) + exp(1) + sub(1) + mul(1)）；bytes = **2·N·elem_bytes**（1 读 1 写） | manifest L168–L174 + test_elu_eval_roofline |

### 0.3 实现算法解读（怎么算）

**结构化阅读协议 7 步**（migration-analysis §3.1）：

1. **入口签名**：`_make_elu_kernel(N, dtype, alpha, output_dtype=None, is_fp8=False, threads=256, npt=8)`；返回 `@tilelang.jit(out_idx=[1])` 工厂 `kernel(threads_arg, npt_arg) -> main`——两参调用式是 GPU 基类 `ParametricUnaryKernel` 的源侧契约（`init_config` 按 `self.kernel(cfg["threads"], cfg["num_per_thread"])` 两参调用）。**该两参依赖不进入 NPU 设计决策**：NPU harness wrapper 已按 K3/K11 适配为单参 `kernel(block_size)`（§0.7 契约权威，其 L100 亲读）。另注：源 builder **无 `@functools.lru_cache` 装饰**（提取件文件头 `import functools` 未使用——verbatim 保真）；NPU 侧按同族模板补 `@functools.lru_cache(maxsize=32)`（host 编译缓存适配，leaky_relu 同款）。
2. **Kernel 结构**：`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg)`——**一维 grid**，`block_size = threads * npt` 为 builder 闭包编译期常量（L18）；`is_fp8` 分支/非 fp8 分支双 prim_func 工厂期 Python if 分派（L20–L59）。
3. **计算原语**（逐语句归步）：
   - **非 fp8 分支（主路径）**：`T.copy(x[bx·bs:(bx+1)·bs], x_reg)` 整块载入 fragment → `for i, j in T.Parallel(threads_arg, npt_arg)` 二级展开，逐元素 `val = x_reg[i·npt+j]`；`zero = T.cast(0, "float32")`、`a = T.cast(alpha, "float32")`、`one = T.cast(1.0, "float32")`、`v32 = T.cast(val, "float32")`；`y_reg[i·npt+j] = T.if_then_else(v32 > zero, val, T.Cast(val.dtype, a * (T.exp(v32) - one)))` → `T.copy(y_reg, y[...])` 写回。**比较在 fp32 域、正支原生直传、负支 fp32 域计算后单次舍回**（§0.1 dtype 语义）。
   - **fp8 分支**：无 fragment，逐元素 `idx = (bx·threads+i)·npt+j` + `if idx < N` guard + 同式 fp32 域计算 + `T.Cast(out_dtype, ...)` 写回（依赖不可达，§0.5 #5；无外部 helper 依赖——与 leaky_relu 不同，本源 fp8 分支自包含）。
4. **内存分配**：`x_reg`/`y_reg` = `T.alloc_fragment((block_size,), dtype)`——GPU **寄存器 fragment** 驻留（每元素恰 1 次 GM 读 + 1 次 GM 写，无 SMEM 中转）。
5. **循环与调度**：仅 `T.Parallel(threads_arg, npt_arg)` 一层元素展开（SIMT block×thread 映射，每线程 npt 元素 ILP）；无 serial、无 Pipelined、无 persistent。
6. **host 侧逻辑**（NPU 落地侧 wrapper，全文亲读）：`SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)` 白名单拒绝（其 L151/L161–165，**fp8 在此被 ValueError 拒绝**）；K11 折叠常量 `_BLOCK_SIZE_FP32=1024` / `_BLOCK_SIZE_NON_FP32=2048`（其 L82–83/L179–188）；custom_op `npub::elu_fwd`（其 L91–L100）：`dtype = getattr(torch, dtype_str)` 后 `_make_elu_kernel(N, dtype, alpha)(block_size)(x)`（**torch.dtype 入参 + 单参 block_size**）；Kernel 类工厂构建 `self.kernel = _make_elu_kernel(self.N_total, self.dtype, self.alpha)`（其 L175）；Op 层：展平/还原、inplace copy-back、alpha finite 校验、`eval_roofline(flops=4N, bytes=2N·elem_bytes)`。
7. **编译配置**：无 pass_configs / 环境变量（GPU 默认管线）。

**计算步骤分解表**：

| 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|------|------|------|------|---------|
| S1 | 整块载入 `T.copy(x_tile, x_reg)` | GM x | fragment x_reg | 读 x |
| S2 | 逐元素 `if_then_else(v32 > 0, val, cast(α·(exp(v32)−1)))`——比较 fp32 域；负支 fp32 域 exp→sub→mul 后单次舍回；正支原生直传 | x_reg + fp32 常量 zero/a/one | y_reg | 分段公式本体（§0.1 全部传播语义） |
| S3 | 整块写回 `T.copy(y_reg, y_tile)` | y_reg | GM y | 写 y |

**数据流图**（源硬件视角，覆盖全部 buffer）：

```text
GM x ──T.copy──> x_reg(GPU register fragment) ──T.Parallel 逐元素 select（负支 fp32 域 exp/sub/mul + 单次舍回）──> y_reg(fragment) ──T.copy──> GM y
常量：zero/a/one 以 fp32 标量内联（编译期 cast）；无其他 buffer、无中间 GM 往返
```

### 0.4 优化手段解读（为什么快）

逐项识别（清单法，migration-analysis §4.1；"识别不出来 ≠ 不存在"已对照——源码中确无 SMEM tiling / mma / warp shuffle / 异步流水 / swizzle / online / persistent 结构）：

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---------|---------|------|----------------|------|
| O1 | register_copy fragment 驻留 | `alloc_fragment` + 整块 `T.copy` 进出 | 每元素恰 1 次 GM 读 + 1 次 GM 写，中间全寄存器 | GPU 寄存器堆 | 硬件强相关（NPU 等价片上层 = UB） |
| O2 | coalesced 连续访问 | 连续 tile 载入/写回（`bx·bs` 连续窗） | 合并访存事务、吃满带宽 | memory coalescing | 意图可移植（访存组织需重排） |
| O3 | block×thread 二级并行 | `T.Parallel(threads_arg, npt_arg)` | 每线程 npt 元素 ILP + SIMT 占用 | CUDA SIMT 层级 | 硬件强相关，需重设计 |
| O4 | fp8 分支 + 逐元素 guard | `is_fp8` 分支、`if idx < N`、`T.Cast(out_dtype, ...)` | 规避 8-bit fragment register_copy 不可靠 | CUDA fp8 fragment | 接口契约死代码（见 §0.5 #5） |
| O5 | **负支 fp32-opmath + 单次舍回 + 标量常量内联** | `v32/zero/a/one` 全 fp32 cast，`T.Cast(val.dtype, ...)` 一次舍回 | 数值路径与 golden（fp32 opmath）一致；比较/计算无原生域逐 op 舍入损失 | —（纯算法层 dtype 路径决策） | **可移植**（NPU transit 结构化承接，§0.6 R2） |
| O6 | 编译期常量折叠 | `block_size = threads*npt` 闭包常量、grid 表达式特化 | host 期折叠、消除运行时开销 | host 层 | 保留 |

**算法层 vs 实现层区分**：本源算法为"单步逐元素映射"，无 online / 无分块累加等算法层结构优化；**O5 是源码中唯一算法层性质的决策**（dtype 域选择：无损升域保序 + fp32 域计算 + 单次舍回），它同时是 §0.6 R2 的直接输入与 D-1 验证的基线。

### 0.5 硬件耦合性分析与 NPU 适配决策

三层判定模型 + 四态处置（migration-analysis §5；判定输入含 Phase R 调研结论 §1.6.0——调研选定 = 基线公式 + α=1 特化评估后不采纳（ROI≈0）+ persistent/UB/transit 落地形态，源算法公式未被替换，仅实现结构重设计）：

| # | 条目 | 层 | 处置 | 依据 |
|---|------|----|------|------|
| 1 | 语义层全集（公式/严格 `>` fp32 域比较/NaN·Inf·±0·exp 下溢溢出传播/`(N,)` 契约/dtype same_as/host 展平+inplace+alpha finite 校验） | 语义层 | **保留** | 语义层不可妥协（映射表前置规则） |
| 2 | O1 register_copy fragment | 优化层 | **等价替换** | fragment → UB：`T.alloc_ub`（映射表「shared memory → L1/UB」「register 累加 → UB 中间缓冲」；同族先例 leaky_relu/lerp/mish kernel 全 UB 链） |
| 3 | O2 coalesced 连续访问 | 优化层 | **保留**（意图承接） | NPU 侧 UB 连续零起点切片 + persistent grid-stride 聚集窗口（pattern-library PL-1.6 实测：grid-stride 优于每核连续分块 +3.2%@256M；kb_stale_check 2026-10-28 复核非 stale） |
| 4 | O3 block×thread 二级并行 | 优化层 | **重新设计** | 映射表「block × thread 两级并行 → 重新设计（本项目 `T.Kernel` 只接受一维 block 数）」；元素级并行改由 v-prefix 全 buffer 向量 op 承接（§0.6 R1） |
| 5 | O4 fp8 explicit_parallel 分支 | 优化层 | **舍弃** | 接口契约死代码：NPU wrapper `SUPPORTED_DTYPES = (float16, bfloat16, float32)`（其 L151）在 builder 之前即 ValueError 拒绝 fp8 → `is_fp8=True` 对本算子**不可达**（wrapper K6 注记 L26–L30 明文"the NPU kernel builder rejects is_fp8=True"）；manifest workloads dtypes 亦仅 fp16/bf16/fp32。参数 `is_fp8` 保留于签名；NPU 实现 `is_fp8=True` 时抛 ValueError（与 wrapper 拒绝语义一致） |
| 6 | O5 负支 fp32-opmath + 单次舍回 + 常量内联 | 算法层（dtype 路径） | **保留**（transit 结构化承接） | 源路径 = golden 路径（§0.1）；NPU fp16/bf16 走 vcast(rint) 升 fp32 → fp32 域链 → vcast(rint) 单次舍回 = 源舍入路径逐位复刻（D-1 A1：cand vs base 0 ulp）；常量以 vbrc 广播（alpha32/zero32）+ vsub 标量字面量（1.0，mish L78 先例）内联 |
| 7 | inplace copy-back | host 层 | **保留** | Op 层语义（kernel 写新 buffer + host copy-back，test_elu_inplace `out is x`），kernel 无感知 |
| 8 | fp8 分支 per-element `if idx < N` 尾 guard | 优化层 | **等价替换** | NPU 尾块统一改为 tile 级 `tail = T.min(block_size, N−t0)` + UB 零起点切片拷出（官方 `examples/elementwise/vec_add_1d.py` 尾块模式 + leaky_relu 同款已验证），全部尾 lane 语义由切片覆盖，无逐元素分支 |

**处置统计：保留 4（#1/#3/#6/#7）/ 等价替换 2（#2/#8）/ 重新设计 1（#4）/ 舍弃 1（#5）**。无存疑项（全部条目有迁移映射表条目佐证，API 细节佐证为 docs/Tilelang.language/ 各 API 文档与 §3.5.3 所列 examples 同类实现文件）。

### 0.6 NPU 算法重设计

对 §0.5「重新设计」项 + 调研选定算法族（§1.6.0）落地为 NPU 侧算法。重设计输入：§0.5 #4（并行结构）+ §0.4 全部优化意图 + Phase R 调研结论（§1.6.0：基线公式四口径并列最优、α=1 特化 bitwise 安全但 ROI≈0 不进基线设计、无算法族替换，选定 = 基线公式 + NPU 重设计落地形态）。

**R1 并行结构重设计**（承接 §0.5 #4，模式库「并行结构重设计」）

- 源方案：`T.Kernel(ceil(N/bs))` × `T.Parallel(threads, npt)` 两级 SIMT 展开，逐元素 `if_then_else`。
- NPU 新算法：**一维 `T.Kernel(num_kernels)` persistent 分核**（`num_kernels = min(ceil(N/block_size), vector_cores)`）+ 核内 `T.serial(num_local_tasks)` 静态边界 grid-stride（`block_id = i·num_kernels + cid`，`if block_id < num_logical` 屏蔽）+ **v-prefix 全 buffer 向量链**（`vcmp` → `vexp` → `vsub` → `vmul` → `vselect`，fp32 域）替代逐元素展开；比较/指数/减/乘/选择均为 UB 全宽度向量 op。
- **语义保持论证**：① 逐元素映射无累加顺序，lane 重排不改结果；每元素恰计算同一公式一次。② 边界：尾块 `tail = T.min(block_size, N−t0)` 切片 + 零起点 UB 拷出覆盖任意 N（含 N < block_size、非 32B 对齐尾、N=1），行为与源整块覆盖语义一致（源非 fp8 分支依赖 block_size 整除，NPU tail 方案覆盖非整除——严格更强）。③ NaN/Inf/±0/exp 下溢：`vcmp("gt")` 逐 lane 产生与标量 `>` 一致的谓词（NaN>0=false → 负支 → vexp(NaN)=NaN → 链传播 NaN），`vselect` 选取支路复现 §0.1 全部传播语义——§1.6.1 A1 机器验证 candidate vs golden **0 ulp / 0 violations**（含 ±Inf/NaN/±0/次正规/max-finite/exp 下溢带 −104/±89 溢出 lane/1e-8 消除区角点）。④ 严格 `>`：`vcmp` 支持 `"gt"` 模式（docs/Tilelang.language/比较操作/T.vcmp.md §2.3，leaky_relu 已验证使用）。

**R2 dtype 路径重设计**（算法替代·dtype 路径；承接 §0.5 #6 / §0.4 O5——**源 fp32-opmath 语义的结构化承接而非修正**；采纳须与 §1.6.0 调研结论一致，见 §1.6.0 R4 dtype 路径行）

- 源方案：比较 fp32 域 + 负支 fp32 域计算 + 单次舍回 + 正支原生直传（§0.1）。
- NPU 新算法：**fp16/bf16 走 fp32 中转链**——`T.vcast(x_ub, x32, round_mode="rint")` 无损升 fp32 → fp32 域 `vcmp/vexp/vsub/vmul/vselect`（alpha 以 fp32 值参与）→ `T.vselect` 在 fp32 域选择（select-then-round）→ `T.vcast(w32→x_ub, round_mode="rint")` **单次舍回**；fp32 走原生域（无 transit，select-then-round 退化为 Out=A 原地写回）。与源的 round-then-select 形态**逐位等价**（正支：rint(x32)=x 无损还原；负支：两侧同为 rint(α·(exp−1))——D-1 A1 cand-vs-base 0 ulp 机器证实）。
- 采纳依据（双触发，PL-1.17 两任务实证 pattern + 本任务第三例）：① **bf16 无 v-prefix 超越/算术 op**——`T.vexp` dtype 矩阵 bf16 ×（docs/Tilelang.language/数学操作/T.vexp.md §2.2.1，亲读）、`T.vmul` bf16 ×、`T.vsub` bf16 ×（各自 §2.2.1）——bf16 原生域链无计算路径，vcast 升 fp32 是唯一路径；② **fp16 golden/source 对齐**——fp16 原生域逐 op 舍入链（vexp fp16 原生可用，T.vexp.md §2.2.1 fp16 √）与 golden 差 ≤1 ulp（A6 机器证据：16448 mismatches / max 32767 ulp），transit 路径与 golden **逐位一致（0 ulp / 0 violations）**。alpha 常量以 fp32 值参与：vbrc 的 fp64→fp32 转换为 IEEE 最近舍入（= np.float32(alpha) = golden opmath scalar 转换）。
- **语义保持论证**：fp16/bf16 → fp32 升变换**数学无损**（值域子集）且保序 → fp32 域比较结果与源 fp32 域比较**严格一致**；fp32 域 exp/sub/mul 逐 op 单次舍入 = 源表达式 `a*(exp(v32)-one)` 的同序同域计算；select-then-round 与源 round-then-select 逐位等价（正支无损还原 / 负支同一舍入对象）；fp32 dtype 无 transit（原生域 = 源域，逐位一致）。机器验证 A1：6/6 行 EQUIV_PASS（§1.6.1）。

**R3 尾块重设计**（承接 §0.5 #8）：见 R1 ②，tile 级 tail + 零起点切片，`if idx < N` 逐元素分支消除。

**源优化意图承接说明**（algorithm-research §4：不得静默丢弃）：

| 源手段 | 意图 | NPU 承接 |
|--------|------|---------|
| O1 fragment 驻留（1 读 1 写 GM） | 消中间 GM 往返 | UB 驻留整 tile，单遍 load→compute→store，GM 流量同为 2N·B 下界（§1.6.0 R3） |
| O2 coalesced | 满带宽连续访问 | UB 连续零起点切片 + persistent grid-stride 聚集窗口（PL-1.6 实测最优访问形态） |
| O3 二级并行 ILP | 元素级并行度 | v-prefix 全 buffer 向量 op（Vector 单元全宽度）+ Developer 自动多缓冲（PL-1.6 实证 lerp 7-pass 链完全隐藏于 MTE2 窗口） |
| O4 fp8 分支 | —（不可达路径） | 不迁移（§0.5 #5 舍弃；参数保留 + ValueError） |
| O5 fp32-opmath + 单次舍回 | 数值路径与 golden 一致 | R2 transit 链逐位承接（D-1 A1 0 ulp）；one 常量以 vsub 标量字面量、alpha/zero 以 vbrc fp32 广播内联（消标量访存意图同承） |
| O6 编译期常量折叠 | host 期特化 | 同款保留：N/block_size/alpha 均工厂闭包常量 + `@lru_cache(32)` 编译缓存（源无 lru_cache，同族模板适配，§0.3 步 1 注） |

### 0.7 标杆实现

- **源算子 verbatim 提取件**：`examples/TileOPs/tileops/kernels/elementwise/elu_fwd/_elu_fwd_kernels.py`（Pattern-B 提取，自包含无外部 helper）。
- **GPU 原件**：`/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2497。
- **NPU harness wrapper（集成契约权威）**：`examples/TileOPs/tileops/kernels/elementwise/elu_fwd/elu_fwd.py`——K3/K11 适配明文（模块 docstring L14–L23：factory 签名降为 `(N, dtype, alpha)`、返回 callable 收单参 `block_size`）、custom_op 单参调用 `_make_elu_kernel(N, dtype, alpha)(block_size)(x)`（其 L100）、Kernel 类工厂构建 `self.kernel = _make_elu_kernel(self.N_total, self.dtype, self.alpha)`（其 L175，入参为 torch.dtype 对象）、折叠常量 `_BLOCK_SIZE_FP32=1024`/`_BLOCK_SIZE_NON_FP32=2048`（其 L82–83）、K6 fp8 死代码注记（其 L26–L30）、`ascend_mode = "Developer"`（其 L148）、`SUPPORTED_DTYPES`（其 L151）；harness 调用方 `examples/TileOPs/tests/ops/test_elu.py`（import 该 wrapper 的 `EluFwdKernel`）。**GPU 基类两参调用式（`init_config` 的 `self.kernel(cfg["threads"], cfg["num_per_thread"])`）仅为源侧事实，永不进入 NPU 设计决策行**（leaky_relu Stage 2 检视教训）。
- **同族已迁移标杆（本仓已验证）**：`examples/leaky_relu/_make_leaky_relu_kernel/`（DESIGN.md r1 + `_make_leaky_relu_kernel.py` + verify_equiv.py + RETROSPECTIVE.md——persistent 分核 + fp32 transit + L0 六项套件，**最接近模板**）；`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/`（E1/E2 单参工厂契约同形）。
- **vexp 生产先例**：`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`——`T.vexp(x_f32_ub, buf1_ub)`（L77）+ **标量字面量 v-op 操作数** `T.vadd(buf1_ub, 1.0, buf2_ub)`（L78）+ 全 dtype fp32 域计算策略（transit）+ 尾块切片；同族规格文件 `examples/TileOPs/tileops/kernels/elementwise/mish/`。
- **精度/性能契约**：`examples/TileOPs/tests/ops/test_elu.py`、`examples/TileOPs/benchmarks/ops/bench_elu.py`（manifest 驱动，`workloads_to_params(include_extra=True)`）。
- **manifest**：`examples/TileOPs/tileops/manifest/elementwise.yaml` EluFwdOp 段（L143–L174）。

---

## 1. 概述

### 1.1 算子名称

`_make_elu_kernel`（`elu_fwd` 项目；@tilelang.jit builder 函数名严格沿用源签名）。

### 1.2 功能描述

NPU 侧逐元素 ELU 前向激活：对展平 `(N,)` 连续输入按 `y = x (x>0)；y = alpha·(exp(x)−1) (x≤0)` 做分段映射，输出同 shape 同 dtype。支撑 `EluFwdOp` 的 4 个 manifest workloads（smoke-1m [1048576] fp32|fp16|bf16 / mlp-hidden [2048,4096] / mlp-hidden-wide [2048,8192] / gan-feat [16,256,64,64]，均 fp16|bf16）。本节描述的是 **§0.6 迁移决策后的 NPU 侧算法**（与源算法的结构差异：两级 SIMT 展开 → persistent 一维分核 + UB 向量链；源 fp32-opmath 负支语义由 transit 链逐位承接；`kernel(threads_arg, npt_arg)` 两参 → `kernel(block_size)` 单参 K11 契约）。

### 1.3 数学公式

```text
y[i] = x[i]                        若 x[i] >  0   （严格 >，比较在 fp32 域）
y[i] = alpha · (exp(x[i]) − 1)     若 x[i] <= 0   （fp32 域计算，单次舍回）
```

FLOPs 口径（manifest roofline §1.3 / L168–L174）：compare-and-select(1) + exp(1) + sub(1) + mul(1) = **4 FLOPs/元素**；访存 = 2·N·elem_bytes（1 读 1 写）。算术强度 4/(2B) = 2/B flop/B：fp16 = 1.0、fp32 = 0.5——远低于 Vector 单元平衡点，**带宽受限算子**。

### 1.4 算法描述

分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的算法（与源算法非直译关系）：

1. **分核**：persistent 一维分核——`num_logical = ceildiv(N, block_size)` 个逻辑 tile，固定启动 `num_kernels = min(num_logical, vector_cores)` 个物理核（纯 Vector 翻倍核数，§5.5 实查 20→40），核内 `T.serial(num_local_tasks)` 静态边界 grid-stride。
2. **单遍 tile 流水**（每 tile）：GM→UB 同 dtype 载入（`tail` 切片）→ [fp16/bf16: `vcast(rint)` 升 fp32，域输入 = x32] → fp32 域 `vcmp`（"gt"）→ `vexp` → `vsub(w32, 1.0, w32)`（标量字面量操作数，mish L78 先例）→ `vmul(alpha32, w32, w32)`（原地）→ `vselect`（fp16/bf16：`vselect(cond, x32, w32, w32)` Out=B，结果驻留 w32、仅同迭代消费；fp32：`vselect(cond, x_ub, w32, x_ub)` Out=A 写回载入 buffer，gqa_prefill L875 先例）→ [fp16/bf16: `vcast(rint)` 单次舍回至 x_ub] → UB→GM 拷出 `x_ub[0:tail]`（**两路线统一拷出源 = x_ub**，与 §1.5/§3.1/§3.3/§4.3/§4.4 交叉一致）。
3. **常量预备**（每 kernel 一次，persistent 循环外）：`vbrc(0.0, zero32)`、`vbrc(alpha, alpha32)`（fp32 值参与）。
4. dtype 路径：fp32 原生域；fp16/bf16 fp32 中转（§0.6 R2——源 fp32-opmath 语义的结构化承接，PL-1.17 第三例）。

### 1.5 数据流图

```text
                                     （每 kernel 一次）
                      0.0 ────────────vbrc──> zero32 (UB, fp32, block 宽)
                      alpha(f64 标量)──vbrc──> alpha32(UB, fp32, block 宽)

每 tile（persistent 串行，两路线结果落点分列）：
GM x[t0:t0+tail] ──T.copy──> x_ub(UB, 输入 dtype)

[fp16/bf16 路线 use_transit=True]
   x_ub ──vcast(rint)──> x32(UB, fp32，无损升域)
   x32, zero32 ──vcmp("gt")──> cond(UB, bool)
   x32 ──vexp──> w32(UB, fp32)                      # 全 buffer 计算；正轴大值 lane 的 exp 溢出
                                                    # Inf 被后续 vselect 屏蔽（lane 独立，无 trap）
   w32 ──vsub(w32, 1.0, w32)──> w32                 # 标量字面量 1.0，原地（vsub.md §2.3.3）
   alpha32, w32 ──vmul(alpha32, w32, w32)──> w32    # 原地：w32 = alpha·(exp(x)−1)
   cond, x32, w32 ──vselect(Out=B 原地)──> w32 = y32（仅同迭代消费）
   w32 ──vcast(rint)──> x_ub（⑥ 单次舍回，复用载入 buffer）
   x_ub ──T.copy[0:tail]──> GM y

[fp32 路线 use_transit=False（无独立 x32，域即 x_ub，无 vcast）]
   x_ub, zero32 ──vcmp("gt")──> cond(UB, bool)
   x_ub ──vexp──> w32(UB, fp32)
   w32 ──vsub(w32, 1.0, w32)──> w32
   alpha32, w32 ──vmul(alpha32, w32, w32)──> w32
   cond, x_ub, w32 ──vselect(Out=A 原地)──> x_ub = y（⑤ 写回载入 buffer，gqa_prefill L875 先例）
   x_ub ──T.copy[0:tail]──> GM y
```

> 拷出源一致性（leaky_relu Stage 2 检视问题 2 教训，逐路线字面推演）：两路线的 UB→GM 拷出源**统一为 x_ub**——transit 路线经 ⑥ 降域回写 x_ub、fp32 路线经 ⑤ Out=A 原地写回 x_ub；不存在「结果驻留 buffer ≠ 拷出源」的组合（transit 结果驻留 w32 → ⑥ 回写 x_ub → ⑦ 拷 x_ub；fp32 结果直接驻留 x_ub → ⑦ 拷 x_ub）。§1.4/§3.1/§3.3/§4.3/§4.4 五处交叉一致。

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

调研对象：同一数学语义 `y = x (x>0) else α·(e^x−1)` 的算法族。算子特征判定（M0/Phase 1）：**单步逐元素**（无规约/窗口/矩阵）→ **轻量调研**（algorithm-research.md §2），四问逐问回答并给依据。

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1，此处只做初判；命中 algorithm-candidates.md ALG-elementwise 条目——该族 R1 手段清单：公共子表达式消除/除法转乘倒数/cast 链合并/恒等变形，逐项对照）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法/输入公式）** | `sel(x>0, x, α·(e^x−1))`（fp32 域负支）：fp32 域 vcmp+vexp+vsub+vmul+vselect = 5 op（manifest 口径 4 FLOPs/元素） | — | — | ✅（基线） |
| 1 | expm1 形式 | 负支 `α·expm1(x)`（select 保留） | **容差内等价**（数学恒等；舍入路径偏离 golden 的直算 `exp(x)−1` 形式——A2 机器证据 16692 mismatches@α=1.0） | 小 |x| 相对精度更高 | ❌ 双重否决：① `vexpm1` 不存在——docs/Tilelang.language/ 全部 16 个子目录已枚举 + grep `expm1` 零命中（`T.ceildiv/rsqrt/sigmoid/vabs/vadd/vcos/vdiv/verf/vexp2/vexp/vln/vlog2/vmul/vpow/vrelu/vsin/vsqrt/vsub/vtanh` 全清单亲核，无 expm1 家族 API）；② 非 golden-bitwise（golden 保真优先，PL-1.17 先例） |
| 2 | max 条件式 | `max(x, α·(e^x−1))` | **结构性不等价**——`e^x > 1+x` 对一切 x≠0 严格成立 → 正轴上 `α·(e^x−1) > x`，max 恒取负支表达式（A4 机器证据 EQUIV_FAIL：32765 mismatches ≈ 正半轴全量） | −1 op（若等价） | ❌ 非等价，直接淘汰 |
| 3 | 重构式 | `α·e^x − α`（select 保留） | **容差内等价但非 golden-bitwise**——两次舍入（α·e^x 舍一次、减 α 舍一次）vs 基线单次（A5 机器证据 15179 mismatches@α=0.1；α=2^k 时恰等但契约允许任意 finite α） | 理论上可省 one 常量缓冲 | ❌ 非 golden-bitwise；且 one 常量已由 vsub 标量字面量承载（零缓冲），重构式无收益 |
| 4 | α=1 特化（省 vmul） | `sel(x>0, x, e^x−1)`（α=1 时 1.0·v 为 IEEE 恒等，跳过 vmul） | **数学恒等（bitwise）**——A3 机器验证 3/3 行 0 ulp / 0 violations | −1 vmul/元素（4→3 FLOPs） | ⚠️ 纳入对比；**评估后不进基线设计**：计算链优化在 copy-floor 形态下 ROI≈0（CONST-copy-floor-method：lerp 7-pass 链 ≥16M 全档被 MTE2 窗口完全隐藏 delta ≤1.1µs，本链 7-op 同判；−1 op 的墙钟收益 ≤0.15µs@16M），而代价是工厂期第 3 分支维度（2 dtype 路线 × 2 α 形态）与伪代码复杂化——**保留为 Stage 4 可选杠杆**（机器证据已留存，启用零风险，§6.3） |
| 5 | fp16 原生域链 | fp16 域 vcmp/vexp(原生 √)/vsub/vmul/vselect 逐 op 舍入 | **容差内等价但偏离源路径**——源比较与负支均在 fp32 域（§0.1），原生域逐 op 舍入 vs golden 最大 32767 ulp（A6 机器证据 16448 mismatches） | 省 2×vcast | ❌ golden/source 对齐优先（PL-1.17 先例：transit 0 ulp 优于原生 1 ulp）；bf16 原生域更不可行（vexp bf16 ×） |
| 6 | exp2 降代换 | `e^x = vexp2(x·log2e)` | 数学恒等（浮点舍入另计） | 无 | ❌ +1 vmul（log2e 预乘）换 1 个 vexp，op 数不减反增；无 vexp 吞吐劣势证据，弃选 |
| 7 | 公共子表达式消除 | — | 结构上无重复子式（单 exp、单比较） | 无 | ❌ 无 CSE 空间 |

R1 结论：**基线公式即公式级最简形态**（候选 1/3/5/6 弃选有机器证据或结构性理由，候选 2 结构性不等价，候选 4 bitwise 安全但 ROI≈0 留作杠杆，无 CSE 空间）；cast 链已由 §0.6 R2 定为最小单次舍回（2×vcast，bf16 路线本就必需）。

**R2 在线算法**：**天然单遍，无需在线变体**。结构依据（algorithm-research §3 R2 判据）：输出 `y[i]` 仅依赖 `x[i]`，无 running 统计量（和/max/计数）、无跨元素依赖——逐元素映射本身即单遍流式；源算法亦为单遍（§0.4 无 online 手段），一致。收益口径不适用（无两遍/三遍结构可化）。

**R3 算法复杂度对比**（带宽受限算子——访存为主导项；口径：FLOPs 含每元素全部向量 op（manifest 4/元素口径）；Bytes 为 GM 流量含输入读+输出写、不含 tile 常量广播；缓冲为 UB 峰值驻留）：

| 算法候选 | FLOPs（op/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------|---------------|---------|-------------|---------------------|
| 基线 `sel(x>0, x, α·(e^x−1))`（+fp32 transit 落地形态） | manifest 口径 4（fp32 域 5 op 链；fp16/bf16 另加 2×vcast 属 dtype 路径） | **2N·B**（1 读 + 1 写，无中间 GM 往返——tile 驻留 UB）= **I/O 下界** | 1（每元素每张量恰读一次写一次） | O(block_size)：fp32 17B/elem、fp16/bf16 19B/elem（§4.5） | N/block_size 个完全独立 tile；persistent 零跨核同步 |
| 候选 4 α=1 特化 | 3（−1 vmul） | 2N·B（同） | 1 | 同基线（alpha32 若保留分配则同） | 同上 |
| 候选 1 expm1（假设 API 存在） | 3（−1 vsub，expm1 替代 exp+sub） | 2N·B（同） | 1 | −4B/elem（省 one 操作数……实际 one 已零缓冲） | 同上 |
| 候选 2 max 形式 | 2（若等价） | 2N·B（同） | 1 | 同基线 | 同上 |

四口径结论：基线在 FLOPs 并列（仅结构性不等价的候选 2 少 op、候选 4 −1 属计算链优化——**PL-1.6 copy-floor 实证 ≥16M 档计算链被 MTE2 窗口完全隐藏（ROI≈0）**）、访存已达 I/O 下界、单遍、缓冲最小档。**基线并列最优，无算法族替换**（与 leaky_relu/lerp 调研同构结论）。

**R4 硬件亲和性评估**（逐候选对照检查清单，algorithm-research §3 R4；评估对象 = 基线 + transit 落地形态）：

| 检查项 | 基线（+transit 落地形态） | 结论 |
|--------|--------------------------|------|
| 计算单元匹配 | vcmp/vexp/vsub/vmul/vselect/vcast/vbrc 全为 Vector 单元 op（逐元素/超越/比较/选择/转换族）；无 MAC 密集段，无 Cube 需求 | ✅ 亲和 |
| 片上容量 | 工作集 = 17–19B/elem × block_size ≤ 8192 → ≤155.6KB < UB 192KB（§4.5） | ✅ 有界 |
| 对齐/整除 | UB 切片零起点（官方 vec_add_1d 对齐模式）；尾轴对齐由 GM 侧连续 1-D + [0:tail] 拷出消化；v-prefix 全宽度 lane 并行（lane 宽度/lane 组大小为编译器实现细节，不作设计依赖），尾 tile 陈旧 lane 由拷出切片屏蔽、浪费 O(1)/tile | ✅ |
| 静态边界 | N 为 builder 工厂特化编译期常量 → num_logical/num_local_tasks/tail 上界全部 trace 期折叠（leaky_relu/lerp 同款 host 常量折叠） | ✅ |
| 流水可融合性 | 单阶段 copy→compute→copy；Developer 自动多缓冲（§6.3） | ✅ |
| 跨核结构 | 无跨核归并（逐元素独立 tile），persistent 零核间同步 | ✅ |
| dtype 路径 | **bf16 无 vexp/vmul/vsub（T.vexp.md/T.vmul.md/T.vsub.md §2.2.1 bf16 ×，逐一亲读）→ fp32 transit 唯一路径；fp16 transit 为 golden/source 对齐（A1 0 ulp vs A6 原生域 16448 mismatches 机器证据）**；vexp 约束 src/dst 同形状同 dtype（T.vexp.md §2.1）→ 独立 w32 输出缓冲 | ✅（transit 形态） |

**调研结论（选定算法族 + 依据）**：

1. **选定**：基线公式 `sel(x>0, x, α·(e^x−1))`（fp32 域负支）+ §0.6 重设计落地形态（persistent 一维分核 + UB tile 驻留 + fp32 域 vcmp/vexp/vsub/vmul/vselect 链 + fp16/bf16 fp32 中转）。依据：R3 四口径基线并列最优（访存达 2N·B 下界）；R4 全项亲和；候选 1/3/5/6 弃选有据、候选 2 结构性不等价、候选 4 ROI≈0（PL-1.6 实测）留作 Stage 4 杠杆。**结构差异（vs 源算法）**：算法族不变（同公式同 dtype 路径），实现结构由两级 SIMT 展开 + 寄存器 fragment → persistent 一维分核 + UB 向量链 + tile 级尾块。
2. **调研范围声明**（「无更优替代」举证）：① 参考表 algorithm-candidates.md ALG-elementwise 条目命中行全部评估（R1 手段清单见 R1 表）；② 恒等变形自查覆盖 expm1/max/重构式/exp2 降代换/α=1 特化/CSE 七族；③ 本仓同类实现 leaky_relu（同族单步逐元素+标量参，persistent+transit 已验证形态）、lerp_tensor（同形 E1/E2 契约）、mish（vexp 生产先例 + 标量操作数先例 + fp32 域全 dtype 策略）、vec_add_1d（官方尾块模式）；④ pattern-library elementwise.md PL-1.6/PL-1.17（copy-floor 与 transit 双证，kb_stale_check 2026-10-28 复核非 stale）+ constants.md 带宽常数（STALE 条目按设计期估算处理）；⑤ R2 结构判据（逐元素天然单遍）；⑥ docs/Tilelang.language/ 全 16 子目录枚举（expm1/exp2/超越函数家族清单亲核）。本地源已覆盖该族全部已知算法形态；未启用互联网检索（简单算子轻量调研，algorithm-research §6：通常无需联网）。
3. **设计期估算下界（D-2，协议 `_shared/standards/hardware-cost-model.md` §2）**：估算下界 = max(流量项, 发射项, 容量项)。
   - 流量项（主导）：引用 pattern-library/constants.md「MTE2 有效带宽随总流量退化」实测条目 `CONST-mte2-degradation`（**版本戳：tilelang 0.1.2+ed787bb / Ascend 910B2C / CANN 8.5.0 / 48 vector core；kb_stale_check 2026-10-28 判定 STALE（当前 stamp 88ee8c302e）——按「设计期估算 + Stage 4 实测校准」口径引用**）——1M 档 33.9 GB/s/core、16M 档 30.2 GB/s/core。本设备 910B3 / 40 vector core（§5.5 实查），**跨设备缩放按核数比例折算，标注为设计期假设**（40/48 ≈ 0.83 → 1M ≈ 28.2、16M ≈ 25.2 GB/s/core；40 核聚合 ~1.0–1.3 TB/s 量级）。
   - 发射项：7 op/元素（fp16/bf16 transit 链：2 vcast + vcmp + vexp + vsub + vmul + vselect）/ 5 op（fp32）——lerp 7-pass 链在 ≥16M 已被 MTE2 窗口完全隐藏（`CONST-copy-floor-method` 实测 delta ≤1.1µs@≥16M、0.34µs@16M；同条 STALE，同口径引用），7-op 链同判隐藏，发射项 < 流量项。
   - 容量项：UB 驻留 ≤155.6KB < 192KB（§4.5），非约束（轻量调研·单步逐元素豁免，hardware-cost-model.md §2）。
   - **估算下界行**（设计期，待 Stage 4 实测校准）：

| workload | N | dtype | GM 流量 | 估算下界 |
|----------|---|-------|---------|---------|
| smoke-1m | 2²⁰ | fp32 | 8.4 MB | ≈ 8 µs |
| smoke-1m | 2²⁰ | fp16/bf16 | 4.2 MB | ≈ 4 µs |
| mlp-hidden [2048,4096] | 2²³ | fp16/bf16 | 33.6 MB | ≈ 28–34 µs |
| mlp-hidden-wide [2048,8192] | 2²⁴ | fp16/bf16 | 67.1 MB | ≈ 55–67 µs |
| gan-feat [16,256,64,64] | 2²⁴ | fp16/bf16 | 67.1 MB | ≈ 55–67 µs |

   常数未覆盖量：910B3 的 MTE2 曲线（跨设备缩放假设）、1:1 R/W 混合（PL-1.6 为 lerp 3:2 混合）、L2 驻留档（bench harness 同输入复用可能命中 43.1 GB/s/core L2 口径——ada_layer_norm 补充条目，估算将偏保守）——**均为未实测假设，估算依据 = PL-1.6 曲线 × 核数比例；Stage 4 以 msprof 实测校准并回写**。
   - **canonical 估算下界行**（hardware-cost-model.md §2 格式；代表 workload = smoke-1m fp32，N=2²⁰）：**估算下界: ≈8 µs（流量 ≈7.4 µs = 8.4 MB ÷ ~1.13 TB/s〔1M 档 33.9 GB/s/core × 40/48 核数缩放，设计期假设〕 + 发射 ≈0〔7-op 链被 MTE2 窗口完全隐藏，CONST-copy-floor-method 实测 ≥16M delta ≤1.1 µs，STALE 条目按设计期估算口径引用〕 + 容量 0〔轻量调研·单步逐元素豁免，hardware-cost-model.md §2；UB 驻留 ≤155.6KB < 192KB〕，引用 CONST-mte2-degradation〔STALE〕；版本戳 tilelang 0.1.2+ed787bb / CANN 8.5.0 / Ascend 910B2C / 48c，当前工具链 88ee8c302e 需 Stage 4 重验）**。其余 workload 行见上表（同口径拆分：流量项主导、发射项被隐藏、容量项豁免）。

#### 1.6.1 数学等价优化（公式级）

**（a）公式级降算优化：采纳 0 项进基线设计（α=1 特化评估后不采纳，其余候选弃选）；dtype 路径采纳 1 项（A1，实为源语义的结构承接）。**

依据：§1.6.0 R1 候选表 8 行逐一评估——候选 1（expm1）API 缺失（docs 全 16 子目录枚举 + grep 零命中）且非 golden-bitwise（A2 机器证据）、候选 2（max）结构性不等价（A4 等价性验证未通过）、候选 3（重构式）非 golden-bitwise（A5）且无缓冲收益、候选 4（α=1 特化）**bitwise 安全（A3 机器验证 0 ulp）但墙钟收益 ROI≈0**（CONST-copy-floor-method：计算链被 MTE2 窗口完全隐藏；−1 vmul 的收益 ≤0.15µs@16M，代价是工厂分支维度加倍与伪代码复杂化——与 leaky_relu R3 弃选 −1-op vmax 候选的同款 ROI 判据一致），**不进基线设计、机器证据留存、保留为 Stage 4 可选杠杆**（启用方式见 §6.3）；候选 5（fp16 原生域）偏离源 fp32-opmath 路径（A6 机器证据）、候选 6（exp2 降代换）op 数不降反增、无 CSE 空间；且基线访存已达 I/O 下界 2N·B，计算链在 ≥16M 档被 MTE2 窗口完全隐藏（PL-1.6 实测）——**公式级无进基线设计的降算/降访存空间**，按原式（源 fp32 域形态）设计。**机器证据归属**（gate-and-retry.md §1）：弃选/否决候选（A2/A4/A5/A6）与评估后不采纳项（A3）的机器证据行归档于 §1.6.3.1，本节仅保留采纳项 A1 的结果表（(b)）。

**（b）dtype 路径等价性（§0.6 R2 采纳项 A1，四要素 + 机器验证）**：

- **原式**（源基线，逐语句）：`zero = f32(0)；a = f32(α)；one = f32(1.0)；v32 = f32(x)；y = sel(v32 > zero, x, cast_D(a·(exp(v32) − one)))`（比较 fp32 域、正支原生直传、负支 fp32 域计算 + 单次舍回——D ∈ {fp16, bf16}）。
- **优化后公式**（NPU transit 候选，select-then-round 形态）：`x32 = rint(f32·x)（无损）；cond = x32 > 0；w32 = exp(x32)；w32 = w32 − 1.0；w32 = f32(α)·w32；y = rint_D(sel(cond, x32, w32))`（单次舍回）。
- **等价性论证**：① f16/bf16 ⊂ f32，升变换逐值无损且保序 → `x32 > 0` 与源 `v32 > 0` 严格一致（同一值域同序）；② fp32 域 `exp→sub→mul` 与源负支 `a·(exp(v32)−one)` 同序同域逐 op 单次舍入，中间量逐位一致；③ select-then-round vs 源 round-then-select：正支 `rint_D(x32) = x`（升域无损的逆 = 恒等），负支两侧同为 `rint_D(w32)`——**逐位等价**；④ 与 golden（`F.elu(x.float(), α).to(D)`，fp32 opmath + 单次舍回）同路径 → 逐位一致。fp32 dtype 无 transit（原生域 = 源域 = golden 域，逐位一致）。±0 角点：NPU 负支给出 `α·(1−1) = +0.0`，golden ATen 恒等支给出 −0.0——**IEEE 值相等**（== 成立），L0-5 断言按值等价处理（§8.3/§9.1#3）。
- **收益量化**：非 FLOPs 收益——**正确性/精度收益**：bf16 路线唯一可行路径（vexp/vmul/vsub bf16 ×）；fp16 路线消除原生域逐 op 舍入分歧（A6 原生域 16448 mismatches / max 32767 ulp → transit 0 mismatches / 0 ulp），与 golden 逐位一致消除 Stage 3 精度门禁返工风险（lerp 实证此类分歧 Stage 3 才暴露损失 2485s）；代价 +2 vcast/tile（fp16/bf16），同指令形态于 bf16 路线本就必需。
- **机器验证（D-1，`verify_equiv.py` 已执行，torch CPU + 角点 ±Inf/NaN/±0/次正规/max-finite(torch.finfo)/exp 下溢带 −104/−144/±89 溢出 lane/1e-8 消除区；每 case 派生种子可复现）**：

| 采纳项 | dtype | alpha | cand vs golden max ulp | cand vs golden 违反 | 基线 vs golden max ulp | 基线 vs golden 违反 | cand vs 基线 ulp | 结论 |
|--------|-------|-------|------------------------|---------------------|------------------------|---------------------|------------------|------|
| A1 fp32-transit | float16 | 1.0 | 0 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | float16 | 0.5 | 0 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | bfloat16 | 1.0 | 0 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | bfloat16 | 0.5 | 0 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | float32 | 1.0 | 0 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | float32 | 0.5 | 0 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |

**（c）弃选候选机器证据归属**：按 gate 规范（`.agents/skills/_shared/standards/gate-and-retry.md` §1），本节仅保留采纳优化项（A1）的等价性验证结果表；弃选/否决候选（A2/A4/A5/A6）与评估后不采纳项（A3）的机器证据行（mismatches/violations 计数 + 佐证路径或「未文档化」标注）**归档于 §1.6.3.1**——同由脚本 `examples/elu_fwd/_make_elu_kernel/verify_equiv.py` 产出，与本节 (b) A1 表为同一次执行（2026-10-28 复跑核对一致）。

**优化结论**：采纳 A1——fp16/bf16 走 fp32 transit 单次舍回（机器验证 6/6 EQUIV_PASS，脚本 `verify_equiv.py`，结果表见上 (b)；A3 特化 bitwise 安全但不进基线设计，弃选/不采纳机器证据归档 §1.6.3.1）；公式级无进一步进基线设计的优化空间（R1 全部候选弃选有机器证据/结构性理由/ROI≈0 判据 + 基线访存已达 I/O 下界 2N·B），最终公式 = 源式 fp32 域形态 `sel(x>0, x, α·(e^x−1))` + A1 dtype 路径，与 §1.4/§3.1/§4.4 一致。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案全部循环与标量计算点盘点（逐项判定）：

| # | 计算点 | 类型 | 替代方案 | API 佐证 | 结论 |
|---|--------|------|---------|---------|------|
| V1 | 逐元素 `if_then_else(v32>0, val, α·(exp(v32)−1))`（源算法核心计算） | 逐元素标量选择+超越计算 | `vcmp("gt")` + `vexp` + `vsub`（标量 1.0）+ `vmul` + `vselect` 全 buffer 向量链 | docs/Tilelang.language/比较操作/T.vcmp.md、数学操作/T.vexp.md（fp16/fp32 √）、数学操作/T.vsub.md（标量 src1 §2.1 + 原地 §2.3.3）、数学操作/T.vmul.md、条件操作/T.vselect.md；先例 mish L77–L81（vexp 生产链 + 标量操作数）、gqa_prefill（vcmp+vselect 生产使用） | **向量化 ✅** |
| V2 | 逐元素 `if idx < N` 尾 guard（源 fp8 分支） | 逐元素标量分支 | tile 级 `tail = T.min(block_size, N−t0)` 切片 + 零起点 UB 拷出 | 官方 `examples/elementwise/vec_add_1d.py` 尾块模式；leaky_relu 同款已验证 | **向量化 ✅（消除）** |
| V3 | 标量常量 zero/one/alpha（源内联 fp32 cast） | 标量广播 | zero32/alpha32：`T.vbrc` 每 kernel 一次（persistent 循环外）；one：`T.vsub(w32, 1.0, w32)` 标量字面量操作数（免 one32 缓冲） | docs/Tilelang.language/shape操作/T.vbrc.md §2.2.2（scalar→tensor，fp32 ✓；标量源须 @T.prim_func 体内 let 绑定——leaky_relu L229–240 先例）；T.vsub.md §2.1（src1 标量）+ mish L78 先例 | **向量化 ✅** |
| V4 | fp16/bf16→fp32 域转换（源 v32 cast 的向量形态） | 逐元素 cast | `T.vcast(..., round_mode="rint")` 全 buffer（升域 + 降域各一） | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 rint / bf16→f32 rint / f32→f16·bf16 rint） | **向量化 ✅** |
| V5 | persistent 核内 `for i in T.serial(num_local_tasks)`（tile 调度循环） | tile 级标量循环体 | 不可向量化——**理由**：任务映射循环（每迭代处理一个不相交 tile，循环携带 block_id/t0/tail 的 block 级索引算术，O(1)/tile 非逐元素热点）；tile 间无逐元素计算可合并 | — | **保留（理由：block 级索引/任务映射，可接受类别）** |
| V6 | `tail = T.min(block_size, N − t0)` / `block_id = i·num_kernels + cid` / `if block_id < num_logical` | 标量算术/分支 | 不可向量化——**理由**：block 级索引与动态边界处理（依赖 N 的尾块语义，无法静态化为全 tile——N 本身工厂特化常量但 tail 随 tile 变化；零起点切片方案已将逐元素分支静态化，此标量仅 O(1)/tile） | — | **保留（理由：动态边界 + block 级索引）** |
| V7 | host 侧 N/alpha 校验、block_size UB guard、展平 reshape、lru_cache | host 标量 | 不在 kernel 内（host 元数据/契约校验/编译缓存） | — | **保留（理由：host 侧元数据）** |

**向量化结论**：7 个循环/标量计算点中 4 处以 v-prefix 向量操作等价替代（V1 逐元素选择+超越链 → vcmp+vexp+vsub+vmul+vselect、V2 逐元素尾 guard → tile 级 tail 切片消除、V3 标量常量 → vbrc 广播 + 标量字面量操作数、V4 dtype 域转换 → vcast，API 佐证见上表），3 处保留（V5 tile 调度循环——任务映射类 / V6 block 级索引与动态边界 / V7 host 元数据，逐项理由见上表，无"实现简单"式保留）；本结论与 §6 循环结构一致（tile 内无显式逐元素循环，v-prefix 全 buffer 形态）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

- **I/O layout（契约，非设计变量）**：输入/输出均为展平连续 1-D `(N,)`（host `contiguous().reshape(-1)`，Op 层契约；manifest 任意 ND 输入经 host 展平）。
- **候选枚举**：本算子核内仅一条数据轴（展平元素轴），**无第二非 batch 维**——属"纯 1-D 逐元素"单候选轴情形（豁免多轴枚举，理由：无广播轴、无归约轴、无窗口轴；ND→1D 展平在 host 视图完成，核内无重排自由度）。候选：①（唯一主选）**连续元素轴 lane 映射**——UB buffer `(block_size,)` 一维连续，v-prefix op 全宽度 lane 并行（lane 宽度/lane 组大小为编译器实现细节，本设计不依赖具体数值）；②（枚举即弃）核内重排布局——无收益场景（无跨轴访问、无 bank 冲突问题域），且引入 repack 代价，弃选（量化理由：重排仅为负收益，PL-1.1 转置链 µs 级开销纯增）。
- **向量化宽度与尾浪费**：v-prefix op 全 buffer 全宽度 lane 并行（设计依赖的仅是「整 buffer 计算 + `[0:tail]` 拷出」的屏蔽机制，官方 vec_add_1d 同款，不依赖具体 lane 宽度数值）；尾 tile `[tail:]` 陈旧 lane 由拷出切片屏蔽（正确性无碍，尾浪费 O(1)/tile）。**附加说明**：vexp 对正轴大值 lane（x ≥ 89）产生 exp 溢出 Inf——发生在被 vselect 屏蔽的 lane 上（该 lane cond=true 选 x32），lane 独立无 trap，mish 全 buffer v-op 同形先例；L0-5 角点覆盖。
- **与 §3.3/§4/§6 一致性**：主选轴 = 展平连续轴，UB 驻留缓冲一维 `(block_size,)`，计算链无跨步系数——§3.3 伪代码、§4 缓冲表、§6 循环结构均按此布局（三方一致）。
- **实验裁决模式**：不触发——判定不依赖未实证常数（单候选轴；带宽常数引用 PL-1.6 已实测条目——STALE 条目按设计期估算处理并留 Stage 4 校准条款，非判定裕度依赖）。

**布局决策结论**：数据布局 = GM → UB（一维连续 `(block_size,)`，零起点切片）→ Vector 单元（v-prefix 全宽度 lane）→ UB（**结果统一写回载入 buffer x_ub：fp16/bf16 经 ⑥ 降域复用写回、fp32 经 ⑤ Out=A 原地写回**）→ GM；主选轴 = 展平连续元素轴（单候选轴、一维分块），NZ 分形布局不适用（纯逐元素无矩阵语义、无跨轴访问自由度）；尾 tile 陈旧 lane 浪费 O(1)/tile 由 `[0:tail]` 拷出屏蔽（lane 宽度为编译器实现细节）——与 §3.3 伪代码、§4 缓冲表、§6 循环结构三方一致。

#### 1.6.3.1 公式级弃选/否决候选机器证据（非采纳项归档，原 §1.6.1(c) 移入）

> **归属说明**：按 gate 规范（`.agents/skills/_shared/standards/gate-and-retry.md` §1），§1.6.1 仅保留采纳优化项的等价性验证结果表；弃选/否决候选的机器证据行归档本小节。全部行由脚本 `examples/elu_fwd/_make_elu_kernel/verify_equiv.py` 产出（torch CPU、每 case 派生种子可复现，2026-10-28 复跑采纳项 6/6 通过与 §1.6.1(b) 一致）。措辞注记：机器验证未通过的候选在本表一律记为「等价性验证未通过」。

| 候选 | dtype | alpha | mismatch 元素数 | 容差违反 | 机器证据摘要（verify_equiv.py） | 佐证路径（仓库相对路径）或「未文档化」标注 | 处置 |
|------|-------|-------|----------------|---------|-------------------------------|----------------------------------------------------|------|
| A3 α=1 特化（省 vmul） | fp16 / bf16 / fp32 | 1.0 | 0（3 行全 0 ulp） | 0 | 等价性验证通过（bitwise 恒等：cand vs golden 0 ulp / 0 violations，脚本 run_a3 段） | `examples/elu_fwd/_make_elu_kernel/verify_equiv.py`（run_a3）+ ROI 判据 pattern-library elementwise.md PL-1.6（CONST-copy-floor-method 实测：≥16M 计算链被 MTE2 窗口完全隐藏） | 评估后不采纳——ROI≈0（−1 vmul 收益 ≤0.15µs@16M < 工厂分支维度加倍代价），留作 Stage 4 杠杆（§1.6.1(a)/§6.3） |
| A2 expm1 形式 | float32（公式域） | 1.0 | 16692 | 0 | 等价性验证未通过 golden-bitwise（容差内，16692 mismatches——expm1 vs 直算 exp(x)−1 舍入路径分歧，脚本 run_rejected A2 段）；**vexpm1 API 缺失（未文档化）** | 未文档化标注：docs/Tilelang.language/ 全 16 子目录枚举亲核（数学操作/ 目录 vexp/vexp2/vadd/vmul/vsub/vpow/vsin/vcos/vtanh/verf/vln/vlog2/vsqrt/vdiv/vabs/vsigmoid 全清单，无 expm1 家族）+ grep expm1 零命中（§1.6.0 R1#1）；机器证据 `examples/elu_fwd/_make_elu_kernel/verify_equiv.py`（run_rejected A2） | 弃选（API 缺失 + 非 golden-bitwise） |
| A2 expm1 形式 | float32（公式域） | 0.1 | 16165 | 0 | 同上（16165 mismatches，容差内） | 同上（同脚本 run_rejected A2 段，alpha=0.1 case） | 同上 |
| A4 max 形式 | float32（公式域） | 1.0 | 32765 | 26990 | **等价性验证未通过**（26990 容差违反——`e^x > 1+x` 对一切 x≠0 严格成立，正轴上 α·(e^x−1) > x，max 恒取负支表达式，结构性不等价；脚本 run_rejected A4 段） | 结构论证 §1.6.0 R1#2（数学恒等式层面，非 API 缺失类断言，无「未文档化」问题）；机器证据 `examples/elu_fwd/_make_elu_kernel/verify_equiv.py`（run_rejected A4） | 弃选（结构性不等价，非容差可弥合） |
| A5 重构式 α·e^x−α | float32（公式域） | 0.1 | 15179 | 0 | 等价性验证未通过 golden-bitwise（容差内，两次舍入 vs 基线单次舍入；脚本 run_rejected A5 段）；缓冲动机不成立——one 常量已由 vsub 标量字面量承载（零缓冲） | docs/Tilelang.language/数学操作/T.vsub.md §2.1（src1 标量操作数）+ `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` L78（vadd 标量字面量生产先例）+ `examples/elu_fwd/_make_elu_kernel/verify_equiv.py`（run_rejected A5） | 弃选（非 golden-bitwise + 无缓冲收益） |
| A6 fp16 原生域链 | float16 | 1.0 | 16448 | 0 | 等价性验证未通过 golden-bitwise（容差内，max 32767 ulp——逐 op fp16 舍入 vs 源 fp32 opmath；脚本 run_rejected A6 段 + cand_fp16_native） | docs/Tilelang.language/数学操作/T.vexp.md §2.2.1（fp16 √——API 可用但舍入路径偏离源/golden）+ T.vexp.md/T.vmul.md/T.vsub.md §2.2.1（bf16 ×——原生域对 bf16 更不可行）+ pattern-library elementwise.md PL-1.17（leaky_relu/lerp transit 0 ulp 优于原生域双证）+ `examples/elu_fwd/_make_elu_kernel/verify_equiv.py`（run_rejected A6） | 弃选（golden/source 对齐优先） |

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（Stage 0 prompt 显式指定；落地侧 wrapper `ascend_mode = "Developer"`（其 L148）同证）。

### 2.2 选型理由

1. 调度 prompt 契约 + wrapper `ascend_mode` 契约（最高优先级，双源一致）。
2. 算子形态匹配：单阶段 copy→compute→copy 逐 tile 流水，无跨核协作、无手动流水插桩需求——Developer 自动同步/自动多缓冲已覆盖（同族先例 leaky_relu/lerp/mish 同判）。
3. v-prefix API（vcast/vcmp/vexp/vsub/vmul/vselect/vbrc）均为 Developer Op 形态（docs/Tilelang.language/ 各 §1 标注 `[Developer Op]`）。

### 2.3 模式影响

- 同步：编译器自动插入 UB 读写栅栏，无手动 `set_flag/wait_flag`（§7）。
- 多缓冲：auto multi-buffer（Stage 2 编译期自动 stage 化），PL-1.6 实证可将 7-op 计算链完全隐藏于 MTE2 窗口。
- 模块头 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`（leaky_relu/mish 同款，防 Expert/Developer 混跑进程互相污染——kernel_base CG-2026-0010 同源问题域）。

## 3. API 映射设计

### 3.1 公式拆解

优化后公式（= §1.6.1(b) A1 采纳式 + §1.6.0 选定落地形态）拆解为逐 tile 原语序列：

```text
D = dtype（fp16|bf16|fp32）；域输入 = x32（transit）或 x_ub（fp32，x32 ≡ x_ub）
① 载入    x_ub[0:tail]      ← GM x[t0:t0+tail]
② 升域    x32[0:tail]       ← rint(fp32 ← x_ub)            （仅 fp16/bf16）
③ 比较    cond[0:tail]      ← gt(域输入, zero32)            （fp32 域，严格 >）
④ 指数    w32[0:tail]       ← exp(域输入)                   （全 buffer；被屏蔽 lane 的溢出 Inf 无害）
⑤ 减一    w32[0:tail]       ← w32 − 1.0（标量字面量，原地）
⑥ 乘α    w32[0:tail]       ← alpha32 ⊙ w32（原地）          （= alpha·(exp(x)−1)，源同序同域）
⑦ 选择    结果 ← sel(cond, 域输入, w32)（原地写回，按路线分列）：
          transit（fp16/bf16）: w32[0:tail] ← sel(cond, x32, w32)   （Out=B，结果驻留 w32，仅同迭代被 ⑧ 消费）
          fp32:                x_ub[0:tail] ← sel(cond, x_ub, w32)  （Out=A 原地写回载入 buffer，gqa_prefill L875 先例）
⑧ 降域    x_ub[0:tail]      ← rint(D ← w32)                 （仅 fp16/bf16；复用载入 buffer）
⑨ 写出    GM y[t0:t0+tail]  ← x_ub[0:tail]                  （两路线统一拷出源 = x_ub，§1.5/§3.3/§4.3/§4.4 一致）
预备（每 kernel 一次）⑩ zero32 ← vbrc(0.0)；⑪ alpha32 ← vbrc(alpha)
```

### 3.2 TileLang API 映射

| 语义步骤 | API（v 前缀优先） | 关键参数 | 文档佐证（亲读） |
|---------|------------------|---------|-----------------|
| ① 载入/⑨ 写出 | `T.copy` | `T.copy(x[t0:t0+tail], x_ub[0:tail])` 零起点切片 | docs/Tilelang.language/内存操作/T.copy.md；官方 vec_add_1d / leaky_relu / mish 先例 |
| ②⑧ 域转换 | `T.vcast` | `round_mode="rint"`（f16→f32 / bf16→f32 仅 rint；f32→f16/bf16 rint） | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1 |
| ③ 比较 | `T.vcmp` | `T.vcmp(x32, zero32, cond, "gt")`（fp32 路线比较对象 = x_ub），cond 为 UB bool | docs/Tilelang.language/比较操作/T.vcmp.md §2.1/§2.3（"gt" 合法；bool 仅片上） |
| ④ 指数 | `T.vexp` | `T.vexp(x32, w32)`——src/dst 同形状同 dtype（§2.1 约束 → 独立 w32）；fp16 √/fp32 √/bf16 ×（bf16 经 transit 后仅 fp32 域使用） | docs/Tilelang.language/数学操作/T.vexp.md §2.1/§2.2.1 |
| ⑤ 减一 | `T.vsub` | `T.vsub(w32, 1.0, w32)`——src1 标量字面量 + dst≡src0 原地 | docs/Tilelang.language/数学操作/T.vsub.md §2.1（src1 tensor 或标量）/§2.2.2（标量广播）/§2.3.3（原地支持）；标量操作数须 kernel 内定义；先例 mish L78 `T.vadd(buf1_ub, 1.0, buf2_ub)` |
| ⑥ 乘α | `T.vmul` | `T.vmul(alpha32, w32, w32)`（原地；fp32 域 ✓，bf16 ×——transit 后仅 fp32 域使用） | docs/Tilelang.language/数学操作/T.vmul.md §2.2.1 |
| ⑦ 选择 | `T.vselect` | transit：`T.vselect(cond, x32, w32, w32)`（Out=B 原地，结果驻留 w32）；fp32：`T.vselect(cond, x_ub, w32, x_ub)`（Out=A 原地写回载入 buffer） | docs/Tilelang.language/条件操作/T.vselect.md §2.2.1（fp16/fp32/bf16 ✓）；Out=A 生产先例 gqa_prefill L875 |
| ⑩⑪ 常量广播 | `T.vbrc` | `T.vbrc(0.0, zero32)` / `T.vbrc(float(alpha), alpha32)`——标量源为 @T.prim_func 体内 let 绑定局部变量 | docs/Tilelang.language/shape操作/T.vbrc.md §2.2.2（scalar→tensor ✓）；leaky_relu L229–240 let 绑定先例（工厂作用域标量触发 rank 诊断——RETROSPECTIVE 教训） |
| 分配 | `T.alloc_ub` | `(block_size,)` 各缓冲；cond 用 `"bool"` dtype | docs/Tilelang.language/内存操作/T.alloc_ub.md；bool UB 先例 gqa_prefill L709、sparse_mla L230 |
| 核/循环 | `T.Kernel(num_kernels, is_npu=True)` / `T.serial` / `T.ceildiv` / `T.min` | persistent 一维分核 | leaky_relu / lerp_tensor 先例；docs/Tilelang.language/数学操作/T.ceildiv.md |

### 3.3 计算伪代码

```python
def _make_elu_kernel(N, dtype, alpha, output_dtype=None, is_fp8=False,
                     threads=256, npt=8):
    # 外层 7 参签名保持源形态（wrapper 仅传 3 位置参，defaults 兼容，§3.5.2/§10.3）；
    # threads/npt 为 K11 折叠后的历史参数：NPU 侧不消费（保留仅为签名兼容 + lru_cache 键空间一致）。
    # host 校验（builder 期；ValueError 非 assert，python -O 安全）：
    #   N <= 0            -> ValueError（§9.3 防御，源无此项）
    #   is_fp8=True       -> ValueError（契约死代码，§0.5 #5）
    #   alpha 非 finite   -> ValueError（承接源 host 校验语义，test_elu_rejects_nonfinite_alpha）
    #   output_dtype 非 None 且 != dtype -> ValueError（manifest same_as(input)）
    #   dtype 归一化后不在 {float16,bfloat16,float32} -> ValueError
    # dtype 入参归一化（leaky_relu RETROSPECTIVE 教训）：wrapper __init__ 传 torch.dtype 对象
    #   （其 L175）、custom_op 链 getattr(torch, dtype_str) 后亦为 torch.dtype（其 L100）、
    #   直调可传 str——torch 2.7.1 的 torch.dtype 无 .name 属性（本环境实测 hasattr=False），
    #   str(torch.float16) == "torch.float16"（带前缀）：
    dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]
    use_transit   = dtype_key in ("float16", "bfloat16")       # §0.6 R2
    default_block = 1024 if dtype_key == "float32" else 2048   # wrapper _BLOCK_SIZE_*（K11 折叠）
    vector_cores  = NPUUtils.get().get_aicore_num() * 2        # 实查，纯 Vector 翻倍（§5.5）

    # host 编译缓存：@functools.lru_cache(maxsize=32) 施加于本外层 builder 函数
    # （键 = (N, dtype, alpha, output_dtype, is_fp8, threads, npt)；leaky_relu L124 同款适配，§0.3 步 1）

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(block_size=default_block):                      # E2 单参契约（K11；leaky_relu L194 同形）
        # host guard（kernel 期）：
        #   block_size <= 0 或 > 8192 -> ValueError（§4.5 UB 预算）
        num_logical     = ceildiv(N, block_size)               # 编译期常量
        num_kernels     = min(num_logical, vector_cores)       # ≤ 物理核
        num_local_tasks = ceildiv(num_logical, num_kernels)    # 静态边界

        if use_transit:                                        # 工厂期 Python if（leaky_relu L218 同款，规避 C11）
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    zero_val  = 0.0                            # 标量源体内 let 绑定（vbrc rank 陷阱规避）
                    alpha_val = float(alpha)
                    x_ub     = T.alloc_ub((block_size,), dtype_key)   # 载入 + ⑧ 降域回写目标
                    x32      = T.alloc_ub((block_size,), "float32")   # 升域输入
                    w32      = T.alloc_ub((block_size,), "float32")   # exp→sub→mul→select 工作缓冲
                    cond     = T.alloc_ub((block_size,), "bool")
                    zero32   = T.alloc_ub((block_size,), "float32")   # 常量（循环外填充）
                    alpha32  = T.alloc_ub((block_size,), "float32")
                    T.vbrc(zero_val, zero32)               # ⑩ 每 kernel 一次
                    T.vbrc(alpha_val, alpha32)             # ⑪（fp32 值参与）
                    for i in T.serial(num_local_tasks):    # 静态边界 grid-stride
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0   = block_id * block_size
                            tail = T.min(block_size, N - t0)          # 尾块（静态上界）
                            T.copy(x[t0:t0+tail], x_ub[0:tail])       # ①
                            T.vcast(x_ub, x32, round_mode="rint")     # ② 无损升域
                            T.vcmp(x32, zero32, cond, "gt")           # ③ 严格 >（fp32 域）
                            T.vexp(x32, w32)                          # ④ 全 buffer（屏蔽 lane 溢出 Inf 无害）
                            T.vsub(w32, 1.0, w32)                     # ⑤ 标量字面量，原地
                            T.vmul(alpha32, w32, w32)                 # ⑥ 原地：α·(exp−1)
                            T.vselect(cond, x32, w32, w32)            # ⑦ Out=B 原地（结果驻留 w32，仅同迭代消费）
                            T.vcast(w32, x_ub, round_mode="rint")     # ⑧ 单次舍回复用载入 buffer
                            T.copy(x_ub[0:tail], y[t0:t0+tail])       # ⑨ 拷出源 = x_ub
        else:                                                  # fp32 原生域（无 vcast，域即 x_ub）
            @T.prim_func
            def main(x: T.Tensor((N,), dtype_key), y: T.Tensor((N,), dtype_key)):
                with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                    zero_val  = 0.0
                    alpha_val = float(alpha)
                    x_ub     = T.alloc_ub((block_size,), dtype_key)   # 载入 + ⑦ Out=A 选择输出（结果驻留 buffer）
                    w32      = T.alloc_ub((block_size,), "float32")
                    cond     = T.alloc_ub((block_size,), "bool")
                    zero32   = T.alloc_ub((block_size,), "float32")
                    alpha32  = T.alloc_ub((block_size,), "float32")
                    T.vbrc(zero_val, zero32)                 # ⑩
                    T.vbrc(alpha_val, alpha32)               # ⑪
                    for i in T.serial(num_local_tasks):
                        block_id = i * num_kernels + cid
                        if block_id < num_logical:
                            t0   = block_id * block_size
                            tail = T.min(block_size, N - t0)
                            T.copy(x[t0:t0+tail], x_ub[0:tail])       # ①
                            T.vcmp(x_ub, zero32, cond, "gt")          # ③（域输入 = x_ub）
                            T.vexp(x_ub, w32)                         # ④
                            T.vsub(w32, 1.0, w32)                     # ⑤
                            T.vmul(alpha32, w32, w32)                 # ⑥
                            T.vselect(cond, x_ub, w32, x_ub)          # ⑦ Out=A 原地写回载入 buffer（gqa L875 先例）
                            T.copy(x_ub[0:tail], y[t0:t0+tail])       # ⑨ 拷出源 = x_ub（两路线统一）
        return main
    return kernel
```

注：① 两 dtype 路线在工厂期以 Python if 静态选择（`use_transit` 为闭包常量，整支 prim_func 分置——leaky_relu L218–L309 同款结构，规避 TVM-script if 块作用域限制 C11；全链无 in-body Python if——α 特化不进基线设计，§1.6.1(a)）。② **fp32 路线不分配独立 x32**——vcmp/vexp 以载入 buffer x_ub 为域输入，选择结果经 Out=A 写回 x_ub（gqa_prefill L875 `T.vselect(ub_cond, ub_f32_ND, ub_neg, ub_f32_ND)` 同形生产先例，亲读核对），⑨ 拷出源与 transit 路线统一为 x_ub。③ transit 路线 ⑦ 为 Out=B 原地形态，结果仅同迭代被 ⑧ 消费——跨迭代携带禁令与诊断指纹见 §9.1#9/§9.2（TRAP-vselect-inplace-carried-state）。④ ⑤⑥ 原地 vsub/vmul 均为「同迭代消费、无 loop-carried」安全形态（lerp 原地 vsub/vmul/vadd 链 bit-exact 同形先例；vsub.md §2.3.3 明文支持原地）。⑤ alpha32/zero32 分配无条件（19B/17B 预算含之，§4.5），填充与消费均在闭包常量控制下——避免 in-body 条件分配的 C11 作用域风险。

### 3.4 API 可行性确认

| API | 存在性证据 | 形态先例（examples/） |
|-----|-----------|----------------------|
| T.vexp（fp32，src/dst 同形状同 dtype） | docs/Tilelang.language/数学操作/T.vexp.md §2.1/§2.2.1（fp32 √，bf16 ×） | mish L77 `T.vexp(x_f32_ub, buf1_ub)`（生产）+ perf_opt L128/L157/L178 |
| T.vsub（标量 src1 + 原地） | docs/Tilelang.language/数学操作/T.vsub.md §2.1/§2.2.2/§2.3.3 | mish L78 标量字面量先例（vadd 形态）；T.vsub.md §2.4 示例二原地先例（`T.vsub(x_ub, max_ub, x_ub)`） |
| T.vcmp "gt" + bool UB | docs/Tilelang.language/比较操作/T.vcmp.md（cmp_mod 列表含 "gt"；bool 输出仅片上） | gqa_prefill `_gqa_prefill_fwd_kernel.py`（vcmp+vand+vselect 生产链）；leaky_relu 已验证 |
| T.vselect | docs/Tilelang.language/条件操作/T.vselect.md（fp16/fp32/bf16 √） | gqa_prefill L875（Out=A 原地写回生产先例）；leaky_relu 已验证（Out=A/Out=B 双形态） |
| T.vmul fp32（原地） | docs/Tilelang.language/数学操作/T.vmul.md（fp32 √，bf16 ×） | lerp_tensor 全链 vmul；lerp 原地 vmul 链 bit-exact |
| T.vcast rint | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 rint / bf16→f32 rint / f32→f16·bf16 rint） | leaky_relu/lerp/mish transit 链（三任务实证 PL-1.17） |
| T.vbrc scalar→UB | docs/Tilelang.language/shape操作/T.vbrc.md §2.2.2 示例 1（标量广播，fp32 ✓） | brc 官方示例；leaky_relu L239–240（体内 let 绑定先例） |
| T.alloc_ub + "bool" | docs/Tilelang.language/内存操作/T.alloc_ub.md | gqa_prefill L709 / sparse_mla L230（bool UB 先例） |
| T.copy 零起点切片 | docs/Tilelang.language/内存操作/T.copy.md | vec_add_1d（官方尾块）、leaky_relu/mish |
| T.Kernel(is_npu=True) + T.serial persistent | leaky_relu / lerp_tensor / mish | persistent 分核已验证形态 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（ascend-constraints.md 强制检测项）

| 检测项 | 结果 |
|--------|------|
| 三维 Kernel | ✅ 不涉及——一维 `T.Kernel(num_kernels)` |
| L0C 容量 | ✅ 不涉及——纯 Vector，无 Cube/L0C 路径 |
| GEMM 非整除 | ✅ 不涉及——无 GEMM；逐元素尾块另有 tail 方案（§6.4） |
| L1 分形限制 | ✅ 不涉及 |
| UB 容量 | ✅ 有界——19B/elem × 8192 = 155.6KB < 192KB（§4.5）；host guard 拒绝超限 block_size |
| bool/i1 写 GM | ✅ 规避——cond 仅片上消费（vcmp.md §2.3 限制条款已核对），vselect/vexp/vsub/vmul 输出均为 fp32/原生 dtype |
| bf16 v-算术/超越缺失 | ✅ 绕开——transit 后全部计算在 fp32 域（§0.6 R2；T.vexp.md/T.vmul.md/T.vsub.md bf16 × 已逐一亲读） |
| 一维 Kernel 核数超发 | ✅ persistent 分核（§5.5） |
| in-place 参数 | ✅ 契约保持——kernel 始终写独立 y，inplace 为 Op 层 copy-back（§0.2） |

#### 3.5.2 参考实现差异说明

| 差异点 | 源（GPU） | NPU 设计 | 理由 |
|--------|----------|---------|------|
| 并行组织 | `T.Parallel(threads, npt)` 两级 SIMT | persistent 一维分核 + v-prefix 全 buffer | §0.5 #4（一维 Kernel 限制） |
| 片上驻留 | register fragment | UB | §0.5 #2 |
| dtype 域 | 比较/负支 fp32 域 + 单次舍回（正支原生直传） | **fp16/bf16 fp32 中转单次舍回（select-then-round）**——逐位承接源路径 | §0.6 R2（bf16 vexp/vmul/vsub 缺失 + golden/source 对齐，D-1 A1 0 ulp） |
| 尾块 | 整块 fragment（依赖 block_size 整除 N 的 grid 收尾）/逐元素 guard（fp8） | tile 级 tail 切片（覆盖非整除——严格强于源） | §0.5 #8 |
| fp8 分支 | is_fp8 双 prim_func + `if idx < N` | 舍弃（ValueError） | §0.5 #5（契约死代码） |
| E2 工厂签名 | `kernel(threads_arg, npt_arg)`（GPU 基类 `init_config` 的 `self.kernel(cfg["threads"], cfg["num_per_thread"])` 两参依赖——**源侧事实**） | **`kernel(block_size=default_block)` 单参**（K11：threads×npt 积折叠为 block_size；default_block 按 dtype fp32 1024 / fp16·bf16 2048，= wrapper `_BLOCK_SIZE_*`） | NPU harness wrapper 已按 K3/K11 移植并以单参调用 factory（elu_fwd.py L100/L175，折叠常量 L82–83，亲读）；threads/npt 属迁移规则「后端相关参数可适配」类目；leaky_relu/lerp 已验证件同形 |
| dtype 入参形态 | GPU 基类经 builder args 传 dtype | **torch.dtype 与 str 双形态接受并归一**：`dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]`（torch 2.7.1 无 `.name`，本环境实测），transit 分支/guard/默认 block 一律基于 `dtype_key` | wrapper `__init__` 传 `self.dtype`（torch.dtype，其 L175），custom_op 链经 `getattr(torch, dtype_str)` 后亦为 torch.dtype（其 L100）；torch 对象与 str 元组直接 `in` 比较恒 False，会误走 fp32 路线（leaky_relu Stage 2 检视问题 1④ + Stage 3 复核教训） |
| alpha 特化 | 无（α 恒参与 vmul） | 不特化（α=1 亦走通用链）——A3 bitwise 安全但 ROI≈0（§1.6.1(a)），留 Stage 4 杠杆 | CONST-copy-floor-method（计算链隐藏实证） |
| host 编译缓存 | builder 无 lru_cache（提取件 functools 导入未用） | `@functools.lru_cache(maxsize=32)` 于外层 builder（N/dtype/alpha 为键，含 alpha 特化空间） | 同族模板适配（leaky_relu L124 同款） |

#### 3.5.3 本项目同类实现参考

1. `examples/leaky_relu/_make_leaky_relu_kernel/_make_leaky_relu_kernel.py` —— 同族（单输入单标量参逐元素 Parametric builder）已验证迁移：persistent 分核、fp32 transit、UB guard、L0 六项套件、E2 单参工厂契约（**最接近模板**；本设计在其结构上增加 vexp/vsub 两级计算与 vexp 独立输出缓冲约束）。
2. `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` —— **vexp 生产先例**（L77）+ 标量字面量 v-op 操作数先例（L78 `T.vadd(buf1_ub, 1.0, buf2_ub)`）+ 全 dtype fp32 域计算策略 + 尾块切片 + trace 期 dtype 分支（L69/L84 in-body if 形态佐证）。
3. `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` —— E1/E2 单参工厂契约 `_make_X(N, dtype) -> kernel(block_size=default_block) -> main` 同形模板；`perf_opt/` 子目录为 persistent nc 调优标杆（Stage 4 参考）。
4. `examples/elementwise/vec_add_1d.py` —— 官方一维逐元素 + 尾块零起点切片模式。
5. `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` —— vcmp+bool UB+vselect Out=A 原地写回生产先例（L875）。
6. `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` —— NPUUtils 物理核数查询与 per-factory 重查先例。
7. verbatim 源件：`examples/TileOPs/tileops/kernels/elementwise/elu_fwd/_elu_fwd_kernels.py`。

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
| 生成方式 | kernel 内新 buffer（`out_idx=[1]`）；inplace 为 Op 层 copy-back（test_elu_inplace：`out is x`） |

### 4.3 中间缓冲区（全部 UB，`(block_size,)` 一维）

| 缓冲 | dtype | 字节/元素 | 角色 | 复用 |
|------|-------|----------|------|------|
| `x_ub` | 输入 dtype | 2（fp16/bf16）/ 4（fp32） | 载入 + 降域回写目标（transit）/ **选择输出与结果驻留 buffer（fp32 路线，⑦ Out=A 原地写回）** | 载入后原值死后复用；两路线 ⑨ 拷出源均为 x_ub |
| `x32` | fp32 | 4 | 升域输入（仅 transit 路线分配；fp32 路线无独立 x32，域即 x_ub） | — |
| `w32` | fp32 | 4 | exp→sub→mul→select 工作缓冲（④ vexp 输出受 §2.1 同 dtype 约束需独立缓冲；⑤⑥⑦ 原地推进，结果仅同迭代被 ⑧/⑨ 消费） | vsub/vmul/vselect 原地写回（迭代内消费，§9.1#9） |
| `cond` | bool | 1 | vcmp 谓词 | 片上消费，不落 GM |
| `zero32` | fp32 | 4 | 比较常量（每 kernel 一次 vbrc） | persistent 循环外填充 |
| `alpha32` | fp32 | 4 | alpha 常量（每 kernel 一次 vbrc；分配无条件——§3.3 注⑤） | 同上 |

### 4.4 内存搬运路径

```text
[fp16/bf16] GM x ──MTE2──> UB(x_ub) ──vcast──> UB(x32) ──vcmp/vexp/vsub/vmul/vselect（Out=B → 结果驻留 w32）──> UB(w32) ──vcast──> UB(x_ub) ──MTE2──> GM y
[fp32]      GM x ──MTE2──> UB(x_ub) ──vcmp/vexp/vsub/vmul/vselect（Out=A 结果原地写回 x_ub）──────────────────────────────────> UB(x_ub) ──MTE2──> GM y
```

纯 Vector 路径：GM↔UB 直连（MTE2/MTE3），无 L1/L0 介入；计算全在 Vector 单元 + UB。

### 4.5 UB 内存预算

| dtype | 缓冲字节/元素 | block_size 上界（196608B UB） | 保守 cap | 默认 block（E2 契约） |
|-------|--------------|------------------------------|---------|----------------------|
| float32 | 4+4+1+4+4 = **17B**（fp32 路线无独立 x32：vcmp/vexp 以 x_ub 为域输入） | 11565 | **8192** | K11 折叠默认 = **1024**（wrapper `_BLOCK_SIZE_FP32`） |
| float16/bfloat16 | 2+4+4+1+4+4 = **19B** | 10347 | **8192** | K11 折叠默认 = **2048**（wrapper `_BLOCK_SIZE_NON_FP32`） |

- host guard：`block_size > 8192 → ValueError`（leaky_relu 同款 guard 形态，`python -O` 安全用 ValueError 非 assert）。
- auto multi-buffer（Developer 默认）的 UB 占用口径：朴素 ×2 模型（2×19×8192=311KB > 192KB）**不是平台实测口径**——TRAP-UB-multibuffer-inflation（pattern-library traps-compiler.md，status=verified，kb_stale_check 2026-10-28 复核非 stale）实测：膨胀后 footprint 是「结构类平台值」且与 staging 数解耦（transit≈20–21 / fp32≈26–27 B/elem；基础分配差系数 1.10–1.12 且结构依赖）。本算子 resident **19B/elem × 8192 = 155.6KB，与 leaky_relu resident 完全同值**（leaky_relu L1 套件 bs=8192 fp16/bf16 实测编译运行通过；fp32 路线 17B 同其先例）→ **guard 8192 保留（同族同字节实测先例直接覆盖）**；**默认 block（1024/2048）双缓冲峰值 34.8KB / 77.8KB 在含 ×2 朴素模型在内的一切口径下安全**。**Stage 4 调大 block 前必须以编译探针/BishengIR `ub overflow requires` 报文复算，并连带复验 num_kernels**（平台值随 num_local_tasks 变化；风险点 §9.1#4）。

### 4.6 动态轴定义

- 唯一动轴：N（展平元素总数）。处理方式：**builder 工厂按 N 特化**（N 闭包常量 + `lru_cache`；非 kernel 内动态 shape）——N 为编译期常量，num_logical/num_local_tasks/tail 上界全部 trace 期折叠；任意 N 的尾部由 §6.4 tail 方案覆盖。与源 Op 契约一致（`N_total` constructor 参数、每 N 一 kernel）。

### 4.7 JIT 配置

- `@tilelang.jit(out_idx=[1], target="npuir")`；E2 工厂 `kernel(block_size=default_block)`（**单参**，K11 折叠；default_block 按 dtype fp32 1024 / fp16·bf16 2048 = wrapper `_BLOCK_SIZE_*`，leaky_relu L194 同形）。block_size 经工厂闭包在 trace 期折叠为 grid / UB 尺寸编译期常量。
- `TILELANG_ASCEND_MODE=Developer`（模块头 setdefault）。
- pass_configs：默认（无特殊项；Stage 4 可评估 auto multi-buffer stage 数）。

## 5. Tiling 策略

### 5.1 计算类型

纯 Vector（逐元素比较/超越/算术/选择），无 Cube 参与；仅需 UB，无 L1/L0A/L0B/L0C。无 CV 融合（§8.2 说明）。

### 5.2 Block 划分

- **block_size（tile 内元素数）= E2 工厂单参注入**（K11：GPU threads×npt 积折叠为单参 block_size，§3.5.2；不再由 threads_arg×npt_arg 计算）。
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

- 目标设备 **Ascend 910B3**。本设计环境 `import tilelang` 因 dev root 私有 TVM 包缺失不可用，无法直接调用 `NPUUtils.get().get_aicore_num()`；采用**等价实查路径**：ctypes 直调 CANN runtime `libruntime.so` 的 `rtGetAiCoreCount`——与 `NPUUtils` 内部实现**同一运行时调用**（`tilelang/utils/npu_utils.cpp` L124–L137 `getAiCoreNum()` 即封装 `rtGetAiCoreCount`，leaky_relu 任务亲读核对）。查询代码与返回值：

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
| mlp-hidden [2048,4096] | 2²³ | fp16/bf16（2048） | 4096 |
| mlp-hidden-wide [2048,8192] | 2²⁴ | fp16/bf16（2048） | 8192 |
| gan-feat [16,256,64,64] | 2²⁴ | fp16/bf16（2048） | 8192 |

**要素③ 规模判定与分核方案**：**极大规模**——全部 manifest workload 的 num_logical（512–8192）均 > 物理 Vector 核数 40，且无法通过增大 block_size 缩减到 ≤40（UB cap 8192 → smoke-1m fp16 已是 512 个 tile，2²⁴ 需 2048 个，均 ≫ 40）。方案（core-split-strategy.md §1 要素③ 极大规模档）：

- 固定启动 `num_kernels = min(num_logical, 40)`（编译期常量；小 N 自动退化为 num_logical ≤ 40，同模板全覆盖）。
- 核内串行：`for i in T.serial(num_local_tasks)`，`num_local_tasks = ceildiv(num_logical, num_kernels)`——**静态边界**（N/block_size/40 皆 trace 期常量）。
- 任务映射：`block_id = i × num_kernels + cid`（grid-stride），`if block_id < num_logical` 屏蔽越界。
- 负载分布：1M fp16 → 40 核 × ≤13 任务；2²³ → 40 × ≤103；2²⁴ → 40 × ≤205——核启动 40 次（对比超发 8192 次串行调度），负载不均 ≤1 任务/核（≤0.5%，可接受）。
- **失败信号**（供 Stage 3/4 对照）：① num_local_tasks 依赖运行时值（非静态）→ 违反标准；② msprof 核串行排队/启动开销占比异常 → 检查 num_kernels 是否误用逻辑核数；③ 末核任务数 2 倍 → 核对 min() 与 ceildiv 上界。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 循环 | 形态 | 边界 | 选择理由 |
|------|------|------|---------|
| 核间 | `T.Kernel(num_kernels, is_npu=True)`（隐式 block 并行） | `min(num_logical, 40)` 编译期常量 | persistent 分核（极大规模档，§5.5） |
| 核内任务 | `T.serial(num_local_tasks)` | 静态（ceildiv(num_logical, num_kernels)，trace 期常量） | grid-stride 任务映射，摊薄核启动开销；静态边界满足 core-split 标准 |
| tile 内计算 | 无显式循环——v-prefix 全 buffer op 隐式向量并行（§1.6.2 V1–V4） | block_size | Vector 单元全宽度 lane 并行 |

无 `T.Pipelined`（单阶段 tile 流水，Developer 自动多缓冲承担 stage 化）；无 `T.Parallel`（v-prefix op 已是全 buffer 向量形态，无需再展开）。

### 6.2 循环伪代码

见 §3.3（`for i in T.serial(num_local_tasks)` + `if block_id < num_logical` + tail 切片即完整循环结构）。

### 6.3 流水线优化

- **Developer 自动多缓冲**：copy→compute→copy 每 tile 单阶段，编译器自动 stage 化 GM→UB 载入与 UB→GM 写出（双缓冲），PL-1.6 实证该形态下计算链（lerp 7-pass）在 ≥16M 档完全隐藏于 MTE2 窗口——本算子 7-op（transit）/5-op（fp32）链与 lerp 同级，同判。
- **grid-stride 聚集窗口**：40 核聚集于移动窗口访问 GM（PL-1.6 实测优于每核连续分块 +3.2%@256M）。
- **Stage 4 可选方向**（不在本设计展开，按 ROI 排序）：① block_size 调优（UB 倍数复算 + 编译探针，§4.5 条款）；② **A3 α=1 特化启用**（省 vmul——机器证据已留存 §1.6.3.1，启用零风险；预期收益 ≤0.15µs@16M，仅当 msprof 实测显示计算链暴露时才值得）；③ stage 数评估；④ mte2_ratio 判定标量削减（PL-1.6 判定式：mte2_ratio ≥0.95 时无收益）。

### 6.4 尾块处理

- **方案**：tile 级动态尾块 + 零起点切片——`tail = T.min(block_size, N − t0)`；载入 `T.copy(x[t0:t0+tail], x_ub[0:tail])`；v-prefix 整 buffer 计算覆盖 `[tail:]` 陈旧 lane（无污染：vexp/vsub/vmul/vselect 均 lane 独立；被屏蔽 lane 的 exp 溢出 Inf 同理无害——§1.6.3）；**拷出仅 `[0:tail]`**（官方 vec_add_1d 模式）。
- 覆盖场景：任意 N（含 N < block_size、尾 = 1、非 32B 对齐尾）；`tail` 为 trace 期可折叠表达式上界（N 编译期常量），`if block_id < num_logical` 保证最后核不越界。
- 无 padding+crop（无需 host 侧补内存——切片方案无 GM 越界风险，优于 padding 方案的额外流量）。

## 7. 同步策略

### 7.1 同步模式

**Developer 自动同步**（编程模式匹配，§2）：编译器在 GM↔UB 搬运与 Vector 计算之间自动插入栅栏；无手动 `set_flag/wait_flag`/`pipe_barrier`。

### 7.2 同步点说明

| 位置 | 同步机制 | 说明 |
|------|---------|------|
| GM→UB 载入 → vcast/vcmp 链 | 自动（UB WAR/RAW 栅栏） | Developer 编译器插入 |
| vexp/vsub/vmul/vselect 原地链（w32/x_ub）→ [fp16/bf16] vcast 降域 / [fp32] UB→GM 拷出 | 自动（RAW） | UB 内依赖（结果 buffer w32/x_ub 的消费） |
| vcast → UB→GM 拷出 | 自动（RAW） | — |
| vbrc 常量填充（循环外） → 首次 vcmp/vmul 消费 | 自动（RAW） | 每 kernel 一次 |
| persistent 串行迭代间 | 自动（tile 间无跨迭代数据依赖；buffer 全量复用由自动栅栏保证） | — |
| 跨核 | 无（tile 两两不相交，无共享输出；无 workspace 往返） | 零核间同步 |

### 7.3 pass_configs 配置

默认（无手动 pass_configs）；`TILELANG_ASCEND_MODE=Developer` 模块级固定（§2.3）。Stage 4 如需显式 stage 数再评估。

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现，独立于 NPU 算法）

```python
import torch
import torch.nn.functional as F

def golden_elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU 参考：移植源仓测试锚点（test_elu.py ref_program L40）。

    以 §0.1 语义为唯一依据（y = x if x>0 else alpha*(exp(x)-1)，fp32 opmath
    + 单次舍回路径），**不复刻 NPU 算法**（无 vcast transit / vcmp/vexp/
    vselect / persistent 结构知识）。CPU 可跑，不依赖 NPU。
    """
    return F.elu(x.detach().float(), alpha).to(x.dtype)
```

- 独立性核验：golden 输入直接取 `F.elu` 语义（§0.1），与 NPU 侧 transit/向量链实现零共享。
- 参数化 alpha：`test_elu_param` 用例（alpha=0.5）由 golden 第二参覆盖；inplace 用例（test_elu_inplace）为 Op 层语义（kernel 无感知）。

### 8.2 精度标准与 CV 融合说明

| dtype | atol | rtol | 来源 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | test_elu.py `_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 同上 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 同上 |

预期：transit 路径 vs golden **逐位一致**（§1.6.1 A1 机器验证 0 ulp / 0 violations——限于公式/舍入路径；vexp 硬件 ULP 属 Stage 3 门禁范畴，§9.1#1）；fp32 路线公式路径逐位一致（同上 vexp 附注）；NaN/Inf 角点 IEEE 传播逐位一致；±0 角点值相等（-0.0→+0.0 零符号差异豁免，§0.1/§9.1#3）。

**CV 融合设计**：不适用——纯 Vector 算子，无 Cube 段（§5.1），无 `Scope("Cube")/Scope("Vector")` 协同、无 sync_block_set/wait 需求（本节即 design-template §8.2 要求的显式结论）。

### 8.3 L0 门槛测试计划（六项 + 附加；完整 L1/L2/Boundary 由 tilelang-op-develop 展开）

| # | 项 | 用例 |
|---|----|------|
| L0-1 | smoke 全 dtype | N=2²⁰，fp16/bf16/fp32，randn |
| L0-2 | 大规模抽查 | N=2²⁴，fp16/bf16（205 任务/核多迭代——原地 v-op 无 loop-carried 纪律兜底） |
| L0-3 | 非整除尾块 | N ∈ {1000, 4097, 5000}（单 partial tile / 尾=1 / 非 32B 对齐尾），fp16+fp32 |
| L0-4 | 微型 shape | N ∈ {1, 7, 32}（单非满 tile / num_logical=1 退化），fp16/bf16/fp32 |
| L0-5 | 角点与特殊值 | ±Inf / NaN / ±0 / 次正规（finfo.min×0.5）/ max-finite（torch.finfo(d).max，勿手写字面量）/ exp 下溢带（−104、−144）/ exp 溢出 lane（±89，正轴溢出 lane 被 vselect 屏蔽）/ 小值消除区（±1e-8、±1e-7、±2⁻²⁴）/ alpha=0.5 非默认参数；断言规则：**值断言（assert_close equal_nan=True）+ NaN/Inf 逐位检查，±0 零符号差异豁免**（-0.0→+0.0 为源公式与 ATen 条件方向的已知值等价点，§0.1/§9.1#3——与 leaky_relu 的 sign-bit strict 不同，因 ELU 负支含减法零） |
| L0-6 | persistent 分核边界 | num_logical ∈ {39, 40, 41}（fp16 block 2048 → N = 39·2048 / 40·2048 / 40·2048+1）：不满核 / 恰好满核 / 掩蔽+不均任务激活 |
| 附加 | E2 双配置复用 | 同一工厂先后以 block_size=2048 与 1024 单参调用（kernel(2048)(x) / kernel(1024)(x)）均编译运行通过（E2 单参契约，jit wrapper 按 block_size 缓存编译产物；leaky_relu L0-extra 同款） |
| 附加 | UB guard + N/alpha 防御 | block_size=8193 → ValueError（fp16/bf16/fp32 三档）；N=0 / N<0 → ValueError；alpha=inf → ValueError（builder 防御，§9.3） |
| 附加 | is_fp8 拒绝 | `_make_elu_kernel(16, "float16", 1.0, is_fp8=True)` → ValueError（§0.5 #5 死代码语义） |

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 风险 | 等级 | 缓解 |
|---|------|------|------|
| 1 | **vexp 硬件 ULP 与 golden 的 exp 实现差异**：D-1 的 bitwise 主张限于公式/舍入路径（torch CPU 侧模拟）；NPU vexp（hivm.hir.vexp）与 golden 侧 exp（ATen opmath，CPU std::exp / SLEEF vec / NPU torch 各自实现）可能在 fp32 域相差 ≤1 ulp，经 fp16/bf16 舍回通常湮灭、偶发 1 ulp | 中 | 测试容差覆盖（fp16 1e-3 / bf16 1.6e-2 / fp32 1e-5）；L0-1/L0-5 全 dtype 容差断言 + 角点；本设计不声明 vexp 自身的 bitwise 等价（与 leaky_relu 无超越函数的情形不同，如实标注） |
| 2 | vexp 全 buffer 计算：正轴大值 lane（x ≥ 89）exp 溢出为 +Inf，驻留 w32 被屏蔽 lane；无 FP trap，lane 独立 | 低 | vselect 屏蔽 + [0:tail] 拷出（mish 全 buffer v-op 同形先例）；L0-5 含 ±89 角点；诊断指纹：若输出正轴出现 Inf → 检查 vselect 条件方向/Out 形态 |
| 3 | ±0 零符号值等价点：NPU 负支给 +0.0、golden ATen 恒等支给 −0.0（x=−0.0 时）——IEEE 值相等 | 低 | L0-5 断言规则调整（值断言 + NaN/Inf 逐位、±0 豁免，§8.3）；不作为 bitwise 失败 |
| 4 | auto multi-buffer UB 占用：朴素 ×2 模型下 8192 档 2×19×8192=311KB 超限（默认 1024/2048 双缓冲峰值 34.8/77.8KB 全口径安全） | 中 | TRAP-UB-multibuffer-inflation（verified，kb_stale_check 2026-10-28 非 stale）：footprint 与 staging 数解耦（结构类平台值 transit≈20–21 / fp32≈26–27 B/elem），×2 非平台口径；本算子 resident 19B×8192=155.6KB **与 leaky_relu resident 同值**（其 L1 bs=8192 实测编译运行通过）→ guard 8192 保留；Stage 4 调 block 前以编译探针/BishengIR `ub overflow requires` 报文复算 + 连带复验 num_kernels（§4.5） |
| 5 | 物理核数跨设备差异（本设备 20/40；同族记录他设备 24/48） | 低 | 工厂期 `NPUUtils.get().get_aicore_num()*2` 现查现用（ada_layer_norm 同款），部署自适应 |
| 6 | D-2 估算常数跨设备外推 + **CONST-mte2-degradation / CONST-copy-floor-method 为 STALE 条目**（stamp ed787bb/CANN 8.5.0/910B2C，当前 88ee8c302e/CANN 9.0.0/910B3；kb_stale_check 2026-10-28 实跑判定） | 低 | §1.6.0 已按「设计期估算 + Stage 4 实测校准」口径引用并标注；PL-1.6/PL-1.17/TRAP-* 各条目经同一 kb_stale_check 复核非 stale；Stage 4 msprof 实测校准回写 |
| 7 | fp8 参数误用：is_fp8=True / output_dtype≠dtype 在 NPU 无对应路径 | 低 | host ValueError（§3.3 契约校验），报错语义与 wrapper SUPPORTED_DTYPES 拒绝一致 |
| 8 | alpha fp32 常量精度：vbrc(alpha) 的 fp64→fp32 转换必须为 IEEE 最近舍入（= np.float32(alpha) = golden opmath scalar 转换），A1 等价性以该值为前提 | 中 | L0-5 含 alpha=0.5 逐位断言；Stage 3 实现后抽验 `alpha32` 内容（T.print 或小 N 对拍） |
| 9 | 原地 v-op 别名形态（⑤⑥⑦ 三连原地：vsub/vmul Out=C=w32、vselect Out=B=w32（transit）/Out=A=x_ub（fp32）；kb_search 命中 TRAP-vselect-inplace-carried-state） | 低 | 该 trap 触发边界 = **Out=B 形态 + 输出作为跨 `T.serial` 迭代携带状态**（argmax running 类，status=verified）；本设计无 loop-carried 状态——每 tile 由 ① 全量重载、④ 全量重算，⑦ 输出仅同迭代被 ⑧/⑨ 消费（lerp 原地 vsub/vmul/vadd 链多迭代 bit-exact 同形先例；fp32 Out=A 另有 gqa_prefill L875 生产先例）。防呆 §9.2；L0-2（205 任务/核多迭代）兜底；诊断指纹 = 输出停留 tile-0 局部值 |
| 10 | vsub 标量字面量操作数形态（`T.vsub(w32, 1.0, w32)`）在本工具链的编译兼容性：文档 §2.1/§2.2.2 支持 + mish L78 vadd 标量先例，但 vsub+标量+原地三要素组合无逐字同形先例 | 低 | **回退方案**：若 Stage 3 编译报错，改 one32 缓冲形态（`T.vbrc(1.0, one32)` + `T.vsub(w32, one32, w32)`）——+4B/elem（fp16/bf16 19→23B、fp32 17→21B），8192 档 resident 188.4KB 仍 <192KB（inflated 估计 >192KB，届时同步把 guard 降至 4096 并留痕）；不阻塞设计 |
| 11 | bf16 无 vexp/vmul/vsub（docs 三处 §2.2.1 bf16 ×，逐一亲读） | 低 | transit 路线唯一可行（已采纳，§0.6 R2）；负向断言证据：T.vexp.md/T.vmul.md/T.vsub.md dtype 矩阵单元格 |
| 12 | `cond`（bool/i1）缓冲对齐：vbrc.md §2.3——i1 dst 尾轴需 16 对齐否则越界风险（该条针对 vbrc bool dst；本设计 vbrc 仅 fp32，cond 由 vcmp 产生全宽度消费） | 低 | cond 不经 vbrc、不写 GM（vcmp.md §2.3 条款已核对）；L0-3 尾块用例覆盖 |

### 9.2 常见错误（Stage 3 防呆，含同族 RETROSPECTIVE 教训逐字承接）

- 不要把 `num_kernels` 写成 num_logical（超发 → 串行调度开销，core-split 失败信号②）。
- 不要用 `@T.prim_func` 内 Python if 引用循环变量（TVM-script if 块作用域限制，C11/lerp 先例——dtype 分支放工厂期；本设计全链无 in-body if）。
- 不要对 bool cond 做 GM 拷出（vcmp.md §2.3 明令禁止，编译失败）。
- 不要省略 alpha 的 fp32 域参与（误用 fp16 域 alpha 会引入舍入路径分歧，破坏 A1 等价前提）；vbrc 标量源必须在 @T.prim_func **体内**（with T.Kernel 块内）普通赋值 let 绑定——写在工厂作用域即使是有名局部变量也会触发 `The input vector and output vector must have same rank` 编译诊断（vbrc rank 陷阱，RETROSPECTIVE 教训）。
- E2 契约：NPU harness wrapper 以**单参**调用 factory——`_make_elu_kernel(N, dtype, alpha)(block_size)(x)`（`examples/TileOPs/tileops/kernels/elementwise/elu_fwd/elu_fwd.py` L100；折叠常量 `_BLOCK_SIZE_FP32=1024`/`_BLOCK_SIZE_NON_FP32=2048`，其 L82–83）。**禁止**按 GPU 基类两参形态 `self.kernel(cfg["threads"], cfg["num_per_thread"])` 实现——该依赖是源侧专用，NPU wrapper 不存在（接口契约核对必须查「落地侧现状」，leaky_relu Stage 2 检视问题 1 教训）。
- dtype 入参一律先归一化再分支：`dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]`（§3.3；torch 2.7.1 无 `.name`，本环境实测 hasattr=False）；禁止把 torch.dtype 对象直接与 str 元组做 `in` 比较（恒 False，会误走 fp32 路线、fp16/bf16 输出错误升域结果）。
- 原地 v-op 链（⑤⑥⑦）仅限「同迭代消费」：输出必须在同一 `T.serial` 迭代内被 ⑧/⑨ 消费，禁止把原地输出当作跨迭代携带状态（TRAP-vselect-inplace-carried-state；诊断指纹 = 输出停留 tile-0 局部值）。
- 测试数据 dtype 极值一律从 `torch.finfo(torch.<dtype>).max` 取，不要手写十进制字面量（fp32 max 十进制近似赋 fp32 张量会 overflow 抛 RuntimeError——RETROSPECTIVE 教训）。
- 多 dtype 路线冻结前逐路线推演「① 载入 → … → ⑨ 拷出」字面 buffer 数据流：结果驻留 buffer（transit ⑧ 回写 x_ub / fp32 ⑦ Out=A 写回 x_ub）必须与 ⑨ 拷出源一致（leaky_relu Stage 2 检视问题 2 教训；本设计 §1.5 已按路线分列推演）。
- 同一 kernel 内多条 dtype 分支共享 API 约束时，逐分支单独审计约束遵守情况（不要凭「一条分支修好了」推断另一条也正确——RETROSPECTIVE 教训）。
- 涉及 torch.dtype 的 builder 归一化用 `str(dtype).split(".")[-1]`，不用 `.name`（torch 2.7.1 无该属性）。

### 9.3 特殊场景处理

- **N=0/负 N**：源无显式拦截（Op 层校验为 `numel != N_total`，空张量可穿透）——NPU builder 防御性 `N <= 0 → ValueError`（适配项，早于编译失败）；L0 附加用例覆盖（§8.3）。
- **alpha = inf/NaN**：承接源 host 校验语义（test_elu_rejects_nonfinite_alpha）→ builder `math.isfinite(alpha)` 失败抛 ValueError；L0 附加用例覆盖。
- **alpha = 0.0**：负支退化为 α·(exp(x)−1) = ±0——链路无需特判（公式自然覆盖）；L0-5 可加 case。
- **inplace=True**：Op 层 copy-back（kernel 无感知，始终写新 buffer）——契约与源一致（test_elu_inplace）。
- **fp32 大 N**：无 transit，链最短（5 op）；§1.6.0 估算下界已列。
- **alpha ≠ 1（如 0.5）**：通用链覆盖（vmul 在链）；A3 特化为 Stage 4 杠杆（§1.6.1(a)/§6.3）。

## 10. 交付清单

### 10.1 目录结构

```text
examples/elu_fwd/_make_elu_kernel/
├── DESIGN.md            # 本文档
├── verify_equiv.py      # D-1 等价性机器验证（已执行，§1.6.1 结果表）
├── RETROSPECTIVE.md     # Stage 1 复盘（Phase 8）
└── {op}.py              # Stage 3 产出：kernel + golden + 内嵌 L0 套件
```

### 10.2 文件清单（Stage 3 输入契约）

| 文件 | 内容 | 状态 |
|------|------|------|
| DESIGN.md | 全部设计决策（§0–§10） | 本文档 |
| verify_equiv.py | A1 采纳项 6 行（结果表 §1.6.1(b)）+ A3 证据 3 行 + A2/A4/A5/A6 弃选证据 5 行（结果表归档 §1.6.3.1）；torch CPU 独立可跑（每 case 派生种子可复现） | ✅ 已产出并执行（采纳项 EQUIV_PASS，2026-10-28 复跑核对一致） |
| RETROSPECTIVE.md | Stage 1 复盘 | 随本文档产出 |
| {op}.py | `_make_elu_kernel` builder + `golden_elu` + L0 套件 | 待 Stage 3 |

### 10.3 命名规范

- builder 函数名/参数名：`_make_elu_kernel(N, dtype, alpha, output_dtype=None, is_fp8=False, threads=256, npt=8)`（逐字保持源签名；**threads/npt 保留于签名但 NPU 侧不消费**——K11 已折叠为内层单参 block_size，保留仅为签名兼容与 `lru_cache` 键空间一致，§3.5.2）；`@functools.lru_cache(maxsize=32)` 施加于该外层 builder（leaky_relu L124 同款）。
- E2 工厂：`kernel(block_size=default_block)`（default_block 按 dtype：fp32 1024 / fp16·bf16 2048，= wrapper `_BLOCK_SIZE_*`，leaky_relu L194 同形）；prim_func 名 `main`，参数 `(x, y)` 名与顺序保持源形态（迁移规则）。
- golden：`golden_elu`。

### 10.4 实现顺序（Stage 3 建议）

1. builder 骨架 + host 契约校验（§3.3）→ 2. persistent 分核 + fp32 路线（先通 fp32 全 L0）→ 3. transit 分支（fp16/bf16，vexp/vsub 引入）→ 4. L0-5 角点（含 vexp 溢出 lane / exp 下溢带 / ±0 值等价规则）+ L0-6 分核边界 → 5. E2 复用 + UB guard + is_fp8/alpha 防御用例。
