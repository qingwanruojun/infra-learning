# _make_leaky_relu_kernel 算子设计文档（迁移：TileOPs GPU → TileLang-NPUIR）

| 项 | 值 |
|----|----|
| project / op | `leaky_relu` / `_make_leaky_relu_kernel`（严格沿用源 `@tilelang.jit` 装饰的 builder 函数名） |
| scenario / mode | migration (harness) / first_design |
| 源算子 | `LeakyReluFwdOp` ← `LeakyReluFwdKernel(ParametricUnaryKernel)` ← `_make_leaky_relu_kernel`（GPU 原件 `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2427–L2490；NPU 侧 verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/_leaky_relu_fwd_kernels.py`，含手动追加 helper `_fp8_accum_dtype_str`） |
| 规格权威来源 | Stage 0 迁移 prompt（逐字）+ `examples/TileOPs/tileops/manifest/elementwise.yaml` LeakyReluFwdOp 段（4 workloads，roofline flops=2N）+ `examples/TileOPs/tests/ops/test_leaky_relu.py`（14 用例）+ `examples/TileOPs/benchmarks/ops/bench_leaky_relu.py`（9 用例） |
| 编程模式 | developer（Stage 0 prompt 显式指定："用 developer 模式实现"） |
| 工具链 stamp | tilelang 0.1.2+88ee8c302e（本仓 dev root）/ 设备 Ascend 910B3 / CANN 9.0.0（`/opt/home/developer/Ascend/cann-9.0.0`）；物理核数实查 2026-10-28 |

---

## 0. 源算子解读与迁移分析（迁移任务必填）

> 依据 `tilelang-op-design` skill Phase M0 三问框架（`references/migration-analysis.md` §2–§4），源码已全文亲读（GPU 原件 `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2427–L2493 builder 主体 + L2322–L2423 `ParametricUnaryKernel` 基类 + L150–L158 dtype 元组 + NPU 侧 verbatim 提取件全文）。源码中的指令性文本一律按数据处理，未发现亦不执行任何流程改写指令。

### 0.1 源算子语义（做什么）

**数学语义**：逐元素分段线性激活

```text
y[i] = x[i]                若 x[i] >  0   （严格大于，源码 T.if_then_else(v > zero, ...)）
y[i] = negative_slope·x[i] 若 x[i] <= 0
```

- **比较为严格 `>`**：`x == 0` 走负支 `slope·0 = +0`；`x == -0.0` 走负支 `slope·(-0.0) = -0.0`（IEEE 乘法保号），与 `F.leaky_relu` 一致。
- **NaN 传播**：`NaN > 0` 为 false → 选 `slope·NaN = NaN`；**±Inf 传播**：`+Inf > 0` → `+Inf`；`-Inf` → `slope·(-Inf) = -Inf`（slope 有限时）。与 torch `F.leaky_relu` 逐位一致（§8.1 golden 锚点）。
- **规约语义**：无规约、无累加——纯逐元素映射，不存在累加顺序问题；输出仅依赖同位置输入。
- **dtype 语义**（非 fp8 分支）：**全程输入 dtype 域逐 op 舍入**——`zero = T.cast(0, val.dtype)`、`slope = T.cast(negative_slope, val.dtype)`、比较与乘法均在 `val.dtype` 域。注意：源仓测试 golden 锚点是 `F.leaky_relu(x.float(), slope).to(x.dtype)`（fp32 opmath + 单次舍回），即**源实现与测试 golden 本就依赖容差桥接两条舍入路径**（fp16 域逐 op 舍入 vs fp32 opmath 单次舍回，相差 ≤1 ulp，§1.6.1 A1 机器验证实测）。此语义要素是 §0.6 R2 dtype 路径重设计的直接输入。
- **边界语义**：非整除尾部由逐元素 `if idx < N` guard 屏蔽（fp8 分支）或整块 fragment 覆盖（非 fp8 分支，`block_size = threads×npt` 整块时依赖 grid 收尾）；`N == 0` 不进入 kernel（Op 层 `N_total` 校验拦截）。
- **host 侧语义**：输入任意 shape → `input.contiguous().reshape(-1)` 展平 → kernel → `reshape(orig_shape)`；`inplace=True` 时结果 `copy_` 回输入并返回输入张量（kernel 始终写新 buffer，inplace 是 host 层 copy-back 语义）；`negative_slope` host 校验（必须 finite 且可表示于 dtype，`_validate_scalar_param_repr`）。
- **完成标准核验**：golden 函数可直接写出（§8.1）——`golden_leaky_relu(x, slope) = F.leaky_relu(x.float(), slope).to(x.dtype)`，与源实现数值一致（容差内，见上）。

### 0.2 源算子输入输出

| 项 | 值 | 依据 |
|----|----|------|
| 输入 `x` | 展平 `(N,)` 一维连续张量；NPU 契约 dtype **float16 \| bfloat16 \| float32**（manifest `input.dtype`）；host 支持任意 4D/ND，展平后 N = product(shape) | manifest elementwise.yaml L108 + Op 层 `_eager_forward` |
| 输出 `y` | `(N,)`，**dtype same_as(input)**（`out_dtype = output_dtype or dtype`，非 fp8 路径 `output_dtype=None` → `out_dtype == dtype`）；`output.shape == input.shape`（shape_rules） | 源码 L2437/L2464 + manifest L115 |
| 参数 `negative_slope` | Python float，default 0.01；constructor 期标量（烘焙进 kernel 常量）；host 校验 finite + dtype 可表示 | 源码 L2484 + Op 层校验 |
| 参数 `output_dtype` | GPU 侧仅 fp8 输出流程使用；非 fp8 契约下必为 None/=dtype。NPU 保留参数，非 None 且 ≠ dtype 时 ValueError | 源码 L2437 + manifest output same_as(input) |
| 参数 `is_fp8` / `threads` / `npt` | 见 §0.5 处置；`threads=256, npt=4|8`（fp32 npt=4，其余 8），`block_size = threads×npt` | 源码 L2438 + 基类 default_config |
| 动态轴 | N（展平后元素总数，含非整块尾部）；builder 按 N 工厂特化（`@lru_cache(32)`） | manifest workloads（smoke-1m / gan-feat / gan-feat-deep / fc-wide）|
| 不变量 | `y[i] = f(x[i])`，1 读 1 写，无形状变换、无 layout 变换（连续 1-D） | §0.3 数据流 |

### 0.3 实现算法解读（怎么算）

**结构化阅读协议 7 步**（migration-analysis §3.1）：

1. **入口签名**：`_make_leaky_relu_kernel(N, dtype, negative_slope, output_dtype=None, is_fp8=False, threads=256, npt=8)`，`@functools.lru_cache(maxsize=32)`；返回 `@tilelang.jit(out_idx=[1])` 工厂 `kernel(threads_arg, npt_arg) -> main`——E2 契约与基类 `init_config` 的 `self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])`（L2417）对应。
2. **Kernel 结构**：`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg)` —— **一维 grid**，`block_size = threads * npt` 为 builder 闭包编译期常量（L2438）。
3. **计算原语**（逐语句归步）：
   - **非 fp8 分支**（主路径）：`T.copy(x[bx·bs:(bx+1)·bs], x_reg)` 整块载入 fragment → `for i, j in T.Parallel(threads_arg, npt_arg)` 二级展开，逐元素 `y_reg[i·npt+j] = T.if_then_else(val > zero, val, slope·val)`（`zero`/`slope` 均为 `val.dtype` 埸常量）→ `T.copy(y_reg, y[...])` 写出。
   - **fp8 分支**：无 fragment，逐元素 `idx = (bx·threads+i)·npt+j` + `if idx < N` guard + `T.cast(val, "float16")` 累积域计算 + `T.Cast(out_dtype, result)` 写回（依赖 helper `_fp8_accum_dtype_str() -> "float16"`）。
4. **内存分配**：`x_reg`/`y_reg` = `T.alloc_fragment((block_size,), dtype)` —— GPU **寄存器 fragment** 驻留（两次 GM 访问/元素，无 SMEM 中转）。
5. **循环与调度**：仅 `T.Parallel(threads_arg, npt_arg)` 一层元素展开（SIMT block×thread 映射，每线程 npt 元素 ILP）；无 serial、无 Pipelined、无 persistent。
6. **host 侧逻辑**：`ParametricUnaryKernel.__init__`——dtype 合法性（`SUPPORTED_DTYPES = _FLOAT_DTYPES = (fp16, bf16, fp32)`，L154–158/L2342，**fp8 在此处已被 ValueError 拒绝**）、fp8 output dtype 推导（本契约不可达）、`default_config{threads:256, npt:4|8}`、builder 调用与 `init_config` 编译缓存；`LeakyReluFwdKernel._builder_args()` 返回 `(negative_slope,)`；Op 层：展平/还原、inplace copy-back、`eval_roofline(flops=2N, bytes=2N·elem_bytes)`。
7. **编译配置**：无 pass_configs / 环境变量（GPU 默认管线）。

**计算步骤分解表**：

| 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|------|------|------|------|---------|
| S1 | 整块载入 `T.copy(x_tile, x_reg)` | GM x | fragment x_reg | 读 x |
| S2 | 逐元素 `if_then_else(val>0, val, slope·val)` | x_reg + 常量 zero/slope | y_reg | 分段线性公式本体 |
| S3 | 整块写回 `T.copy(y_reg, y_tile)` | y_reg | GM y | 写 y |

**数据流图**（源硬件视角，覆盖全部 buffer）：

```text
GM x ──T.copy──> x_reg(GPU register fragment) ──T.Parallel 逐元素 select/mul──> y_reg(fragment) ──T.copy──> GM y
常量：zero/slope 在 val.dtype 域内联（编译期标量）；无其他 buffer、无中间 GM 往返
```

### 0.4 优化手段解读（为什么快）

逐项识别（清单法，migration-analysis §4.1；"识别不出来 ≠ 不存在"已对照——源码中确无 SMEM tiling / mma / warp shuffle / 异步流水 / swizzle / online 结构）：

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---------|---------|------|----------------|------|
| O1 | register_copy fragment 驻留 | `alloc_fragment` + 整块 `T.copy` 进出 | 每元素恰 1 次 GM 读 + 1 次 GM 写，中间全寄存器 | GPU 寄存器堆 | 硬件强相关（NPU 等价片上层 = UB） |
| O2 | coalesced 连续访问 | 连续 tile 载入/写回（`bx·bs` 连续窗） | 合并访存事务、吃满带宽 | memory coalescing | 意图可移植（访存组织需重排） |
| O3 | block×thread 二级并行 | `T.Parallel(threads_arg, npt_arg)` | 每线程 npt 元素 ILP + SIMT 占用 | CUDA SIMT 层级 | 硬件强相关，需重设计 |
| O4 | fp8 explicit_parallel + fp16 accum | fp8 分支：逐元素 guard + cast fp16 | 规避 8-bit fragment register_copy 不可靠 | CUDA fp8 fragment | 接口契约死代码（见 §0.5 #5） |
| O5 | builder `lru_cache(32)` + 编译期常量折叠 | `@functools.lru_cache`、闭包 `block_size` | host 编译缓存、grid/块常量折叠 | host 层 | 保留（与硬件无关） |
| O6 | 标量常量内联 | `T.cast(negative_slope/0, val.dtype)` | 消除标量访存 | — | 保留（NPU 转为 vbrc 常量广播，§0.6） |

**算法层 vs 实现层区分**：本源算法为"单步逐元素映射"，**无算法层优化**（无 online / 无分块累加 / 无数学恒等变形）；O1–O4 全部为实现层。

### 0.5 硬件耦合性分析与 NPU 适配决策

三层判定模型 + 四态处置（migration-analysis §5；判定输入含 Phase R 调研结论 §1.6.0）：

| # | 条目 | 层 | 处置 | 依据 |
|---|------|----|------|------|
| 1 | 语义层全集（公式/严格 `>`/NaN·Inf·±0 传播/`(N,)` 契约/dtype same_as/host 展平+inplace） | 语义层 | **保留** | 语义层不可妥协（映射表前置规则） |
| 2 | O1 register_copy fragment | 优化层 | **等价替换** | fragment → UB：`T.alloc_ub`（映射表「shared memory → L1/UB」「register 累加 → UB 中间缓冲」；同族先例 lerp_tensor kernel 全 UB 链） |
| 3 | O2 coalesced 连续访问 | 优化层 | **保留**（意图承接） | NPU 侧 UB 连续零起点切片 + persistent grid-stride 聚集窗口（pattern-library PL-1.6 实测：grid-stride 优于每核连续分块 +3.2%@256M）；访存组织重排见 §0.6 R1 |
| 4 | O3 block×thread 二级并行 | 优化层 | **重新设计** | 映射表「block × thread 两级并行 → 重新设计（本项目 `T.Kernel` 只接受一维 block 数）」；元素级并行改由 v-prefix 全 buffer 向量 op 承接（§0.6 R1） |
| 5 | O4 fp8 explicit_parallel 分支 | 优化层 | **舍弃** | 接口契约死代码：基类 `SUPPORTED_DTYPES = _FLOAT_DTYPES = (fp16, bf16, fp32)`（elementwise.py L154–158, L2342）在 builder 之前即以 ValueError 拒绝 fp8 输入 → `is_fp8=True` 对本算子**不可达**；NPU manifest workloads dtypes 亦仅 fp16/bf16/fp32。参数 `is_fp8` 保留于签名；NPU 实现 `is_fp8=True` 时抛 ValueError（与基类拒绝语义一致），`_fp8_accum_dtype_str` helper 不迁移 |
| 6 | O5 lru_cache + 常量折叠 | host 层 | **保留** | host 层与硬件无关；NPU 侧同款 `@lru_cache(32)` builder 缓存 |
| 7 | inplace copy-back | host 层 | **保留** | Op 层语义（kernel 写新 buffer + host `copy_`），kernel 无感知 |
| 8 | fp8 分支 per-element `if idx < N` 尾 guard | 优化层 | **等价替换** | NPU 尾块统一改为 tile 级 `tail = T.min(block_size, N−t0)` + UB 零起点切片拷出（官方 `examples/elementwise/vec_add_1d.py` 尾块模式 + 演进队列 VP-2026-0003 模板），全部尾 lane 语义由切片覆盖，无逐元素分支 |

**处置统计：保留 4（#1/#3/#6/#7）/ 等价替换 2（#2/#8）/ 重新设计 1（#4）/ 舍弃 1（#5）**。无存疑项（全部条目有迁移映射表条目佐证，API 细节佐证为 docs/Tilelang.language/ 各 API 文档与 §3.5.3 所列 examples 同类实现文件）。

### 0.6 NPU 算法重设计

对 §0.5「重新设计」项 + 调研选定算法族（§1.6.0）落地为 NPU 侧算法。重设计输入：§0.5 #4（并行结构）+ §0.4 全部优化意图 + Phase R 调研结论（§1.6.0：基线公式四口径并列最优，无算法族替换，选定 = 基线公式 + NPU 重设计落地形态）。

**R1 并行结构重设计**（承接 §0.5 #4，模式库「并行结构重设计」）

- 源方案：`T.Kernel(ceil(N/bs))` × `T.Parallel(threads, npt)` 两级 SIMT 展开，逐元素 `if_then_else`。
- NPU 新算法：**一维 `T.Kernel(num_kernels)` persistent 分核**（`num_kernels = min(ceil(N/block_size), vector_cores)`）+ 核内 `T.serial(num_local_tasks)` 静态边界 grid-stride（`block_id = i·num_kernels + cid`，`if block_id < num_logical` 屏蔽）+ **v-prefix 全 buffer 向量链**（`vcmp` → `vmul` → `vselect`，每 tile 3 op/元素域）替代逐元素展开；比较/乘/选择均为 UB 全宽度向量 op。
- **语义保持论证**：① 逐元素映射无累加顺序，lane 重排不改结果；每元素恰计算同一公式一次。② 边界：尾块 `tail = T.min(block_size, N−t0)` 切片 + 零起点 UB 拷出覆盖任意 N（含 N < block_size、非 32B 对齐尾、N=1），行为与源 guard 语义一致。③ NaN/Inf/±0：`vcmp("gt")` 逐 lane 产生与标量 `>` 一致的谓词（NaN>0=false），`vselect` 选取 `slope·x` 支路复现 NaN/±0 传播——§1.6.1 A1 机器验证 candidate vs golden **0 ulp / 0 violations**（含 ±Inf/NaN/±0/次正规角点）。④ 严格 `>`：`vcmp` 支持 `"gt"` 模式（docs/Tilelang.language/比较操作/T.vcmp.md §2.3）。

**R2 dtype 路径重设计**（算法替代·dtype 路径；承接 §0.1 识别的"golden 容差桥接"语义要素；模式库「算法替代」，采纳须与 §1.6.0 调研结论一致——见 §1.6.0 R4 dtype 路径行）

- 源方案：非 fp8 路径全程**输入 dtype 域逐 op 舍入**（slope/zero cast 到 `val.dtype`）。
- NPU 新算法：**fp16/bf16 走 fp32 中转链**——`T.vcast(x_ub, x32, round_mode="rint")` 无损升 fp32 → fp32 域 `vcmp/vmul/vselect`（slope 以 fp32 值参与）→ `T.vcast(x32→x_ub, round_mode="rint")` **单次舍回**；fp32 走原生域（无 transit）。
- 采纳依据（双触发，PL-1.17 两任务实证 pattern）：① **bf16 无 v-prefix 算术**——`T.vmul` dtype 矩阵 bf16 ×（docs/Tilelang.language/数学操作/T.vmul.md §2.2.1，亲读），bf16 原生域链无计算路径，vcast 升 fp32 是唯一路径；② **fp16 golden 对齐**——测试 golden 锚点经 fp32 opmath + 单次舍回（test_leaky_relu.py `ref_program`），fp16 原生域逐 op 舍入与 golden 差 ≤1 ulp（A1 实测 base vs golden 1 ulp），transit 路径与 golden **逐位一致（0 ulp / 0 violations）**——lerp_tensor 实证此类分歧设计期机器可拦截（避免 Stage 3 精度返工）。slope 常量以 fp32 值参与：`f32(0.01) = 0.009999999776…` 与 golden 的 fp32 opmath 标量一致；fp16 域 slope（0.010002…）是弃选路径的差异源。
- **语义保持论证**：fp16/bf16 → fp32 升变换**数学无损**（值域子集），fp32 域比较结果与原生域比较**严格一致**（升变换保序），fp32 乘法后单次 rint 舍回 = golden 舍入路径（机器验证 A1：6/6 行 EQUIV_PASS）；容差内等价成立且严格更近 golden（ulp 0 ≤ 基线 1）。fp32 dtype 走原生域 = golden 原生计算，逐位一致。

**R3 尾块重设计**（承接 §0.5 #8）：见 R1 ③，tile 级 tail + 零起点切片，`if idx < N` 逐元素分支消除。

**源优化意图承接说明**（algorithm-research §4：不得静默丢弃）：

| 源手段 | 意图 | NPU 承接 |
|--------|------|---------|
| O1 fragment 驻留（1 读 1 写 GM） | 消中间 GM 往返 | UB 驻留整 tile，单遍 load→compute→store，GM 流量同为 2N·B 下界（§1.6.0 R3） |
| O2 coalesced | 满带宽连续访问 | UB 连续零起点切片 + persistent grid-stride 聚集窗口（PL-1.6 实测最优访问形态） |
| O3 二级并行 ILP | 元素级并行度 | v-prefix 全 buffer 向量 op（Vector 单元全宽度）+ Developer 自动多缓冲（PL-1.6 实证 ≥16M 计算链完全隐藏于 MTE2 窗口） |
| O6 标量常量内联 | 消标量访存 | `T.vbrc` 常量广播（zero32/slope32 每 kernel 一次，置于 persistent 循环外，§1.6.2 V5） |

### 0.7 标杆实现

- **源算子 verbatim 提取件**：`examples/TileOPs/tileops/kernels/elementwise/leaky_relu/_leaky_relu_fwd_kernels.py`（Pattern-B 提取，含 `_fp8_accum_dtype_str` helper）。
- **GPU 原件**：`/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2427–L2490。
- **同族已迁移标杆（本仓已验证）**：`examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py`（base，persistent 分核 + fp32 transit + L0 六项套件）与 `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py`（persistent nc=48 全档贴 copy 地板）；同族规格文件 `examples/TileOPs/tileops/kernels/elementwise/mish/`。
- **精度/性能契约**：`examples/TileOPs/tests/ops/test_leaky_relu.py`、`examples/TileOPs/benchmarks/ops/bench_leaky_relu.py`。

---

## 1. 概述

### 1.1 算子名称

`_make_leaky_relu_kernel`（`leaky_relu` 项目；@tilelang.jit builder 函数名严格沿用源签名）。

### 1.2 功能描述

NPU 侧逐元素 LeakyReLU 前向激活：对展平 `(N,)` 连续输入按 `y = x (x>0)；y = negative_slope·x (x≤0)` 做分段线性映射，输出同 shape 同 dtype。支撑 `LeakyReluFwdOp` 的 4 个 manifest workloads（smoke-1m / gan-feat[16,256,64,64] / gan-feat-deep[16,512,32,32] / fc-wide[2048,4096]，dtype fp16|bf16|fp32）。本节描述的是 **§0.6 迁移决策后的 NPU 侧算法**（与源算法的结构差异：两级 SIMT 展开 → persistent 一维分核 + UB 向量链；dtype 域逐 op 舍入 → fp16/bf16 fp32 中转单次舍回）。

### 1.3 数学公式

```text
y[i] = x[i]                    若 x[i] >  0
y[i] = negative_slope · x[i]   若 x[i] <= 0        （严格 >；NaN/Inf/±0 按 §0.1 传播）
```

FLOPs 口径（manifest roofline.md §1.3）：compare-and-select(1) + mul(1) = **2 FLOPs/元素**；访存 = 2·N·elem_bytes（1 读 1 写）。

### 1.4 算法描述

分析对象 = §1.6.0 调研选定、经 §0.6 重设计落地的算法（与源算法非直译关系）：

1. **分核**：persistent 一维分核——`num_logical = ceildiv(N, block_size)` 个逻辑 tile，固定启动 `num_kernels = min(num_logical, vector_cores)` 个物理核（纯 Vector 翻倍核数，§5.5 实查），核内 `T.serial(num_local_tasks)` 静态边界 grid-stride。
2. **单遍 tile 流水**（每 tile）：GM→UB 同 dtype 载入（`tail` 切片）→ [fp16/bf16: `vcast(rint)` 升 fp32] → fp32 域 `vcmp(x32, zero32, "gt")` → `vmul(slope32, x32, sx32)` → `vselect(cond, x32, sx32, sx32)`（原地）→ [fp16/bf16: `vcast(rint)` 单次舍回] → UB→GM 拷出 `[0:tail]`。
3. **常量预备**（每 kernel 一次，persistent 循环外）：`vbrc(negative_slope, slope32)`、`vbrc(0.0, zero32)`。
4. dtype 路径：fp32 原生域；fp16/bf16 fp32 中转（§0.6 R2，PL-1.17 双触发）。

### 1.5 数据流图

```text
                                     （每 kernel 一次）
                     slope(f64 标量)──vbrc──> slope32(UB, fp32, block 宽)
                     0.0 ────────────vbrc──> zero32 (UB, fp32, block 宽)

每 tile（persistent 串行）：
GM x[t0:t0+tail] ──T.copy──> x_ub(UB, 输入 dtype)
   │ [fp16/bf16]  x_ub ──vcast(rint)──> x32(UB, fp32)          [fp32: x32 ≡ x_ub]
   │              x32, zero32 ──vcmp("gt")──> cond(UB, bool)
   │              slope32, x32 ──vmul──> sx32(UB, fp32)
   │              cond, x32, sx32 ──vselect(原地)──> sx32 = y32
   │ [fp16/bf16]  sx32 ──vcast(rint)──> x_ub(UB, 输入 dtype)     （复用载入 buffer）
   └──T.copy──> GM y[t0:t0+tail]                                 （仅 [0:tail] 拷出）
```

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

调研对象：同一数学语义 `y = x (x>0) else slope·x` 的算法族。算子特征判定（M0/Phase 1）：**单步逐元素**（无规约/窗口/矩阵）→ **轻量调研**（algorithm-research.md §2），四问逐问回答并给依据。

**R1 等价化简公式候选**（基线在表；正式等价论证与收益量化在 §1.6.1，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线（源算法/输入公式）** | `sel(x>0, x, slope·x)`：vcmp+vselect+vmul 3 op（cmp+sel 计 1 flop + mul 1 flop = 2 FLOPs/元素） | — | — | ✅（基线） |
| 1 | 乘数选择式 | `x · sel(x>0, 1, slope)` | 数学恒等 | **零收益**：同样 3 op（vcmp+vselect+vmul），且需 one32+slope32 双常量缓冲（UB 预算 21B/elem > 基线 17B） | ❌ op 数同、UB 更差 |
| 2 | vmax 条件式 | `max(x, slope·x)`：vmul+vmax 2 op | **条件等价**——仅 `slope ∈ [0,1]` 时成立（slope>1 时负支 `slope·x < x`，max 误选 x；slope<0 时负支变正，全错） | −1 op/元素 | ❌ 契约允许任意 finite slope（host 校验只查 finite），非普遍等价；机器证据见 §1.6.1 A2（slope=2.0 时 65521/65548 违反） |
| 3 | max+min 分解式 | `max(x,0) + slope·min(x,0)`：vmax+vmin+vmul+vadd 4 op | 数学恒等 | **负收益**：+1 op/元素 | ❌ op 数更多 |
| 4 | 公共子表达式消除 | — | 结构上无重复子式（单 cmp + 单 mul） | 无 | ❌ 无 CSE 空间 |
| 5 | 超越函数降代换 | — | 公式无超越函数 | 无 | ❌ 不适用 |

R1 结论：**基线公式即公式级最简形态**（对照 algorithm-candidates.md ALG-elementwise 条目：该族 R1 手段为公共子表达式消除/除法乘倒数/cast 链合并——本公式无除法、无重复子式、cast 链已由 §0.6 R2 定为最小单次舍回；恒等变形自查 1–3 全部负收益或条件等价）。

**R2 在线算法**：**天然单遍，无需在线变体**。结构依据（algorithm-research §3 R2 判据）：输出 `y[i]` 仅依赖 `x[i]`，无 running 统计量（和/max/计数）、无跨元素依赖——逐元素映射本身即单遍流式；源算法亦为单遍（§0.4 无 online 手段），一致。收益口径不适用（无两遍/三遍结构可化）。

**R3 算法复杂度对比**（带宽受限算子——算术强度 2/(2B) = 1/B flop/B：fp16 = 0.5、fp32 = 0.25，远低于 Vector 单元平衡点，**访存为主导项**；口径：FLOPs 含每元素全部向量 op；Bytes 为 GM 流量含输入读+输出写、不含 tile 常量广播；缓冲为 UB 峰值驻留）：

| 算法候选 | FLOPs（op/元素） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-----------------|---------------|---------|-------------|---------------------|
| 基线 `sel(x>0, x, slope·x)`（+fp32 transit） | 3 op/元素域 op（fp32 域 vcmp+vmul+vselect；fp16/bf16 另加 2×vcast，但属于 dtype 路径而非公式代价） | **2N·B**（1 读 + 1 写，无中间 GM 往返——tile 驻留 UB）= **I/O 下界** | 1（每元素每张量恰读一次写一次） | O(block_size)：fp32 17B/elem、fp16/bf16 19B/elem（§4.5） | N/block_size 个完全独立 tile；persistent 零跨核同步 |
| 候选 1 乘数选择 | 3 op（同） | 2N·B（同） | 1 | +4B/elem（双常量） | 同上 |
| 候选 2 vmax（仅 slope∈[0,1]） | 2 op（−1） | 2N·B（同） | 1 | −4B/elem（单常量） | 同上 |
| 候选 3 max+min | 4 op（+1） | 2N·B（同） | 1 | 同基线 | 同上 |

四口径结论：基线在 FLOPs 并列次优（仅条件等价的候选 2 少 1 op）、访存已达 I/O 下界、单遍、缓冲最小档；候选 2 的 −1 op 属计算链优化——**PL-1.6 copy-floor 实证 ≥16M 档计算链被 MTE2 窗口完全隐藏（ROI≈0）**，且其条件等价性与契约冲突，弃选。**基线并列最优，无算法族替换**（与 lerp 调研同构结论）。

**R4 硬件亲和性评估**（逐候选对照检查清单，algorithm-research §3 R4）：

| 检查项 | 基线（+transit 落地形态） | 结论 |
|--------|--------------------------|------|
| 计算单元匹配 | vcmp/vmul/vselect/vcast/vbrc 全为 Vector 单元 op（逐元素/比较/选择族）；无 MAC 密集段，无 Cube 需求 | ✅ 亲和 |
| 片上容量 | 工作集 = 17–19B/elem × block_size ≤ 8192 → ≤156KB < UB 192KB（§4.5） | ✅ 有界 |
| 对齐/整除 | UB 切片零起点（官方 vec_add_1d 对齐模式）；尾轴 32B 对齐由 GM 侧连续 1-D + [0:tail] 拷出消化；向量宽度 fp32×4 lane，尾 tile 浪费 ≤1 lane 组 | ✅ |
| 静态边界 | N 为 builder 工厂特化编译期常量 → num_logical/num_local_tasks/tail 上界全部 trace 期折叠（lerp 同款 host 常量折叠） | ✅ |
| 流水可融合性 | 单阶段 copy→compute→copy；Developer 自动多缓冲（UB budget ×2 stage，§6.3） | ✅ |
| 跨核结构 | 无跨核归并（逐元素独立 tile），persistent 零核间同步 | ✅ |
| dtype 路径 | bf16 无 v-prefix 算术（T.vmul.md §2.2.1 bf16 ×，亲读）→ fp32 transit 唯一路径；fp16 transit 为 golden 对齐（A1 机器验证） | ✅（transit 形态） |

**调研结论（选定算法族 + 依据）**：

1. **选定**：基线公式 `sel(x>0, x, slope·x)` + §0.6 重设计落地形态（persistent 一维分核 + UB tile 驻留 + fp32 域 vcmp/vmul/vselect 链 + fp16/bf16 fp32 中转）。依据：R3 四口径基线并列最优（访存达 2N·B 下界）；R4 全项亲和；候选 1/3 op 数劣势、候选 2 条件等价且计算链隐藏 ROI≈0（PL-1.6 实测）。**结构差异（vs 源算法）**：算法族不变（同公式），实现结构由两级 SIMT 展开 + 寄存器 fragment → persistent 一维分核 + UB 向量链 + fp32 中转 dtype 路径。
2. **调研范围声明**（「无更优替代」举证）：① 参考表 algorithm-candidates.md ALG-elementwise 条目命中行全部评估（R1 手段清单见 R1 表）；② 恒等变形自查覆盖乘数选择/极值分解/条件 vmax 四族；③ 本仓同类实现 lerp_tensor（同族单步逐元素，persistent+transit 已验证形态）、vec_add_1d（官方尾块模式）；④ pattern-library elementwise.md PL-1.6/PL-1.17（copy-floor 与 transit 双证）+ constants.md 带宽常数；⑤ R2 结构判据（逐元素天然单遍）。本地源已覆盖该族全部已知算法形态；未启用互联网检索（简单算子轻量调研，algorithm-research §6：通常无需联网）。
3. **设计期估算下界（D-2，协议 `_shared/standards/hardware-cost-model.md` §2）**：估算下界 = max(流量项, 发射项, 容量项)。
   - 流量项（主导）：引用 pattern-library/constants.md「MTE2 有效带宽随总流量退化」实测条目（**版本戳：tilelang 0.1.2+ed787bb / Ascend 910B2C / CANN 8.5.0 / 48 vector core，PL-1.6**）——1M 档 33.9 GB/s/core、16M 档 30.2 GB/s/core。本设备 910B3 / 40 vector core（§5.5 实查），**跨设备缩放按核数比例折算，标注为设计期假设**（40/48 ≈ 0.83 → 1M ≈ 28.2、16M ≈ 25.2 GB/s/core 聚合 ~1.0–1.3 TB/s 量级）。
   - 发射项：5 op/元素（fp16/bf16 transit 链）/fp32 3 op——lerp 7-pass 链在 ≥16M 已被 MTE2 窗口完全隐藏（PL-1.6 实测），5-op 链同判隐藏，发射项 < 流量项。
   - 容量项：UB 驻留 ≤156KB < 192KB（§4.5），非约束。
   - **估算下界行**（设计期，待 Stage 4 实测校准）：

| workload | N | dtype | GM 流量 | 估算下界 |
|----------|---|-------|---------|---------|
| smoke-1m | 2²⁰ | fp32 | 8.4 MB | ≈ 8 µs（1M 档聚合 ~1.13 TB/s 假设） |
| smoke-1m | 2²⁰ | fp16/bf16 | 4.2 MB | ≈ 4 µs |
| gan-feat [16,256,64,64] | 2²⁴ | fp16/bf16 | 67.1 MB | ≈ 55–67 µs（16M 档聚合 ~1.0–1.2 TB/s 假设） |
| gan-feat-deep [16,512,32,32] | 2²³ | fp16/bf16 | 33.6 MB | ≈ 28–34 µs |
| fc-wide [2048,4096] | 2²³ | fp16/bf16 | 33.6 MB | ≈ 28–34 µs |

   常数未覆盖量：910B3 的 MTE2 曲线（跨设备缩放假设）、1:1 R/W 混合（PL-1.6 为 lerp 3:2 混合）——**均为未实测假设，估算依据 = PL-1.6 曲线 × 核数比例；Stage 4 以 msprof 实测校准并回写**。

#### 1.6.1 数学等价优化（公式级）

**（a）公式级降算优化：结论 = 无优化空间。**

依据：§1.6.0 R1 候选表 5 项逐一评估——候选 1 op 数相同且 UB 预算更差、候选 2 条件等价（slope∈[0,1] 才成立）与契约（任意 finite slope）冲突、候选 3 op 数 +1、无 CSE 空间（单 cmp+单 mul 无重复子式）、无超越函数可降代换；且基线访存已达 I/O 下界 2N·B，计算链在 ≥16M 档被 MTE2 窗口完全隐藏（PL-1.6 实测，计算链优化 ROI≈0）——**公式级无降算/降访存空间**，按原式设计。D-1 豁免条款不适用于下述（b）项（dtype 路径采纳项仍机器验证）。

**（b）dtype 路径等价性（§0.6 R2 采纳项 A1，四要素 + 机器验证）**：

- **原式**（源基线，fp16/bf16 域）：`zero_d = 0(D)；slope_d = f D(slope)；y = sel(x > zero_d, x, slope_d · x)`（逐 op 原生域舍入，D ∈ {fp16, bf16}）。
- **优化后公式**（transit 候选）：`x32 = rint(f32·x)（无损）；y32 = sel(x32 > 0, x32, f32(slope)·x32)；y = rint_D(y32)`（单次舍回）。
- **等价性论证**：① f16/bf16 ⊂ f32，升变换逐值无损且保序 → 比较结果严格一致；② fp32 域乘法精确执行 f32(slope)·x32；③ 单次 rint 舍回 = golden 锚点（`F.leaky_relu(x.float(), slope).to(dtype)`）的 fp32 opmath + 单次舍回路径 → 与 golden 逐位一致；④ 与源基线（原生域逐 op 舍入）为**容差内等价**（舍入路径差异 ≤1 ulp，远小于测试容差 fp16 1e-3 / bf16 1.6e-2），且严格更近 golden。fp32 dtype 无 transit（原生域 = golden 域，逐位一致）。
- **收益量化**：非 FLOPs 收益——**精度收益**：candidate vs golden 最大 ulp 1→0、违反率 >0→0（消除 Stage 3 精度门禁返工风险，lerp 实证此类分歧 Stage 3 才暴露损失 2485s）；代价 +2 vcast/tile（fp16/bf16），同指令形态于 bf16 路线本就必需。
- **机器验证（D-1，`verify_equiv.py` 已执行，torch CPU + fp64 锚 + 角点 ±Inf/NaN/±0/次正规/65504）**：

| 采纳项 | dtype | slope | cand vs golden max ulp | cand vs golden 违反 | 基线 vs golden max ulp | 基线 vs golden 违反 | 结论 |
|--------|-------|-------|------------------------|---------------------|------------------------|---------------------|------|
| A1 fp32-transit | float16 | 0.01 | 0 | 0 | 1 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | float16 | 0.2 | 0 | 0 | 1 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | bfloat16 | 0.01 | 0 | 0 | 1 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | bfloat16 | 0.2 | 0 | 0 | 1 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | float32 | 0.01 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |
| A1 fp32-transit | float32 | 0.2 | 0 | 0 | 0 | 0 | **EQUIV_PASS** |

**（c）弃选候选机器证据（非采纳项）**：候选 2 vmax 条件式——`y = max(x, slope·x)` 在 slope=0.01 时 EQUIV_PASS（0 mismatch），**slope=2.0 时 EQUIV_FAIL（65521/65548 元素违反，≈全部负半轴）**——机器证实其非普遍等价性，支撑弃选（脚本同 `verify_equiv.py` A2 段）。

**优化结论**：采纳 A1——fp16/bf16 走 fp32 transit 单次舍回（机器验证 6/6 EQUIV_PASS，脚本 `verify_equiv.py`，结果表见上）；公式级无进一步优化空间（R1 全部化简候选负收益或条件等价弃选 + 基线访存已达 I/O 下界 2N·B），最终公式 = 基线 `sel(x>0, x, slope·x)` + A1 dtype 路径，与 §1.4/§3.1/§4.4 一致。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案全部循环与标量计算点盘点（逐项判定）：

| # | 计算点 | 类型 | 替代方案 | API 佐证 | 结论 |
|---|--------|------|---------|---------|------|
| V1 | 逐元素 `if_then_else(val>0, val, slope·val)`（源算法核心计算） | 逐元素标量选择 | `vcmp("gt")` + `vselect` + `vmul` 全 buffer 向量链 | docs/Tilelang.language/比较操作/T.vcmp.md、docs/Tilelang.language/条件操作/T.vselect.md、docs/Tilelang.language/数学操作/T.vmul.md；先例 gqa_prefill `_gqa_prefill_fwd_kernel.py`（vcmp+vselect 生产使用） | **向量化 ✅** |
| V2 | 逐元素 `if idx < N` 尾 guard（源 fp8 分支） | 逐元素标量分支 | tile 级 `tail = T.min(block_size, N−t0)` 切片 + 零起点 UB 拷出 | 官方 `examples/elementwise/vec_add_1d.py` 尾块模式；VP-2026-0003 | **向量化 ✅（消除）** |
| V3 | 标量 slope/zero 常量（源内联 cast） | 标量广播 | `T.vbrc(slope, slope32)` / `T.vbrc(0.0, zero32)` 每 kernel 一次（persistent 循环外） | docs/Tilelang.language/shape操作/T.vbrc.md §2.2.2（scalar→tensor，fp32 ✓） | **向量化 ✅** |
| V4 | fp16/bf16→fp32 域转换（源无此项） | 逐元素 cast | `T.vcast(..., round_mode="rint")` 全 buffer | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 rint / bf16→f32 rint / f32→f16·bf16 rint） | **向量化 ✅** |
| V5 | persistent 核内 `for i in T.serial(num_local_tasks)`（tile 调度循环） | tile 级标量循环体 | 不可向量化——**理由**：任务映射循环（每迭代处理一个不相交 tile，循环携带 block_id/t0/tail 的 block 级索引算术，O(1)/tile 非逐元素热点）；tile 间无逐元素计算可合并 | — | **保留（理由：block 级索引/任务映射，可接受类别）** |
| V6 | `tail = T.min(block_size, N − t0)` / `block_id = i·num_kernels + cid` / `if block_id < num_logical` | 标量算术/分支 | 不可向量化——**理由**：block 级索引与动态边界处理（依赖 N 的尾块语义，无法静态化为全 tile——N 本身工厂特化常量但 tail 随 tile 变化；零起点切片方案已将逐元素分支静态化，此标量仅 O(1)/tile） | — | **保留（理由：动态边界 + block 级索引）** |
| V7 | host 侧 N 校验 / block_size UB guard / 展平 reshape | host 标量 | 不在 kernel 内（host 元数据/契约校验） | — | **保留（理由：host 侧元数据）** |

**向量化结论**：7 个循环/标量计算点中 4 处以 v-prefix 向量操作等价替代（V1 逐元素选择链 → vcmp+vselect+vmul、V2 逐元素尾 guard → tile 级 tail 切片消除、V3 标量常量 → vbrc 广播、V4 dtype 域转换 → vcast，API 佐证见上表），3 处保留（V5 tile 调度循环——任务映射类 / V6 block 级索引与动态边界 / V7 host 元数据，逐项理由见上表，无"实现简单"式保留）；本结论与 §6 循环结构一致（tile 内无显式逐元素循环，v-prefix 全 buffer 形态）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

- **I/O layout（契约，非设计变量）**：输入/输出均为展平连续 1-D `(N,)`（host `contiguous().reshape(-1)`，Op 层契约；manifest 任意 4D/ND 输入经 host 展平）。
- **候选枚举**：本算子核内仅一条数据轴（展平元素轴），**无第二非 batch 维**——属"纯 1-D 逐元素"单候选轴情形（豁免多轴枚举，理由：无广播轴、无归约轴、无窗口轴；4D→1D 展平在 host 视图完成，核内无重排自由度）。候选：①（唯一主选）**连续元素轴 lane 映射**——UB buffer `(block_size,)` 一维连续，v-prefix op 全宽度 lane 并行，fp32 域 128B/4-lane 组对齐；②（枚举即弃）核内重排布局——无收益场景（无跨轴访问、无 bank 冲突问题域），且引入 repack 代价，弃选（量化理由：重排仅为负收益，PL-1.1 转置链 µs 级开销纯增）。
- **向量化宽度与尾浪费**：fp32 域 4 lane/32B 组；尾 tile `tail` 非 32B 整除时仅最后一个 lane 组部分有效，v-prefix 整 buffer 计算覆盖陈旧 lane 但 `[0:tail]` 拷出屏蔽（正确性无碍，浪费 ≤7/8 组/尾 tile，总量 O(1)/tile）。
- **与 §3.3/§4/§6 一致性**：主选轴 = 展平连续轴，UB 驻留缓冲一维 `(block_size,)`，累加/计算链无跨步系数——§3.3 伪代码、§4 缓冲表、§6 循环结构均按此布局（三方一致）。
- **实验裁决模式**：不触发——判定不依赖未实证常数（单候选轴；带宽常数引用 PL-1.6 已实测条目）。

**布局决策结论**：数据布局 = GM → UB（一维连续 `(block_size,)`，零起点切片）→ Vector 单元（v-prefix 全宽度 lane）→ UB（降域复用载入 buffer）→ GM；主选轴 = 展平连续元素轴（单候选轴、一维分块），NZ 分形布局不适用（纯逐元素无矩阵语义、无跨轴访问自由度）；fp32 域按 128B/4-lane 组对齐，尾 tile 浪费 O(1)/tile 由 `[0:tail]` 拷出屏蔽——与 §3.3 伪代码、§4 缓冲表、§6 循环结构三方一致。

---

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（Stage 0 prompt 显式指定："用 developer 模式实现"）。

### 2.2 选型理由

1. 调度 prompt 契约（最高优先级）。
2. 算子形态匹配：单阶段 copy→compute→copy 逐 tile 流水，无跨核协作、无手动流水插桩需求——Developer 自动同步/自动多缓冲已覆盖（同族先例 lerp_tensor 同判）。
3. v-prefix API（vcast/vcmp/vmul/vselect/vbrc）均为 Developer Op 形态（docs/Tilelang.language/ 各 §1 标注 `[Developer Op]`）。

### 2.3 模式影响

- 同步：编译器自动插入 UB 读写栅栏，无手动 `set_flag/wait_flag`（§7）。
- 多缓冲：auto multi-buffer（Stage 2 编译期自动 stage 化），PL-1.6 实证可将 5-op 计算链完全隐藏于 MTE2 窗口。
- 模块头 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`（lerp 同款，防 Expert/Developer 混跑进程互相污染——kernel_base CG-2026-0010 同源问题域）。

## 3. API 映射设计

### 3.1 公式拆解

优化后公式（= §1.6.1(b) A1 采纳式 + §1.6.0 选定落地形态）拆解为逐 tile 原语序列：

```text
D = dtype（fp16|bf16|fp32）；S = fp32（D≠fp32 时）或 D（fp32 时）
① 载入    x_ub[0:tail]      ← GM x[t0:t0+tail]
② 升域    x32[0:tail]       ← rint(fp32 ← x_ub)            （仅 fp16/bf16）
③ 比较    cond[0:tail]      ← gt(x32, zero32)
④ 乘      sx32[0:tail]      ← slope32 ⊙ x32
⑤ 选择    sx32[0:tail]      ← sel(cond, x32, sx32)          （原地写回）
⑥ 降域    x_ub[0:tail]      ← rint(D ← sx32)                （仅 fp16/bf16；复用载入 buffer）
⑦ 写出    GM y[t0:t0+tail]  ← x_ub[0:tail]
预备（每 kernel 一次）⑧ slope32 ← vbrc(negative_slope)；⑨ zero32 ← vbrc(0.0)
```

### 3.2 TileLang API 映射

| 语义步骤 | API（v 前缀优先） | 关键参数 | 文档佐证（亲读） |
|---------|------------------|---------|-----------------|
| ① 载入/⑦ 写出 | `T.copy` | `T.copy(x[t0:t0+tail], x_ub[0:tail])` 零起点切片 | docs/Tilelang.language/内存操作/T.copy.md；官方 vec_add_1d / lerp 先例 |
| ②⑥ 域转换 | `T.vcast` | `round_mode="rint"`（f16→f32 / bf16→f32 仅 rint；f32→f16/bf16 rint） | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1 |
| ③ 比较 | `T.vcmp` | `T.vcmp(x32, zero32, cond, "gt")`，cond 为 UB bool | docs/Tilelang.language/比较操作/T.vcmp.md §2.1/§2.3（"gt" 合法；bool 仅片上） |
| ④ 乘 | `T.vmul` | `T.vmul(slope32, x32, sx32)`（fp32 域 ✓） | docs/Tilelang.language/数学操作/T.vmul.md §2.2.1 |
| ⑤ 选择 | `T.vselect` | `T.vselect(cond, x32, sx32, sx32)`（cond 真→A=x32） | docs/Tilelang.language/条件操作/T.vselect.md §2.2.1（fp16/fp32/bf16 ✓） |
| ⑧⑨ 常量广播 | `T.vbrc` | `T.vbrc(negative_slope, slope32)` / `T.vbrc(0.0, zero32)` | docs/Tilelang.language/shape操作/T.vbrc.md §2.2.2（scalar→tensor ✓） |
| 分配 | `T.alloc_ub` | `(block_size,)` 各缓冲；cond 用 `"bool"` dtype | docs/Tilelang.language/内存操作/T.alloc_ub.md；bool UB 先例 gqa_prefill L709、sparse_mla L230 |
| 核/循环 | `T.Kernel(num_kernels, is_npu=True)` / `T.serial` / `T.ceildiv` / `T.min` | persistent 一维分核 | lerp_tensor 先例；VP-2026-0003 模板 |

### 3.3 计算伪代码

```python
def _make_leaky_relu_kernel(N, dtype, negative_slope, output_dtype=None,
                            is_fp8=False, threads=256, npt=8):
    # 接口契约保持（§3.5）；host 校验：
    #   is_fp8=True -> ValueError（契约死代码，§0.5 #5）
    #   output_dtype 非 None 且 != dtype -> ValueError（manifest same_as(input)）
    #   N <= 0 -> ValueError；dtype 不在 {fp16,bf16,fp32} -> ValueError
    # block_size = threads * npt（源语义保持，E2 契约 block 语义）
    vector_cores = NPUUtils.get().get_aicore_num() * 2        # 实查，纯 Vector 翻倍
    use_transit = dtype in ("float16", "bfloat16")            # §0.6 R2
    num_logical    = ceildiv(N, block_size)                   # 编译期常量
    num_kernels    = min(num_logical, vector_cores)           # ≤ 物理核
    num_local_tasks = ceildiv(num_logical, num_kernels)       # 静态边界

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(threads_arg, npt_arg):                         # E2 契约保持
        block_size = threads_arg * npt_arg
        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(num_kernels, is_npu=True) as (cid, _):
                x_ub    = T.alloc_ub((block_size,), dtype)    # 载入 + 降域复用
                x32     = T.alloc_ub((block_size,), "float32")  # fp32 域（fp32 路线即 x_ub）
                sx32    = T.alloc_ub((block_size,), "float32")
                cond    = T.alloc_ub((block_size,), "bool")
                zero32  = T.alloc_ub((block_size,), "float32")  # 常量（循环外填充）
                slope32 = T.alloc_ub((block_size,), "float32")
                T.vbrc(0.0, zero32)                           # ⑨ 每 kernel 一次
                T.vbrc(negative_slope, slope32)               # ⑧（fp32 值参与）
                for i in T.serial(num_local_tasks):           # 静态边界 grid-stride
                    block_id = i * num_kernels + cid
                    if block_id < num_logical:
                        t0   = block_id * block_size
                        tail = T.min(block_size, N - t0)      # 尾块（静态上界）
                        T.copy(x[t0:t0+tail], x_ub[0:tail])   # ①
                        if use_transit:
                            T.vcast(x_ub, x32, round_mode="rint")     # ② 无损升域
                        # （fp32 路线：x32 即 x_ub，无 vcast）
                        T.vcmp(x32, zero32, cond, "gt")               # ③ 严格 >
                        T.vmul(slope32, x32, sx32)                    # ④
                        T.vselect(cond, x32, sx32, sx32)              # ⑤ 原地
                        if use_transit:
                            T.vcast(sx32, x_ub, round_mode="rint")    # ⑥ 单次舍回
                        T.copy(x_ub[0:tail], y[t0:t0+tail])           # ⑦
        return main
    return kernel
```

注：fp32 路线（use_transit=False）跳过 ②⑥，`x32/sx32/cond/zero32/slope32` 与 `x_ub` 的关系按 3.3 注释——实现期可将 fp32 路线的域缓冲直接绑 `x_ub`（dtype=fp32），两分支由工厂期 Python if 静态选择（TVM-script if 块作用域限制，lerp 同款规避）。

### 3.4 API 可行性确认

| API | 存在性证据 | 形态先例（examples/） |
|-----|-----------|----------------------|
| T.vcmp "gt" + bool UB | docs/Tilelang.language/比较操作/T.vcmp.md（cmp_mod 列表含 "gt"；bool 输出仅片上） | gqa_prefill `_gqa_prefill_fwd_kernel.py`（vcmp+vand+vselect 生产链） |
| T.vselect | docs/Tilelang.language/条件操作/T.vselect.md（fp16/fp32/bf16 √） | 同上；argmax kernel |
| T.vmul fp32 | docs/Tilelang.language/数学操作/T.vmul.md（fp32 √，bf16 ×——transit 后仅 fp32 域使用） | lerp_tensor 全链 vmul |
| T.vcast rint | docs/Tilelang.language/数据类型转换操作/T.vcast.md §2.2.1（f16→f32 rint / bf16→f32 rint / f32→f16·bf16 rint） | lerp_tensor transit 链（两任务实证 PL-1.17） |
| T.vbrc scalar→UB | docs/Tilelang.language/shape操作/T.vbrc.md §2.2.2 示例 1（标量广播，fp32 ✓） | brc 官方示例 |
| T.alloc_ub + "bool" | docs/Tilelang.language/内存操作/T.alloc_ub.md | gqa_prefill L709 / sparse_mla L230（bool UB 先例） |
| T.copy 零起点切片 | docs/Tilelang.language/内存操作/T.copy.md | vec_add_1d（官方尾块）、lerp_tensor |
| T.Kernel(is_npu=True) + T.serial persistent | lerp_tensor / ada_layer_norm | persistent 分核已验证形态 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查（ascend-constraints.md 强制检测项）

| 检测项 | 结果 |
|--------|------|
| 三维 Kernel | ✅ 不涉及——一维 `T.Kernel(num_kernels)` |
| L0C 容量 | ✅ 不涉及——纯 Vector，无 Cube/L0C 路径 |
| GEMM 非整除 | ✅ 不涉及——无 GEMM；逐元素尾块另有 tail 方案（§6.4） |
| L1 分形限制 | ✅ 不涉及 |
| UB 容量 | ✅ 有界——19B/elem × 8192 = 155.6KB < 192KB（§4.5）；host guard 拒绝超限 block_size |
| bool/i1 写 GM | ✅ 规避——cond 仅片上消费（vcmp.md §2.3 限制条款已核对），vselect 输出为 fp32 |
| bf16 v-算术缺失 | ✅ 绕开——transit 后全部算术在 fp32 域（§0.6 R2；T.vmul.md bf16 × 已核对） |
| 一维 Kernel 核数超发 | ✅ persistent 分核（§5.5） |
| in-place 参数 | ✅ 契约保持——kernel 始终写独立 y，inplace 为 Op 层 copy-back（§0.2） |

#### 3.5.2 参考实现差异说明

| 差异点 | 源（GPU） | NPU 设计 | 理由 |
|--------|----------|---------|------|
| 并行组织 | `T.Parallel(threads, npt)` 两级 SIMT | persistent 一维分核 + v-prefix 全 buffer | §0.5 #4（一维 Kernel 限制） |
| 片上驻留 | register fragment | UB | §0.5 #2 |
| dtype 域 | 输入域逐 op 舍入 | fp16/bf16 fp32 中转单次舍回 | §0.6 R2（bf16 算术缺失 + golden 对齐） |
| 尾块 | 整块 fragment（无 guard，依赖整除）/逐元素 guard（fp8） | tile 级 tail 切片 | §0.5 #8 |
| fp8 分支 | explicit_parallel + fp16 accum | 舍弃（ValueError） | §0.5 #5（契约死代码） |
| E2 工厂签名 | `kernel(threads_arg, npt_arg)` | **保持不变**（block_size = threads_arg×npt_arg 内部折叠） | 接口参数不变更（迁移公共规则） |

#### 3.5.3 本项目同类实现参考

1. `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py` —— 同族（单步逐元素 Parametric builder）已验证迁移：persistent 分核、fp32 transit、UB guard、L0 套件、E1/E2 工厂契约（**最接近模板**）。
2. `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/_make_lerp_tensor_kernel.py` —— persistent nc=48 全档贴 copy 地板（Stage 4 调优标杆）。
3. `examples/elementwise/vec_add_1d.py` —— 官方一维逐元素 + 尾块零起点切片模式。
4. `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/_gqa_prefill_fwd_kernel.py` —— vcmp+bool UB+vselect 生产先例。
5. `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/_ada_layer_norm_kernel.py` —— NPUUtils 物理核数查询与 per-factory 重查先例。
6. verbatim 源件：`examples/TileOPs/tileops/kernels/elementwise/leaky_relu/_leaky_relu_fwd_kernels.py`。

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
| 生成方式 | kernel 内新 buffer（`out_idx=[1]`）；inplace 为 Op 层 copy-back |

### 4.3 中间缓冲区（全部 UB，`(block_size,)` 一维）

| 缓冲 | dtype | 字节/元素 | 角色 | 复用 |
|------|-------|----------|------|------|
| `x_ub` | 输入 dtype | 2（fp16/bf16）/ 4（fp32） | 载入 + 降域回写目标 | 载入后原值死后复用 |
| `x32` | fp32 | 4 | 升域输入（fp32 路线绑 x_ub 不独占） | — |
| `sx32` | fp32 | 4 | slope·x 中间 + 选择输出（原地） | vselect 原地写回 |
| `cond` | bool | 1 | vcmp 谓词 | 片上消费，不落 GM |
| `zero32` | fp32 | 4 | 比较常量（每 kernel 一次 vbrc） | persistent 循环外填充 |
| `slope32` | fp32 | 4 | slope 常量（每 kernel 一次 vbrc） | 同上 |

### 4.4 内存搬运路径

```text
GM x ──MTE2──> UB(x_ub) ──[vcast]──> UB(x32) ──vcmp/vmul/vselect（Vector 单元，UB 内）──> UB(sx32) ──[vcast]──> UB(x_ub) ──MTE2──> GM y
```

纯 Vector 路径：GM↔UB 直连（MTE2），无 L1/L0 介入；计算全在 Vector 单元 + UB。

### 4.5 UB 内存预算

| dtype | 缓冲字节/元素 | block_size 上界（196608B UB） | 保守 cap | 默认 block（E2 契约） |
|-------|--------------|------------------------------|---------|----------------------|
| float32 | 4+4+1+4+4 = **17B**（x_ub/x32 合并） | 11565 | **8192** | threads×npt = 256×4 = **1024** |
| float16/bfloat16 | 2+4+4+1+4+4 = **19B** | 10347 | **8192** | threads×npt = 256×8 = **2048** |

- host guard：`block_size > 8192 → ValueError`（lerp 同款 guard 形态，`python -O` 安全用 ValueError 非 assert）。
- auto multi-buffer（Developer 默认）会按 stage 数倍增 UB 占用——默认 block（1024/2048）下双缓冲峰值 = 2×预算仍 < 192KB（fp16: 2×19×2048=77.8KB；fp32: 2×17×1024=34.8KB）；** Stage 4 若调大 block 需按多缓冲倍数复算**（风险点 §9.1）。

### 4.6 动态轴定义

- 唯一动轴：N（展平元素总数）。处理方式：**builder 工厂按 N 特化**（源契约 `lru_cache` + N 闭包常量；非 kernel 内动态 shape）——N 为编译期常量，num_logical/num_local_tasks/tail 上界全部 trace 期折叠；任意 N 的尾部由 §6.4 tail 方案覆盖。与源 Op 契约一致（`N_total` constructor 参数、每 N 一 kernel）。

### 4.7 JIT 配置

- `@tilelang.jit(out_idx=[1], target="npuir")`；E2 工厂 `kernel(threads_arg, npt_arg)`（保持源签名；block_size = threads_arg×npt_arg 编译期折叠）。
- `TILELANG_ASCEND_MODE=Developer`（模块头 setdefault）。
- pass_configs：默认（无特殊项；Stage 4 可评估 auto multi-buffer stage 数）。

## 5. Tiling 策略

### 5.1 计算类型

纯 Vector（逐元素比较/乘/选择），无 Cube 参与；仅需 UB，无 L1/L0A/L0B/L0C。

### 5.2 Block 划分

- **block_size（tile 内元素数）= threads_arg × npt_arg**（源语义 L2438 保持；E2 契约传入）。
- 默认：fp32 1024、fp16/bf16 2048（源 default_config threads=256, npt=4/8 折叠）；UB cap 8192（§4.5）。
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

**要素② 物理核数依据（实查记录，2026-10-28）**：

- 目标设备 **Ascend 910B3**。本设计环境 `import tilelang` 因 dev root 私有 TVM 包缺失不可用，无法直接调用 `NPUUtils.get().get_aicore_num()`；采用**等价实查路径**：ctypes 直调 CANN runtime `libruntime.so` 的 `rtGetAiCoreCount`——与 `NPUUtils` 内部实现**同一运行时调用**（`tilelang/utils/npu_utils.cpp` L124–L137 `getAiCoreNum()` 即封装 `rtGetAiCoreCount`，亲读核对）。查询代码与返回值：

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
| smoke-1m | 2²⁰ | fp32（1024） | 1024 |
| smoke-1m | 2²⁰ | fp16/bf16（2048） | 512 |
| gan-feat [16,256,64,64] | 2²⁴ | fp16/bf16（2048） | 8192 |
| gan-feat-deep [16,512,32,32] | 2²³ | fp16/bf16（2048） | 4096 |
| fc-wide [2048,4096] | 2²³ | fp16/bf16（2048） | 4096 |

**要素③ 规模判定与分核方案**：**极大规模**——全部 manifest workload 的 num_logical（512–8192）均 > 物理 Vector 核数 40，且无法通过增大 block_size 缩减到 ≤40（UB cap 8192 → smoke-1m fp16 已是 512 个 tile，2²⁴ 需 2048 个，均 ≫ 40）。方案（core-split-strategy.md §1 要素③ 极大规模档）：

- 固定启动 `num_kernels = min(num_logical, 40)`（编译期常量；小 N 自动退化为 num_logical ≤ 40，同模板全覆盖）。
- 核内串行：`for i in T.serial(num_local_tasks)`，`num_local_tasks = ceildiv(num_logical, num_kernels)`——**静态边界**（N/block_size/40 皆 trace 期常量）。
- 任务映射：`block_id = i × num_kernels + cid`（grid-stride），`if block_id < num_logical` 屏蔽越界。
- 负载分布：1M fp16 → 40 核 × ≤13 任务；2²⁴ → 40 × ≤205；2²³ → 40 × ≤103——核启动 40 次（对比超发 8192 次串行调度），负载不均 ≤1 任务/核（≤0.5%，可接受）。
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

- **Developer 自动多缓冲**：copy→compute→copy 每 tile 单阶段，编译器自动 stage 化 GM→UB 载入与 UB→GM 写出（双缓冲），PL-1.6 实证该形态下计算链（lerp 7-pass）在 ≥16M 档完全隐藏于 MTE2 窗口——本算子 5-op（transit）/3-op（fp32）链更短，同判。
- **grid-stride 聚集窗口**：40 核聚集于移动窗口访问 GM（PL-1.6 实测优于每核连续分块 +3.2%@256M）。
- Stage 4 可选方向（不在本设计展开）：block_size 调优（UB 倍数复算）、stage 数、mte2_ratio 判定标量削减（PL-1.6 标量削减判定式：mte2_ratio ≥0.95 时无收益）。

### 6.4 尾块处理

- **方案**：tile 级动态尾块 + 零起点切片——`tail = T.min(block_size, N − t0)`；载入 `T.copy(x[t0:t0+tail], x_ub[0:tail])`；v-prefix 整 buffer 计算覆盖 `[tail:]` 陈旧 lane（无污染：选择/乘均为 lane 独立）；**拷出仅 `[0:tail]`**（官方 vec_add_1d 模式）。
- 覆盖场景：任意 N（含 N < block_size、尾 = 1、非 32B 对齐尾）；`tail` 为 trace 期可折叠表达式上界（N 编译期常量），`if block_id < num_logical` 保证最后核不越界。
- 无 padding+crop（无需 host 侧补内存——切片方案无 GM 越界风险，优于 padding 方案的额外流量）。

## 7. 同步策略

### 7.1 同步模式

**Developer 自动同步**（编程模式匹配，§2）：编译器在 GM↔UB 搬运与 Vector 计算之间自动插入栅栏；无手动 `set_flag/wait_flag`/`pipe_barrier`。

### 7.2 同步点说明

| 位置 | 同步机制 | 说明 |
|------|---------|------|
| GM→UB 载入 → vcast/vcmp 链 | 自动（UB WAR/RAW 栅栏） | Developer 编译器插入 |
| vselect 原地写回 → vcast 降域 | 自动（RAW） | UB 内依赖 |
| vcast → UB→GM 拷出 | 自动（RAW） | — |
| persistent 串行迭代间 | 自动（tile 间无跨迭代数据依赖；buffer 全量复用由自动栅栏保证） | — |
| 跨核 | 无（tile 两两不相交，无共享输出；无 workspace 往返） | 零核间同步 |

### 7.3 pass_configs 配置

默认（无手动 pass_configs）；`TILELANG_ASCEND_MODE=Developer` 模块级固定（§2.3）。Stage 4 如需显式 stage 数再评估。

## 8. 验证方案

### 8.1 Golden 函数（PyTorch 参考实现，独立于 NPU 算法）

```python
import torch
import torch.nn.functional as F

def golden_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """LeakyReLU 参考：移植源仓测试锚点（test_leaky_relu.py ref_program）。

    以 §0.1 语义为唯一依据（y = x if x>0 else slope·x，fp32 opmath + 单次
    舍回路径），**不复刻 NPU 算法**（无 vcast transit / vcmp/vselect /
    persistent 结构知识）。CPU 可跑，不依赖 NPU。
    """
    return F.leaky_relu(x.detach().float(), negative_slope).to(x.dtype)
```

- 独立性核验：golden 输入直接取 `F.leaky_relu` 语义（§0.1），与 NPU 侧 transit/向量链实现零共享。
- 参数化 slope：`test_leaky_relu_param` 用例（slope=0.2）由 golden 第二参覆盖。

### 8.2 精度标准

| dtype | atol | rtol | 来源 |
|-------|------|------|------|
| float32 | 1e-5 | 1e-5 | test_leaky_relu.py `_get_tolerances` |
| float16 | 1e-3 | 1e-3 | 同上 |
| bfloat16 | 1.6e-2 | 1.6e-2 | 同上 |

预期：transit 路径 vs golden **逐位一致**（§1.6.1 A1 机器验证 0 ulp / 0 violations）；fp32 路线逐位一致；NaN/Inf/±0 角点 IEEE 传播逐位一致。

### 8.3 L0 门槛测试计划（六项；完整 L1/L2/Boundary 由 tilelang-op-develop 展开）

| # | 项 | 用例 |
|---|----|------|
| L0-1 | smoke 全 dtype | N=2²⁰，fp16/bf16/fp32，randn |
| L0-2 | 大规模抽查 | N=2²⁴，fp16/bf16 |
| L0-3 | 非整除尾块 | N ∈ {1000, 4097, 5000}（单 partial tile / 尾=1 / 非 32B 对齐尾），fp16+fp32 |
| L0-4 | 微型 shape | N ∈ {1, 7, 32}（单非满 tile / num_logical=1 退化），fp16/bf16/fp32 |
| L0-5 | 角点与特殊值 | ±Inf / NaN / ±0 / 次正规（±5.96e-8）/ max-finite（65504）/ slope=0.2 非默认参数；断言 NaN↔NaN、Inf↔Inf 逐位 |
| L0-6 | persistent 分核边界 | num_logical ∈ {39, 40, 41}（fp16 block 2048 → N = 39·2048 / 40·2048 / 40·2048+1）：不满核 / 恰好满核 / 掩蔽+不均任务激活 |
| 附加 | E2 双配置复用 | 同一工厂 kernel(256, 8) 与 kernel(128, 8)（block 2048/1024）先后编译运行均通过 |
| 附加 | UB guard | block_size=8193 → ValueError（fp16/bf16/fp32 三档） |

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 风险 | 等级 | 缓解 |
|---|------|------|------|
| 1 | `cond`（bool/i1）缓冲对齐：vbrc.md §2.3——i1 dst 尾轴需 16 对齐否则越界风险（该条针对 vbrc bool dst；本设计 vbrc 仅 fp32，cond 由 vcmp 产生全宽度消费） | 低 | cond 不经 vbrc、不写 GM（vcmp.md §2.3 条款已核对）；L0-3 尾块用例覆盖 |
| 2 | slope fp32 常量精度：vbrc(negative_slope, slope32) 的 fp64→fp32 转换必须为 IEEE 最近舍入（= np.float32(slope)），A1 等价性以该值为前提 | 中 | L0-5 含 slope=0.2 逐位断言；Stage 3 实现后抽验 `slope32` 内容（T.print 或小 N 对拍） |
| 3 | vselect 原地写回（Out=B 同缓冲） | 低 | lane 独立无读后写冲突；lerp 原地链（vsub/vmul/vadd in-place）已验证同形态；L0 全量兜底 |
| 4 | auto multi-buffer 倍增 UB 占用：block_size 调大时双缓冲峰值 = 2×19B×block | 中 | §4.5 表按双缓冲复核；host guard 上限 8192 已留裕量（155.6KB 单缓冲，双缓冲 77.8KB@2048）；Stage 4 调 block 必须复算 |
| 5 | 物理核数跨设备差异（本设备 20/40；同族记录他设备 24/48） | 低 | 工厂期 `NPUUtils.get().get_aicore_num()*2` 现查现用（ada_layer_norm 同款），部署自适应 |
| 6 | D-2 估算常数跨设备外推（PL-1.6 测于 910B2C/48c，本机 910B3/40c；1:1 R/W vs lerp 3:2 混合） | 低 | §1.6.0 已标注未实测假设；Stage 4 msprof 实测校准回写 |
| 7 | fp8 参数误用：is_fp8=True / output_dtype≠dtype 在 NPU 无对应路径 | 低 | host ValueError（§3.3 契约校验），报错语义与源基类一致 |
| 8 | pattern-library 引用条目跨工具链 stamp（PL-1.6/PL-1.17 实证于 tilelang 0.1.2+ed787bb~1990aa9fe4 / CANN 8.5.0 / 910B2C；当前 0.1.2+88ee8c302e / CANN 9.0.0 / 910B3） | 低 | 两条目 front-matter status=verified（kb_stale_check 未判 stale）；引用处均带版本戳；Stage 3/4 首跑即重验（L0 套件 + msprof） |

### 9.2 常见错误（Stage 3 防呆）

- 不要把 `num_kernels` 写成 num_logical（超发 → 串行调度开销，core-split 失败信号②）。
- 不要用 `@T.prim_func` 内 Python if 引用循环变量（TVM-script if 块作用域限制，C11/lerp 先例——分支放工厂期）。
- 不要对 bool cond 做 GM 拷出（vcmp.md §2.3 明令禁止，编译失败）。
- 不要省略 slope 的 fp32 域参与（误用 fp16 域 slope 会引入 0.010002 vs 0.00999999 差异，破坏 A1 等价前提）。
- E2 契约：`self.kernel(cfg["threads"], cfg["num_per_thread"])` 两参调用形态必须保持（wrapper init_config 依赖）。

### 9.3 特殊场景处理

- **N=0**：Op 层 `N_total` 校验拦截，不进 kernel；builder 防御性 `N <= 0 → ValueError`。
- **inplace=True**：Op 层 copy-back（kernel 无感知，始终写新 buffer）——契约与源一致。
- **slope = 0.0**：退化为 y = sel(x>0, x, ±0)（负支全 ±0），链路无需特判（公式自然覆盖）；L0-5 可加 case。
- **fp32 大 N**：无 transit，链最短（3 op）；§1.6.0 估算下界已列。

## 10. 交付清单

### 10.1 目录结构

```text
examples/leaky_relu/_make_leaky_relu_kernel/
├── DESIGN.md            # 本文档
├── verify_equiv.py      # D-1 等价性机器验证（已执行，§1.6.1 结果表）
├── RETROSPECTIVE.md     # Stage 1 复盘（Phase 8）
└── {op}.py              # Stage 3 产出：kernel + golden + 内嵌 L0 套件
```

### 10.2 文件清单（Stage 3 输入契约）

| 文件 | 内容 | 状态 |
|------|------|------|
| DESIGN.md | 全部设计决策（§0–§10） | 本文档 |
| verify_equiv.py | A1 采纳项 6 行 + A2 弃选证据 2 行；torch CPU 独立可跑 | ✅ 已产出并执行（EQUIV_PASS） |
| RETROSPECTIVE.md | Stage 1 复盘 | 随本文档产出 |
| {op}.py | `_make_leaky_relu_kernel` builder + `golden_leaky_relu` + L0 套件 | 待 Stage 3 |

### 10.3 命名规范

- builder 函数名/参数名：`_make_leaky_relu_kernel(N, dtype, negative_slope, output_dtype=None, is_fp8=False, threads=256, npt=8)`（逐字保持源签名）。
- E2 工厂：`kernel(threads_arg, npt_arg)`；prim_func 名 `main`（源一致）。
- golden：`golden_leaky_relu`。

### 10.4 实现顺序（Stage 3 建议）

1. builder 骨架 + host 契约校验（§3.3）→ 2. persistent 分核 + fp32 路线（先通 fp32 全 L0）→ 3. transit 分支（fp16/bf16）→ 4. L0-5 角点 + L0-6 分核边界 → 5. E2 复用 + UB guard 用例。
