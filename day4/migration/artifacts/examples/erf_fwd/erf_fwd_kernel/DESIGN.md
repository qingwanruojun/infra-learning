# erf_fwd_kernel 算子设计文档（迁移任务：TileOPs GPU → TileLang-NPUIR）

> project=erf_fwd, op=erf_fwd_kernel；场景 migration-harness（TileOPs 同构仓迁移）；编程模式 Developer。
> 源算子：TileOPs `ErfFwdKernel(FloatUnaryKernel)`（op_func = `T.erf(T.cast(x, "float32"))`）。
> 本文档章节序号遵循 tilelang-op-design design-template：§0 迁移分析 → §1 概述（§1.6 算法调研与优化分析）→ §2–§10。

## 0. 源算子解读与迁移分析

> 执行序：Phase M0（本节 0.1–0.4，源码逐层实读）→ Phase R（§1.6.0，调研四问）→ Phase M1（0.5–0.6，判定输入含 §1.6.0 调研结论）。源码阅读清单（全部亲自 Read 核对）：GPU kernel `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L680–879（`UnaryKernel` 基类）、L1297（`FloatUnaryKernel`）、L344–363（`_make_unary_regcopy` 默认策略 builder）、L2118–2127（`ErfFwdKernel`）；GPU op `/mnt/workspace/learn/refs/TileOPs/tileops/ops/elementwise/math_unary.py` L207–211；GPU test `/mnt/workspace/learn/refs/TileOPs/tests/ops/test_unary_math.py`（容差 L69–74、edge 用例 L396–404）；落地侧 wrapper `examples/TileOPs/tileops/kernels/elementwise/erf_fwd/erf_fwd.py`（全文 219 行逐字核对）；NPU 侧抽取参考 `examples/TileOPs/tileops/kernels/elementwise/erf_fwd/_erf_fwd_kernels.py`；NPU 侧 test/bench `examples/TileOPs/tests/ops/test_unary_math.py`、`examples/TileOPs/benchmarks/ops/bench_unary_elementwise.py`；manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` L225–260（ErfFwdOp 条目）；同族已落地标杆 `examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py`（全文）。

### 0.1 源算子语义（做什么）

- **数学语义**：逐元素误差函数 `y[i] = erf(x[i]) = (2/√π)·∫₀^x e^(−t²) dt`，值域 (−1, 1)，奇函数。单输入单输出、无规约、无跨元素依赖（纯逐点映射）。
- **dtype 语义（中间精度契约）**：源 `op_func`（elementwise.py L2126–2127）为 `T.erf(T.cast(x, "float32"))` —— **三种输入 dtype 一律先升域 fp32、在 fp32 域调用 erf、单次舍回输出 dtype**（fp32 中转 / opmath 形态）。fp32 输入为原生域（cast 恒等）。这是语义层的精度契约，不是可优化的实现细节。
- **边界语义**：erf 全域连续有界。`erf(±0) = ±0`（保号）；`erf(±inf) = ±1`；`erf(NaN) = NaN`；大 |x|（|x| ≳ 4）输出在输出 dtype 舍入后恒为 ±1；subnormal 输入 → erf ≈ 2x/√π ≈ 0。无空维度、无非整除 shape 语义问题（flatten 后 1-D）。
- **语义完成标准达成**：golden 函数 = `torch.erf(x)`（源测试锚 `ref_fn = torch.erf`，GPU test L402 / NPU test L221）；NPU 侧测试 `ref_program` 显式为 `ref_fn(x.float()).to(x.dtype)`（NPU test L164–168，fp32-opmath 形态）——与 §0.1 dtype 契约一致，已能写出数值一致的 golden。

### 0.2 源算子输入输出

| 项 | 内容 | 来源（亲自核对） |
|----|------|------|
| 输入 | `x: (N_total,)` 扁平连续 1-D，dtype ∈ {float16, bfloat16, float32}；Op 层 `flatten → kernel → reshape`（erf_fwd.py O6，原始多维 shape 由 Op 层还原） | erf_fwd.py L86–87、L186 |
| 输出 | `(N_total,)`，dtype = same_as(input)（`self.output_dtype = dtype`） | erf_fwd.py L187 |
| 输出 shape | `output.shape == input.shape`（manifest shape_rules） | elementwise.yaml L236–237 |
| 调用面（落地侧现状，唯一权威） | `_erf_fwd_wrapped(N, dtype_str, block_size, x)` → `erf_fwd_kernel(N, dtype)(block_size)(x)`（**两段式 curried**：factory(N, dtype) → builder(block_size) → main(x) 返回 y，`out_idx=[1]`）——**非 GPU 侧 `(threads_arg, npt_arg)` 两参调用式** | erf_fwd.py L100–117 逐字核对 |
| K11 block_size 常量 | `_BLOCK_SIZE_FP32 = 1024`、`_BLOCK_SIZE_NON_FP32 = 2048`（GPU threads(256)×npt(4/8) 折叠） | erf_fwd.py L91–92 |
| Kernel 类契约 | `ErfFwdKernel(N_total, dtype, config=None)`；`SUPPORTED_DTYPES = (fp16, bf16, fp32)`；`ascend_mode = "Developer"`（CG-2026-0010：调用面锁定 Developer）；config 键 = `{"block_size"}`；K8 无 autotune、`init_config(config)` 无 tune 参 | erf_fwd.py L131–206 |
| dtype 入参形态 | wrapper 传 `getattr(torch, dtype_str)` 即 **torch.dtype 对象**——factory 内必须归一化（leaky_relu 教训：torch 对象与 str 集合 `in` 比较恒 False） | erf_fwd.py L116 |

### 0.3 实现算法解读（怎么算）

**源码结构化阅读协议逐层结论**（协议见 migration-analysis.md §3.1）：

1. **入口签名**：GPU factory `_make_unary_regcopy(N, dtype, op_func, output_dtype, threads=256, num_per_thread=8)`（elementwise.py L344）；`block_size = threads * num_per_thread`； ErfFwdKernel 默认策略 `register_copy`（UnaryKernel.DEFAULT_STRATEGY，L700），默认 threads=256、npt=4（fp32）/8（fp16/bf16）（`_strategy_npt`）。
2. **Kernel 结构**：`@tilelang.jit(out_idx=[1])` 内 `kernel(threads_arg, npt_arg)` → `main(x: (N,) dtype, y: (N,) out_dtype)`；`with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx` —— **一维 grid**，block 数 = ceildiv(N, block_size)。
3. **计算原语**：仅 3 句——`T.copy(x[bx*bs:(bx+1)*bs], x_reg)`；`for i, j in T.Parallel(threads_arg, npt_arg): y_reg[i*npt_arg+j] = T.erf(T.cast(x_reg[i*npt_arg+j], "float32"))`；`T.copy(y_reg, y[...])`。计算语句全部归入「载入 → 逐点 fp32-cast+erf → 写回」三步，无臆造步骤。
4. **内存分配**：`T.alloc_fragment((block_size,), dtype)` ×2（x_reg / y_reg）——CUDA fragment（寄存器驻留），无 SMEM、无多级缓冲。
5. **循环与调度**：`T.Parallel(threads_arg, npt_arg)` 两级 SIMT 展开（256 线程 × 4/8 元素/线程），线性索引 `i*npt_arg+j`；无 T.serial / T.Pipelined。
6. **host 侧逻辑**：Op 层 `input.contiguous().reshape(-1)` → kernel → `.reshape(orig_shape)`（math_unary.py 经 erf_fwd.py O6 同构移植）；`functools.lru_cache(maxsize=32)` 工厂缓存；fp8 包络 helper（`_wrap_fp8_accumulation`）对本 op 不生效（FloatUnaryKernel 拒 fp8，manifest 仅 3 float dtype）。
7. **编译配置**：无 pass_configs / 环境变量依赖（GPU 侧）。

**数据流图（源硬件视角）**：`GM(x) → T.copy → fragment x_reg → [cast fp32 → erff] 逐线程 → fragment y_reg → T.copy → GM(y)`。全部 buffer 已覆盖：x、y（GM）、x_reg、y_reg（fragment），无来源不明数据。

**host 侧逻辑清单**：dtype 校验（SUPPORTED_DTYPES ValueError）、lru_cache、flatten/reshape、config 解析（block_size=threads×npt）。无预处理/后处理数值变换。

### 0.4 优化手段解读（为什么快）

逐项识别（识别不出来 ≠ 不存在；本算子为单步逐元素最简形态，源码仅 4 类语句，清单完备）：

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---------|---------|------|----------------|------|
| S1 | register_copy 策略（fragment 驻留） | `T.alloc_fragment` + 载入/计算/写回三段 | 数据驻留寄存器，消局部内存往返 | CUDA 大寄存器堆 + SIMT fragment | 实现层，硬件强相关 → **等价替换**（UB buffer 链） |
| S2 | fp32 中转（opmath） | `T.cast(x, "float32")` 后调 `T.erf` | 亚 fp32 dtype 单次舍回的精度语义（CUDA 对 fp16/bf16 走 fp32 opmath） | **无**（纯算法/精度层） | **保留**（结构承接为 transit 链，§0.6 R2） |
| S3 | threads×npt 两级并行展开 | `T.Parallel(threads, npt)` 256×4/8 | SIMT 占用与访存合并 | CUDA warp 调度 | **重新设计**（一维 persistent 分核 + 向量化整 tile 链，§0.6 R1） |
| S4 | lru_cache 工厂缓存 | `@functools.lru_cache(maxsize=32)` | 消 host 重复编译查找 | 无 | **保留**（NPU 落地 kernel 同款） |
| S5 | K11 block_size 折叠 | threads×npt → 单 block_size（1024/2048） | 调用面简化（host 契约） | 无 | **保留**（wrapper 常量已定，§0.2） |
| S6 | GPU 硬件 erff 指令（`T.erf` → CUDA `erff`） | op_func 内单 transcendental | 单指令超越函数精度/吞吐 | CUDA SFU 特殊函数单元 | **重新设计**（NPU 无硬件 erf，证据见 §0.5-C2 / §1.6.0） |

无 SMEM tiling / coalesced 显式控制 / mma / warp shuffle / cp.async / swizzle / persistent（源 kernel 无此类特征——单步逐元素无需）。S2 为算法层优化（可移植），其余为实现层。

### 0.5 硬件耦合性分析与 NPU 适配决策

> 判定输入含 Phase R 调研结论（§1.6.0）：源算法与调研候选同台比较；四态处置依据 = migration-analysis.md §5.3 映射表条目 / docs 条目 / examples 佐证。

| 条目 | 层 | 处置 | 依据 |
|------|----|------|------|
| C1 数学语义 / I/O 契约 / 边界语义（§0.1–0.2） | 语义层 | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| C2 erf 计算 = CUDA 硬件 `erff`（S6） | 算法层 | **重新设计** → A&S 7.1.26（§0.6 R1，承接 §1.6.0 选定） | ① docs 证据：`docs/Tilelang.language/数学操作/T.verf.md` §3——NPU 侧 erf 入口 `T.verf` 的 lowering 为 `arith::ConstantOp + arith::DivFOp + hivm::VMulOp + hivm::VAddOp`（**多项式 mul/add/div 分解，非单条硬件 erf 指令**）；② 实测证据：7 阶泰勒展开在 |x|=1.5 误差 0.145、x=2 误差 1.58、x=3 误差 39.1（本任务 verify_equiv.py REJ-1 表机器复现，与用户 910B3 实测数字一致）——不满足源 edge 用例（x=3、x=100，GPU test L401）更不满足 fp32 atol 1e-5；③ `docs/Tilelang.language/` 全目录检索无 vrec/verfc 类硬件 erf 补充指令 |
| C3 fp32 中转精度语义（S2） | 算法层 | **保留**（结构承接：fp16/bf16 走 vcast rint 升域 → fp32 域链 → vcast rint 单次舍回） | golden 即 fp32-opmath 形态（NPU test L164–168）；同族先例 expm1/elu 同款 transit（CASE-elementwise-elu-migration「结构承接」型；expm1 landed kernel L208–292）；bf16 无任何 v-prefix 算术/超越函数（T.vabs.md / T.vdiv.md / T.vexp.md dtype 矩阵 bf16 ×）→ transit 亦为可行性强制 |
| C4 register_copy fragment 驻留（S1） | 优化层 | **等价替换** → UB buffer 链 + T.copy | 映射表「shared memory → UB（Vector 数据）」条目；expm1 landed kernel 同构（GM→UB→v 链→GM） |
| C5 threads×npt 两级并行（S3） | 优化层 | **重新设计** → 一维 persistent 分核 + 整 tile 向量化 v-op 链 | 映射表「block × thread 两级并行 → block 级任务划分 + 向量化元素并行（**重新设计**，本项目 T.Kernel 只接受一维 block 数）」条目；expm1 §0.6 R1 同款 |
| C6 lru_cache / K11 block_size（S4/S5） | host 层 | **保留** | 纯 host 契约，与硬件无关；wrapper 现状（§0.2）即按此契约书写 |
| C7 GPU grid 逐 tile 启动（ceildiv(N, bs) 个 block） | 优化层 | **等价替换** → persistent 核内串行（逻辑核数 ≫ 物理核数，§5.5） | docs/开发指南.md §3.3（物理核数限制与分核策略）；core-split-strategy.md §1 要素③「极大规模」方案 |

**统计**：保留 3 项（C1/C3/C6）、等价替换 2 项（C4/C7）、重新设计 2 项（C2/C5）、舍弃 0 项。无静默丢弃：S1→C4、S2→C3、S3→C5、S4/S5→C6、S6→C2 逐项承接。

### 0.6 NPU 算法重设计

> 重设计对照 §1.6.0 调研结论（Phase R 先于本节执行）；每个重设计项给出「源方案 → NPU 新算法 → 语义保持论证」三要素。

- **R1（算法替代，承接 C2 + §1.6.0 选定）**
  - 源方案：`y = T.erf(fp32(x))`（CUDA 硬件 erff，单 transcendental）。
  - NPU 新算法：**Abramowitz & Stegun 7.1.26 有理逼近**（§1.6.0 调研选定，候选对比与淘汰依据见 §1.6.0 R1/R3/R4）：
    `t = 1/(1 + 0.3275911·|x|)`；`erf(x) = sign(x)·(1 − t·(A1 + t·(A2 + t·(A3 + t·(A4 + t·A5))))·e^(−x²))`，A1..A5 = 0.254829592, −0.284496736, 1.421413741, −1.453152027, 1.061405429。
  - 语义保持论证：**容差内等价**——fp32 域 op-by-op 链最大绝对误差 5.868e-07（verify_equiv.py 实测，6M 随机点 + 23 角点，3 dtype 全部 0 违反，EQUIV_PASS），≪ fp32 atol 1e-5（17× 裕度）；fp16/bf16 单次舍回后 0 违反（包络 1e-3 / 1.6e-2）。边界语义逐一核对：erf(±0)=±0（NPU 链 x=−0 时 s=+1、r=+0，IEEE 值相等、零号差豁免，同 expm1 L0-5 先例）；erf(±inf)=±1（x²=+inf → e=exp(−inf)=0 → r=1，sign 由 vcmp/vselect 还原，verify_equiv corners ok）；erf(NaN)=NaN（逐 op NaN 传播，vcmp(NaN<0)=False → s=+1，输出 NaN，equal_nan 断言）；erf(±max_finite)=±1（x² 溢出 → e=0）；subnormal → 输出 ≈0（|err| ≤ 1e-8 ≪ atol）；大 |x| 由 e=exp(−x²) 下溢自然饱和到 ±1（corners ±3/±4/±6/±100/±max 全部 1 ulp 内）——**用户关键事实 1/2/3 均已在设计中复核并机器验证**。验证方式：§8 golden（按 §0.1 语义，torch.erf）+ L0 角点套件（§8.3）。
- **R2（transit 结构承接，承接 C3）**
  - 源方案：全部 dtype 先 cast fp32 再计算（GPU fragment 内隐式）。
  - NPU 新算法：fp16/bf16 路线 `T.copy → T.vcast(x, x32, "rint") 升域 → fp32 域 v-prefix 链 → T.vcast(x32, x, "rint") 单次舍回 → T.copy`（与 expm1 landed kernel L247–292 同构）；fp32 路线原生域无 transit。标量参与量（P、A1..A5、±1、0）以 fp32 字面量/常量缓冲参与（A2 优化，§1.6.1）。
  - 语义保持论证：fp16→fp32 升域无损（fp16 为 fp32 子集）；单次舍回与源 opmath 单次舍回舍入路径一致（GOLD-DIAG：CPU 上 `torch.erf(fp16原生)` 与 `torch.erf(x.float()).to(fp16)` 200000/200000 位等价，fp16/bf16 双双成立）→ golden 双形态任一皆数值一致。
- **R3（并行结构重设计，承接 C5/C7）**
  - 源方案：`T.Kernel(ceildiv(N, bs), threads=256)` 逐 tile 启动 + `T.Parallel(256, npt)` SIMT 展开。
  - NPU 新算法：一维 `T.Kernel(num_kernels, is_npu=True)` persistent 分核（`num_kernels = min(num_logical, vector_cores)` ≤ 物理向量核数）+ 核内 `T.serial(num_local_tasks)` grid-stride（`block_id = i·num_kernels + cid`，`num_local_tasks = ceildiv(num_logical, num_kernels)` 编译期 IntImm 静态边界）+ `if block_id < num_logical` 纯名常量 guard；元素级并行交给整 tile v-op 链（每 v-op 处理 (bs,) 全向量）。
  - 语义保持论证：逐点映射无跨元素累加 → 并行度/分组变化不触碰数值（无规约顺序问题）；grid-stride 映射只改 tile→core 分派，不改每元素计算；guard 语义 = 源 ceildiv 尾块语义。
- **R4（尾块处理）**
  - 源方案：ceildiv 尾块由 T.Parallel 边界自然覆盖（block_size 均匀切分，GPU 无越界问题因 copy 边界对齐）。
  - NPU 新算法：tile 级 `tail = T.min(bs, N − t0)`；全部 v-op 作用**整 buffer**（(bs,) 无切片——v-op 切片仅接受静态边界，TRAP-expert-v-operands P10），stale [tail:] lane 垃圾值不落 GM；store 仅 `x_ub[0:tail] → y[t0:t0+tail]`（零原点 UB 切片，官方 vec_add_1d.py 模式，expm1 §0.6 R3 已落地验证）。
  - 语义保持论证：store 边界 = 源 copy 边界；stale lane 无 GM 副作用；T.copy 接受 runtime 索引切片（traps-compiler.md 第六轮补充实证）。

**§0.4 优化意图承接核对表**：S1→C4（UB 链承接「数据驻留片上」意图）；S2→R2（transit 链逐字承接 fp32-opmath 意图）；S3→R3（persistent + 向量化承接「占满算力」意图）；S6→R1（A&S 承接「单指令超越函数」意图——NPU 无该硬件，以多项式链在带宽受限窗口内隐藏计算代价替代，§1.6.0 R4）；S4/S5→C6 原样保留。**无静默丢弃。**

### 0.7 标杆实现

- 同族已落地标杆（结构模板）：`examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py`（elementwise 族 harness 迁移已落地 kernel：factory 契约 / persistent 分核 / transit 链 / L0–L2+Boundary 嵌入式测试全套）。
- 源侧参考：`/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2118（ErfFwdKernel）、`/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L344（regcopy builder）。
- 测试锚：GPU/NPU 双侧 `tests/ops/test_unary_math.py`（erf 子集：L1 fixture 1M×3dtype randn + L4 edge fp32 4K [0, ±3, ±100] + rejects_non_float + eval_roofline）。
- 同类案例索引：CASE-elementwise-expm1-migration / CASE-elementwise-elu-migration / CASE-elementwise-leakyrelu-migration（pattern-library/cases.md，harness 迁移第 1–3 例；本任务为第 4 例同族）。

## 1. 概述

### 1.1 算子名称

`erf_fwd_kernel`（项目 `erf_fwd`；落地侧产品位置 = examples/TileOPs/tileops/kernels/elementwise/erf_fwd/ 下的 erf_fwd_kernel/ 产品子目录，Stage 3 创建，同族 expm1_kernel 布局；Stage 5 将 erf_fwd.py L73 的 import 切换至该产品）。

### 1.2 功能描述

逐元素误差函数 forward：对扁平化 `(N,)` 输入张量施加 `y = erf(x)`，输出 same_as(input)。支持 float16 / bfloat16 / float32；manifest workloads：smoke-1m（3 dtype）、elementwise-16m [4096,4096]（3 dtype）、elementwise-64m [8192,8192]（fp16/bf16）、elementwise-256m [16384,16384]（fp16/bf16）。

### 1.3 数学公式

$$\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}\,dt$$

NPU 计算公式（§0.6 R1 重设计后，容差内等价）：

$$t = \frac{1}{1 + p\,|x|},\quad p = 0.3275911;\qquad \mathrm{erf}(x) = \mathrm{sign}(x)\cdot\Big(1 - t\,\big(A_1 + t(A_2 + t(A_3 + t(A_4 + t A_5)))\big)\,e^{-x^2}\Big)$$

$$A_1..A_5 = 0.254829592,\ -0.284496736,\ 1.421413741,\ -1.453152027,\ 1.061405429$$

### 1.4 算法描述

**NPU 侧算法（迁移决策后）**：输入按 `(N,)` 扁平 1-D 处理；persistent 分核将 `ceildiv(N, bs)` 个逻辑 tile 均摊到 ≤40 个向量核（grid-stride，静态边界）；每 tile 执行：

1. `T.copy` 载入 GM→UB（fp16/bf16 载入原 dtype）；
2. fp16/bf16：`vcast rint` 升域 fp32（fp32 路线跳过）；
3. fp32 域 22-op v-prefix 链（§3.3 伪代码逐 op 对应 §1.3 公式）；
4. fp16/bf16：`vcast rint` 单次舍回原 dtype（复用载入 buffer）；
5. `T.copy` 写回 UB→GM（仅 `[0:tail]`）。

与源算法的结构差异：CUDA 硬件 erff → A&S 7.1.26 多项式链（§0.6 R1）；fragment 两级 SIMT → persistent 分核 + 整 tile 向量链（R3）；fp32 中转语义原样承接（R2）。算法选型依据 = §1.6.0 调研结论。

### 1.5 数据流图

```
GM(x, dtype) ──T.copy──▶ x_ub/x16_ub (UB, dtype)
   fp16/bf16:  x16_ub ──vcast(rint)──▶ x32 (UB, fp32)     [fp32: 直接以 x_ub 为域输入]
   域内(fp32):  x ──vcmp(lt, zeros)──▶ cond(bool UB)
               x ──vabs──▶ ax ──vmul(×P)──vadd(+1)──▶ denom ──vdiv(ones,·)──▶ t
               x ──vmul(x·x)──▶ x2 ──vmul(×−1)──▶ vexp──▶ e
               ones×A5 ──▶ h ──[4×(vmul(h·t)+vadd(+Ai))]──▶ poly ──vmul(×t)──vmul(×e)──▶ m
               ones ──vsub(−m)──▶ r;   cond ──vselect(±1)──▶ s;   r ──vmul(×s)──▶ out32
   fp16/bf16:  out32 ──vcast(rint)──▶ x16_ub（复用载入 buffer）
GM(y, dtype) ◀──T.copy([0:tail])── out（拷出源 buffer）
```

覆盖全部 buffer：x/y（GM）、x16_ub、x32、ax、x2、t、h、e、cond、ones/neg_one/zeros（常量）、out（复用 x_ub/x32，§4.3）。

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：与 `erf` 同一数学语义（全域 |误差| ≤ fp32 atol 1e-5 包络）的**算法族**；输入公式 / 源算法（CUDA erff 硬件指令）只是基线候选之一。调研深度：erf 为「多步超越函数链」→ 按标准执行完整调研（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。执行位置：Phase M0 之后、Phase M1 之前（§0.5 判定输入即本节结论）。

**R1 等价化简公式候选表（含基线）**：

| # | 候选 | 公式要点 | 等价性初判 | 收益方向 | 结论 |
|---|------|---------|-----------|---------|------|
| B0（基线） | 源算法：CUDA 硬件 `erff`（fp32 cast 后单指令） | `T.erf(fp32(x))` | 数学精确（~1 ulp） | 0 计算 / 最小延迟 | NPU 无对应硬件（T.verf lowering 反证）→ 不可行 |
| C1 | **T.verf（NPU 原生 erf 入口）** | 7 阶泰勒展开 `(2/√π)(x − x³/3 + x⁵/10 − x⁷/42)` | **非等价**（截断误差发散） | 1 op 最短链 | **淘汰**：docs T.verf.md §3 lowering = ConstantOp+DivFOp+VMulOp+VAddOp（多项式分解，非硬件 erf）；误差实测 0.145@1.5 / 39.1@3（verify_equiv.py REJ-1 表，与 910B3 用户实测一致）；源 edge 用例 x=3、x=100 直接击穿 |
| C2 | **A&S 7.1.26 有理逼近**（Abramowitz & Stegun 7.1.26，5 项 t 多项式 × e^(−x²)） | §1.3 公式 | **容差内等价**（D-1 机器验证 EQUIV_PASS） | 全域最大 |err| 5.868e-07（fp32） | **选定** |
| C3 | tanh 型有理近似（GELU-tanh 族） | `erf(x) ≈ tanh(αx+βx³)` | 非等价（误差 ~1e-3） | ~3 op 极短链 | 淘汰：1e-3 ≫ fp32 atol 1e-5 |
| C4 | 高阶 minimax 有理逼近（Cody–Waite / ercf 型） | ≥7 项 | 等价（更准） | 精度裕度 | 淘汰：C2 已有 17× 精度裕度，更高阶仅增发射 op 数，带宽受限无收益 |
| C5 | 分域混合（\|x\|≤0.5 泰勒 + 大域 A&S） | 双公式 + vselect | 等价 | 省 ~8 op@小域 | 淘汰：增加 cmp/sel/分支复杂度，带宽受限下小域省 op 无收益（§1.6.0 R3 口径） |

R1 调研范围声明：恒等变形自查（erf 奇函数性、erfc 补函数关系、erf=2Φ(x√2)−1 换底——均不降 op 数）；pattern-library（.agents/skills/tilelang-op-optimize/references/pattern-library/）同类实现所用公式；docs/Tilelang.language/数学操作/ 全目录（T.verf/vtanh/vpow/vexp2 等入口逐一核对 dtype 矩阵与 lowering）；用户实测候选（A&S 7.1.26）。C4 的「无更省 op 的同精度公式」为**未文档化假设**（依据：经典手册族谱中 5 项 t-多项式是满足 1e-7 级精度的最低阶标准形式；即使存在更省项形式，收益上限 = 省 2 op/tile ≈ 1µs，带宽主导下不可辨——列 §9 待确认风险不阻塞）。

**R2 在线算法**：**无**。结构依据：erf 为逐点无状态映射，`erf(x_i)` 仅依赖 `x_i`，元素间依赖图为零边——「在线/单遍」 Reformulation 的收益口径（扫描遍数、中间缓冲、UB 驻留）对零依赖图无作用对象；源算法本身已是单遍流式（每元素独立、单次 GM 扫描）。参考表（algorithm-candidates.md 逐元素族）无在线条目命中。结论：无在线变体（结构依据充分，非空白断言）。

**R3 复杂度四口径对比表**（口径：FLOPs = 每 tile 每元素向量操作数，exp/div/cmp/sel 各记 1；访存 = GM 读+写字节；扫描遍数 = GM 数据完整扫过次数；中间缓冲 = 每 tile UB 驻留 (bs,) 工作缓冲数。基线在表）：

| 候选 | FLOPs/elem | 访存 Bytes（N 元素） | 扫描遍数 | 中间缓冲峰值 | 判定 |
|------|-----------|--------------------|---------|-------------|------|
| B0 源算法（硬件 erff） | 1 | 2N·eB | 1 | 0（fragment） | NPU 不可行 |
| C1 T.verf 泰勒 | ~10 | 2N·eB | 1 | ~2 | **精度淘汰** |
| **C2 A&S 7.1.26（选定）** | ~22 | 2N·eB（**与基线相同**——全部中间量驻留 UB，无 GM 往返） | 1 | 6 工作 + 3 常量（(bs,4B) 各） | **选定** |
| C3 tanh 型 | ~3 | 2N·eB | 1 | ~2 | 精度淘汰 |
| C4 高阶 minimax | >26 | 2N·eB | 1 | >7 | 无收益淘汰 |

关键读法：全部候选**访存与扫描遍数完全相同**（单入单出逐点算子，无 GM 往返差异）——erf 是带宽受限算子（roofline AI：fp32 0.125 flop/B、fp16 0.25 flop/B，manifest `flops=N, bytes=2N·eB`），**算法选择由精度与发射 op 数/UB 驻留决定，不由 FLOPs 决定**；C2 的 22 op 链在带宽饱和区隐藏于 MTE2 窗口（CONST-copy-floor-method：7-pass fp32-transit 链 ≥16M 全 dtype delta ≤1.1µs——〔引用条目前置重验：工具链漂移风险，CANN 8.5.0 stamp〕首跑重验承诺：Stage 4 copy-floor 探针复测本 22-op 链暴露量）。

**R4 硬件亲和性逐候选评估**：

| 检查项 | C2 A&S（选定） | C1 T.verf | C3 tanh | C4 minimax |
|--------|---------------|-----------|---------|-----------|
| 计算单元匹配 | 纯 Vector 逐元素链：vabs/vmul/vadd/vsub/vdiv/vexp/vcmp/vselect 全部 fp32 ✓（docs dtype 矩阵逐一核对） | ✓ | T.vtanh 存在 fp32 ✓ | ✓ |
| bf16 支持 | transit 强制（T.vabs/vdiv/vexp bf16 ×）→ fp32 域统一 | bf16 ×（verf 矩阵） | bf16 × | 同 C2 需 transit |
| 片上容量 | 37–41 B/elem × bs4096 = 152–168KB < 192KB（§4.5，CONST-capacity-910B2C〔stale〕） | 更小 | 更小 | 更大（>9 缓冲） |
| 对齐整除 / 静态边界 | 1-D 连续；N/bs host int → 全静态 ✓ | ✓ | ✓ | ✓ |
| 流水可融合性 | persistent + 整 tile 链，multi-buffer 覆盖 MTE2 窗口 ✓ | ✓ | ✓ | 链更长暴露风险↑ |
| 精度（决定项） | 5.868e-07 ≪ 1e-5 ✓ | 0.145@1.5 ✗ | ~1e-3 ✗ | ✓（过冗） |

**D-2 设计期估算下界**（协议 = _shared/standards/hardware-cost-model.md §2；常数 = pattern-library/constants.md，含版本戳与 stale 标注）：

- 设备与核数：Ascend 910B3，物理 AI Core **实查 = 20**（`NPUUtils.get().get_aicore_num()` 为权威查询——本设计期环境 dev-root TVM 缺失无法导入 tilelang，按 CONST-aicore-910B2C 910B3 数据点记载的等价路径 ctypes 直调 `libruntime.so rtGetAiCoreCount`，2026-10-29 实测 ret=0 → **20**，与该条目 2026-10-28 三次独立复跑一致；纯 Vector 算子翻倍 → **40 向量核**。落地 kernel 工厂期现查，禁经验值）。
- 流量项（CONST-mte2-degradation〔引用条目前置重验：CANN 8.5.0 stamp，910B2C/48 核测量——910B3/40 核外推，Stage 4 对账〕，MTE2 按元素档位取值，每核流量 = 总字节/40）：
  | workload | 总流量 | 每核流量 | 档位带宽 | 流量项 |
  |----------|--------|---------|---------|--------|
  | smoke-1m fp16 | 4.19MB | 105KB | 33.9GB/s（1M 档） | 3.1µs |
  | smoke-1m fp32 | 8.39MB | 210KB | 33.9GB/s（1M 档，fp32 外推） | 6.2µs |
  | 16M fp16 | 67.1MB | 1.68MB | 30.2GB/s（16M 档） | 55.6µs |
  | 16M fp32 | 134.2MB | 3.36MB | 30.2GB/s（16M 档，fp32 外推） | 111µs |
  | 64M fp16 | 268.4MB | 6.71MB | 20.9GB/s（64M 档） | 321µs |
  | 256M fp16 | 1.074GB | 26.8MB | 18.4GB/s（256M 档） | 1.46ms；按 256M 3:1 混合聚合地板 1.26TB/s 折算 852µs → **报告区间 0.85–1.46ms**（两口径测量条件不同：48 核 3:1 混合 vs 本算子 1:1） |
- 发射项：22 v-op/tile × ~0.5µs/op ≈ **11µs**（tile 链时延下界，CONST-vector-launch-overhead〔stale，首跑重验〕；大 N 下隐藏于 MTE2 窗口，CONST-copy-floor-method〔stale，首跑重验〕）。
- 容量项：纯 Vector 单引擎、无跨引擎 GM 往返 → **豁免**（轻量调研口径流量项必算已完成）。
- **估算下界行**：`smoke-1m ≈ 11µs（发射主导）；16M fp16 ≈ 56µs；16M fp32 ≈ 111µs；64M fp16 ≈ 321µs；256M fp16 ≈ 0.85–1.46ms（流量主导）——引用 CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C（stale 条目重验承诺见 §9）；未实测假设：fp32/64M/256M 档位带宽外推、vdiv 吞吐≈其他 v-op（无实测常数，未文档化假设，依据：链在带宽饱和区隐藏，影响限于小 N 时延 ≤ 数 µs）`。
- 候选排序结论：唯一满足精度 + 全项硬件亲和的候选 = **C2 A&S 7.1.26**（无 R3 并列，无实验裁决需求；§1.6.3 单轴豁免）。

**调研结论**：选定算法族 = A&S 7.1.26 有理逼近 + fp32 transit（fp16/bf16）；关键依据 = R1 精度唯一达标候选（D-1 机器验证 5.868e-07 ≪ atol）+ R2 无在线变体（逐点零依赖图，结构依据）+ R3 全候选访存/遍数相同、由精度与发射数决胜 + R4 逐项硬件亲和通过；与基线的结构差异 = CUDA 硬件 erff 单指令 → 22-op v-prefix 多项式链（§0.6 R1，优化意图承接见 §0.5/§0.6）；「无更优替代」的调研范围 = R1 表 C1–C5 全候选 + docs/Tilelang.language/数学操作/ 全目录 + .agents/skills/tilelang-op-optimize/references/pattern-library/ 同类实现 + 恒等变形自查（R1 段），C4 更省项公式为未文档化假设（§9 R7 待确认）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 A&S 7.1.26 链（非源公式直译）。采纳项四要素齐全；**等价性机器验证（D-1）结果表内嵌于本节**（脚本 `examples/erf_fwd/erf_fwd_kernel/verify_equiv.py`，随算子目录留存，Stage 2 维度 8 重跑复核）。

**A1（采纳）：erf 直接计算 → A&S 7.1.26 容差内等价替代**

- 原式：`y = erf(x)`，NPU 直接实现入口 T.verf = 7 阶泰勒（精度发散，§1.6.0 C1）。
- 优化后公式：§1.3 A&S 7.1.26（t = 1/(1+p|x|)；erf = sign(x)·(1 − t·poly₅(t)·e^(−x²))）。
- 等价性论证：容差内等价。A&S 手册引述 |ε| ≤ 1.5e-7（近似论误差）；fp32 域 op-by-op 链舍入后实测最大绝对误差 5.868e-07（vs fp64 真值，6M 随机 + 23 角点）；最敏感区为 x→0 的 `1 − m` 相消——m→1 时绝对误差有界 ~1e-7（链舍入 ulp(1) 量级），输出绝对误差仍 ~5.9e-7 ≪ atol 1e-5（erf 在 0 附近平坦、包络 atol 主导，无 rtol 风险）；fp16/bf16 transit 单次舍回后 0 违反。舍入影响评估：fp32 链每 op 舍入 ≤0.5 ulp，Horner 8 步 + 尾 4 步累计 ~5e-7，已被 D-1 全覆盖。
- 收益量化：**可行性收益**（唯一满足精度的候选，vs C1 精度发散）；代价 = 22 op/elem vs 基线约定 1 op/elem，带宽受限（AI ≤ 0.25 flop/B）下隐藏于 MTE2 窗口（§1.6.0 R3），估算下界不变（§1.6.0 D-2 行）。

**A2（采纳）：标量常量字面量化（常量缓冲 9 → 3）**

- 原式：P、A1..A5、−1、+1、0 共 9 个常量各经 `T.vbrc` 物化为 (bs,4B) UB 常量缓冲（bs=4096 时 9×16KB = 144KB）。
- 优化后公式：vadd/vmul/vsub/vdiv 的**第二操作数标量形态**（文档依据：T.vadd.md §2.1「src1 tensor或scalar」+ §2.3.4「仅支持src1为标量」；T.vmul.md §2.1/§2.3.4 同构；T.vsub.md 同构；T.vdiv.md §2.3.4「仅支持src2为标量」）——P 乘入 vmul、+1 乘入 vadd、A2/A3/A4/A1 加入 Horner vadd、−1 乘入 vmul 全部走字面量；仅 vdiv 的 src1（分子 1）、vcmp 的 B（0）、vselect 的 A/B（±1）因「src0 不支持标量 / vcmp 文档仅张量对 / vselect 文档仅张量」保留 3 个常量缓冲（ones_ub、zeros_ub、neg_one_ub）；A5 初始化改 `vmul(ones_ub, A5字面量, h)` 免第 4 个 vbrc。
- 等价性论证：字面量与 vbrc 广播值是**同一 fp32 数**（Python float → fp32 舍入与经缓冲舍入同一路径）；D-1 A2 断言机器验证 `torch.tensor(v, fp32).expand(8) == torch.full((8,), v, fp32)` 全常量 EQUIV_PASS。标量字面量直传不经 vbrc let 绑定触发面（CONST-vsub-crossbuf-scalar 补注：标量字面量不经 let 绑定、无 rank 断言触发）。
- 收益量化：UB 常量缓冲 9→3（−56KB @bs2048 / −112KB @bs4096）；发射 op 数省 6 次一次性 vbrc；**使 fp32 路线 bs=4096 档在 192KB 内可行**（§4.5 算术）。〔注：vadd/vmul 标量字面量为文档支持、CONST-vsub-crossbuf-scalar 仅对 vsub 做过生产实证——首跑 L0 覆盖，回退 = vbrc 常量缓冲形态（UB 预算已含回退列，§4.5）。〕

**无更优化空间项（显式结论）**：

- 公共子表达式：x² 仅 exp 支路使用、|x| 仅 t 支路使用、poly 仅 m 使用——链中无重复计算的子式，CSE 无对象。
- 稳定化变形：无 softmax 型跨元素 max——A&S 链的数值稳定性已由 §1.6.0/D-1 角点覆盖（inf/nan/极值全过），无需减 max 类变换。
- 算子降代换：vexp→vexp2 同 op 数（−log2e 折入同一 vmul）无收益；vpow(denom,−1) 替代 vdiv —— 淘汰（T.vpow.md §2.1：指数 B 必须为 tensor，需 −1 缓冲 + 幂内部为 exp/log 复合链，未文档化假设：不快于 vdiv 且无精度收益）。
- 访存削减：中间量已全驻留 UB（唯一访存 = 输入载入 + 输出写回，即 manifest bytes 口径），无 GM 往返可削。
- 归约合并：零归约结构（逐点算子），不适用。

**D-1 等价性机器验证结果表**（`python3 examples/erf_fwd/erf_fwd_kernel/verify_equiv.py`，torch CPU，fp64 参照，2026-10-29 执行，exit 0）：

| 验证项 | dtype | 测试点数 | 包络违反数 | max abs err vs fp64 | max ulp vs golden（尾域） | 特殊位 | 结论 |
|--------|-------|---------|-----------|--------------------|--------------------------|--------|------|
| A1 A&S 链（随机 5 尺度 + 极值域 + 角点 23 值） | float32 | 6,000,023 | **0** | **5.868e-07** | 1.00（\|x\|≥3 套件） | ok | **EQUIV_PASS** |
| A1 同上 | float16 | 5,000,023 | **0** | 2.443e-04 | 0.00（尾域） | ok | **EQUIV_PASS** |
| A1 同上 | bfloat16 | 5,000,023 | **0** | 1.951e-03 | 0.00（尾域） | ok | **EQUIV_PASS** |
| A2 标量字面量 fp32 恒等（8 常量） | — | 8 | 0 | 0（按构造相等） | — | — | **EQUIV_PASS** |

逐套件明细（fp32）：randn 5.835e-07 / near-zero 5.868e-07 / mid(×3) 5.850e-07 / uniform±6 5.621e-07 / tail±[3,6] 4.442e-08 / extreme(1e0..1e38) 1.778e-07 / corners 9.715e-08——全部 0 违反。ulp 注记：near-zero 套件 ulp 值（4.2e6）为度量伪影（golden ~1e-9 量级时 |err|~5e-7 除以亚正规 ulp 所致），该域由 atol 包络治理（违反数 0），有效 ulp 判读域为尾域（|x|≥3）≤1 ulp 与 fp16 全域 ≤3 ulp。

**REJ-1（T.verf 泰勒禁用机器证据，非采纳项）**：7 阶泰勒 fp64 误差表——x=0.5: 9.8e-06；x=1.0: 4.48e-03；x=1.5: 1.451e-01；x=2.0: 1.576；x=3.0: 39.11——与用户 910B3 实测（|x|=0.5 约 1e-5、|x|=1 约 4.5e-3、1.5 约 0.15、2 约 1.6、3 约 39）**逐点吻合**，独立复核成立；「T.verf 不可用于全范围 erf」由 docs lowering 结构 + 本表双证。

**GOLD-DIAG（golden 形态判定）**：CPU 上 `torch.erf(fp16 原生)` vs `torch.erf(x.float()).to(fp16)` 位等价 200000/200000；bf16 同样 200000/200000——golden 可用 `torch.erf(x.detach().cpu()).to(x.dtype)` 形态（expm1 同款），与 NPU 测试 ref_program 的显式 fp32-transit 形态数值一致。

**优化结论**：采纳 A1（A&S 7.1.26 容差内等价替代）与 A2（标量常量字面量化，常量缓冲 9→3）共 2 项，均经 verify_equiv.py 机器验证 EQUIV_PASS；显式无优化空间项 4 类（CSE / 稳定化 / 降代换 / 访存削减 / 归约合并均无对象或无收益，论证见上）；优化后公式（§1.3）= §3.1 公式拆解唯一输入。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案全部循环与标量计算点盘点（结论与 §6 循环结构一致，无未盘点点）：

| # | 计算/循环点 | 形态 | 向量化替代 | API 佐证 | 结论 |
|---|------------|------|-----------|---------|------|
| V1 | 逐元素算术全链（\|x\|、x²、P·\|x\|、+1、1/denom、Horner、×t、×e、1−m、×s、exp） | 逐元素 | v-prefix 整 tile 向量链（vabs/vmul/vadd/vsub/vdiv/vexp） | docs 数学操作各 md（dtype 矩阵 fp32 √，§3.4 表） | **全部向量化** ✓ |
| V2 | Horner 4 次迭代循环 | 常量次数循环 | 直接展开为 8 个 v-op（无 T.serial——常量迭代次数、无跨 tile 携带） | 展开后即 V1 | **向量化** ✓（编译期展开） |
| V3 | 逐元素符号判定/还原 | 逐元素条件 | vcmp(lt) + vselect(±1) + vmul | T.vcmp.md（bool 片上配合 vselect，不可写 GM 条款已核对）、T.vselect.md | **向量化** ✓ |
| V4 | 标量常量广播（p/A/±1/0） | 逐元素填充 | 标量字面量第二操作数（A2）+ 3 常量缓冲 vbrc | T.vadd/vmul/vsub/vdiv.md 标量形态；T.vbrc.md 标量 src | **向量化** ✓ |
| V5 | t = 1/denom 的除法 | 逐元素 | T.vdiv（tensor-tensor，ones_ub 作 src1） | T.vdiv.md §2.3.4「仅支持 src2 为标量，不支持 src1」→ 分子必须张量；vrec/vrc 倒数 API：docs/Tilelang.language/ **全 20 子目录 grep 检索零命中**（含创建操作/索引与元素操作/条件操作/排序操作/逻辑操作/原子操作） | 向量化 ✓（vdiv）；「专用倒数 API 缺失」为已检索佐证的负向断言 |
| V6 | tile/core 索引（block_id = i·num_kernels + cid、t0、tail） | block 级标量 | 不可替代 | 每 tile 每核 O(1) 标量（非逐元素热点）；任务映射计算属「block 级索引/任务映射计算」豁免类别 | **保留**（理由：block 级索引，3 个标量/tile，非热点） |
| V7 | host 侧 N/bs/num_kernels/num_local_tasks | host 标量 | 不适用 | host 元数据豁免类别 | **保留**（host 侧） |
| V8 | 尾块边界 tail = min(bs, N−t0) | tile 级标量 + 动态边界 | 不可静态化（N 运行期给定；padding 方案反而引入额外 GM 写） | 「依赖动态 shape 的边界处理」豁免类别；且 v-op 切片仅静态边界（traps-compiler.md P10）→ 整 buffer v-op + [0:tail] store 是唯一正确形态 | **保留**（理由：动态边界，expm1 已验证模式） |
| V9 | if block_id < num_logical guard | tile 级标量分支 | 不可替代（persistent 尾核掩码） | num_kernels 整除不尽时必须；guard 条件为编译期可折叠纯名比较（num_logical/num_kernels 均 Python int 静态） | **保留**（理由：任务掩码语义，静态边界折叠） |

无「实现简单/照源码写」类理由；2D T.Parallel 标量化陷阱（TRAP-parallel2d-bm-ge2-scalarize）不适用——本算子为 1-D 整 tile v-op 链，无 2D T.Parallel 融合 if_then_else 结构。

**向量化结论**：向量化替代 5/5 处可替代计算点（V1 算术链 / V2 Horner 展开 / V3 vcmp+vselect / V4 标量广播 / V5 vdiv），保留 3 处（V6 block 级索引、V7 host 元数据、V8 动态尾边界 + V9 掩码 guard——理由类别分别为 block 级任务映射 / host 侧元数据 / 动态 shape 边界处理，均属可接受理由清单）；结论与 §6 一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**豁免声明**：本算子为 1-D 扁平逐元素映射（I/O 契约即 `(N,)` 连续），**仅一个候选轴**（尾轴连续）——按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明仅一个候选轴豁免」处理。核对项：

- 向量轴 = 连续尾轴，fp16/bf16 无打包加速（CONST-vector-launch-overhead：f16≈f32 逐元素速率）→ 不存在「换轴提速」空间；
- 广播轴对齐不适用（常量全 tile 统一广播，无按轴广播反转问题）；
- 核内布局 = 1-D (bs,) UB buffer；lane 映射 = v-op 隐式连续；GPU 源码的 threads×npt 轴选择（§0.3）是输入而非结论——NPU 侧整 tile v-op 天然覆盖，无 SIMT 轴需保留；
- 布局候选枚举豁免后无弃选项，无实验裁决需求（§1.6.0 R4 同结论）。

**布局决策结论**：主选布局 = 1-D (bs,) 连续 UB buffer + 连续尾轴向量映射（单轴豁免）；与 §3.3 伪代码（整 tile v-op 无切片）、§4.3（(bs,) buffer 形状）、§6（无 2D 循环）三方一致；无备选布局、无实验裁决计划（豁免成立：仅一个候选轴）。

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（wrapper 契约锁定：erf_fwd.py L168 `ascend_mode = "Developer"`（CG-2026-0010）；任务参数 TILELANG_ASCEND_MODE=Developer）。

### 2.2 选型理由

1. 逐元素 v-prefix 链为 Developer 模式标准形态（官方 vec_add_1d.py 模式族）；无手动流水/双缓冲/跨核协同需求（单引擎、无规约）。
2. 同族标杆 expm1 landed kernel 即 Developer 模式落地（`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` + 模块级 ASCEND_MODE 常量），wrapper 侧 ascend_mode="Developer" 与之配套。
3. 用户显式指定 Developer（op_requirements 编程模式字段）。

### 2.3 模式影响

- 同步：自动同步（RAW/WAR 由编译器插入 pipe 屏障），无手动 set_flag/wait_flag（§7）。
- 多缓冲：auto-multi-buffer 开启时 BishengIR 占用膨胀 ~1.7x（CONST-capacity-910B2C〔stale〕；显式逐 buffer alloc 结构下 r9 数据点实测 ≈ 手工核算 ×0.981）——UB 预算按显式 alloc 口径核算并留 CG-2026-0008 +12% 裕度（§4.5）。
- kernel 产品需 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` 模块级自锁（Expert/Developer 混进程隔离，expm1 同款）。

## 3. API 映射设计

### 3.1 公式拆解

A&S 7.1.26 链（§1.6.1 优化后公式 = 本节唯一输入）拆解为 fp32 域 11 类子计算：cond=sign 判定、|x|、x²、P·|x|、+1、取倒、−x²、exp、Horner×4、×t、×e、1−m、±1 选择、符号施加。

### 3.2 TileLang API 映射

| 计算逻辑 | TileLang API（v-prefix） | 关键参数/形态 | 文档佐证（已亲自打开核对） |
|---------|--------------------------|--------------|---------------------------|
| 载入/写回 | T.copy | runtime 索引切片 `[t0:t0+tail]` / 零原点 `[0:tail]` | traps-compiler.md 第六轮（T.copy 接受 runtime 切片）+ 官方 vec_add_1d 模式 |
| 升域/舍回 | T.vcast | `round_mode="rint"`；f16→f32 rint / bf16→f32 rint / f32→f16 rint / f32→bf16 rint 全在 dtype 矩阵 | T.vcast.md §2.2.1 L29/30/35/36/42 |
| 符号判定 | T.vcmp | `(x, zeros_ub, cond, "lt")` tensor-tensor；bool 输出仅片上配合 vselect | T.vcmp.md §2.2/§2.3（bool 不可写 GM） |
| 绝对值 | T.vabs | `(x, ax)`；fp32 √ bf16 ×（transit 规避） | T.vabs.md §2.2.1 |
| 乘/加/减 | T.vmul / T.vadd / T.vsub | 第二操作数标量字面量（A2）；原地 dst≡src（§2.3.3 原地条款） | T.vmul/vadd/vsub.md §2.1/§2.3 |
| 取倒 | T.vdiv | `(ones_ub, denom, t)` tensor-tensor（src1 标量不支持） | T.vdiv.md §2.3.4 |
| 指数 | T.vexp | `(nx2, e)` fp32 √ bf16 × | T.vexp.md §2.2.1 |
| 常量物化 | T.vbrc | 标量 src（1.0/−1.0/0.0），let 绑定于 traced body 内 | T.vbrc.md §2.1 标量形态 + §2.4 示例 1；TRAP-expert-v-operands let 绑定位置 |
| 符号选择 | T.vselect | `(cond, neg_one_ub, ones_ub, s)` 4 参 | T.vselect.md §2.1 |

已枚举并淘汰的候选 API：T.verf（精度，REJ-1）、T.vtanh（精度）、T.vpow（指数须张量 + 复合链，§1.6.1）、vrc/vrec（不存在，V5 已检索佐证）、T.vexp2（同 op 数无收益）。

### 3.3 计算伪代码

```python
# 基于 TileLang API 的计算流程伪代码（fp32 路线；transit 路线差异以 [T] 标注）
# 工厂（§3.5.2 契约）：erf_fwd_kernel(N, dtype, output_dtype=None, block_size=None,
#                                     threads=256, num_per_thread=8)  # lru_cache
#   dtype 归一化: dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]
#   -> 返回 curried builder kernel(bs=default_block) -> main(x, y) (out_idx=[1])
@tilelang.jit(out_idx=[1], target="npuir")
def kernel(bs=default_block):          # bs 默认: fp32 1024 / fp16·bf16 2048 (K11)
    num_logical  = ceildiv(N, bs)                    # 编译期 IntImm
    num_kernels  = min(num_logical, vector_cores)    # ≤ 40（§5.5 实查×2）
    num_local_tasks = ceildiv(num_logical, num_kernels)   # 静态边界
    @T.prim_func
    def main(x: T.Tensor((N,), dk), y: T.Tensor((N,), dk)):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            # UB 分配（§4.3）；常量缓冲一次性物化（vbrc 标量 let 绑定在 body 内）
            x_ub, ax_ub, x2_ub, t_ub, h_ub, e_ub = alloc_ub((bs,), f32) ×6   # [T] + x16_ub(dtype), x32_ub(f32)
            cond_ub = alloc_ub((bs,), "bool")
            ones_ub, neg_one_ub, zeros_ub = alloc_ub((bs,), f32) ×3
            T.vbrc(1.0, ones_ub); T.vbrc(-1.0, neg_one_ub); T.vbrc(0.0, zeros_ub)
            for i in T.serial(num_local_tasks):            # persistent 核内串行（静态）
                block_id = i * num_kernels + cid
                if block_id < num_logical:                 # 编译期可折叠 guard
                    t0 = block_id * bs
                    tail = T.min(bs, N - t0)
                    # 1) 载入 GM→UB
                    T.copy(x[t0:t0+tail], x_ub[0:tail])    # [T] 载入到 x16_ub
                    # [T] 2) vcast(x16_ub, x32_ub, rint) 升域 fp32（无损）
                    # 3) fp32 域 22-op 链（v-op 全整 buffer，stale lane 不落 GM）
                    T.vcmp(x_ub, zeros_ub, cond_ub, "lt")          # cond = x<0
                    T.vabs(x_ub, ax_ub)                            # a = |x|
                    T.vmul(x_ub, x_ub, x2_ub)                      # x²
                    T.vmul(ax_ub, 0.3275911, ax_ub)                # p·a（原地）
                    T.vadd(ax_ub, 1.0, ax_ub)                      # denom（原地）
                    T.vdiv(ones_ub, ax_ub, t_ub)                   # t = 1/denom
                    T.vmul(x2_ub, -1.0, x2_ub)                     # −x²（原地）
                    T.vexp(x2_ub, e_ub)                            # e = exp(−x²)
                    T.vmul(ones_ub, 1.061405429, h_ub)             # h = A5
                    T.vmul(h_ub, t_ub, h_ub); T.vadd(h_ub, -1.453152027, h_ub)  # A4
                    T.vmul(h_ub, t_ub, h_ub); T.vadd(h_ub,  1.421413741, h_ub)  # A3
                    T.vmul(h_ub, t_ub, h_ub); T.vadd(h_ub, -0.284496736, h_ub)  # A2
                    T.vmul(h_ub, t_ub, h_ub); T.vadd(h_ub,  0.254829592, h_ub)  # A1
                    T.vmul(t_ub, h_ub, h_ub)                       # ×t（前导 t 因子）
                    T.vmul(h_ub, e_ub, x2_ub)                      # m = t·poly·e → x2_ub（复用）
                    T.vsub(ones_ub, x2_ub, e_ub)                   # r = 1 − m → e_ub（复用）
                    T.vselect(cond_ub, neg_one_ub, ones_ub, h_ub)  # s = ±1 → h_ub（复用）
                    T.vmul(e_ub, h_ub, x_ub)                       # out = r·s → x_ub（复用；回退 out_ub §9）
                    # [T] 4) vcast(x32_ub, x16_ub, rint) 单次舍回（复用载入 buffer）
                    # 5) 写回 UB→GM（仅 [0:tail]）
                    T.copy(x_ub[0:tail], y[t0:t0+tail])            # [T] 拷出源 = x16_ub
```

注：Horner 系数字面量按 Horner 求值序书写（A4→A1）；实现在 v-op 全整 buffer 上执行（伪代码以 x_ub 简写域输入，transit 路线域输入 = x32_ub）；「结果落位 buffer = GM 拷出源」统一形态（CONST-vsub-crossbuf-scalar 意图，vmul 变体首跑重验，回退 dedicated out_ub）。

### 3.4 API 可行性确认

| API | dtype 矩阵关键项 | 形态限制 | 风险等级 |
|-----|-----------------|---------|---------|
| T.vcast | f16→f32 rint √ / bf16→f32 rint √ / f32→f16·bf16 rint √ | — | 低（expm1 生产已用） |
| T.vabs | fp32 √ bf16 × | shape 一致 | 低（fp32 域使用） |
| T.vmul/vadd/vsub 标量 | fp32 √（第二操作数） | 标量须 T.Kernel 内 T.Scope 外定义（各 md §2.3.2） | 中（文档√/生产实证仅 vsub——首跑 L0 覆盖 + 回退 vbrc） |
| T.vdiv | fp32 √ bf16 × | src1 不可标量 → ones_ub 形态 | 低 |
| T.vexp | fp32 √ bf16 × | 1-5D | 低（expm1 生产已用；≤1 ulp 域差风险 §9） |
| T.vcmp | fp32 √ | A/B 同 shape、bool 仅片上 | 低 |
| T.vselect | fp32 √ bf16 √ | 4 参同 shape | 低 |
| T.vbrc | 标量 src fp32 √ | 标量 let 绑定须在 traced body 内（工厂作用域命名局部量触发 rank 断言——任务事实 5 + TRAP 实证） | 低（绑定位置已按规范设计） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

- 三维 Kernel：不涉及（一维 T.Kernel）✓；GPU 专用 API（T.erf CUDA 侧）：已重设计为 v-prefix 链 ✓；GEMM 非整除：不涉及（非 GEMM）✓；L0C：不涉及（纯 Vector）✓。
- bf16 无 v-prefix 算术（多 md dtype 矩阵）→ transit 路线规避 ✓；v-op 切片静态边界限制 → 整 buffer v-op 设计 ✓；bool 不可写 GM → cond 仅片上 ✓；vbrc 标量 let 绑定位置 → body 内绑定 ✓。
- 强制检测清单（ascend-constraints.md）逐项通过，无遗留。

#### 3.5.2 参考实现差异说明（工厂契约 / wrapper 契约）

- **工厂契约（两段式，E2/K11）**：`erf_fwd_kernel(N, dtype, output_dtype=None, block_size=None, threads=256, num_per_thread=8)` → 返回值恒为 curried builder（`kernel(bs=default_block)`）；`threads/num_per_thread` 保留于签名（源形态保真 + lru_cache 键空间）但 NPU 侧不消费；`output_dtype` 校验 same_as(input)；N≤0 / 非 float dtype → ValueError（host 侧防御，survive python -O 用 ValueError 不用 assert）。
- **wrapper 现状逐字核对结论**：erf_fwd.py L117 `return erf_fwd_kernel(N, dtype)(block_size)(x)` 已是两段式形态（**与 GPU 抽取件的 `kernel(threads_arg, npt_arg)` 不同——落地侧现状为唯一权威，勿照搬 GPU 两参式**，leaky_relu 教训条目 1）；L116 传 torch.dtype 对象 → 工厂归一化（教训条目 4）。**Stage 5 集成零 wrapper 编辑**：仅将 L73 `from ._erf_fwd_kernels import erf_fwd_kernel` 切换为 conductor 产品（同工厂签名），expm1/expm1 集成模式。L57–60 Stage 5 注释与之一致。
- 与 GPU 抽取件（`_erf_fwd_kernels.py`）的差异 = 本设计全部 §0.6 重设计项（算法/并行/尾块），非语法翻译。

#### 3.5.3 本项目同类实现参考

- `examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py`（同族第 3 例已落地：工厂契约 / persistent / transit / UB guard / L0-L2 测试骨架——本设计结构模板）
- `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/`（首例 harness 迁移的 TileOPs 集成侧；其 Stage 1 工件含教训来源：examples/leaky_relu/_make_leaky_relu_kernel/ 的 RETROSPECTIVE.md）
- `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/`（标量字面量 vsub 先例 L78 同形）
- 官方形态：`examples/` 下 vec_add_1d.py 模式（1-D elementwise + persistent + 尾块零原点切片）

## 4. 数据规格与内存规划

### 4.1 输入张量

`x: (N_total,)`，dtype ∈ {fp16, bf16, fp32}，contiguous（Op 层保证 flatten+contiguous）；动态轴：N_total 运行期给定（工厂入参，编译期常量化——每 N 特化编译，lru_cache 键）。

### 4.2 输出张量

`y: (N_total,)`，dtype = same_as(input)（manifest 契约 + 工厂 output_dtype 校验）；`out_idx=[1]`，`main(x)` 返回 y。

### 4.3 中间缓冲区（每 tile，UB）

| 缓冲 | shape/dtype | 角色 | 复用关系 |
|------|-------------|------|---------|
| x_ub | (bs, fp32) | fp32 路线载入 buffer + 域输入 | 死亡后复用为最终 out（拷出源）|
| x16_ub | (bs, fp16/bf16) | transit 载入 + 舍回 dst + 拷出源 | 仅 transit 路线 |
| x32_ub | (bs, fp32) | transit 域输入 | 死亡后复用为 out32（拷出源）|
| ax_ub | (bs, fp32) | \|x\| → P·a → denom（原地链） | Horner 开始后死亡 |
| x2_ub | (bs, fp32) | x² → −x² → vexp 输入 | 死亡后复用为 m |
| t_ub | (bs, fp32) | 1/denom | Horner 后死亡（×t 消费）|
| h_ub | (bs, fp32) | Horner 累加器 → poly | 死亡后复用为 s |
| e_ub | (bs, fp32) | exp(−x²) | 死亡后复用为 r |
| cond_ub | (bs, bool) | vcmp 输出 | vselect 消费；仅片上 |
| ones_ub / neg_one_ub / zeros_ub | (bs, fp32) ×3 | 常量（vdiv src1 / vselect A·B / vcmp B） | 核级一次物化，跨 tile 复用 |

复用纪律：全部 v-op 走 3-buffer 形式（dst ∉ src）或原地 dst≡src（vadd/vmul/vsub/vdiv §2.3.3 文档条款；vsub 原地另有生产实证）；每条 dtype 路线单独数据流推演——fp32 路线拷出源 = x_ub，transit 路线拷出源 = x16_ub（教训条目 2：无回写步路线沿用统一拷贝源即恒等输出错误——本设计 fp32 路线最终 vmul dst=x_ub 即为该路线的「回写」，两路线拷出源不同，已在伪代码 [T] 标注）。

### 4.4 内存搬运路径

`GM(x) → (MTE2) → UB[x16_ub/x_ub] → vcast → UB fp32 域缓冲组 → v-prefix 链 → vcast → UB[x16_ub/x_ub] → (MTE3) → GM(y)`。纯 Vector 单引擎；无 L1/L0A/L0B/L0C 参与；无跨引擎往返（容量项豁免依据）。

### 4.5 UB 内存预算（per element 口径，bs 元素 tile）

| 路线 | 工作缓冲 | 常量 | bool | 合计 B/elem | bs=2048 | bs=4096 | +12% (CG-2026-0008) |
|------|---------|------|------|------------|---------|---------|---------------------|
| fp32 | 6×4B | 3×4B | 1B | **37** | 75.8KB | 151.6KB | 169.8KB ✓ |
| transit (fp16/bf16) | 5×4B + x16 2B + x32 4B | 3×4B | 1B | **39** | 79.9KB | 159.7KB | 178.9KB ✓ |
| 回退（out_ub 独立） | +4B | — | — | 41 / 43 | 84/88KB | 168/176KB | 188KB ✓ / 197KB ✗→回退时 transit cap 收紧 2048 |

- 上限 192KB/核（CONST-capacity-910B2C〔stale：UB 192KB/AIV + 膨胀条款〕；显式 alloc 结构 r9 实测 ≈ 手工 ×0.981——本设计按显式逐 buffer alloc，实际余量优于 +12% 悲观口径）。
- **UB guard**：`_UB_MAX_BLOCK = {float32: 4096, float16: 4096, bfloat16: 4096}`，`bs > cap` → ValueError（工厂 builder 内 host 校验，expm1 同款）；默认 bs = 1024（fp32）/ 2048（fp16/bf16）均低于 cap。
- 对照教训条目 5：UB 论证引用 pattern-library 平台值口径（192KB + r9/CG-2026-0008 两套系数），不自设 ×2 线性模型。

### 4.6 动态轴定义

仅 N_total（运行期）。bs / num_kernels / num_local_tasks / tail 均为编译期可折叠量（N、bs 为 Python int；cid 运行期但仅出现在 block_id 与 guard）。无运行期 shape 分支。

### 4.7 JIT 配置

`@tilelang.jit(out_idx=[1], target="npuir")`；无 pass_configs（Developer 自动同步 + auto-multi-buffer 默认）；lru_cache(maxsize=32) 于工厂（S4 承接）；每 (N, dtype, bs) 组合一次编译。

## 5. Tiling 策略

### 5.1 计算类型

纯 Vector 逐元素（无 Cube / 无规约 / 无跨核依赖）→ 仅需 UB；单步多 op 链（22-op）；无融合后处理。

### 5.2 Block 划分

- **Block 尺寸（bs）**：fp32 默认 1024、fp16/bf16 默认 2048（K11 wrapper 契约常量 `_BLOCK_SIZE_FP32/_BLOCK_SIZE_NON_FP32`，GPU threads(256)×npt(4/8) 折叠）；上限 = §4.5 UB guard 4096。
- **Tile Shape**：1-D `(bs,)`；非整除处理 = **tile 级 tail（Kernel 内动态 block 方案）**：`tail = T.min(bs, N − t0)`，v-op 整 buffer、store `[0:tail]`（§0.6 R4）——无 padding、无 host 侧 crop。

### 5.3 约束分析

- 32B 对齐：T.copy 切片边界由 MTE 处理（vec_add_1d 官方模式同款）；尾 tile 最小 1 元素可用（expm1 L0-4 N=1 用例同构）。
- bf16 transit 强制；v-op 无切片（静态边界限制）→ 整 buffer 设计（§1.6.2 V8）。
- auto-multi-buffer 膨胀（悲观 +12%）已在 §4.5 预算内。

### 5.4 注意事项

- bs 变更不改语义（纯分块参数）；lru_cache 按 (N, dtype, bs) 特化。
- 每逻辑 tile 计算量恒定 → 各核负载均衡仅受 num_local_tasks 整除余数影响（≤1 tile 差）。

### 5.5 分核策略（物理核数适配）⭐

**要素① 逻辑核数计算**：`num_logical_kernels = ceildiv(N, bs)`（1-D 输出网格）。manifest workloads：smoke-1m fp32 → 1024 / fp16·bf16 → 512；16M fp32 → 16384 / fp16·bf16 → 8192；64M fp16 → 32768；256M fp16 → 131072。

**要素② 物理核数依据（实查，禁假设值）**：

- 查询代码（落地 kernel 内，工厂期现查）：`vector_cores = NPUUtils.get().get_aicore_num() * 2`（纯 Vector 算子翻倍——core-split-strategy.md §1 要素②）。
- **本设计期实查记录（2026-10-29）**：设计环境 tilelang 不可导入（dev-root 私有 TVM 包缺失，与 CONST-aicore-910B2C 条目记载环境一致），按该条目记载的等价实查路径 ctypes 直调 CANN libruntime：`python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)"` → **ret=0, aicore_num=20**（与 910B3 设备 + npu-smi 25.5.0 报告的 910B3 一致）；纯 Vector 翻倍 → **40**。权威查询 NPUUtils.get().get_aicore_num() 由 Stage 3 kernel 工厂期在设备上执行（预期 20，异常即报环境错误，不猜测）。
- 版本戳：CONST-aicore-910B2C 910B3 数据点（tilelang 0.1.2+88ee8c302e / CANN 9.0.0 / libruntime 25.5.0 / 2026-10-28，三次独立复跑）+ 本任务 2026-10-29 ctypes 第四次复跑一致。〔该条目在 154 条 stale 清单内——工具链漂移风险标注；本条已在**当前环境**实测重验通过，重验承诺已履行。〕

**要素③ 规模判定与分核方案**：全部 manifest workloads 满足 `num_logical ≫ 40`（最小 smoke-1m 亦 512–1024 ≫ 40）→ **极大规模**：固定启动内核数 = `num_kernels = min(num_logical, 40)` ≤ 物理向量核数；核内 `T.serial(num_local_tasks)` 串行处理多逻辑块（`num_local_tasks = ceildiv(num_logical, num_kernels)`——**静态值**：N、bs、num_kernels 均 Python int，trace 时折叠 IntImm）；grid-stride 映射 `block_id = i·num_kernels + cid` + `if block_id < num_logical` 掩码；无可调 block 使核数对齐物理核整数倍的意义（逻辑核数远超核数，负载不均仅尾核 −1 tile）。方案 = core-split-strategy.md §1 要素③「极大规模」标准形态（persistent 核内串行），与 expm1 §0.6 R1 同构。

## 6. 循环与调度结构

### 6.1 循环结构总结

| 层 | 结构 | 选择依据 |
|----|------|---------|
| 核间 | `T.Kernel(num_kernels, is_npu=True)`（一维，persistent） | §5.5 要素③ |
| 核内任务 | `T.serial(num_local_tasks)`（静态边界 grid-stride） | 极大规模摊薄核启动开销；边界静态（S1-CORES 要求） |
| tile 内元素 | **无显式循环**——22 个 v-op 各处理整 (bs,) 向量 | §1.6.2 V1–V5 全向量化结论；T.Parallel 不使用（Developer v-op 链形态，vec_add_1d 同款） |
| Horner | 编译期展开 4×(vmul+vadd) | §1.6.2 V2 |

T.Pipelined / T.Persistent 不显式使用：Developer 模式 auto-multi-buffer 覆盖 tile 间流水（copy-floor 证据：向量链隐藏于 MTE2 窗口）；T.Kernel 静态 num_kernels 本身即 persistent 语义。

### 6.2 循环伪代码

见 §3.3（三方一致：§1.6.3 布局 = 1-D 连续尾轴 / §4.3 缓冲形状 = (bs,) / 本节结构 = serial×guard + 整 tile v-op 链）。

### 6.3 流水线优化

- tile 间：auto-multi-buffer（Developer 默认）使下一 tile 载入与当前 tile 计算重叠；CONST-copy-floor-method〔stale，重验承诺〕实证 7-pass 链 ≥16M 全隐藏——22-op 链暴露量列 Stage 4 copy-floor 探针复测项（§9 R6）。
- tile 内：链式 RAW 依赖由 Developer 自动同步串行；无手动 double-buffer 需求。

### 6.4 尾块处理

`tail = T.min(bs, N − t0)`；v-op 整 buffer（stale lane 垃圾经 vexp 可产生 inf/nan——**不落 GM**，store 仅 [0:tail]）；expm1 生产验证同构。N=1 退化 tile（tail=1）由 T.copy 零原点切片覆盖。

## 7. 同步策略

### 7.1 同步模式

**Developer 自动同步**：RAW/WAR 依赖由编译器插入 pipe 屏障，无手动 set_flag/wait_flag/pipe_barrier；无跨核同步（每 tile 独立、无核间归并）。

### 7.2 同步点说明

隐式同步点 = 每 tile 内 v-op 链的 RAW 依赖序（§3.3 顺序即依赖序：vcmp/vabs/vmul-x² ← 载入；vdiv ← vmul-P/vadd-1；vexp ← vmul-−1；Horner ← vdiv + h 初始化；m ← Horner+vexp；r ← m；s ← cond；out ← r·s；store ← out + 载入 buffer 复用 WAR）。与编程模式匹配（Developer 自动同步——门禁「同步」项依据）。跨引擎（Cube/Vector）同步：不适用。

### 7.3 pass_configs 配置

无显式 pass_configs（默认 Developer 行为）；模块级 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` 自锁（expm1 同款）。若 Stage 4 需关闭 auto-multi-buffer 做对照，经 perf_opt 阶段 env 注入，不改产品代码默认。

## 8. 验证方案

### 8.1 Golden 函数

```python
def golden_erf(x: torch.Tensor) -> torch.Tensor:
    """erf 参考：torch.erf（源仓测试锚 ref_fn = torch.erf，GPU test L402 / NPU test L221）。

    golden IS the fp32-opmath + single round-back path（GOLD-DIAG 机器验证：
    CPU 上 torch.erf(原生 fp16/bf16) 与 torch.erf(x.float()).to(dtype) 位等价
    200000/200000 —— verify_equiv.py）。独立于 NPU 算法（不含 vcast transit /
    A&S 链 / persistent 结构知识）。CPU 运行，不需 NPU 设备。
    """
    return torch.erf(x.detach().cpu()).to(x.dtype)
```

独立性：以 §0.1 语义为依据（源仓 ref_fn 移植），未复刻 NPU 的 A&S 公式 / transit 链 / 分核结构——验证不 Self-证。优先移植源仓参考实现 ✓（torch.erf 即源测试锚）。

### 8.2 精度标准

- 容差（源仓 `test_unary_math.py::_get_tolerances` L69–74，GPU/NPU 双侧一致，逐字核对）：**fp16 (atol 1e-3, rtol 1e-3)；bf16 (1.6e-2, 1.6e-2)；fp32 (1e-5, 1e-5)**。
- 断言形态：`torch.testing.assert_close(out, ref, rtol, atol, equal_nan=True)` + NaN/Inf 位断言（`_assert_special_bits`：±inf 符号必须一致、NaN 位置 both-NaN 即可——payload 可变）；±0 零号差豁免（erf(−0): golden −0.0 vs NPU +0.0，IEEE 值相等，expm1 L0-5 先例）。
- 精度裕度链：A&S 5.868e-07（fp32 链）≪ 1e-5（17×）；NPU vexp 与 torch.exp ≤1 ulp 域差（expm1 先例口径）经 m=poly·e、r=1−m 传播后仍 ≪ atol；fp16/bf16 单次舍回 ≤1.5 ulp ≪ 包络（D-1 机器验证背书）。

### 8.3 L0 门槛测试计划（gate；L1/L2/Boundary 全套交 tilelang-op-develop）

1. **L0-1 smoke**：3 dtype × N=2^20 randn，默认 bs。
2. **L0-2 大规模**：fp16/bf16 N=2^24（多轮核内串行 + 常量缓冲跨 tile 复用覆盖）。
3. **L0-3 非整除尾**：fp16/fp32 × N ∈ {1000, 4097, 5000}（partial tile / tail=1 类）。
4. **L0-4 tiny**：N ∈ {1, 7, 32}（单非满 tile / num_logical=1 退化）。
5. **L0-5 角点与特殊值**（torch.finfo 取极值，禁手写字面量——教训条目 7）：每 dtype 角点集 = ±inf / NaN / ±0 / ±subnormal_min(2^-24, 2^-133, 2^-149) / ±finfo.tiny / ±1 / ±2 / ±3 / ±4 / ±6 / ±100（源 edge 用例值）/ ±finfo.max；断言 = 包络(equal_nan) + 特殊位；erf(±inf)=±1 与 erf(±max)=±1 精确值断言。
6. **L0-6 persistent 边界**：num_logical ∈ {vc−1, vc, vc+1}（fp16 bs 2048；vc 运行期实查）+ bf16 vc+1（transit 链过 guard）。
7. **L0-extra**：E2 双 bs 复用（1024/2048 同工厂）；两段式集成形态 `erf_fwd_kernel(N, "float16", block_size=2048)(2048)(x)`；单段 `(x)` 直调须 build 期失败；UB guard 拒绝 bs=4097；N≤0 / output_dtype≠dtype / 非 float dtype 拒绝；**标量字面量 v-op 形态首跑验证**（fp32 路线即 A2 形态，PASS 即同时履行 CONST/trap stale 条目重验承诺）。

## 9. 风险点与注意事项

| # | 风险 | 等级 | 缓解 / 验证路径 |
|---|------|------|----------------|
| R1 | vadd/vmul 标量字面量第二操作数 + vmul 跨缓冲 dst=载入 buffer：文档支持（§2.3/§2.1）但生产实证仅 vsub 形态（CONST-vsub-crossbuf-scalar） | 中 | L0-1 首跑即覆盖（fp32 路线全程该形态）；回退 = vbrc 常量缓冲 + dedicated out_ub（UB 预算回退列 §4.5 已含）；恒等指纹检查（输出≡输入即 dst 失效）纳入 L0 |
| R2 | NPU vexp 与 torch.exp 域差 ≤1 ulp（expm1 先例）→ 经 m/r 链传播 | 低 | 传播后绝对误差仍 ≪ atol（§8.2 分析）；L0 包络断言把关 |
| R3 | bf16 全链无 v-prefix 算术（多 md 矩阵）→ 必须 transit | 已规避 | transit 路线设计（§0.6 R2）；bf16 L0 用例覆盖 |
| R4 | 154 条 pattern-library 条目 stale（工具链漂移）——本设计引用 CONST-capacity-910B2C / CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-copy-floor-method / CONST-vsub-crossbuf-scalar / TRAP-expert-v-operands / CONST-aicore-910B2C 均 stale | 中 | 逐条「引用条目前置重验」标注 + 验证路径：CONST-aicore **已重验**（本任务 ctypes 实测）；CONST-vsub/TRAP → L0 首跑（编译期/数值期即暴露）；CONST-mte2/launch/copy-floor/capacity → Stage 4 copy-floor 探针 + msprof 对账（hardware-cost-model §3 偏差行）；不引用 status=overturned 条目（已核对 INDEX） |
| R5 | T.verf 泰勒不可用的负向断言若被推翻（工具链升级引入硬件 erf） | 低 | REJ-1 表 + T.verf.md §3 lowering 证据绑定工具链 stamp；toolchain 升级后 kb_stale_check 触发重验 |
| R6 | 22-op 链发射开销在中等 N（1M–4M）暴露量高于估算（链长 = expm1 的 5.5×） | 中 | §1.6.0 估算下界已计入（smoke-1m 11µs 发射主导）；Stage 4 copy-floor 探针量化暴露量，必要时 perf_opt 阶段评估 Horner Estrin 重排（同 op 数降依赖深度） |
| R7 | C4「无更省 op 同精度公式」为未文档化假设 | 低 | 收益上限 2 op（≈1µs）带宽主导不可辨；列待确认不阻塞 |
| R8 | golden 双形态（原生 dtype vs 显式 fp32 transit）在 NPU 设备侧 torch 实现差异 | 低 | GOLD-DIAG CPU 位等价已证；NPU 侧 golden 在 host CPU 执行（expm1 同款 detach().cpu()），不触设备实现 |
| R9 | N=0 / 负 N / dtype 不合法 | 已规避 | 工厂 host 防御 ValueError（§3.5.2；Op 层 numel 校验之上双保险） |

## 10. 交付清单

### 10.1 目录结构

```
examples/erf_fwd/erf_fwd_kernel/          # 本 Stage 1 工件（conductor 编排目录）
├── DESIGN.md                             # 本文档
├── verify_equiv.py                       # D-1 等价性机器验证（EQUIV_PASS, exit 0）
└── RETROSPECTIVE.md                      # Stage 1 复盘（Phase 8）

examples/TileOPs/tileops/kernels/elementwise/erf_fwd/
├── erf_fwd.py                            # wrapper（Stage 5 仅切 L73 import，零编辑）
├── _erf_fwd_kernels.py                   # GPU 抽取参考（保留）
└── erf_fwd_kernel/                       # Stage 3 产品目录（同族 expm1_kernel 布局）
    ├── erf_fwd_kernel.py                 # kernel + golden + L0-L2/Boundary 嵌入式测试
    └── history_version/
```

### 10.2 文件清单

| 文件 | 阶段 | 说明 |
|------|------|------|
| DESIGN.md | Stage 1（本阶段） | 设计冻结基线 |
| verify_equiv.py | Stage 1（本阶段） | 已执行，EQUIV_PASS；Stage 2 重跑复核 |
| erf_fwd_kernel.py | Stage 3 | 按 §3.3 伪代码 + §3.5.2 契约实现 |
| REVIEW.md | Stage 2 | 检视 |
| perf_opt/ | Stage 4 | copy-floor 探针 + msprof（R6） |

### 10.3 命名规范

工厂 `erf_fwd_kernel`（与 wrapper import 名一致）；golden `golden_erf`；产品文件 `erf_fwd_kernel.py`；常量 `_BLOCK_SIZE_FP32/_BLOCK_SIZE_NON_FP32/_UB_MAX_BLOCK/_TOLERANCE/_DEFAULT_BLOCK`（expm1 命名族）。

### 10.4 实现顺序

1. 工厂骨架 + dtype 归一化 + host 防御（§3.5.2）→ 2. fp32 路线链（含 A2 标量字面量形态首验）→ 3. transit 路线 → 4. persistent 分核 + 尾块 → 5. golden + L0 套件（§8.3）→ 6. L1/L2/Boundary → 7. Stage 5 import 切换（erf_fwd.py L73）。
