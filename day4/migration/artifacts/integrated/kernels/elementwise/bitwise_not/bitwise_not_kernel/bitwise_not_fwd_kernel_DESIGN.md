# bitwise_not_fwd_kernel 算子设计文档（迁移任务：TileOPs GPU → TileLang-NPUIR）

> project=bitwise_not, op=bitwise_not_fwd_kernel；场景 migration-harness（TileOPs 同构仓迁移）；编程模式 Developer。
> 源算子：TileOPs `BitwiseNotFwdKernel(UnaryKernel)`（DEFAULT_STRATEGY="direct"；op_func 按 dtype 分支：bool→`x == 0`、uint8→`x ^ 255`、其余整型→`x ^ (-1)`）。
> 本文档章节序号遵循 tilelang-op-design design-template：§0 迁移分析 → §1 概述（§1.6 算法调研与优化分析）→ §2–§10。
> 工具链环境：Ascend 910B3 / tilelang 0.1.2 dev-root（0.1.1.10+ubuntu.20.4.cann900 集成侧同族先例）/ CANN 9.0.0 / torch 2.7.1+torch_npu / 2026-10-29。

## 0. 源算子解读与迁移分析

> 执行序：Phase M0（本节 0.1–0.4，源码逐层实读）→ Phase R（§1.6.0，调研四问）→ Phase M1（0.5–0.6，判定输入含 §1.6.0 调研结论）。源码阅读清单（全部亲自 Read 核对）：GPU kernel `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L140–147（`_BITWISE_DTYPES`）、L305–320（`_make_unary_direct` direct 策略 builder）、L680–879（`UnaryKernel` 基类）、L2273–2289（`BitwiseNotFwdKernel`）；GPU op `/mnt/workspace/learn/refs/TileOPs/tileops/ops/elementwise/bitwise.py` L79–83（`BitwiseNotFwdOp(UnaryOp)`）、`/mnt/workspace/learn/refs/TileOPs/tileops/ops/elementwise/_base.py` L578–681（`UnaryOp` host 逻辑）；落地侧 wrapper `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not.py`（全文 250 行逐字核对）；NPU 侧抽取参考 `examples/TileOPs/tileops/kernels/elementwise/bitwise_not/_bitwise_not_fwd_kernels.py`（全文 67 行）；NPU 侧 test/bench `examples/TileOPs/tests/ops/test_bitwise_not.py`、`examples/TileOPs/benchmarks/ops/bench_bitwise_not.py`；workload `examples/TileOPs/tileops/workloads/elementwise.py` L43–67（`BitwiseNotWorkload.gen_inputs`）；manifest `examples/TileOPs/tileops/manifest/elementwise.yaml` L266–305（BitwiseNotFwdOp 条目）；同族已落地标杆 `examples/TileOPs/tileops/kernels/elementwise/{expm1,erf_fwd,leaky_relu,elu_fwd}/`（结构模板 erf_fwd_kernel_DESIGN.md 全文）。

### 0.1 源算子语义（做什么）

- **数学语义**：逐元素按位取反 `y[i] = ~x[i]`，单输入单输出、无规约、无跨元素依赖（纯逐点映射）。**按位整数/布尔运算，全程精确、无浮点舍入**——任意输入位置的输出由输入位的逐位补唯一确定。
- **dtype 语义（按源 `op_func` 三分支，elementwise.py L2283–2289）**：
  | 输入 dtype | 源公式 | 数学语义 |
  |-----------|--------|---------|
  | `torch.bool` | `x == T.cast(0, "bool")` | 逻辑非（True↔False） |
  | `torch.uint8` | `T.bitwise_xor(x, T.cast(255, "uint8"))` | 逐位取反（xor 全 1 字节 0xFF） |
  | int8/int16/int32/int64 | `T.bitwise_xor(x, T.cast(-1, x.dtype))` | 逐位取反（xor 全 1；补码 −1 位型 = 全 1） |
  三分支在各自 dtype 域内均为**同一位运算**（按位取反）的等价实现；无中间精度提升/截断（对比 erf 的 fp32 opmath——本算子为精确整数域，无 transit 需求）。
- **边界语义**：整数域无 NaN/Inf；角点 = 各位宽极值（int8 −128/127、uint8 0/255、int64 ±2⁶³ 等）与交替位型（0x55/0xAA），均由逐位取反精确覆盖。bool 域遵守 **torch.bool 存储契约 {0,1}**（`workloads/elementwise.py` L61–62 gen_inputs 以 `rand > 0.5` 生成，仅含 0/1）；越出 {0,1} 的存储值在 torch 契约外（行为未定义），本设计以 {0,1} 域等价性为准（§1.6.1 A3 机器验证）。空维度：Op 层 flatten 后 1-D，N≥1（N=0 防御见 §0.6 R4）。非整除 shape：`ceildiv(N, threads)` 尾块由 GPU copy 边界自然覆盖，NPU 侧尾块方案见 §0.6 R3。
- **语义完成标准达成**：golden 函数 = `torch.bitwise_not(x)`（源测试锚 `ref_program`，NPU test L45–46），逐位精确比较（`exact_compare`，test_base.py L50–51，无容差）——已能写出与源算子数值一致的 golden。

### 0.2 源算子输入输出

| 项 | 内容 | 来源（亲自核对） |
|----|------|------|
| 输入 | `x: (N_total,)` 扁平连续 1-D，dtype ∈ {bool, uint8, int8, int16, int32, int64}（`_BITWISE_DTYPES`，elementwise.py L140–147）；Op 层 `flatten → kernel → reshape`（_base.py L657–661，原始多维 shape 由 Op 层还原） | _base.py L660 |
| 输出 | `(N_total,)`，dtype = same_as(input)（`self.output_dtype = dtype`） | _base.py / wrapper bitwise_not.py L220 |
| 输出 shape | `output.shape == input.shape`（manifest shape_rules） | elementwise.yaml L277–278 |
| workloads（manifest，bench 契约） | smoke-1m `[1048576]` × {bool, uint8, int32}；elementwise-16m `[4096,4096]` × {int32, int64}；elementwise-64m `[8192,8192]` × {int16, int32}；elementwise-256m `[16384,16384]` × {int32} | elementwise.yaml L280–289 |
| 调用面（落地侧现状，唯一权威） | `_bitwise_not_wrapped(N, dtype_str, block_size, x)` → `bitwise_not_fwd_kernel(N, dtype)(block_size)(x)`（**两段式 curried**：factory(N, dtype) → builder(block_size) → main(x) 返回 y，`out_idx=[1]`）——**非 GPU 抽取件的 `kernel(threads_arg)` 单参调用式** | bitwise_not.py L130–152 逐字核对 |
| K11 block_size 常量 | `_BLOCK_SIZE = 256`（全 dtype 统一；GPU direct 策略 threads=256、npt 不参与 direct 路径的折叠口径） | bitwise_not.py L96–122 |
| Kernel 类契约 | `BitwiseNotFwdKernel(N_total, dtype, config=None)`；`SUPPORTED_DTYPES = _BITWISE_DTYPES`；`ascend_mode = "Developer"`（CG-2026-0010：调用面锁定 Developer）；config 键 = `{"block_size"}`；K8 无 autotune；K5 `supported_archs = None` | bitwise_not.py L161–249 |
| dtype 入参形态 | wrapper 传 `getattr(torch, dtype_str)` 即 **torch.dtype 对象**——factory 内必须归一化（leaky_relu 教训：torch 对象与 str 集合 `in` 比较恒 False） | bitwise_not.py L146 |
| roofline 契约 | `flops = N`（每元素 1 次位运算）、`bytes = 2·N·elem_bytes`（读 + 写，elem_bytes: bool/uint8/int8=1, int16=2, int32=4, int64=8） | elementwise.yaml L291–296 |

### 0.3 实现算法解读（怎么算）

**源码结构化阅读协议逐层结论**（协议见 migration-analysis.md §3.1）：

1. **入口签名**：GPU factory `_make_unary_direct(N, dtype, op_func, output_dtype=None, threads=256)`（elementwise.py L305，`@functools.lru_cache(maxsize=32)`）；`BitwiseNotFwdKernel.DEFAULT_STRATEGY = "direct"`（L2280）→ `_build_kernel` 走 direct 分支（L790–794）。
2. **Kernel 结构**：`@tilelang.jit(out_idx=[1])` 内 `kernel(threads_arg)` → `main(x: (N,) dtype, y: (N,) out_dtype)`；`with T.Kernel(T.ceildiv(N, threads_arg), threads=threads_arg) as bx`——一维 grid，block 数 = ceildiv(N, 256)，每 block 256 线程。
3. **计算原语**：仅 3 句——`for i in T.Parallel(threads_arg): idx = bx * threads_arg + i; y[idx] = op_func(x[idx])`。op_func 内联后为单条逐元素位运算（x==0 / xor 255 / xor −1，按编译期 dtype 分支选择）。计算语句全部归入「载入 → 逐点取反 → 写回」三步，无臆造步骤。
4. **内存分配**：无 `alloc_shared`/`alloc_fragment`——direct 策略为 **Global → Register → Compute → Register → Global**（模块 docstring L8–9），标量经寄存器直通，无片上共享内存。
5. **循环与调度**：`T.Parallel(threads_arg)` 单层 SIMT 展开（256 线程，每线程 1 元素）；无 T.serial / T.Pipelined。
6. **host 侧逻辑**：Op 层 `input.contiguous().reshape(-1)` → kernel → `.reshape(orig_shape)`（_base.py L657–661）；`SUPPORTED_DTYPES` 校验（L711–715）；bool 强制 direct 策略告警（L742–751，对本 op 是 no-op——DEFAULT_STRATEGY 已是 direct，wrapper 记为 K12）；fp8 包络 helper（`_wrap_fp8_accumulation`，L785）对本 op 不生效（OUTPUT_DTYPE=None 且 dtype 非 fp8，包装器透传原 op_func）；`lru_cache` 工厂缓存；无预处理/后处理数值变换。
7. **编译配置**：无 pass_configs / 环境变量依赖（GPU 侧）。

**数据流图（源硬件视角）**：`GM(x) → load（每线程 1 元素）→ CUDA register → 逐点位运算 → register → store → GM(y)`。全部 buffer 已覆盖：x、y（GM）、线程级寄存器直通（无具名片上 buffer），无来源不明数据。

**host 侧逻辑清单**：dtype 校验（SUPPORTED_DTYPES ValueError，test_bitwise_not.py L56–65 锚定 float dtype 拒绝）、lru_cache、flatten/reshape、config 解析（block_size=threads=256）。无预处理/后处理数值变换。

### 0.4 优化手段解读（为什么快）

逐项识别（识别不出来 ≠ 不存在；本算子为单步逐元素最简形态，源码仅 4 类语句，清单完备）：

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---------|---------|------|----------------|------|
| S1 | direct 策略（1 元素/线程） | `T.Kernel(ceildiv(N,threads), threads=threads)` + `T.Parallel(threads)` | SIMT 占用率拉满 + 合并访存；最简 codegen | CUDA SIMT warp 调度 + 大寄存器堆 | 实现层，硬件强相关 → **重新设计**（§0.6 R1：persistent 分核 + 整 tile 向量链） |
| S2 | xor 全 1 替代 `T.bitwise_not` | op_func 三分支均以 xor/比较实现取反；类 docstring L2276–2278：**"Uses XOR with -1 (all-ones) because T.bitwise_not fails on vectorized int4 CUDA types"** | 规避 GPU 前端 `T.bitwise_not` 对向量化 int4 CUDA 类型的缺陷 | CUDA codegen 的 int4 向量化缺陷（GPU 编译器特定） | **等价替换**（§0.6 R2：NPU 原生 `T.vnot` 即逐位取反向量 op，docs T.vnot.md 实证支持整型全宽——GPU workaround 的「正确取反」意图由原生 op 直接承接） |
| S3 | per-dtype op_func 分支（bool==0 / uint8×255 / int×−1） | 运行期 dtype 分支选择公式 | 适配三域语义（bool 逻辑非 vs 整型位取反）与 dtype 字面量类型 | 无（纯算法/契约层） | **保留**（NPU 侧承接为「dtype 路线分派」，§0.6 R2；bool 分支语义等价替换为 xor-1，§1.6.1 A3） |
| S4 | lru_cache 工厂缓存 | `@functools.lru_cache(maxsize=32)` | 消 host 重复编译查找 | 无 | **保留**（NPU 落地 kernel 同款） |
| S5 | `@tilelang.jit(out_idx=[1])` 输出自动分配 | direct builder 装饰器 | host 免手写输出分配 | 无 | **保留**（uint8 分配缺陷以 int8 位视图规避，§0.6 R2；见 §9 R3） |
| S6 | 尾块 ceildiv 覆盖 | `T.ceildiv(N, threads_arg)` | 非整除 shape 正确性 | 无（纯边界语义） | **保留语义 / 等价替换机制**（§0.6 R3：tile 级 tail + 零起点 UB 切片） |

无 SMEM tiling / coalesced 显式控制 / mma / warp shuffle / cp.async / swizzle / persistent（源 kernel 无此类特征——单步逐元素无需）。S3 为算法/契约层（可移植），S1/S2 为实现层，其余 host 层。

### 0.5 硬件耦合性分析与 NPU 适配决策

> 判定输入含 Phase R 调研结论（§1.6.0）：源算法与调研候选同台比较；四态处置依据 = migration-analysis.md §5.3 映射表条目 / docs 条目 / examples 佐证 / 本任务 910B3 实测事实（调度 prompt 事实 1–5，均已复核）。

| 条目 | 层 | 处置 | 依据 |
|------|----|------|------|
| C1 数学语义 / I/O 契约 / 边界语义（§0.1–0.2） | 语义层 | **保留** | 语义层无条件保留（migration-analysis.md §5.4 规则 1） |
| C2 direct 策略两级 SIMT 并行（S1） | 优化层 | **重新设计** → 一维 persistent 分核 + 整 tile 向量化 v-op 链 | 映射表「block × thread 两级并行 → block 级任务划分 + 向量化元素并行（**重新设计**，本项目 T.Kernel 只接受一维 block 数）」条目；同族 expm1/erf §0.6 R1 同款；PL-1.22 persistent 模板（910B3 第二证） |
| C3 xor 全 1 workaround（S2） | 算法层 | **等价替换** → `T.vnot`（int8/16/32/64）；uint8 以 int8 位视图走同一 vnot | ① docs 证据：`docs/Tilelang.language/逻辑操作/T.vnot.md` §2.2.1 dtype 矩阵——uint8/int8/int16/int32/int64 全 √（bool ×），§3 lowering = `hivm::VNotOp`（原生逐位取反向量 op）；② 实测证据：本任务实测事实 1——`T.vnot` 对 int8/int16/int32/int64 正确（含动态 tail，N=1024 与 N=1000/block=256 均通过）；③ 恒等性：补码 `~x ≡ x ⊕ (2^w−1)`，−1 位型 = 全 1（§1.6.1 A1 机器验证 0 违反） |
| C4 bool 分支 `x == 0`（S3） | 算法层 | **等价替换** → int8 位视图 + `T.vxor(x, 1)`（shaped buffer）+ view 回 bool | ① docs 证据：T.vnot.md/T.vxor.md §2.3——**bool（i1）无法 load/store GM**（原生 bool 路径被排除）；② 实测证据：本任务实测事实 3——原生 `dtype="bool"` 能编译但执行挂死（200s 超时，禁止使用）；事实 4——int8 视图 + `T.fill(ones, 1)` + `T.vxor` + view 回 bool 实测通过，且 **vxor 不接受标量操作数**（与 T.vxor.md §2.1「src1 或 Python 标量」的文档表述冲突——以实测为准，shaped buffer 形态，§9 R1）；③ 等价性：torch.bool 存储契约 {0,1} 上 `b^1 ≡ ¬b ≡ (b==0)`（§1.6.1 A3 机器验证 0 违反） |
| C5 uint8 分支 `x ^ 255`（S3/S5） | 算法层 | **等价替换** → int8 位视图 + `T.vnot` + view 回 uint8 | ① 实测证据：本任务实测事实 2——uint8 走 `@tilelang.jit(out_idx=[1])` 时计算字节正确但输出张量被分配为 torch.float32（dtype 分配缺陷）；替代方案「out_idx=[] + 调用方预分配」会改动 wrapper 调用面（违背零 wrapper 编辑集成契约），**int8 位视图**为实测通过的无缝方案（位级等价：uint8/int8 同字节表示，vnot 逐位取反与符号解释无关，§1.6.1 A2 机器验证 0 违反） |
| C6 lru_cache / out_idx=[1] / K11 block_size=256（S4/S5/S6-host） | host 层 | **保留** | 纯 host 契约，与硬件无关；wrapper 现状（§0.2）即按此契约书写，Stage 5 仅切 import |
| C7 GPU grid 逐 tile 启动（ceildiv(N, 256) 个 block） | 优化层 | **等价替换** → persistent 核内串行（逻辑核数 ≫ 物理核数，§5.5） | docs/开发指南.md §3.3（物理核数限制与分核策略）；core-split-strategy.md §1 要素③「极大规模」方案；同族 erf §0.5-C7 同款 |

**统计**：保留 3 项（C1/C6/C7-语义）、等价替换 3 项（C3/C4/C5）、重新设计 2 项（C2、C7-机制）、舍弃 0 项。无静默丢弃：S1→C2、S2→C3、S3→C4/C5、S4/S5→C6、S6→C7 逐项承接（承接核对表见 §0.6 末）。

### 0.6 NPU 算法重设计

> 重设计对照 §1.6.0 调研结论（Phase R 先于本节执行）；每个重设计项给出「源方案 → NPU 新算法 → 语义保持论证」三要素。

- **R1（并行结构重设计，承接 C2/C7）**
  - 源方案：`T.Kernel(ceildiv(N, 256), threads=256)` 逐 tile 启动 + `T.Parallel(256)` 每线程 1 元素。
  - NPU 新算法：一维 `T.Kernel(num_kernels, is_npu=True)` persistent 分核（`num_kernels = min(num_logical, vector_cores)` ≤ 物理向量核数 40，§5.5 实查）+ 核内 `T.serial(num_local_tasks)` grid-stride（`block_id = i·num_kernels + cid`，`num_local_tasks = ceildiv(num_logical, num_kernels)` 编译期 IntImm 静态边界）+ `if block_id < num_logical` 纯名常量 guard；元素级并行交给整 tile v-op 链（每 v-op 处理 (block_size,) 全向量——T.vnot/T.vxor 单 op 覆盖整 tile）。
  - 语义保持论证：逐点映射无跨元素累加 → 并行度/分组变化不触碰数值（位运算精确，无规约顺序问题）；grid-stride 映射只改 tile→core 分派，不改每元素计算；guard 语义 = 源 ceildiv 尾块语义。
- **R2（dtype 路线重设计，承接 C3/C4/C5 + §1.6.0 选定）**
  - 源方案：单一 direct kernel，op_func 运行期按 dtype 三分支（==0 / ^255 / ^−1）。
  - NPU 新算法：**编译期 dtype 路线分派**（factory 级分支，TVM-script 解析器 if-block 变量作用域限制规避——leaky_relu L218 同款），三路线：
    1. **int8/int16/int32/int64（原生域）**：`T.copy → T.vnot(x_ub, y_ub) → T.copy`（1 v-op/tile）；
    2. **uint8（int8 位视图域）**：host 侧 `x.view(torch.int8)` → 路线 1 kernel → 输出 `.view(torch.uint8)`（零拷贝 reinterpret，builder 内包装）；
    3. **bool（int8 位视图域 + xor-1）**：host 侧 `x.view(torch.int8)` → `T.fill(ones_ub, T.cast(1, "int8"))`（核级一次）+ `T.vxor(x_ub, ones_ub, y_ub)` → 输出 `.view(torch.bool)`。
  - 语义保持论证：路线 1 = 逐位恒等（~x ≡ x⊕全1，D-1 A1 全 0 违反）；路线 2 = 位级等价（同字节表示 + 逐位取反与符号解释无关，D-1 A2 全 0 违反）；路线 3 = {0,1} 存储契约域等价（b^1 ≡ ¬b ≡ b==0，D-1 A3 全 0 违反；越契约存储值行为差异已披露，§9 R4）。**原生 bool 与原生 uint8 路径在 NPU 被实测排除**（事实 2/3），位视图是唯一同时满足「零 wrapper 编辑 + 实测通过」的路线。
- **R3（尾块处理，承接 S6/C7）**
  - 源方案：ceildiv 尾块由 T.Parallel 边界自然覆盖（每线程 1 元素，无越界）。
  - NPU 新算法：tile 级 `tail = T.min(block_size, N − t0)`；v-op 作用**整 buffer**（(block_size,) 无切片——v-op 切片仅接受静态边界），stale [tail:] lane 垃圾值不落 GM；store 仅 `y_ub[0:tail] → y[t0:t0+tail]`（零原点 UB 切片，官方 vec_add_1d 模式，expm1/erf 生产验证同构）。**动态 tail 正确性已实测**：事实 1——N=1000/block=256（tail=232）通过。
  - 语义保持论证：store 边界 = 源 copy 边界；stale lane 无 GM 副作用；T.copy 接受 runtime 索引切片（traps-compiler.md 第六轮补充实证）。
- **R4（host 防御，设计适配）**
  - 源方案：Op 层 numel 校验（N=0 可通过 `_validate_input`——numel 0 == N_total 0）。
  - NPU 新算法：factory 内 `N <= 0 → ValueError`（survive `python -O` 用 ValueError 不用 assert，leaky 同款）；dtype 不在六 dtype 集合 → ValueError（承接源 SUPPORTED_DTYPES 校验，test_bitwise_not.py L56–65 语义）；block_size 超 UB 预算 → ValueError（§4.5 guard）。
  - 语义保持论证：源 Op 层对 N=0 的行为是「空 kernel 启动」——NPU T.Kernel(0) 非法（num_kernels=min(0,·)=0），防御性拒绝为**设计适配**（与 leaky_relu §9.3 同口径），不改合法域语义。

**§0.4 优化意图承接核对表**：S1→R1（persistent + 整 tile 向量链承接「占满算力 + 最简 codegen」意图）；S2→R2 路线 1（原生 vnot 直接承接「正确逐位取反」意图——GPU workaround 的动因（CUDA int4 缺陷）在 NPU 不存在，docs T.vnot.md §3 原生 VNotOp）；S3→R2 路线分派（bool 语义由 xor-1 等价承接、uint8 由位视图承接、int 域由 vnot 承接）；S4/S5→C6 原样保留（lru_cache + out_idx=[1]）；S6→R3（tail 机制承接边界语义）。**无静默丢弃。**

### 0.7 标杆实现

- 同族已落地标杆（结构模板）：`examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py` 与 `erf_fwd/erf_fwd_kernel/erf_fwd_kernel.py`（elementwise 族 harness 迁移已落地 kernel：factory 契约 / persistent 分核 / L0–L2+Boundary 嵌入式测试全套）；设计文档模板 = `erf_fwd_kernel_DESIGN.md`（同族第 4 例，本任务为第 5 例）。
- 源侧参考：`/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2273（BitwiseNotFwdKernel）、L305（direct builder）。
- 测试锚：NPU 侧 `examples/TileOPs/tests/ops/test_bitwise_not.py`（6 dtype fixture + exact_compare + rejects_float）；bench `examples/TileOPs/benchmarks/ops/bench_bitwise_not.py`（manifest 驱动 4 workloads）。
- 同类案例索引：CASE-elementwise-expm1-migration / erf / elu / leakyrelu（pattern-library/cases.md，harness 迁移第 1–4 例；本任务为第 5 例同族）；PL-1.22 persistent 模板（两证合入）；PL-1.21 910B3 elementwise 集成基线（四例对照锚点）。

## 1. 概述

### 1.1 算子名称

`bitwise_not_fwd_kernel`（项目 `bitwise_not`；落地侧产品位置 = examples/TileOPs/tileops/kernels/elementwise/bitwise_not/ 下的 bitwise_not_kernel/ 产品子目录，Stage 3 创建，同族 expm1_kernel 布局；Stage 5 将 bitwise_not.py L64 的 import 切换至该产品，零 wrapper 编辑）。

### 1.2 功能描述

逐元素按位取反 forward：对扁平化 `(N,)` 输入张量施加 `y = ~x`，输出 same_as(input)。支持 bool / uint8 / int8 / int16 / int32 / int64（manifest 契约）；manifest workloads：smoke-1m（bool/uint8/int32）、elementwise-16m（int32/int64）、elementwise-64m（int16/int32）、elementwise-256m（int32）。**全整数/布尔域精确位运算，无浮点、无容差**（exact_compare 验收）。

### 1.3 数学公式

$$y_i = \sim x_i$$（即逐位取反，bitwise not）

按 dtype 的 NPU 计算公式（§0.6 R2 重设计后，逐位/契约域精确等价）：

| 输入 dtype | NPU 公式 | 等价性 |
|-----------|---------|--------|
| int8/int16/int32/int64 | `y = vnot(x)`（原生 ~x） | 逐位恒等（D-1 A1） |
| uint8 | `x.view(int8)` → `y8 = vnot(x8)` → `y8.view(uint8)` | 位级恒等（D-1 A2） |
| bool | `x.view(int8)` → `y8 = vxor(x8, 1)` → `y8.view(bool)` | {0,1} 契约域恒等（D-1 A3） |

### 1.4 算法描述

**NPU 侧算法（迁移决策后）**：输入按 `(N,)` 扁平 1-D 处理；编译期 dtype 路线分派（factory 级三分支，见 §1.3）；persistent 分核将 `ceildiv(N, bs)` 个逻辑 tile 均摊到 ≤40 个向量核（grid-stride，静态边界）；每 tile 执行：

1. `T.copy` 载入 GM→UB（原生域 dtype 或 int8 视图域，载入即位视图后的张量——view 在 host 侧完成，kernel 内全 int8/原生整型域）；
2. int 路线：`T.vnot(x_ub, y_ub)`；bool 路线：`T.vxor(x_ub, ones_ub, y_ub)`（ones_ub 核级 `T.fill` 物化一次）；
3. `T.copy` 写回 UB→GM（仅 `[0:tail]`）；bool/uint8 路线由 host 侧 builder 包装 view 回原 dtype。

与源算法的结构差异：CUDA 每线程 1 元素 SIMT 直通 → persistent 分核 + 整 tile 向量 op；GPU xor-全 1 workaround → NPU 原生 vnot（op 数 2→1，常量缓冲 1→0）；bool `x==0` → int8 视图 xor-1（等价）；uint8 `^255` → int8 位视图 vnot（等价）。算法选型依据 = §1.6.0 调研结论。

### 1.5 数据流图

```
GM(x, dtype) ──host view(torch.int8)──▶ GM(x8)   [仅 bool/uint8 路线；零拷贝 reinterpret]
GM(x, dtype/x8) ──T.copy──▶ x_ub (UB, 域 dtype)
   int 路线:    x_ub ──T.vnot──▶ y_ub
   bool 路线:   ones_ub ──T.fill(1, int8)──▶ (核级一次)
                x_ub ──T.vxor(·, ones_ub)──▶ y_ub
GM(y, dtype/y8) ◀──T.copy([0:tail])── y_ub
   host: y8 ──view(torch.uint8 / torch.bool)──▶ GM(y)   [仅 bool/uint8 路线]
```

覆盖全部 buffer：x/y（GM）、x8/y8（host 视图，与 x/y 同一存储）、x_ub、y_ub、ones_ub（bool 路线常量）。无来源不明数据。

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

> 调研对象：与「逐元素按位取反」同一数学语义的**算法族**——所有能逐位精确算出 `~x` 的算法结构；输入公式 / 源算法（per-dtype xor/比较）只是基线候选之一。调研深度：单步逐元素（无规约/无窗口/无矩阵）→ **轻量调研**（四问各一结论，逐问给依据；候选表 + 四口径表仍完整给出）。执行位置：Phase M0 之后、Phase M1 之前（§0.5 判定输入即本节结论）。

**R1 等价化简公式候选表（含基线）**：

| # | 候选 | 公式/结构要点 | 等价性初判 | 收益方向 | 结论 |
|---|------|-------------|-----------|---------|------|
| B0（基线） | 源算法：per-dtype 三分支（bool `x==0`；uint8 `x^255`；int `x^(−1)`） | xor 全 1 / 比较零 | 逐位精确（各 dtype 域恒等） | 源已是最小公式形态 | 基线；uint8/bool 分支需等价替换（实测约束，§0.5） |
| C1 | **NPU 原生 `T.vnot`**（逐位取反向量 op） | `~x`，单 op | 数学恒等（补码 `~x ≡ x⊕(2^w−1)`；−1 位型 = 全 1） | 每 tile 省 1 op + 省全 1 常量缓冲（vs B0 int 路线的 fill+vxor 两 op） | **选定**（int8/16/32/64 + uint8 位视图） |
| C2 | **bool：int8 视图 + `vxor(x,1)`**（常量 1 shaped buffer） | `b ⊕ 1` | {0,1} 契约域恒等（0¹=1、1¹=0 = ¬b = b==0） | 可行性收益（vnot 对 bool 位视图**非**等价：~0=0xFF≠1） | **选定**（bool 专用） |
| C3 | bool：vcmp(x==0) 产出 i1 | 比较产出 bool 中间量 | 语义等价但 **bool(i1) 无法 load/store GM**（T.vnot.md/T.vxor.md §2.3 同款限制条款） | — | **淘汰**（i1 落盘被 docs 明文禁止；实测事实 3 旁证原生 bool 域不可用） |
| C4 | 标量逐元素循环（`for i in T.serial(bs): y[i] = ...`） | 逐元素标量 | 等价但反向优化 | — | **淘汰**（§1.6.2 V1：向量 op 原生存在，标量循环无任何保留理由） |
| C5 | 查表法（256/65536 项 LUT + gather） | 位宽 ≤8 时查表 | 等价 | — | **淘汰**：gather/索引类 API 需求 + LUT 常量搬运 > 1 v-op 直接计算；bitwise 族在所有主流实现（PyTorch/CUDA/向量 ISA）均为直接位指令，无查表先例 |

R1 调研范围声明：恒等变形自查（补码恒等式族：~x≡x⊕(2^w−1)≡−1−x≡−(x+1)——同为 1 op，无更省形态；De Morgan 律不适用于单目取反）；docs/Tilelang.language/ 全 16 子目录枚举检索位运算入口（`逻辑操作/` vnot/vxor/vand/vor/vshl/vshr 六 op 全部打开核对 dtype 矩阵——**位运算 op 全集即此六者**，无其他取反入口）；`数学操作/`（无位运算）；同族实现（expm1/erf/leaky/elu landed kernel——浮点域无位运算先例）；pattern-library（elementwise.md / cases.md 无 bitwise 条目）。C5 的「无查表先例」为模型知识级断言（标注：未文档化假设，收益方向为负，不影响选型）。

**R2 在线算法**：**无**。结构依据：按位取反为逐点无状态映射，`~x_i` 仅依赖 `x_i`，元素间依赖图为零边——「在线/单遍」重构的收益口径（扫描遍数、中间缓冲、UB 驻留）对零依赖图无作用对象；源算法本身已是单遍流式（每元素独立、单次 GM 扫描）。参考表（algorithm-candidates.md 逐元素族）无在线条目命中。结论：无在线变体（结构依据充分，非空白断言）。

**R3 复杂度四口径对比表**（口径：FLOPs = 每 tile 每元素向量操作数（vnot/vxor 各记 1，fill 为核级一次性）；访存 = GM 读+写字节 = 2N·eB（manifest 口径）；扫描遍数 = GM 数据完整扫过次数；中间缓冲 = 每 tile UB 驻留 (bs,) 工作缓冲数。基线在表）：

| 候选 | v-op 数/tile | 访存 Bytes（N 元素） | 扫描遍数 | 中间缓冲峰值 | 判定 |
|------|-------------|--------------------|---------|-------------|------|
| B0 源算法直译（fill 全1 + vxor） | 2 | 2N·eB | 1 | 2 工作 + 1 常量 | 可行但多 1 op + 1 常量缓冲 |
| **C1/C2 选定（vnot / vxor-1）** | **1**（bool 1 + 核级 1 fill） | 2N·eB（**与基线相同**——单入单出，无 GM 往返） | 1 | 2 工作（x_ub/y_ub）+ bool 1 常量 | **选定** |
| C3 vcmp i1 | 1 | — | — | — | **可行性淘汰**（i1 不可落盘） |
| C4 标量循环 | bs 次标量 | 2N·eB | 1 | 同 | **反向优化淘汰** |
| C5 查表 + gather | ≥2 + LUT 流量 | 2N·eB + LUT | 1 | +LUT 缓冲 | 访存/缓冲劣化淘汰 |

关键读法：全部可行候选**访存与扫描遍数完全相同**（单入单出逐点算子，无 GM 往返差异）——bitwise_not 是带宽受限算子（manifest `flops=N, bytes=2N·eB`），**算法选择由 v-op 发射数、常量缓冲与 dtype 路线可行性决定，不由 FLOPs 决定**；C1 相对 B0 直译每 tile 省 1 次 v-op 发射（~0.5µs 量级，CONST-vector-launch-overhead）+ 1 个 (bs,) 常量缓冲。

**R4 硬件亲和性逐候选评估**：

| 检查项 | C1/C2（选定） | B0 直译 | C3 vcmp | C4 标量 |
|--------|--------------|---------|---------|---------|
| 计算单元匹配 | 逐元素位运算 → Vector 单元原生 op（VNotOp/VXorOp，docs §3 lowering）✓ | ✓ | ✓ | ✗（标量流水，向量单元空转） |
| dtype 支持 | vnot：uint8/int8/16/32/64 √（docs §2.2.1）+ 实测事实 1；vxor：int 全宽 √；bool 域 × → int8 位视图规避 ✓ | vxor int √ | bool i1 落盘 × | ✓ |
| 片上容量 | bs=256 时 x_ub+y_ub+ones ≤ 768B/int8 域、≤4KB/int64 ≪ 192KB（§4.5）✓ | +1 常量 | — | — |
| 对齐整除 / 静态边界 | 1-D 连续；N/bs host int → 全静态 ✓（动态 tail 实测通过，事实 1） | ✓ | — | — |
| 流水可融合性 | persistent + 整 tile 单 op 链，multi-buffer 覆盖 MTE2 窗口 ✓（PL-1.6 copy-floor：7-pass 链 16M 全隐藏——1-op 链更甚） | ✓ | — | — |
| 跨核结构 | 无（逐点独立，无归并）✓ | 同 | — | — |

**D-2 设计期估算下界**（协议 = `_shared/standards/hardware-cost-model.md` §2；常数 = `tilelang-op-optimize/references/pattern-library/constants.md`，含版本戳与 stale 标注；纯 Vector 核数 = 实查 20 × 2 = **40**）：

- 物理核数实查记录（要素②，2026-10-29）：设计环境 tilelang 不可导入（dev-root 私有 TVM 包缺失，CONST-aicore-910B2C 条目记载同款环境），按该条目记载的等价路径 ctypes 直调 CANN libruntime：`python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)"` → **ret=0, aicore_num=20**（与该条目 2026-10-28~29 五次独立复跑一致：leaky/elu/erf 三任务设计期+检视侧双证）；纯 Vector 翻倍 → **40 向量核**。权威查询 `NPUUtils.get().get_aicore_num()` 由 Stage 3 kernel 工厂期在设备上执行（预期 20，异常即报环境错误，不猜测）。
- 流量项（CONST-mte2-degradation〔引用条目前置重验：CANN 8.5.0 stamp、910B2C/48 核、fp16 测量——910B3/40 核整型外推，Stage 4 对账〕，档位按 workload 元素数取值，每核流量 = 总字节/40；本算子 1:1 R/W 对称，无 3:1 混合退化）：

  | workload | 总流量 | 每核流量 | 档位带宽 | 流量项 |
  |----------|--------|---------|---------|--------|
  | smoke-1m bool/uint8 | 2.10MB | 52.4KB | 33.9GB/s（1M 档） | **1.5µs** |
  | smoke-1m int32 | 8.39MB | 210KB | 33.9GB/s（1M 档） | **6.2µs** |
  | 16m int32 | 134.2MB | 3.36MB | 30.2GB/s（16M 档） | **111µs** |
  | 16m int64 | 268.4MB | 6.71MB | 30.2GB/s（16M 档） | **222µs** |
  | 64m int16 | 268.4MB | 6.71MB | 20.9GB/s（64M 档） | **321µs** |
  | 64m int32 | 536.9MB | 13.4MB | 20.9GB/s（64M 档） | **642µs** |
  | 256m int32 | 2.147GB | 53.7MB | 18.4GB/s（256M 档） | **2.92ms**（聚合地板 1.26TB/s 口径 1.70ms → 报告区间 **1.7–2.9ms**） |
- 发射项：每 tile 1 v-op（bool +1 核级 fill）× ~0.5µs ≈ **0.5–1µs**（tile 链时延下界，CONST-vector-launch-overhead〔stale，首跑重验〕）；大 N 下隐藏于 MTE2 窗口（CONST-copy-floor-method〔stale，首跑重验〕：7-pass 链 ≥16M 全隐藏——本算子 1-op 链隐藏裕度更大）。
- 容量项：纯 Vector 单引擎、无跨引擎 GM 往返 → **豁免**（轻量调研口径流量项必算已完成）。
- **估算下界行**：`smoke-1m bool/uint8 ≈ 1.5µs、int32 ≈ 6.2µs（发射与流量同量级）；16m int32/int64 ≈ 111/222µs；64m int16/int32 ≈ 321/642µs；256m int32 ≈ 1.7–2.9ms（流量主导）——引用 CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-aicore-910B2C（stale 条目重验承诺见 §9）；未实测假设：整型/位宽档位带宽沿用 fp16 曲线外推、910B3/40 核 vs 910B2C/48 核每核带宽外推`。
- **同族实测校准锚**（PL-1.21，同 bench 口径 910B3）：expm1（4-op 链）16m fp16 实测 128.14µs = 流量下界 55.6µs 的 **2.3×**；本算子 1-op 链预期倍率 ≤ 2.3×（16m int32 → ~150–260µs 区间），差异超 2× 时先查集成口径（TRAP-BENCH-CONFIG-CALIBRATION）再谈算法。
- 候选排序结论：唯一同时满足全 dtype 可行 + 最小发射数的候选 = **C1 vnot + C2 bool xor-1**（无 R3 并列，无实验裁决需求；§1.6.3 单轴豁免）。

**调研结论**：选定算法族 = **dtype 三路线位运算**（int 域原生 vnot；uint8 int8 位视图 + vnot；bool int8 位视图 + vxor-1）；关键依据 = R1 补码恒等式（逐位/契约域精确，非容差）+ R2 无在线变体（逐点零依赖图，结构依据）+ R3 全可行候选访存/遍数相同、由发射数与可行性决胜 + R4 docs dtype 矩阵与实测事实双证；与基线的结构差异 = GPU xor-全 1 workaround → 原生 vnot（op 2→1）+ bool/uint8 位视图路线；「无更优替代」的调研范围 = R1 表 B0/C1–C5 全候选 + docs/Tilelang.language/ 全 16 子目录位运算入口枚举（逻辑操作六 op 全开）+ 同族 4 个 landed kernel + pattern-library elementwise/cases 检索 + 补码恒等式自查；C5 查表先例为未文档化假设（负收益方向，§9 R7 不阻塞）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 dtype 三路线（非源公式直译）。采纳项四要素齐全；**等价性机器验证（D-1）结果表内嵌于本节**（脚本 `examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py`，随算子目录留存，Stage 2 维度 8 重跑复核）。位运算全程精确——D-1 参照 = torch 原生 dtype 位语义（逐位精确相等，无 fp 舍入路径，强于 fp64 参照口径）。

**A1（采纳）：int 域 `x ^ (−1)` → `T.vnot`（补码恒等降 op）**

- 原式：`y = x ⊕ (−1)`（GPU 源公式；−1 在 w 位补码中位型 = 全 1 = 2^w − 1）。
- 优化后公式：`y = ~x`（T.vnot，原生 VNotOp 单 op）。
- 等价性论证：**数学恒等**（逐位，非容差）。补码定义恒等式 `~x ≡ x ⊕ (2^w − 1)`（对任意 w 位整数，含极值 int8 −128/127 … int64 ±2⁶³：取反即按位补，与加法群逆无关）；xor 与取反对极值/交替位型/全 0 全 1 均逐位一致。D-1 机器验证：int8/int16/int32/int64 四 dtype × (1024+1000 随机 + 8 角点) 全部 0 违反（下表）。
- 收益量化：每 tile v-op 发射 2→1（fill 全 1 常量 + vxor → vnot），省 1 次 ~0.5µs 发射（CONST-vector-launch-overhead）+ 1 个 (bs,·) 全 1 常量缓冲（int64 域 2KB/tile）+ 1 次 fill；bool/uint8 路线另见 A2/A3。

**A2（采纳）：uint8 `x ^ 255` → int8 位视图 + vnot + view 回（可行性 + 降 op）**

- 原式：`y = x ⊕ 255`（uint8 域全 1 字节）。
- 优化后公式：`y = vnot(x.view(int8)).view(uint8)`。
- 等价性论证：**位级恒等**。uint8 与 int8 共享同一字节表示（itemsize=1，存储无符号位差异）；vnot 为逐位取反，与符号解释无关；view 为零拷贝 reinterpret（不改位型）。D-1 机器验证：uint8 × (1024+1000 随机 + 6 角点) 0 违反（A2a：^255 vs ~x 恒等；A2b：位视图路线 vs ~x 恒等）。
- 收益量化：**可行性收益**（规避 out_idx=[1] 对 uint8 输出分配为 float32 的实测缺陷——事实 2；「out_idx=[] + 调用方预分配」需改 wrapper 调用面，违背零编辑集成契约）；op 数同 A1（2→1）。

**A3（采纳）：bool `x == 0` → int8 位视图 + `vxor(x, 1)` + view 回（可行性）**

- 原式：`y = (x == 0)`（GPU bool 逻辑非；存储域 {0→1, 1→0}）。
- 优化后公式：`y = vxor(x.view(int8), 1).view(bool)`（1 为 shaped 常量缓冲 `T.fill(ones_ub, T.cast(1, "int8"))`，核级物化一次）。
- 等价性论证：**{0,1} 契约域恒等**。torch.bool 存储契约值域 {0,1}：0⊕1=1（True）、1⊕1=0（False）≡ 逻辑非 ≡ (x==0)。D-1 机器验证：bool × (1024+1000 随机 + [F,T]×2) 0 违反（A3a：GPU 式 x==0 vs golden；A3b：xor 路线 vs golden）。边界披露：存储值越出 {0,1} 时两式分叉（xor→3 truthy vs ==0→0），属 torch 契约外未定义行为，workload/test 输入均契约内（gen_inputs `rand>0.5`，workloads/elementwise.py L61–62），列 §9 R4 跟踪。
- 收益量化：**可行性收益**（vnot 对 bool 位视图非等价：~0=0xFF≠1——事实 4「普通 vnot 不能用于 bool」；原生 bool 域被 docs 禁止落盘 + 实测挂死——事实 3）；每 tile 1 v-op（vxor）+ 核级 1 次 fill，无额外发射增长。T.vxor.md §2.1 标称 src1 支持标量，但本工具链实测拒绝——设计按实测采用 shaped buffer（§9 R1），此处常量 1 物化即该实测约束的落地形态。

**无更优化空间项（显式结论）**：

- 公共子表达式：单 op 无子式，CSE 无对象。
- 稳定化变形：整数/布尔域无浮点极值/相消问题，不适用。
- 算子降代换：vnot/vxor 已是最低代价位运算入口（docs 逻辑操作六 op 全枚举：vand/vor/vshl/vshr 均需 ≥2 op + 常量才能合成取反，反向劣化）；查表法访存劣化（R1 C5）。
- 访存削减：单入单出 2N·eB 已是信息论下界（manifest 口径），中间量全驻留 UB，无 GM 往返可削。
- 归约合并：零归约结构（逐点算子），不适用。

**D-1 等价性机器验证结果表**（`python3 examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py`，torch CPU，原生 dtype 位语义参照，2026-10-29 执行，exit 0）：

| 验证项 | dtype | 测试点数 | 违反数 | 参照 | 结论 |
|--------|-------|---------|--------|------|------|
| A1 xor(−1)→vnot | int8 | 2,032 | **0** | torch.bitwise_not（位精确） | **EQUIV_PASS** |
| A1 xor(−1)→vnot | int16 | 2,032 | **0** | 同上 | **EQUIV_PASS** |
| A1 xor(−1)→vnot | int32 | 2,032 | **0** | 同上 | **EQUIV_PASS** |
| A1 xor(−1)→vnot | int64 | 2,032 | **0** | 同上 | **EQUIV_PASS** |
| A2a xor(255)→not（GPU 基线恒等） | uint8 | 2,032 | **0** | 同上 | **EQUIV_PASS** |
| A2b int8 位视图路线 | uint8 | 2,032 | **0** | 同上 | **EQUIV_PASS** |
| A3a x==0（GPU 基线恒等） | bool | 2,028 | **0** | torch.bitwise_not | **EQUIV_PASS** |
| A3b xor(1) 路线 | bool | 2,028 | **0** | torch.bitwise_not | **EQUIV_PASS** |

测试点构成 = 1024 整 tile 随机 + 1000 尾块随机 + dtype 角点（0/±1/极值/0x55/0xAA 交替位型）；角点含 int8 −128/127、uint8 0/255、int64 ±2⁶³ 等极值。**优化结论**：采纳 A1/A2/A3 共 3 项，均位精确 EQUIV_PASS；显式无优化空间项 5 类；优化后公式（§1.3）= §3.1 公式拆解唯一输入。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

实现方案全部循环与标量计算点盘点（结论与 §6 循环结构一致，无未盘点点）：

| # | 计算/循环点 | 形态 | 向量化替代 | API 佐证 | 结论 |
|---|------------|------|-----------|---------|------|
| V1 | 逐元素按位取反全链（int 域 ~x；bool 域 x⊕1） | 逐元素 | T.vnot / T.vxor 整 tile 向量（(bs,) 全 buffer 单 op） | docs/Tilelang.language/逻辑操作/T.vnot.md、T.vxor.md（§2.2.1 dtype 矩阵 + §2.4 整 buffer 示例） | **全部向量化** ✓ |
| V2 | 全 1 / 常量 1 物化（bool 路线） | 逐元素填充 | T.fill 整 buffer（核级一次，跨 tile 复用） | 本任务实测事实 4（`T.fill(ones_ub, T.cast(1,"int8"))` 910B3 实测通过）+ examples/deepseek_v4/inference/kernel.py L381、deepseek_v32/sparse_mla_fwd.py L96 同形 | **向量化** ✓ |
| V3 | GPU 源 T.Parallel 逐元素展开 | SIMT 循环 | 消除——v-op 整 tile 隐式并行，无显式元素循环 | 同族 landed kernel 形态（expm1/erf：无 T.Parallel，v-op 整 buffer） | **向量化** ✓（结构消除） |
| V4 | tile/core 索引（block_id = i·num_kernels + cid、t0、tail） | block 级标量 | 不可替代 | 每 tile 每核 O(1) 标量（非逐元素热点）；任务映射计算属「block 级索引/任务映射计算」豁免类别 | **保留**（理由：block 级索引，3 个标量/tile，非热点） |
| V5 | host 侧 N/dtype 归一化/view(int8)/view(回) | host 标量 | 不适用 | host 元数据 + 零拷贝 reinterpret 豁免类别（view 不改存储，无逐元素计算） | **保留**（host 侧） |
| V6 | 尾块边界 tail = min(bs, N−t0) | tile 级标量 + 动态边界 | 不可静态化（N 运行期给定；padding 引入额外 GM 写，比 tail 切片更劣） | 「依赖动态 shape 的边界处理」豁免类别；v-op 切片仅静态边界（traps-compiler.md P10）→ 整 buffer v-op + [0:tail] store 是唯一正确形态；动态 tail 已实测（事实 1） | **保留**（理由：动态边界，同族生产模式） |
| V7 | if block_id < num_logical guard | tile 级标量分支 | 不可替代（persistent 尾核掩码） | num_kernels 整除不尽时必须；条件为编译期可折叠纯名比较（leaky/erf 生产同款，PL-1.6 910B3 结构重验存活） | **保留**（理由：任务掩码语义，静态折叠） |

无「实现简单/照源码写」类理由；2D T.Parallel 标量化陷阱（TRAP-parallel2d-bm-ge2-scalarize）不适用——本算子为 1-D 整 tile v-op 链。

**向量化结论**：向量化替代 3/3 处可替代计算点（V1 位运算链 / V2 常量填充 / V3 SIMT 结构消除），保留 4 处（V4 block 级索引、V5 host 元数据、V6 动态尾边界、V7 掩码 guard——理由类别分别为 block 级任务映射 / host 侧元数据 / 动态 shape 边界处理 ×2，均属可接受理由清单）；结论与 §6 一致（§6 无逐元素标量循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**豁免声明**：本算子为 1-D 扁平逐元素映射（I/O 契约即 `(N,)` 连续），**仅一个候选轴**（尾轴连续）——按 skill Phase 2 第 3 项「纯 GEMV/单轴算子可写明仅一个候选轴豁免」处理。核对项：

- 向量轴 = 连续尾轴；CONST-vector-launch-overhead：f16≈f32 逐元素速率无打包加速差——位宽对齐不构成本算子的轴选择变量（int8 域 1B/elem，MTE 段搬运粒度 128B ≫ 1 elem，T.copy 合并由 MTE 处理）；
- 广播轴对齐不适用（常量全 tile 统一，无按轴广播反转问题）；
- 核内布局 = 1-D (bs,) UB buffer；lane 映射 = v-op 隐式连续；GPU 源码的「1 元素/线程」轴选择（§0.3）是输入而非结论——NPU 侧整 tile v-op 天然覆盖，无 SIMT 轴需保留；
- uint8/bool 的 int8 位视图为 **host 侧零拷贝 reinterpret**（GM 存储不变、UB 内全 int8 域），不构成第二种核内布局候选（无重排代价）。

**布局决策结论**：主选布局 = 1-D (bs,) 连续 UB buffer + 连续尾轴向量映射（单轴豁免）；与 §3.3 伪代码（整 tile v-op 无切片）、§4.3（(bs,) buffer 形状）、§6（无 2D 循环）三方一致；无备选布局、无实验裁决计划（豁免成立：仅一个候选轴）。

## 2. 编程模式选型

### 2.1 模式结论

**Developer 模式**（wrapper 契约锁定：bitwise_not.py L199–201 `ascend_mode = "Developer"`（CG-2026-0010）；任务参数 programming_mode=developer）。

### 2.2 选型理由

1. 逐元素 v-prefix 链为 Developer 模式标准形态（官方 vec_add_1d.py 模式族）；无手动流水/双缓冲/跨核协同需求（单引擎、无规约、单 v-op 链）。
2. 同族标杆 expm1/erf/leaky/elu 四个 landed kernel 均 Developer 模式落地（`os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` + 模块级常量），wrapper 侧 ascend_mode="Developer" 与之配套。
3. 用户显式指定 Developer（op_requirements 编程模式字段）。

### 2.3 模式影响

- 同步：自动同步（RAW/WAR 由编译器插入 pipe 屏障），无手动 set_flag/wait_flag（§7）。
- 多缓冲：auto-multi-buffer 开启；UB 预算按显式 alloc 口径核算（bs=256 峰值 ≤4KB，任何膨胀系数下均可忽略，§4.5）。
- kernel 产品需 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` 模块级自锁（Expert/Developer 混进程隔离，同族同款）。

## 3. API 映射设计

### 3.1 公式拆解

dtype 三路线（§1.6.1 优化后公式 = 本节唯一输入）拆解为：载入、位取反（vnot）或 xor 常量（vxor + fill）、写回；bool/uint8 附加 host 位视图对（reinterpret，非计算步）。

### 3.2 TileLang API 映射

| 计算逻辑 | TileLang API（v-prefix） | 关键参数/形态 | 文档佐证（已亲自打开核对） |
|---------|--------------------------|--------------|---------------------------|
| 载入/写回 | T.copy | runtime 索引切片 `[t0:t0+tail]` / 零原点 `[0:tail]`；int8/16/32/64 域 | traps-compiler.md 第六轮（T.copy 接受 runtime 切片）+ 官方 vec_add_1d 模式 + 实测事实 1（int 全宽含动态 tail 通过） |
| 按位取反 | T.vnot | `(x_ub, y_ub)` 整 buffer 两参形态 | T.vnot.md §2.1 签名、§2.2.1 dtype 矩阵（uint8/int8/int16/int32/int64 √，bool ×）、§2.4 示例、§3 VNotOp lowering |
| 按位异或（bool 路线） | T.vxor | `(x_ub, ones_ub, y_ub)` 三缓冲（src1 = shaped buffer——**标量形态文档 §2.1 有载但本工具链实测拒绝**，按实测设计） | T.vxor.md §2.2.1（int8 √）、§2.3（bool ×）+ 实测事实 4 |
| 常量物化（bool 路线） | T.fill | `T.fill(ones_ub, T.cast(1, "int8"))` 核级一次 | 无专页 docs；examples/deepseek_v4/inference/kernel.py L381、deepseek_v32/sparse_mla_fwd.py L96 同形 + 实测事实 4 直证 |
| 输出分配 | @tilelang.jit(out_idx=[1], target="npuir") | main(x) → y | 同族四例 landed kernel 同款 |

已枚举并淘汰的候选 API：T.vand/T.vor/T.vshl/T.vshr（合成取反需 ≥2 op，反向劣化，T.vnot.md 同目录六 op 全枚举核对）；vcmp 系（i1 不可落盘）；T.vcast（本算子**无浮点 transit 需求**——整数域原生计算，与 erf/expm1 的关键差异）。

### 3.3 计算伪代码

```python
# 基于 TileLang API 的计算流程伪代码（int 原生域路线；bool 路线差异以 [B] 标注）
# 工厂契约（§3.5.2）：bitwise_not_fwd_kernel(N, dtype)  # lru_cache, torch.dtype 归一化
#   dtype 路线分派（编译期，factory 级）:
#     bool/uint8  -> run_dtype="int8", needs_unview=True,  op=vxor1|vnot
#     int8/16/32/64 -> run_dtype=dtype_key, op=vnot
#   -> 返回 curried builder kernel(block_size=256) -> main(x) -> y（out_idx=[1]）
@tilelang.jit(out_idx=[1], target="npuir")
def kernel(block_size=256):                    # K11 wrapper 默认 256（§5.2）
    num_logical  = ceildiv(N, block_size)      # 编译期 IntImm
    num_kernels  = min(num_logical, vector_cores)   # ≤ 40（§5.5 实查×2）
    num_local_tasks = ceildiv(num_logical, num_kernels)  # 静态边界
    @T.prim_func
    def main(x: T.Tensor((N,), run_dtype), y: T.Tensor((N,), run_dtype)):
        with T.Kernel(num_kernels, is_npu=True) as (cid, _):
            x_ub = T.alloc_ub((block_size,), run_dtype)
            y_ub = T.alloc_ub((block_size,), run_dtype)
            # [B] ones_ub = T.alloc_ub((block_size,), "int8")
            # [B] T.fill(ones_ub, T.cast(1, "int8"))      # 核级一次，跨 tile 复用
            for i in T.serial(num_local_tasks):          # persistent 核内串行（静态）
                block_id = i * num_kernels + cid
                if block_id < num_logical:               # 编译期可折叠 guard
                    t0 = block_id * block_size
                    tail = T.min(block_size, N - t0)     # 尾块（实测事实 1 覆盖）
                    # 1) 载入 GM→UB（bool/uint8 的 x 已在 host 侧 view 为 int8）
                    T.copy(x[t0:t0+tail], x_ub[0:tail])
                    # 2) 单 v-op 位运算（整 buffer，stale lane 不落 GM）
                    T.vnot(x_ub, y_ub)                   # int 路线
                    # [B] T.vxor(x_ub, ones_ub, y_ub)    # bool 路线（src1 = shaped buffer）
                    # 3) 写回 UB→GM（仅 [0:tail]）
                    T.copy(y_ub[0:tail], y[t0:t0+tail])
        return main
# builder 包装（bool/uint8 路线）：
#   def call(x): return main(x.view(torch.int8)).view(torch.bool / torch.uint8)
```

注：dtype 路线分支在 factory 级完成（两份 @T.prim_func 定义，TVM-script 解析器 if-block 作用域限制规避——leaky_relu L218 同款）；v-op 全整 buffer 无切片；host view 为零拷贝 reinterpret（GM 存储不变）。

### 3.4 API 可行性确认

| API | dtype 矩阵关键项 | 形态限制 | 风险等级 |
|-----|-----------------|---------|---------|
| T.vnot | uint8/int8/int16/int32/int64 √；bool × | 两参 (src,dst) 整 buffer | 低（docs + 实测事实 1 双证） |
| T.vxor | int8/int16/int32/int64/uint8 √；bool ×；浮点 × | 三缓冲同 shape 同 dtype；src1 标量形态实测拒绝（docs §2.1 声称支持——以实测为准） | 低（shaped buffer 形态实测通过，事实 4） |
| T.fill | int8 域实测 √ | (buffer, scalar) | 低（事实 4 直证 + 2 examples 同形） |
| T.copy | int8–int64 全宽 | runtime 切片 / 零原点 | 低（事实 1 实测 + 同族生产） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

- 三维 Kernel：不涉及（一维 T.Kernel）✓；GPU 专用 API（T.bitwise_xor CUDA 语义 / T.cast scalar）：已重设计为 vnot/vxor/fill ✓；GEMM 非整除：不涉及（非 GEMM）✓；L0C：不涉及（纯 Vector）✓。
- bool 不可 load/store GM（T.vnot.md/T.vxor.md §2.3 同款条款）→ int8 位视图路线 ✓；原生 bool 实测挂死（事实 3）→ factory 禁止原生 bool dtype 进入 traced main ✓；v-op 无切片（静态边界限制）→ 整 buffer 设计 ✓。
- uint8 out_idx 分配缺陷（事实 2）→ int8 位视图路线 ✓；vxor 标量 src1 实测拒绝（事实 4）→ shaped buffer ✓。
- 强制检测清单（ascend-constraints.md）逐项通过，无遗留。

#### 3.5.2 参考实现差异说明（工厂契约 / wrapper 契约）

- **工厂契约（两段式，K3/K11）**：`bitwise_not_fwd_kernel(N, dtype)` → curried builder `kernel(block_size=256)` → `main(x)` 返回 y（out_idx=[1]）。**K3 按 wrapper 契约明确丢弃 `output_dtype`/`threads`**（bitwise_not.py L27–28 注释：factory signature drops output_dtype / threads → `(N, dtype)`；与 erf 保留 7 参形态不同——**本 op 以 wrapper 注释为准**）；same_as(input) 契约由结构保证（traced y dtype ≡ 域 dtype ≡ 输入 dtype，bool/uint8 经 view 对还原）。
- **wrapper 现状逐字核对结论**：bitwise_not.py L147 `return bitwise_not_fwd_kernel(N, dtype)(block_size)(x)` 两段式形态（勿照搬 GPU 抽取件 `kernel(threads_arg)` 单参式——leaky_relu 教训条目 1 同款）；L146 传 torch.dtype 对象 → 工厂归一化（`dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]`，leaky 教训条目 4）；L122 `_BLOCK_SIZE = 256` 全 dtype 统一。
- **Stage 5 集成零 wrapper 编辑**：仅将 L64 `from ._bitwise_not_fwd_kernels import bitwise_not_fwd_kernel` 切换为 conductor 产品（同工厂签名 `(N, dtype)` → builder(block_size) → main(x)），expm1/erf 集成模式。wrapper import 名单核对：`__all__ = ["BitwiseNotFwdKernel"]` + L64 单名 import——无 GPU manually-appended helper 残留（leaky RETROSPECTIVE 教训已预核对，bitwise_not.py L66–70 注释明示 fp8 helpers 不需要）。
- 与 GPU 抽取件（`_bitwise_not_fwd_kernels.py`）的差异 = 本设计全部 §0.6 重设计项（并行结构/dtype 路线/尾块/防御），非语法翻译。

#### 3.5.3 本项目同类实现参考

- `examples/TileOPs/tileops/kernels/elementwise/expm1/expm1_kernel/expm1_fwd_kernel.py`（同族已落地：工厂契约 / persistent / UB guard / L0-L2 测试骨架——本设计结构模板）
- `examples/TileOPs/tileops/kernels/elementwise/erf_fwd/erf_fwd_kernel/erf_fwd_kernel.py` + DESIGN.md（同族第 4 例，设计文档模板）
- `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu_kernel/_make_leaky_relu_kernel.py`（factory 级 dtype 分支双 prim_func 形态 + vbrc let 绑定规范先例）
- `examples/TileOPs/tileops/kernels/elementwise/erf_fwd/erf_fwd_kernel/erf_fwd_kernel.py` 内 `T.vxor/vmul` 标量与跨缓冲形态边界（CONST-vsub-crossbuf-scalar 补充实证——本设计 shaped buffer 形态规避标量面）
- 官方形态：`examples/` 下 vec_add_1d.py 模式（1-D elementwise + persistent + 尾块零原点切片）

## 4. 数据规格与内存规划

### 4.1 输入张量

`x: (N_total,)`，dtype ∈ {bool, uint8, int8, int16, int32, int64}，contiguous（Op 层保证 flatten+contiguous）；bool/uint8 在 host 侧经 `view(torch.int8)` 转为位视图张量后进入 kernel（零拷贝 reinterpret，同一存储）；动态轴：N_total 运行期给定（工厂入参，编译期常量化——每 N 特化编译，lru_cache 键）。

### 4.2 输出张量

`y: (N_total,)`，dtype = same_as(input)（manifest 契约）；`out_idx=[1]`，`main(x)` 返回 y；bool/uint8 路线由 builder 包装 `.view(torch.bool/uint8)` 还原对外 dtype（位型不变）。

### 4.3 中间缓冲区（每 tile，UB）

| 缓冲 | shape/dtype | 角色 | 复用关系 |
|------|-------------|------|---------|
| x_ub | (bs, 域 dtype) | 载入 buffer + v-op src0 | 跨 tile 复用（每轮重载入覆盖） |
| y_ub | (bs, 域 dtype) | v-op dst + 拷出源 | 跨 tile 复用 |
| ones_ub | (bs, int8) | bool 路线常量 1（仅 bool 路线分配） | **核级物化一次**，全 local tasks 复用（不随 tile 变） |

v-op 形态纪律：T.vnot/T.vxor 走 dst∉src 三缓冲（vnot 两参天然三缓冲；vxor 文档无原地条款——**不使用原地形态**，与 erf 对 vadd/vmul 原地条款的谨慎口径一致）；无 fp32 transit 缓冲（整数域原生计算——与 erf/expm1 的结构性差异，缓冲数从 9+ 降至 2–3）。

### 4.4 内存搬运路径

`GM(x) → (MTE2) → UB[x_ub] → vnot/vxor → UB[y_ub] → (MTE3) → GM(y)`。纯 Vector 单引擎；无 L1/L0A/L0B/L0C 参与；无跨引擎往返（容量项豁免依据）；bool/uint8 的 host view 不产生搬运（同一存储 reinterpret）。

### 4.5 UB 内存预算（bs 元素 tile，域 dtype 口径）

| 路线 | 工作缓冲 | 常量 | 合计 B/elem | bs=256 | bs=8192（cap） | +12% (CG-2026-0008) @cap |
|------|---------|------|------------|--------|---------------|--------------------------|
| int8 域（int8/uint8/bool） | 2×1B | bool: +1B | 2 / 3 | 512B / 768B | 16 / 24KB | 18 / 27KB ✓ |
| int16 | 2×2B | — | 4 | 1KB | 32KB | 36KB ✓ |
| int32 | 2×4B | — | 8 | 2KB | 64KB | 72KB ✓ |
| int64 | 2×8B | — | 16 | 4KB | 128KB | 143KB ✓ |

- 上限 192KB/核（CONST-capacity-910B2C〔stale：UB 192KB/AIV + 膨胀条款〕；显式逐 buffer alloc 结构实测 ≈ 手工核算，r9 数据点 ratio 0.981——本设计即显式 alloc 形态）。
- **UB guard**：`_UB_MAX_BLOCK = 8192`（全 dtype 统一，int64 cap 档 143KB < 192KB 留裕度），`bs > cap → ValueError`（工厂 builder 内 host 校验，leaky 同款）；默认 bs=256（wrapper K11）远低于 cap，任何 multi-buffer 膨胀系数下峰值可忽略。
- 对照教训条目 5：UB 论证引用 pattern-library 平台值口径（192KB + r9/CG-2026-0008 两套系数），不自设线性模型。

### 4.6 动态轴定义

仅 N_total（运行期）。bs / num_kernels / num_local_tasks / tail 均为编译期可折叠量（N、bs 为 Python int；cid 运行期但仅出现在 block_id 与 guard）。无运行期 shape 分支。

### 4.7 JIT 配置

`@tilelang.jit(out_idx=[1], target="npuir")`；无 pass_configs（Developer 自动同步 + auto-multi-buffer 默认）；`functools.lru_cache(maxsize=32)` 于工厂（S4 承接）；每 (N, dtype) 组合一次编译（bs 为 builder 参数，同工厂双 bs 复用——expm1 E2 形态）。

## 5. Tiling 策略

### 5.1 计算类型

纯 Vector 逐元素（无 Cube / 无规约 / 无跨核依赖）→ 仅需 UB；单步单 op 链（1 v-op/tile；bool +核级 fill）；无融合后处理。

### 5.2 Block 划分

- **Block 尺寸（bs）**：全 dtype 统一 **256**（wrapper K11 契约常量 `_BLOCK_SIZE = 256`，bitwise_not.py L122——GPU direct 策略 threads=256、npt 不参与的折叠口径）；上限 = §4.5 UB guard 8192。
- **Tile Shape**：1-D `(bs,)`；非整除处理 = **tile 级 tail（Kernel 内动态 block 方案）**：`tail = T.min(bs, N − t0)`，v-op 整 buffer、store `[0:tail]`（§0.6 R3）——无 padding、无 host 侧 crop。动态 tail 正确性已实测（事实 1：N=1000/block=256 通过）。
- **bs=256 的小 tile 开销预判**：256m int32 时每核 26,215 个 local tasks，per-task 固定开销（2 次 MTE issue + guard）未入常数表（未实测假设，§9 R5）；同族校准锚（PL-1.21）显示 4-op 链实测/下界 ≈ 2.3×，1-op 链预期 ≤2.3×；Stage 4 以 config sweep（bs 是 Kernel config 键与 builder 参数）与 perf_opt 内部 clamp 为缓解路径，不改产品默认契约。

### 5.3 约束分析

- 32B 对齐：T.copy 切片边界由 MTE 处理（vec_add_1d 官方模式同款）；尾 tile 最小 1 元素可用（同族 L0-4 N=1 用例同构）。
- int8 域 1B/elem：MTE 最小段 128B ≫ 1B，T.copy 连续 1-D 无跨步惩罚（CONST-gm-to-ub-bw-dilution：2D 跨步才有形态问题，本算子纯连续）。
- v-op 无切片（静态边界限制）→ 整 buffer 设计（§1.6.2 V6）。

### 5.4 注意事项

- bs 变更不改语义（纯分块参数）；lru_cache 按 (N, dtype) 特化、bs 经 builder 参数复用同一 JIT callable 缓存。
- 每逻辑 tile 计算量恒定 → 各核负载均衡仅受 num_local_tasks 整除余数影响（≤1 tile 差）。
- **bs=256 与同族默认（1024/2048）的差异是 wrapper 契约使然（GPU direct 无 npt 因子）**——性能上若实测证明小 tile 开销显著（§5.2 预判），Stage 4 perf_opt 变体可在同签名内 clamp 内部 tile 尺寸（block_size 参数仍被接受与尊重语义：作为最小分块粒度），该决策权在 Stage 4 实测后行使。

### 5.5 分核策略（物理核数适配）⭐

**要素① 逻辑核数计算**：`num_logical_kernels = ceildiv(N, bs)`（1-D 输出网格，bs=256）。manifest workloads：smoke-1m（N=1,048,576）→ **4096**；16m（N=16,777,216）→ **65,536**；64m（N=67,108,864）→ **262,144**；256m（N=268,435,456）→ **1,048,576**。

**要素② 物理核数依据（实查，禁假设值）**：

- 查询代码（落地 kernel 内，工厂期现查）：`vector_cores = NPUUtils.get().get_aicore_num() * 2`（纯 Vector 算子翻倍——core-split-strategy.md §1 要素②）。
- **本设计期实查记录（2026-10-29）**：设计环境 tilelang 不可导入（dev-root 私有 TVM 包缺失，与 CONST-aicore-910B2C 条目记载环境一致），按该条目记载的等价实查路径 ctypes 直调 CANN libruntime：`python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)"` → **ret=0, aicore_num=20**（与该条目 910B3 数据点五次独立复跑一致：leaky 2026-10-28 双证 / elu 第三复跑 / erf 第四五次复跑 2026-10-29）；纯 Vector 翻倍 → **40**。权威查询 NPUUtils.get().get_aicore_num() 由 Stage 3 kernel 工厂期在设备上执行（预期 20，异常即报环境错误，不猜测）。
- 版本戳：CONST-aicore-910B2C 910B3 数据点（tilelang 0.1.2+88ee8c302e→bd7c60c / CANN 9.0.0 / libruntime 25.5.0 / 2026-10-28~29）+ 本任务 2026-10-29 ctypes 复跑一致。〔该条目在 160 条 stale 清单内——工具链漂移风险标注；本条已在**当前环境**实测重验通过，重验承诺已履行。〕

**要素③ 规模判定与分核方案**：全部 manifest workloads 满足 `num_logical ≫ 40`（最小 smoke-1m 亦 4096 ≫ 40）→ **极大规模**：固定启动内核数 = `num_kernels = min(num_logical, 40)` ≤ 物理向量核数；核内 `T.serial(num_local_tasks)` 串行处理多逻辑块（`num_local_tasks = ceildiv(num_logical, num_kernels)`——**静态值**：N、bs、num_kernels 均 Python int，trace 时折叠 IntImm）；grid-stride 映射 `block_id = i·num_kernels + cid` + `if block_id < num_logical` 掩码；无可调 block 使核数对齐物理核整数倍的意义（逻辑核数远超核数，负载不均仅尾核 −1 tile）。方案 = core-split-strategy.md §1 要素③「极大规模」标准形态（persistent 核内串行，循环边界静态），与 expm1/erf §0.6 R1 同构（PL-1.22 模板第二证 910B3）。

## 6. 循环与调度结构

### 6.1 循环结构总结

| 层 | 结构 | 选择依据 |
|----|------|---------|
| 核间 | `T.Kernel(num_kernels, is_npu=True)`（一维，persistent） | §5.5 要素③ |
| 核内任务 | `T.serial(num_local_tasks)`（静态边界 grid-stride） | 极大规模摊薄核启动开销；边界静态（S1-CORES 要求） |
| tile 内元素 | **无显式循环**——1 个 v-op 处理整 (bs,) 向量 | §1.6.2 V1–V3 全向量化结论；T.Parallel 不使用（Developer v-op 链形态，vec_add_1d 同款） |

T.Pipelined / T.Persistent 不显式使用：Developer 模式 auto-multi-buffer 覆盖 tile 间流水（copy-floor 证据：向量链隐藏于 MTE2 窗口——本算子单 v-op 链隐藏裕度最大）；T.Kernel 静态 num_kernels 本身即 persistent 语义。

### 6.2 循环伪代码

见 §3.3（三方一致：§1.6.3 布局 = 1-D 连续尾轴 / §4.3 缓冲形状 = (bs,) / 本节结构 = serial×guard + 整 tile 单 v-op）。

### 6.3 流水线优化

- tile 间：auto-multi-buffer（Developer 默认）使下一 tile 载入与当前 tile 计算/写回重叠；CONST-copy-floor-method〔stale，重验承诺〕实证 7-pass 链 ≥16M 全隐藏——1-op 链暴露量预期更低，列 Stage 4 copy-floor 探针复测项（§9 R5）。
- tile 内：vnot/vxor 与载入/写回的 RAW/WAR 依赖由 Developer 自动同步串行；无手动 double-buffer 需求。

### 6.4 尾块处理

`tail = T.min(bs, N − t0)`；v-op 整 buffer（stale lane 垃圾位型经 vnot/vxor 仍为垃圾位型——**不落 GM**，store 仅 [0:tail]）；动态 tail 正确性实测通过（事实 1）。N=1 退化 tile（tail=1）由 T.copy 零原点切片覆盖。

## 7. 同步策略

### 7.1 同步模式

**Developer 自动同步**：RAW/WAR 依赖由编译器插入 pipe 屏障，无手动 set_flag/wait_flag/pipe_barrier；无跨核同步（每 tile 独立、无核间归并）。

### 7.2 同步点说明

隐式同步点 = 每 tile 内 copy→v-op→copy 的 RAW/WAR 序（§3.3 顺序即依赖序：vnot/vxor ← x_ub 载入；store ← y_ub；x_ub 重载入 ← 上轮 store WAR——由自动同步覆盖）。与编程模式匹配（Developer 自动同步——门禁「同步」项依据）。跨引擎（Cube/Vector）同步：不适用。

### 7.3 pass_configs 配置

无显式 pass_configs（默认 Developer 行为）；模块级 `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")` 自锁（同族同款）。若 Stage 4 需关闭 auto-multi-buffer 做对照，经 perf_opt 阶段 env 注入，不改产品代码默认。

## 8. 验证方案

### 8.1 Golden 函数

```python
def golden_bitwise_not(x: torch.Tensor) -> torch.Tensor:
    """bitwise_not 参考：torch.bitwise_not（源测试锚 ref_program，test_bitwise_not.py L45–46）。

    精确整数/布尔位语义——独立于 NPU 算法（不含 vnot/vxor 路线、int8 位视图、
    persistent 结构知识）。CPU 运行，不需 NPU 设备。
    """
    return torch.bitwise_not(x.detach().cpu())
```

独立性：以 §0.1 语义为依据（源仓 ref_program 移植），未复刻 NPU 的 dtype 路线/位视图/persistent 结构——验证不 self-证。优先移植源仓参考实现 ✓（torch.bitwise_not 即源测试锚）。

### 8.2 精度标准

- **精确比较**：位运算全程精确，无浮点容差——断言 = `torch.equal(out, ref)`（源测试 `exact_compare`，test_base.py L50–51：「Bitwise NOT is exact integer/bool arithmetic, so the reference uses exact comparison (no fp32 upcast, no tolerance)」）。
- dtype 断言：输出 dtype 必须等于输入 dtype（bool/uint8 路线的 view 还原正确性由 dtype 相等性直接把关——**uint8 路线若误回 int8/float32 即在此拦截**，实测缺陷 fingerprint 检查）。
- bool 值域断言：bool 路线输出存储 ∈ {0,1}（view(torch.bool) 合法性）。

### 8.3 L0 门槛测试计划（gate；L1/L2/Boundary 全套交 tilelang-op-develop）

1. **L0-1 smoke**：6 dtype × N=2^20，workload 同款输入分布（bool `rand>0.5` / uint8 `randint(0,256)` / int `randint(-128,128)`，workloads/elementwise.py L59–67），默认 bs=256，exact_compare。
2. **L0-2 大规模**：int32 N=2^24 + int16/int64 N=2^23（多轮核内串行 + ones_ub 跨 tile 复用覆盖；对齐 manifest 64m/256m 档）。
3. **L0-3 非整除尾**：int8/int32/int64 × N ∈ {1000, 4097, 5000}（bs=256：tail ∈ {232, 1, 184}——**N=1000/bs=256 即实测事实 1 的已验证角点**）。
4. **L0-4 tiny**：N ∈ {1, 7, 255, 256, 257}（单非满 tile / tile 边界精确覆盖 / num_logical=1 退化）。
5. **L0-5 角点值**（逐 dtype，禁手写溢出字面量——`torch.iinfo` 取极值）：int8 {0,±1,−128,127,0x55,0xAA}；uint8 {0,1,255,128,0x55,0xAA}；int16/int32/int64 {0,±1,min,max,0x55..,0xAA..}；bool 枚举：all-False / all-True / 交替；全 0 / 全 1 张量各一。exact + dtype 断言。
6. **L0-6 persistent 边界**：num_logical ∈ {39, 40, 41}（bs=256 → N ∈ {39×256, 40×256, 41×256+1}，vc 运行期实查）—— 少核 / 满核 / 掩码+不均任务。
7. **L0-extra**：E2 双 bs 复用（256/2048 同工厂）；UB guard 拒绝 bs=8193；N≤0 / 非法 dtype（float16/32 拒绝——源 test_bitwise_not_rejects_float_dtype 语义）拒绝；**位视图 fingerprint**（uint8 输出 `dtype==torch.uint8` 且全值域 roundtrip 一致——实测缺陷 2 的回归拦截；bool 输出存储 ∈ {0,1}）；**原生 bool 拒绝**（factory 对 bool 不走原生域——事实 3 回归拦截：若未来误改为原生 bool，L0-1 bool 项挂死/超时即暴露）。

## 9. 风险点与注意事项

| # | 风险 | 等级 | 缓解 / 验证路径 |
|---|------|------|----------------|
| R1 | T.vxor 标量 src1：docs §2.1 声称支持（「src1…或 Python 标量 int/float」），本工具链实测拒绝（事实 4）——**文档与实测冲突** | 低 | 设计采用实测已验证 shaped buffer 形态（fill + 三缓冲）；若 Stage 4 想用标量形态省 fill，须先重验（负向断言举证规则）；T.fill 无专页 docs——examples 双证（deepseek_v4 L381 / deepseek_v32 L96）+ 事实 4 直证，回退 = T.vbrc(let 绑定标量) 物化 ones_ub（leaky zero32 模式） |
| R2 | 原生 bool 域：docs 明文禁止 GM 落盘（T.vnot.md/T.vxor.md §2.3）+ 实测编译通过但执行挂死 200s（事实 3）——「能编译」造成假阳性风险 | 已规避 | factory 级强制 int8 位视图路线（编译期分支，原生 bool 无代码路径）；L0-extra 原生 bool 回归拦截（8.3#7） |
| R3 | uint8 out_idx=[1] 输出分配 float32（事实 2，tilelang JIT 分配缺陷） | 已规避 | int8 位视图路线（A2）；L0 dtype fingerprint 拦截回归（8.3#7）；若工具链修复该缺陷，view 路线仍位等价（无迁移负担） |
| R4 | bool 契约外存储值（{0,1} 之外）两式分叉（xor→truthy vs ==0→0） | 低 | torch.bool 契约外为未定义行为；workload/test 输入契约内（gen_inputs rand>0.5）；披露于 §1.6.1 A3 与本行，不做兼容 |
| R5 | bs=256 小 tile 的 per-task 固定开销在 256m 档（26,215 tasks/核）可能抬高实测时延（估算下界未含该项——**未实测假设**：per-task MTE issue + guard 开销无 constants.md 条目；估算依据 = PL-1.21 同族实测/下界倍率 2.3×（4-op 链）作为上界参照） | 中 | D-2 已报告区间 + 倍率锚；Stage 4 msprof 对账（hardware-cost-model §3 偏差行，>2× 触发归因）；缓解 = config sweep（bs 是 config 键）/ perf_opt 同签名 clamp 内部 tile；mte2_ratio ≥0.95 判据（PL-1.6）判定标量开销是否已隐藏 |
| R6 | 160 条 pattern-library 条目 stale（工具链漂移）——本设计引用 CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-copy-floor-method / CONST-capacity-910B2C / CONST-aicore-910B2C 均 stale | 中 | 逐条「引用条目前置重验」标注 + 验证路径：CONST-aicore **已重验**（本任务 ctypes 实测 ret=0→20）；CONST-mte2/launch/copy-floor/capacity → Stage 4 copy-floor 探针 + msprof 对账（§3 偏差行）；不引用 status=overturned 条目（已核对 INDEX；调度 prompt 预注入 5 条目中 TRAP-tvm-parser-rules/CASE-mish/2 条 VP 为 queue-pending 与设计低相关，未作为决策依据） |
| R7 | C5「无查表先例」与「位运算无更省 op 形态」为模型知识级断言（未文档化假设） | 低 | 收益方向为负（查表访存劣化）；即使存在等价形态，1-op 已是 docs 六位运算 op 的最小发射数，带宽主导下不可辨；列待确认不阻塞 |
| R8 | int64 向量速率无独立实测常数（CONST-vector-launch-overhead 仅 f16/f32 口径「逐元素速率 f16≈f32」） | 低 | 未文档化假设：位运算速率与 dtype 位宽弱相关（VNotOp 逐 lane 执行）；16m int64 workload bench 直接对账（D-2 行 222µs）；偏差 >2× 走 §3 偏差行 |
| R9 | N=0 / 负 N / dtype 不合法 / bs 超预算 | 已规避 | 工厂 host 防御 ValueError（§0.6 R4；Op 层 numel 校验之上双保险） |

## 10. 交付清单

### 10.1 目录结构

```
examples/bitwise_not/bitwise_not_fwd_kernel/   # 本 Stage 1 工件（conductor 编排目录）
├── DESIGN.md                                  # 本文档
├── verify_equiv.py                            # D-1 等价性机器验证（EQUIV_PASS, exit 0）
└── RETROSPECTIVE.md                           # Stage 1 复盘（Phase 8）

examples/TileOPs/tileops/kernels/elementwise/bitwise_not/
├── bitwise_not.py                             # wrapper（Stage 5 仅切 L64 import，零编辑）
├── _bitwise_not_fwd_kernels.py                # GPU 抽取参考（保留）
└── bitwise_not_kernel/                        # Stage 3 产品目录（同族 expm1_kernel 布局）
    ├── bitwise_not_fwd_kernel.py              # kernel + golden + L0-L2/Boundary 嵌入式测试
    └── history_version/
```

### 10.2 文件清单

| 文件 | 阶段 | 说明 |
|------|------|------|
| DESIGN.md | Stage 1（本阶段） | 设计冻结基线 |
| verify_equiv.py | Stage 1（本阶段） | 已执行，3 采纳项 8 行全 EQUIV_PASS；Stage 2 重跑复核 |
| bitwise_not_fwd_kernel.py | Stage 3 | 按 §3.3 伪代码 + §3.5.2 契约实现（factory 级 dtype 三路线 + persistent + 位视图包装） |
| REVIEW.md | Stage 2 | 检视 |
| perf_opt/ | Stage 4 | copy-floor 探针 + msprof（R5）+ bs sweep |

### 10.3 命名规范

工厂 `bitwise_not_fwd_kernel`（与 wrapper import 名一致）；golden `golden_bitwise_not`；产品文件 `bitwise_not_fwd_kernel.py`；常量 `_BLOCK_SIZE`（wrapper 契约 256）/ `_UB_MAX_BLOCK` / `_DEFAULT_BLOCK = 256`（同族命名族）。

### 10.4 实现顺序

1. 工厂骨架 + dtype 归一化 + 路线分派 + host 防御（§3.5.2）→ 2. int 域 vnot 路线 → 3. uint8/bool 位视图路线（builder view 包装 + fill/vxor）→ 4. persistent 分核 + 尾块 → 5. golden + L0 套件（§8.3，含位视图 fingerprint 与原生 bool 拦截）→ 6. L1/L2/Boundary → 7. Stage 5 import 切换（bitwise_not.py L64，零 wrapper 编辑——import 名单已预核对无死 helper，leaky RETROSPECTIVE 教训履行）。
