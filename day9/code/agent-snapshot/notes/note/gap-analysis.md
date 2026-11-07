# TileLang 对 NPU 的表达能力评估、缺口与演进路线

日期：2026-11-05
样本：AminFwdOp、MaxPool1dIndicesFwdOp、FusedAddRMSNormFwdOp、FFTC2COp、GatedDeltaNetPrefillFwdOp
方法：以 5 个真实算子的迁移/调优为样本，逐条记录“想表达什么 → 当前 API 能不能表达 → 绕过方案 / 性能天花板”。
证据规则：文档路径 + 限制条款，或实测报错原文；无法实证的标注“待验证”。

> 缺口编号 G1–G14 为本报告临时编号；与仓库 `.agents/evolution/capability-gaps.md` 的 CG-2026-xxxx 对应关系在条目内注明。
> 工具链版本戳：tilelang 0.1.2（dev root）/ CANN 9.0.0 / Ascend910B3。跨版本复用前需重验。

---

## 1. 结论摘要

TileLang npuir 目前**能覆盖“规整、稠密、单算子、fp16/fp32”的绝大多数 vector 形态**，足以把 5 个样本算子全部跑通；真正的缺口集中在四类：

1. **数据类型面**：bf16 无向量算术、fp64 完全缺失、无复数类型 → 直接决定 bf16 只能 transit、complex128 FFT 不可达。
2. **索引/访存面**：无运行时索引 gather、无 strided vector load、int64 索引运算受限 → 池化索引、FFT bit-reversal、动态形状吃力。
3. **调度/流水面**：MTE2 与 VEC 不重叠、双缓冲收益被 flag 握手吃掉、最慢核拖尾 1.38× → 这是 **ratio 上不去的首要原因**，比“缺某个算子”更关键。
4. **编译期健壮性**：bf16 静默 no-op、UB 精确拟合溢出 64 B、动态 extent 向量 op 报 `int() Var`、2D grid 静默串行 → 属于“应该报错却在静默降级”的一类，最消耗调试时间。

对应地，语言演进应优先做 **第 3 类（流水/调度原语）**，因为它同时决定 3 个纯 vector 算子能否从 ~10–30% 走到 60%。

---

## 2. 能力现状（npuir 向量 API 面）

来源：`tilelang/language/customize_npuir.py`（v-prefix 别名同源）。

| 类别 | 已有 | 缺失/受限 |
|---|---|---|
| 二元算术 | add/sub/mul/div/max/min/pow/or/and/xor/shl/shr | 无 fused multiply-add 语义（`vaddcmul`/`vfma`） |
| 一元 | exp/exp2/log2/ln/sqrt/rsqrt/abs/rec/relu/sigmoid/not | 无 log1p、无 expm1 原生（靠组合） |
| 超越 | vcos/vsin/verf/vtanh | 无 atan2、无复数旋转 |
| 比较/选择 | vcmp/vselect/clamp/min/max | 无三目融合的 mask 链优化 |
| 规约 | reduce(sum/prod/max/min/abssum/absmax/any/all/xori)**且 IR 层支持 `max_with_index_left/right`、`min_with_index_left/right`** | 语言层未暴露 arg-reduce 便捷 API；索引返回结构未文档化 |
| 索引/重排 | gather（**索引只能是编译期 list/tuple**）、transpose/flip/pad/concat/interleave/deinterleave/reshape/bitcast/arange | 无运行时索引 buffer 的 gather、无 strided load/store |
| 归约/扫描 | cumsum | 无 segmented scan |
| 同步 | set_flag/wait_flag/pipe_barrier/block_barrier/subblock_barrier/sync_block_set/wait | 无 async DMA copy 原语（`T.copy` 为同步语义） |
| 原子 | atomic_add、atomic_addx4 | 无 atomic max/min |
| 精度 | fp16/fp32 向量算术 | **bf16 ×、fp64 ×、complex ×** |

---

## 3. 缺口清单

### G1 bf16 向量算术缺失，且缺失时静默 no-op（阻塞级，已确认）
- **层**：Frontend API dtype 矩阵 + BishengIR legalize。
- **证据**：`docs/Tilelang.language/数学操作/T.vabs.md:25`、`T.vexp.md:26` 的 dtype 矩阵中 bf16 为 `×`；`src/transform/npu_loop_vectorize.cc:124` 的 `IsFloat16Or32` 只认 16/32 位浮点。
- **实测后果**：未做 fp32 transit 的 bf16 逐元素算子在当前工具链上**不执行、原样返回，编译器不报错**（外部工作流 7 次实验复现；本仓 erf/leaky_relu 的 DESIGN 也已固化为“bf16 必须 transit”）。
- **阻塞场景**：所有 bf16 元素级/规约算子；本队列 RMSNorm、GDN 的 bf16 格子。
- **绕过**：`vcast(rint)` 升 fp32 → 计算 → `vcast(rint)` 落回（本仓标准做法）。
- **代价**：UB 足迹 2–3×（fp16/bf16 6 B/el 模型），block 尺寸被压小；吞吐按 f32 计。
- **建议**：legalize 阶段对不支持的 dtype 直接报错（硬失败），禁止静默丢弃。

### G2 fp64 向量路径完全缺失（阻塞级，已确认）
- **层**：Frontend API dtype 矩阵。
- **证据**：CG-2026-0018（`T.vexp.md §2.2.1` fp64 ×、`T.vcast.md` 规格表无 f64 行、`T.vbrc.md` dtype 列无 fp64）；本日复核 `T.vabs.md:25` 同样 fp64 ×。
- **阻塞场景**：**FFTC2COp 的 complex128（atol/rtol 1e-8）不可达**；complex64 需要 fp32 旋转因子与 fp32 级联，误差约 2.5e-5（n=4096），可过 1e-4 但裕度不大。
- **绕过**：double-double（two-sum/two-product）模拟，代价约 4–6× 计算与 2× 存储；或把 c128 workload 标为不支持。
- **建议**：短期在 manifest 层明确 complex128 不支持；中期提供 fp64 或 compensated 原语。

### G3 arg-reduce 在语言层未暴露（阻塞级，但可能只是 API 表层问题）
- **层**：Frontend API。
- **证据**：`customize_npuir.py:783` 的 `npuir_reduce` 文档列出 `max_with_index_left/right`、`min_with_index_left/right`；而 `tilelang/language/reduce.py` 只提供 `reduce_max/min/sum/abssum/absmax` 便捷封装，`T.reduce` 的 `reduce_type` 直接透传，**未文档化索引结果怎么取**。
- **阻塞场景**：MaxPool1dIndices（int64 indices、first-valid 与 NaN 语义）、任何 argmax/top-k。
- **绕过**：vcmp + vselect 手工维护 (value,index) 双缓冲，指令数 ×3–4。
- **建议**：暴露 `T.argmax/T.argmin`（返回 value+index 或 index-only），明确 tie-break 与 NaN 规则；这是一条“IR 有能力、前端没开口”的缺口，性价比最高。

### G4 gather 只接受编译期索引（高）
- **层**：Frontend API / BisignIR。
- **证据**：`customize_npuir.py:1146` 签名 `indices: Union[list, tuple]`。
- **阻塞场景**：dilated/strided 池化的窗口取样、FFT bit-reversal 重排、GDN 的动态索引。
- **绕过**：把索引固化成多个常量 list（每 shape 编译一次）；或 interleave/deinterleave + 多 pass 重排（UB 流量 ×2）。
- **建议**：支持 UB 中的索引 buffer（运行时索引），并在 pass 层向量化。

### G5 strided vector load/store 缺失（高）
- **层**：Frontend API。
- **证据**：`npnir_load_nd2nz` 只覆盖 2D NZ 布局；`T.copy` 的切片要求连续/对齐。
- **阻塞场景**：`MaxPool1d k5 s2 p2 d2`（步长 2、膨胀 2）、FFT 偶数/奇数抽取。
- **绕过**：整段搬入 UB 再用 gather/deinterleave。
- **建议**：`T.copy_strided(src, dst, strides)`。

### G6 2D grid 第二维静默串行（高，已实测）
- **层**：TileLangIR pass。
- **证据**：项目记忆 `npuir-1d-grid-required`：`T.Kernel(gx, gy)` 的第二维不并行，需手工压成 1D `cid//gy, cid%gy`。
- **阻塞场景**：GDN 的 `(batch, head, chunk)` 三维任务空间；FFT 的 `(n_blocks, batch)`。
- **建议**：前端把多维 grid 自动压平并做 swizzle，避免“看起来并行、实际串行”的性能陷阱。

### G7 动态 extent 的向量 op 不支持（高，已实测）
- **层**：TileLangIR pass。
- **证据**：`T.vcast` 作用在动态长度切片上报 `int() argument must be ... not 'Var'`（前序会话复现）。
- **阻塞场景**：变长序列（GDN 的 cu_seqlens）、尾块处理。
- **绕过**：N 先 pad 到静态倍数，所有 tile 走全尺寸路径，尾块靠 mask/越界丢弃。
- **建议**：支持动态 extent 的分块向量化，或提供 `T.tail_guard` 原语。

### G8 2D `T.Parallel` 在 bm≥2 时整体标量化（高，已登记）
- **层**：TileLangIR pass（npu_loop_vectorize）。
- **证据**：CG-2026-0015，scalar ratio 0.84–0.95，222 µs vs 47 µs（4.7×）。
- **阻塞场景**：row-reduction 的融合候选、多行条件选择。
- **绕过**：显式向量 op 链，或 per-row (1,W) 展开（op 数 × bm）。

### G9 multitile + bm>1 触发编译器 SIGSEGV（高，已登记）
- **层**：TileLangIR pass。
- **证据**：CG-2026-0014，触发矩阵 bm=1 任意 ✓ / bm>1 单 tile ✓ / bm>1 多 tile ✗。
- **代价**：tiled 归约被迫 bm=1，小 M 负载核利用率被锁死在 M/核数。

### G10 UB 预算模型与实际 planner 存在小量级偏差（中，本日实测）
- **层**：BishengIR PlanMemory。
- **证据**：bf16 (1, 32768) 按 6 B/el slab 模型恰为 196608 B，实际要求 196672 B（多 64 B），报 `ub overflow, requires 1573376 bits while 1572864 bits available`。
- **影响**：精确拟合的 tile 会在编译期失败，且错误信息指向“tiling basic block too large”，难以定位到 64 B。
- **绕过**：预留固定余量（本日改回 8 KiB 安全余量）。
- **建议**：报错里给出各 buffer 的占用明细。

### G11 无复数类型与 FFT 原语（高）
- FFT 需要 real/imag 双实数组 + 旋转因子 LUT；当前只能手工展开。
- 建议：complex64 作为 `(N,2)` fp32 一等类型 + `vcmul/vfft_stage` 原语。

### G12 host 侧 launch 开销 250–450 µs，NPUGraph 不可用（高，影响小 shape 评分）
- 对 `M=1, N=4096/8192/16384` 的 decode 类 workload，kernel 本身几十 µs，launch 开销同量级。
- Roofline Ratio 用 msprof 的 kernel 时间计算可规避污染，但“端到端”口径不可用。
- 建议：修复 NPUGraph 在 npuir 产物上的可用性。

### G13 同步语义偏底层，双缓冲收益被握手抵消（高，性能首因）
- 外部工作流实测：算力时间与搬入时间近似相加（aiv ≈ vec + mte2）；手写双缓冲只抢回 21% 搬运时间，又被 flag 握手成本抵消；3 槽仅快 0.8%。
- **含义**：在 MTE2/VEC 不重叠的硬件路径上，增加缓冲槽不是杠杆。
- 建议：提供 **异步 DMA + 事件依赖** 的表达能力，让编译器生成 MTE2/VEC 流水，而不是要求用户手写 flag。

### G14 核间负载不均，最慢核比中位慢 1.38×（高，性能首因）
- 48 核抢同一带宽，静态持久化切分的尾核拖后。
- 建议：`T.core_map(dynamic)` / work-stealing 原语；或按 tile 成本模型做均衡切分。

---

## 4. 语言扩展路线（演进方案）

### Phase 1（1–2 周）：正确性解锁，零架构改动
| 项 | 内容 | 对应缺口 |
|---|---|---|
| P1.1 | bf16 不支持时**硬报错**，禁止静默 no-op | G1 |
| P1.2 | 暴露 `T.argmax/T.argmin`（含 tie-break/NaN 语义文档） | G3 |
| P1.3 | `T.copy_strided` + UB 索引 gather | G4/G5 |
| P1.4 | 多维 grid 自动压平 + swizzle | G6 |
| P1.5 | UB 预算报错输出 buffer 明细 | G10 |

**收益**：MaxPool1d 索引、FFT 重排、GDN 3D 任务空间可以直接写；预计省掉每算子 2–4 个手工 workaround。

### Phase 2（3–6 周）：性能原语，直接服务 60% 目标
| 项 | 内容 | 对应缺口 |
|---|---|---|
| P2.1 | `T.dma_async` + `T.dep_wait`（编译器生成 MTE2/VEC 流水） | G13 |
| P2.2 | `T.core_map(dynamic)` 动态任务队列 | G14 |
| P2.3 | 动态 extent 分块向量化 / `T.tail_guard` | G7 |
| P2.4 | 2D `T.Parallel` bm≥2 向量化修复；multitile+bm>1 SIGSEGV 修复 | G8/G9 |

**收益**：纯 vector 算子从 ~0.5 TB/s 走向 1.1–1.2 TB/s（实测 torch 上限约 1.2 TB/s），对应 ratio 从 10–30% 到 60–90%。

### Phase 3（Q2）：数据类型与融合
| 项 | 内容 |
|---|---|
| P3.1 | complex64 一等类型 + `vcmul` |
| P3.2 | fp64 或 compensated 原语（决定 c128 FFT） |
| P3.3 | cube+vector 统一调度（`T.gemm` + 向量 epilogue，AIC/AIV 配比控制） |
| P3.4 | NPUGraph 在 npuir 产物上的可用性 |

---

## 5. 需要新增的优化原语（及目标场景）

| 原语 | 语义 | 目标场景 | 现状替代 | 预期收益 |
|---|---|---|---|---|
| `T.argmax/argmin` | 归约并返回 (value,index) | MaxPool1dIndices、top-k、softmax 温度采样 | vcmp+vselect 手工维护 | 指令数 −60%，int64 索引天然支持 |
| `T.copy_strided` | 带 stride/dilation 的向量搬入 | dilated/strided 池化、FFT 抽取 | 整段搬入 + gather | UB 流量 −50% |
| `T.gather_dyn` | 运行时索引 buffer 的 gather | FFT bit-reversal、GDN 索引 | 常量索引多版本编译 | 去掉按 shape 编译爆炸 |
| `T.dma_async`/`T.dep_wait` | 显式异步搬运 + 事件依赖 | 所有 memory-bound elementwise/norm | 手写 set_flag/wait_flag | 理论上限 +21% 搬运重叠，实际取决于握手开销 |
| `T.core_map(dynamic)` | 动态任务分配/窃取 | 48 核尾核拖尾、非均匀 tile 成本 | 静态持久化切分 | 尾核 1.38× 差距 → ~1.05× |
| `T.reduce_bcast` | 归约后立即广播的融合 | RMSNorm/LayerNorm/Softmax | reduce + vbrc 两步 | pass 数 −1，UB 流量 −1 次 |
| `T.fma`/`T.axpy` | 融合乘加/标量乘加 | GDN 状态更新、FFT 蝶形 | vmul+vadd 两步 | 指令数 −30% |
| `T.cmul`/`T.cadd` | 复数乘加（(N,2) fp32） | FFT、RoPE | real/imag 手工展开 | 代码量 −50%，寄存器压力 −30% |
| `T.tail_guard` | 动态尾块安全向量化 | 变长序列、任意 N | 静态 pad 到倍数 | 零填充浪费 −(pad-N)/N |
| `T.atomic_max/min` | 原子极值 | 直方图、max 归约跨核 | atomic_add + 二次 pass | 少一个 pass |

---

## 6. 对本队列的直接映射

| 算子 | 主要缺口 | 迁移期绕法 | 调优主攻 |
|---|---|---|---|
| AminFwdOp | G10 | 8 KiB UB 余量，tile 16384×2 | 核映射与尾核均衡 |
| MaxPool1dIndices | G3/G4/G5 | vcmp+vselect 手工索引；常量 gather | 映射顺序（先验 −5.9%） |
| FusedAddRMSNorm | G1/G12 | bf16 fp32 transit；decode 形状合并 | 大 tile + 减少 pass（不做双缓冲） |
| FFTC2C | G2/G4/G11 | c64 手写蝶形；c128 标注不支持 | 片上整段 FFT，减少 GM 往返 |
| GDN Prefill | G6/G7/G8/G9 | 1D grid 压平；workload 裁剪 | cube/vector 配比与峰值占用 |
