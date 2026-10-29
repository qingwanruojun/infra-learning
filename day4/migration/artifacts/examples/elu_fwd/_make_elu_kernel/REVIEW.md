# _make_elu_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/elu_fwd/_make_elu_kernel/DESIGN.md（836 行，revision 1 后）
- 任务类型: 迁移（migration/harness，函数级）
- 源算子: GPU 原件 `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2496–L2560（`_make_elu_kernel` + `EluFwdKernel`）+ `tileops/ops/elementwise/activations.py` L195（`EluFwdOp`）；NPU 侧 verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/elu_fwd/_elu_fwd_kernels.py`（全 63 行）；落地 wrapper（接口契约权威）`examples/TileOPs/tileops/kernels/elementwise/elu_fwd/elu_fwd.py`（全 202 行）——**全部亲自 Read 完毕**
- 检视维度: 9 项（0–8：源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling 分核 / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（不阻塞，见「检视注记」）
- 工件注入扫描：所读源算子/提取件/wrapper/DESIGN.md 中未发现任何指令性文本（DESIGN §0 头部自述「按数据处理、未发现亦不执行」与本轮独立通读一致）。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑（D-1）**：`python3 examples/elu_fwd/_make_elu_kernel/verify_equiv.py` 实跑（2026-10-28，torch 2.7.1 CPU）→ **RESULT: EQUIV_PASS**。与 §1.6.1(b) 内嵌表**逐格一致**：A1 fp32-transit 6 行（fp16/bf16/fp32 × α∈{1.0,0.5}）cand-vs-golden **0 ulp / 0 violations**、cand-vs-base 0 ulp；与 §1.6.3.1 弃选归档一致：A3 α=1 特化 3 行 0 ulp（bitwise 安全、不采纳留杠杆）、A2 expm1 16692/16165 mismatches（α=1.0/0.1）、A4 max-form 32765 mismatches / **26990 violations（EQUIV_FAIL，结构性不等价）**、A5 重构式 15179、A6 fp16 原生域 16448 mismatches / max 32767 ulp。脚本缺失/失败/不一致 → 无。**等价性机器验证 pass**。
- **design_calc_check（D-5）**：`python3 .agents/tools/design_calc_check.py --design examples/elu_fwd/_make_elu_kernel/DESIGN.md` JSON 摘要逐项消费：
  - `ub_budget`: **skip**（buffer 表按「字节/元素」口径未被机械解析）→ **机械跳过，人工已核 pass**：transit 路线 x_ub(2)+x32(4)+w32(4)+cond(1)+zero32(4)+alpha32(4)=**19B/elem**、fp32 路线 4+4+1+4+4=**17B/elem**（与 §3.3 伪代码 buffer 清单、§4.3、§4.5 三方逐 buffer 复算一致）；UB 上界 196608//19=10347、//17=11565 ✓；保守 cap 8192 → resident 155.6KB/139.3KB < 192KB ✓；默认 block 1024/2048 双缓冲峰值 34.8KB/77.8KB 在含 ×2 朴素模型在内的一切口径下安全 ✓。
  - `l0c_budget`: **fail（机械误判，人工复核 pass）**——detail「block 19x8192 x fp32 accum = 608KB」为解析噪声：核触发了设计 §8.2 **否定句**中的字面 token `Scope("Cube")`（checker 正则 `T\.gemm|load_nd2nz|store_fixpipe|Scope\(...Cube` 不识别否定语境，见 `.agents/tools/design_calc_check.py` L163–L173），随后把 §4.5「19B×8192」UB 算术按 L0C 累加器公式复算。本算子**纯 Vector、零 Cube API、L0C 用量 = 0**（§3.5.1/§5.1；全文无任何 affirmative Cube 调用）——L0C 检查 n/a，人工判定 pass。leaky_relu 同构设计本轮此前判 skip（其 §8.2 措辞未含该 token），同一实质、不同机械表观，佐证误判定性。
  - `core_split`: **skip**（dims 解析不足；「declared logical cores = 32」为 host 侧 `lru_cache(maxsize=32)` 字样解析噪声，与 leaky_relu REVIEW 先例同形态）→ **机械跳过，人工已核 pass**：要素①复算 2²⁰/1024=1024、2²⁰/2048=512、2²³/2048=4096、2²⁴/2048=8192 ✓（与 §5.5 表一致）；`num_kernels = min(num_logical, 40)` ✓；`num_local_tasks = ceildiv(num_logical, num_kernels)`：512/40=12.8→**≤13** ✓、4096/40=102.4→**≤103** ✓、8192/40=204.8→**≤205** ✓，负载不均 ≤1 任务/核 ≤0.5% ✓，全部边界为工厂特化编译期常量（静态）✓。要素②物理核数依据——**检视者独立复跑等价实查路径**：ctypes `rtGetAiCoreCount`（libruntime.so，CANN 9.0.0）→ **ret=0, count=20 → 翻倍 40**，与 §5.5 记录逐字一致；`NPUUtils.getAiCoreNum()` 封装 `rtGetAiCoreCount` 经 `tilelang/utils/npu_utils.cpp` L124–L137 亲读确认；翻倍规则出处 `core-split-strategy.md` §1 要素②（L14）亲读确认；极大规模档方案与要素③标准一致 ✓。
  - `r3_metrics`: **pass**（四口径 FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值齐备，基线在表）。
- **kb_stale_check 实跑（检视者复核项）**：`python3 .agents/tools/kb_stale_check.py --json` → 当前 checker stamp **`tilelang 0.1.2+f388fab185`**（≠ 设计标注 88ee8c302e，工具链已漂移）。判定：**CONST-mte2-degradation、CONST-copy-floor-method = STALE（设计已自认并按「设计期估算 + Stage 4 实测校准」口径引用 ✓ 合规）；PL-1.6-copy-floor、PL-1.17-subfp32-fp32-transit、TRAP-UB-multibuffer-inflation 当前亦判 STALE**——设计 §9.1#6「同一 kb_stale_check 复核非 stale」的表述在写作时点（88ee8c302e）成立，本轮漂移后复跑不再复现。处置依据「stale 不自动否定决策 + 设计须持首跑重验承诺」：三条目 front-matter `status: verified` 亲验无误，且 PL-1.17/TRAP-UB 条目 origin_task 已含 leaky_relu 2026-10-28 **910B3/CANN 9.0.0 上板重验存活注记**（结构面+平台值口径跨大版本存活）、TRAP-vselect-inplace-carried-state 本轮复跑非 stale；设计持承诺（§4.5「Stage 4 调大 block 前编译探针/BishengIR ub overflow 报文复算 + 连带复验 num_kernels」、§9.1#6「Stage 4 msprof 实测校准回写」、§9.1#4/§9.1#12 首跑兜底用例）→ **决策不被 stale 否定**，建议级注记 c 登记（Stage 3 首跑留意）。
- **分核三要素（维度 3）核对**：见 core_split 人工复核行——三要素齐全、实查独立复跑一致、静态边界、负载均衡 ≤1 任务/核。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 检视方式：亲自 Read GPU 原件 L2480–L2560（`_make_elu_kernel` L2497 + `EluFwdKernel` L2548 + `ParametricUnaryKernel` 基类 L2322–L2423）、verbatim 提取件全 63 行、NPU wrapper 全 202 行、`activations.py` L195–L234（EluFwdOp）、test_elu.py、manifest。逐项结论均附源码核对证据。

- **语义理解正确性：pass**。§0.1 分段公式/严格 `>` 与源码一致（提取件 L54–L55/L2538–L2539：`T.if_then_else(v32 > zero, val, T.Cast(val.dtype, a*(T.exp(v32)-one)))`）；「比较在 fp32 域 + 负支 fp32 域计算 + 单次舍回 + 正支原生直传」逐语句核实（L50–L55：`zero/a/one` 均 `T.cast(·,"float32")`、`v32=T.cast(val,"float32")`、`T.Cast(val.dtype, ...)` 一次舍回，正支操作数为未转换的 `val`）；±0/NaN/±Inf/exp 下溢（−104）溢出（±89）传播推演逐值复核成立；**ATen golden 条件方向差异**（`x<0` 选负支 → −0.0 走恒等支返回 −0.0）本轮以 torch 2.7.1 实测复核：`F.elu(tensor([-0.0]),1.0)` 输出 copysign=−0.0 ✓，源实现 −0.0 走负支得 α·(1−1)=+0.0 ✓——「IEEE 值相等、零符号差豁免」论断有据（§8.3 L0-5 断言规则按此定制，与 leaky_relu sign-bit strict 的差异化理由成立）。§0.2 I/O 与签名/manifest 一致：`out_dtype = output_dtype or dtype`（L17/L43）→ 非 fp8 路径 y dtype same_as input ✓；alpha default 1.0（原件 L2551/Op L218）✓；`threads=256, npt=8` 签名默认（L2497–L2498）且「GPU 基类实际 fp32 npt=4」经基类 `default_config`（原件 L2406–L2412）与 `_strategy_npt` 注（L185–L188）亲验 ✓；roofline flops=4N/bytes=2N·elem_bytes 与 manifest L168–L174 + test_elu_eval_roofline（L107–L115 断言 `4*N`/`2*N*itemsize`）一致 ✓。边界语义：非 fp8 分支整块覆盖无逐元素 guard、manifest N 均被默认 block 整除（2²⁰/2²³/2²⁴ ÷ 1024/2048 整除复算 ✓ →「源实现实际从未执行 partial tail」成立）；fp8 分支 `if idx < N` guard（L29）为死代码；源无 N==0 拦截（Op 层校验 `numel != N_total`）→ NPU 侧防御性 ValueError 为**明示适配**非语义偏离 ✓。
- **算法解读完整性：pass**。§0.3 步骤分解 S1/S2/S3 覆盖源码全部计算语句（载入 L47、展开计算 L48–L56、写回 L57），数据流图覆盖全部 buffer（x_reg/y_reg + 常量内联），无臆造步骤；host 侧逻辑（白名单/K11 折叠/custom_op/Kernel 类构建/展平/inplace copy-back/alpha finite 校验）与 wrapper 实文逐条对上；「源 builder 无 lru_cache」归因句存在源侧事实错误（见注记 a——GPU 原件 L2496 **有** `@functools.lru_cache(maxsize=32)`，提取件被剥离；NPU 侧补 lru_cache 的动作恰与 GPU 原件行为一致，决策不受影响）。
- **优化手段识别：pass**。O1–O6 与源码逐项对上（fragment L45–L46、连续 tile L47/L57、两级 `T.Parallel` L48、fp8 分支 L20–L37、fp32-opmath L50–L55、闭包常量 L18）；「无 SMEM/mma/warp shuffle/异步流水/online/persistent」经本检视通读两版源码确认；O5 判为唯一算法层决策（dtype 路径）并与 leaky_relu 的「修正型」正确区分为「结构承接型」——该区分是 A1 cand≡base 0 ulp 的解释性前提，本轮独立复核成立（源与 golden 同为 fp32-opmath+单次舍回路径）。
- **耦合性判定抽检：pass**。8 条目四态处置齐备且依据可核：#4「block×thread → 重新设计」映射表条款引用一致；#5 fp8 舍弃的证据链完整——wrapper `SUPPORTED_DTYPES=(fp16,bf16,fp32)`（L151）+ K6 注记「the NPU kernel builder rejects is_fp8=True」（L28–L30）+ manifest dtypes 三选，均亲验；#3 意图承接引用 PL-1.6/CONST-mte2-degradation「grid-stride 优于每核连续分块 +3.2%@256M」与条目实文逐字一致；#8 尾 guard → tile 级 tail 有官方 vec_add_1d（L33–L38 零起点切片亲读）+ leaky_relu 已验证先例。处置统计 4 保留/2 等价替换/1 重设计/1 舍弃无矛盾。
- **NPU 重设计可行性：pass**。R1（一维 persistent + v-prefix 全 buffer 链）满足一维 Kernel/UB 层级/静态边界约束；R2（fp16/bf16 fp32 transit）满足「bf16 无 vexp/vmul/vsub」硬约束（三文档 §2.2.1 本轮逐一亲读：T.vexp.md L28 bf16 ×、T.vmul.md L27 bf16 ×、T.vsub.md L27 bf16 ×；fp16 vexp √ 亦亲验）；语义保持论证独立推演通过——升域无损保序→比较一致、同序同域逐 op 单次舍入、select-then-round ≡ round-then-select（正支 rint(x32)=x 恒等、负支同一舍入对象）、逐元素映射 lane 重排不改结果、tail `[0:tail]` 拷出屏蔽陈旧/溢出 lane（mish 全 buffer v-op 先例亲读 L76–L81）；「严格强于源」声明核实（源依赖整除，NPU 覆盖任意 N）成立。
- **设计与决策一致性：pass**。§1.2/§1.4/§2/§3/§4/§5/§6/§7 描述的均为 §0.6 重设计后的 NPU 算法；无「§0 判重设计、§3 照抄源方案」脱节；GPU 基类两参调用式（原件 L2417 `self.kernel(cfg["threads"], cfg["num_per_thread"])` 亲验存在）仅出现在 §0.3 源侧事实叙述与 §9.2 防呆禁令中，**未进入任何 NPU 决策行**——leaky_relu 阻塞级教训的正确承接。
- **接口契约核对（wrapper 逐字，维度 0 关键项）：pass**。设计 §0.7/§9.2 对 wrapper 的全部断言命中：单参调用 `_make_elu_kernel(N, dtype, alpha)(block_size)(x)`（其 L100）✓；torch.dtype 入参（L99 `getattr(torch, dtype_str)`、L175 `self.dtype`）✓；K11 折叠 `_BLOCK_SIZE_FP32=1024`/`_BLOCK_SIZE_NON_FP32=2048`（L82–L83）✓；`custom_op("npub::elu_fwd")`（L91）✓；`ascend_mode="Developer"`（L148）✓；`SUPPORTED_DTYPES`（L151）✓；E2 `kernel(block_size=default_block)` 与 lerp/leaky_relu 已验证件同形（leaky_relu 实现 L194 亲读）✓；dtype 归一化 `str(dtype).split(".")[-1]` 的依据（torch 2.7.1 无 `.name`）本轮实测复核：`hasattr(torch.float16,'name')=False`、`str(torch.float16)="torch.float16"` ✓，且与 leaky_relu 已验证实现 L169 逐字同式 ✓。
- **golden 独立性：pass**。§8.1 golden = `F.elu(x.detach().float(), alpha).to(x.dtype)`，移植自源仓测试锚点（test_elu.py **L40 逐字命中**）；无 vcast/vcmp/vselect/persistent 等 NPU 算法知识，非 §0.6 复刻；参数化 alpha（L70–L77）、inplace（L81+）、rejects（L95–L103）用例语义承接核实。

### 1. API 可行性 — pass

§3.2/§3.4 全部 API 具备文档 + 先例双佐证，且本轮逐条亲验：`T.vexp`（§2.1 src/dst 同形同 dtype 约束 → 独立 w32 缓冲的设计动机成立；§2.4 示例）、`T.vsub`（§2.1 src1 标量 + §2.3 第 3 条原地 + 第 2 条标量须在 Kernel 内定义——与设计一致；mish L78 标量字面量生产先例亲验，如实标注「vsub+标量+原地」三要素组合无逐字先例并配回退方案 §9.1#10，处理合规）、`T.vcmp`（"gt" 在 cmp_mod 列表；bool 不可写 GM 条款亲验 = 设计 §3.5.1 规避项依据）、`T.vselect`（§2.2.1 fp16/fp32/bf16 √；Out=B/Out=A 原地双形态均有实证：leaky_relu 实现 transit `vselect(cond,x32,sx32,sx32)` / fp32 `vselect(cond,x_ub,sx32,x_ub)` 上板通过 + gqa_prefill **L875 精确命中** `T.vselect(ub_cond, ub_f32_ND, ub_neg, ub_f32_ND)`）、`T.vcast`（§2.2.1：f16→f32 仅 rint、bf16→f32 仅 rint、f32→f16/bf16 含 rint——设计 round_mode="rint" 全部合法）、`T.vbrc`（§2.2.1 fp32 √ + §2.2.2 scalar 源无 shape 限制；体内 let 绑定要求以 leaky_relu 实现注释 + `repro/TRAP-vbrc-scalar-let-binding.py` 实证件承接 ✓）、`T.copy` 零起点切片（vec_add_1d 官方模式亲验）、`T.alloc_ub "bool"`（gqa L709 与 sparse_mla **L230 精确命中** `T.alloc_ub([1,block],"bool")`）。v-prefix 优先合规；Developer 模式与全链自动同步 API 匹配，无手动 flag。

### 2. 内存层级规划 — pass

搬运路径 GM↔UB 直连（纯 Vector，MTE2/MTE3，无越级、无 L1/L0 介入）✓；全部 buffer 标注 UB 且形状 `(block_size,)` 一维 ✓；UB 预算人工复算见「机械复核」ub_budget 行（17/19B/elem × 8192 = 139.3/155.6KB < 192KB，默认 block 34.8/77.8KB 双缓冲全口径安全）✓；auto multi-buffer 占用按 TRAP-UB-multibuffer-inflation 平台值口径论证（×2 朴素模型显式判为「非平台实测口径」，guard 8192 的依据是 leaky_relu resident 同值 bs=8192 L1 实测先例 + Stage 4 编译探针复算条款）——口径引用与条目实文（transit≈20–21/fp32≈26–27 B/elem、第五源 leaky_relu 重验注记）亲验一致 ✓。l0c_budget 机械 fail 判为误解析（见机械复核），维度内实质 n/a pass。

### 3. Tiling 策略 — pass

Block 划分（E2 单参 block_size，默认 1024/2048 = wrapper 折叠常量同源，UB cap 8192 + host ValueError guard）✓；对齐/非整除处理（零起点切片 + tile 级 tail，覆盖 N<block、尾=1、非 32B 对齐尾）✓；分核三要素齐备且自洽——逻辑核数公式与复算、物理核数实查记录（本检视独立复跑一致：ret=0→20→40）、极大规模方案（固定 40 核 + 静态边界 grid-stride + 掩蔽）、负载不均上界 ≤1 任务/核，全部数字见「机械复核」core_split 行；与 `core-split-strategy.md` §1 三要素标准逐项对表成立；「失败信号」三条供 Stage 3/4 对照，形态合规。

### 4. 技术约束检测 — pass

三维 Kernel/L0C/GEMM 非整除/L1 分形均明示不涉且与 §5.1 纯 Vector 判定一致；bool/i1 不落 GM（vcmp.md 条款亲验）；bf16 v-算术缺失经三文档矩阵亲验后以 transit 绕开；一维核数超发以 persistent 消化；in-place 为 Op 层语义、kernel 无感知（契约保持）。N=0 防御为适配项已标注。无遗漏检测项。

### 5. 循环与同步 — pass

§6.1 三层循环结构与 §3.3 伪代码逐字一致（`T.Kernel(num_kernels)` / `T.serial(num_local_tasks)` 静态边界 / tile 内无显式循环——v-prefix 全 buffer 隐式并行）；无 Pipelined/Parallel 的理由（Developer 自动 stage 化 + 全 buffer 向量形态）与模式选型自洽；§6.4 尾块方案完整；§7 同步点表与 Developer 自动同步一致、零跨核同步判定成立（tile 两两不相交，无 workspace）；vbrc 常量循环外一次性填充与 leaky_relu 已验证形态同构。原地 v-op 链审计（检视要求 #5）：⑤⑥⑦ 输出仅同迭代被 ⑧/⑨ 消费、每 tile ①/④ 全量重载重算、**无 loop-carried 状态**——与 TRAP-vselect-inplace-carried-state 触发边界（running 类携带 + num_full≥3）逐字对表后豁免判定成立；lerp 原地 vsub/vmul/vadd 链（perf_opt 件 L262–L264 亲验，persistent 多迭代 bit-exact）+ leaky_relu Out=B/Out=A 双形态 910B3 多迭代实测先例支撑；§9.2 已写入跨迭代携带禁令 + 诊断指纹；vbrc 标量体内 let 绑定防呆三处落笔（§3.2/§3.3 L440–L441/§9.2）✓。

### 6. 验证方案 — pass

Golden §8.1 独立可跑（CPU，锚点来源 test_elu.py L40 亲验）；精度标准表与 `_get_tolerances` 逐值一致（fp32 1e-5/fp16 1e-3/bf16 1.6e-2）；L0 六项 + 附加计划具体（形状 N∈{1000,4097,5000}/{1,7,32}/分核边界 39·2048/40·2048/40·2048+1、2²⁰/2²⁴、全 dtype、角点集含 ±89 溢出 lane/−104 下溢带/1e-8 消除区/max-finite 从 torch.finfo 取、α=0.5、E2 双配置复用、UB guard/is_fp8/alpha 防御用例）——覆盖本设计全部特异性风险（vexp 屏蔽 lane 溢出、±0 值等价豁免、多迭代原地链、fp8 死代码拒绝）；对 vexp 硬件 ULP 的主张边界如实标注（D-1 bitwise 主张限于公式/舍入路径，上板 vexp 差异由容差 + Stage 3 门禁承接）——不放水也不过度主张。

### 7. 完整性与一致性 — pass

章节齐全（§0.1–0.7、§1–§10 含 §1.6 四子节 + §1.6.3.1 归档）；占位符扫描零命中（禁用未填措辞；`{op}.py` 为交付清单文件名模式，与 leaky_relu 先例同规）；§3.5.3/§0.7 引用的 examples/manifest/test/bench 路径 Glob 全部真实存在（含 verify_equiv.py 本尊、leaky_relu 全套、mish L77/L78、gqa L875/L709、vec_add_1d、lerp perf_opt、ada_layer_norm）；内部一致性重点复核通过：拷出源=x_ub 两路线五处（§1.4/§1.5/§3.1/§3.3/§4.3/§4.4，另加 §1.6.3 结论行）交叉一致；17/19B 与 §3.3 buffer 清单一致；E2/dtype 归一化/lru_cache 条款 §3.3/§3.5.2/§9.2/§10.3 多处一致。建议级注记 a/b/c 见下，不阻塞。

### 8. 算法优化分析 — pass

- **§1.6 结构存在性：pass**。1.6.0/1.6.1/1.6.2/1.6.3 + 1.6.3.1 弃选证据归档五段齐备；弃选证据行按 gate region 规则移出 §1.6.1（revision 1 修复项），本轮核对 §1.6.1 仅含采纳项 A1 结果表、无失败判定字样 ✓。
- **算法调研完整性（四问）：pass**。R1 候选表 8 行含基线（候选 0 在表），命中 `algorithm-candidates.md` ALG-elementwise 条目（本轮亲读 L98–L104）的 R1 手段清单逐项对照——CSE（#7 结构论证无重复子式）、除法转乘倒数（本式无除法，n/a 有据）、cast 链合并（→ R2 单次舍回最小链）；另主动扩展 expm1/max/重构式/α=1/fp16 原生/exp2 降代换五候选，超出清单下限。R2「天然单遍」给出结构判据（y[i] 仅依赖 x[i]）且与源单遍事实一致 ✓。R3 四口径表齐（r3_metrics 机械 pass + 人工复核：2N·B = I/O 下界，本设计无中间 GM buffer，往返漏计检查 n/a ✓）。R4 逐检查项评估，容量项引用 §4.5 复算值、带宽淘汰未用峰值参考（hardware-cost-model「混合读写禁用峰值」条款遵守，引用 CONST-mte2-degradation 曲线值 + 跨设备缩放标注未实测假设 + Stage 4 校准回写；canonical 估算下界行格式与协议 §2 一致，容量项豁免为协议明文允许）✓。「无更优替代」调研范围六条声明（含本地源覆盖 + 联网豁免条款引用）✓。
- **调研结论正确性（独立复核）：pass**。① 等价性独立推演：A1 四要素齐，升域无损保序/同序同域逐 op 舍入/select-then-round ≡ round-then-select 三段推演本轮重新走通，且 **verify_equiv 重跑 6/6 EQUIV_PASS 0 ulp 与内嵌表逐格一致**；② 负向断言逐条复核：「vexpm1 不存在」——本轮重跑 docs/Tilelang.language/ 全 16 子目录枚举 + `grep -ri expm1` 零命中 ✓（目录数 16 亲数一致）；「bf16 无 vexp/vmul/vsub」——三文档 §2.2.1 矩阵单元格逐一亲验 × ✓；「max-form 结构性不等价」——独立推演 `e^x > 1+x (∀x≠0)` 成立 + A4 复跑 26990 容差违反与 32765/65548 元素结构自洽 ✓；「fp16 原生域偏离」——A6 复跑 16448/32767 ulp 一致 ✓；③ 复杂度独立复算与数字一致；④ 候选 4（α=1）ROI≈0 判据：A3 复跑 3 行 0 ulp（bitwise 安全属实）+ CONST-copy-floor-method 实测 delta ≤1.1µs@16M 推出 −1 op ≤0.15µs 量级自洽，「不进基线 + 机器证据留存 + Stage 4 杠杆」处置与 leaky_relu 同款 ROI 判据一致（保守方向弃选，无误伤风险）；⑤ R3 并列候选（基线 vs 候选 4）无纸面拍板——裁决路径 = copy-floor 实测 + 分支代价，合规。
- **调研→设计一致性：pass**。§1.4/§3.1/§6 = 选定算法（基线公式 + transit 落地形态）；§0.5/0.6 处置与调研选族一致（源算法族未替换、结构重设计），源优化意图 O1–O6 逐项承接表在 §0.6 且每项有落点（fragment→UB、coalesced→切片+grid-stride、ILP→v-prefix、fp8→ValueError、fp32-opmath→transit、常量折叠→闭包+lru_cache）——无静默丢弃。
- **数学等价优化（1.6.1）：pass**。采纳 1 项（A1 dtype 路径）四要素齐 + 机器验证；「公式级 0 项进基线」的结论有 R1 全表逐一弃选证据（机器/结构性/ROI 三类）支撑，非空话。
- **向量化替代分析（1.6.2）：pass**。V1–V7 与 §3.3 伪代码/§6 循环结构逐点交叉核对无遗漏（伪代码 tile 内无显式逐元素循环、无未分析的标量热点）；4 处替代均有 API+先例双佐证（与维度 1 同口径）；3 处保留理由具体到本算子（任务映射/block 级索引动态边界/host 元数据），无「实现简单」式理由。
- **向量化轴与布局决策（1.6.3）：pass**。纯 1-D 逐元素单候选轴豁免理由成立（无广播/归约/窗口轴，ND→1D 在 host 视图完成）；候选 ② 核内重排弃选带量化理由（负收益 + PL-1.1 转置链开销引用）；尾浪费 O(1)/tile 的论证不依赖未实证 lane 宽度数值（显式声明为编译器实现细节——合规的「不纸面拍板」形态）；vexp 屏蔽 lane 溢出 Inf 语义与 mish 先例、L0-5 ±89 角点、诊断指纹三处闭环；实验裁决模式不触发的判定（无未实证常数依赖）核属实。**布局决策与 §3.3/§4/§6 三方一致**：主选轴=展平连续轴 ↔ 伪代码一维 buffer ↔ UB 预算按 (block_size,) 计 ↔ 循环结构无跨步。
- **与下游章节一致：pass**。§3.1 以 §1.6.1 优化后公式（A1 式）为输入拆解；§6 与 §1.6.2 结论一致（判定向量化的点无残留标量循环）。

## 检视注记（建议级，不阻塞）

- **注记 a（§0.3 步 1 / §0.6 O6 注 / §3.5.2 末行）源侧事实归因不实**：「源 builder 无 `@functools.lru_cache` 装饰」对 verbatim 提取件为真（其 L8 `import functools` 未使用），对 GPU 原件为**假**——原件 `elementwise.py` **L2496 即有 `@functools.lru_cache(maxsize=32)`**（本轮亲读），装饰器系提取器剥离。NPU 侧「补 lru_cache」的动作恰与 GPU 原件行为一致（决策零影响），但三处措辞把「按源保留」误写为「同族模板适配」。建议 Stage 3 交付文档/后续 DESIGN 修订时将归因更正为「GPU 原件 L2496 有 lru_cache(32)；提取件被剥离装饰器；NPU 侧按原件语义保留」。
- **注记 b（文档头规格权威行）**：「tests/ops/test_elu.py（7 用例）」计数与实文不符——本轮亲数 6 个测试函数（test_elu_op/test_elu_param/test_elu_inplace/test_elu_rejects_non_float_dtype/test_elu_rejects_nonfinite_alpha/test_elu_eval_roofline；fixture 参数化实例另计）。所引具体用例名与断言全部真实存在，纯元数据计数文字项。
- **注记 c（§9.1#6 stale 表述 vs 当前环境）**：设计写作时点（stamp 88ee8c302e）「PL-1.6/PL-1.17/TRAP-* 复核非 stale」成立；本轮复跑时 checker stamp 已漂移至 f388fab185，三条目当前判 STALE（CONST-mte2/CONST-copy-floor 设计已自认 STALE）。因三条目 status=verified 且含 910B3 重验存活注记、设计持全套首跑重验承诺（§4.5 编译探针 / §9.1#4/#6 / L0-2 兜底），决策维持；Stage 3 首跑与 Stage 4 校准须把上述条目按「重验后再消费」处理（kb_stale_check 值点蒸馏时本数据点可作 stamp 漂移案例）。

## 通过依据摘要（供 conductor 路由）

9 维度全部 pass（0 阻塞）；维度 0 九项检查（含接口契约逐字核对与 golden 独立性）全 pass；维度 8 复核要点全 pass（D-1 重跑与内嵌表逐格一致、负向断言逐条亲验、四口径机械+人工双过）；分核物理核数独立复跑 ret=0→20→40 与 §5.5 一致；design_calc_check 两项表观异常（l0c fail / core_split skip）均已定位为机械误判/噪声并人工补位 pass（留痕于机械复核段）。
