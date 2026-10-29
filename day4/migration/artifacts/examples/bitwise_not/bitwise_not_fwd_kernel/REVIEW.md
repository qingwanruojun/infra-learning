# bitwise_not_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md（617 行全文读取）
- 任务类型: 迁移（TileOPs GPU `BitwiseNotFwdKernel` → TileLang-NPUIR，migration-harness 同构仓）
- 源算子: /mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py（L140–147 / L305–320 / L680–879 / L2273–2289）、/mnt/workspace/learn/refs/TileOPs/tileops/ops/elementwise/bitwise.py（L79–83）、/mnt/workspace/learn/refs/TileOPs/tileops/ops/elementwise/_base.py（L578–681）；提取件 examples/TileOPs/tileops/kernels/elementwise/bitwise_not/_bitwise_not_fwd_kernels.py（全文 67 行）、落地 wrapper bitwise_not.py（全文 250 行）——以上全部亲自 Read 核对
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4（详见各维度节内注明，不阻塞通过：①§4.5 UB guard 上限的膨胀系数口径；②§8.3 L0-3 尾块期望值算术笔误；③§1.6.0 R3 表 B0 行 tile 级 op 计数与自述口径的张力；④PL-1.21「2.3×」锚点双重语义表述）
- 工件注入防护: 源码注释 / 提取件 docstring / DESIGN.md 全文均为陈述性内容，未发现任何指令性文本，无披露事项。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: PASS。`python3 examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py`（torch CPU，参照 = torch.bitwise_not 原生 dtype 位语义）exit 0，8 行全部 EQUIV_PASS、违反数全 0：A1 int8/int16/int32/int64 各 2,032 点 0 违反；A2a/A2b uint8 各 2,032 点 0 违反；A3a/A3b bool 各 2,028 点 0 违反。**与 §1.6.1 内嵌结果表逐项一致**（点数、违反数、参照口径均吻合；脚本含契约外存储值披露行，与 §9 R4 一致）。脚本主体已通读：baseline 取 GPU 源公式（xor −1 / xor 255 / ==0），candidate 取 NPU 路线（vnot / int8 位视图 vnot / int8 位视图 xor-1），角点覆盖 0/±1/极值/0x55/0xAA 交替位型——验证语义与设计声明相符，非自证形态。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md`）:
  - `ub_budget`: **skip**（checker 未解析出分层 buffer 表）——机械跳过，人工已核：§4.5 表 bs=256 全部路线 ≤4KB、cap bs=8192 手工口径 ≤128KB，默认契约值远低于 192KB/AIV（CONST-capacity-910B2C）；**人工复核发现一处口径问题**（建议级①）：设计按 +12%（CG-2026-0008，实测于 auto-multi-buffer=**false** / r9 显式 alloc 结构）核算 cap 档（int64 128KB→143KB<192KB），但 §2.3/§7.3 声明 auto-multi-buffer **开启**，该 regime 下常数条目记载膨胀 ~1.7×（hardware-cost-model §1 Phase 4 明确要求容量复算「含 auto-multi-buffer ~1.7x 膨胀系数」）——int64 × bs=8192 在 1.7× 口径需 217.6KB > 192KB，即 `_UB_MAX_BLOCK=8192` 的 dtype 统一上限在 auto-MB 开启 regime 下不能保证 int64 cap 档可编译。默认 bs=256 与全部计划测试点（256/2048）在任何系数下均有巨幅裕度，不影响 Stage 3 契约路径，判**建议级**。
  - `l0c_budget`: **skip**（无 Cube 路径 API）——与设计一致（纯 Vector，L0C n/a），人工核 §4.4 无 L0 参与 ✓。
  - `core_split`: **skip**（解析维度不足，仅见 N=1000）——机械跳过，人工已核：§5.5 三要素齐全且数值自洽（①num_logical=ceildiv(N,256)：4096/65536/262144/1048576 复算全对，26,215 tasks/核复算 ceildiv(1048576,40)=26215 ✓；②物理核数实查：设计环境 tilelang 不可导入，按 CONST-aicore-910B2C 条目记载的等价路径 ctypes 直调 libruntime `rtGetAiCoreCount`——**检视侧独立复跑：ret=0, aicore_num=20**，纯 Vector ×2=40，为该数据点第六次独立复跑；③极大规模 persistent T.serial 静态边界，与 core-split-strategy.md §1 要素③「极大规模」标准形态及 PL-1.22 模板一致）。
  - `r3_metrics`: **pass**（四口径 FLOPs/访存/遍数/缓冲峰值齐备，基线 B0 在表）。
- **D-2 估算独立复算**（流量项，口径 = CONST-mte2-degradation 档位 1M 33.9 / 16M 30.2 / 64M 20.9 / 256M 18.4 GB/s/core + 1.26TB/s 聚合地板，条目数值与 pattern-library 原文核对一致）: smoke-1m bool/uint8 2.10MB/40=52.4KB÷33.9GB/s=1.5µs ✓；smoke int32 6.2µs ✓；16m int32 111µs / int64 222µs ✓；64m int16 321µs / int32 642µs ✓；256m int32 2.92ms、聚合口径 1.70ms ✓——7 个数字全部复算吻合。发射项 ~0.5µs/op（CONST-vector-launch-overhead ✓）；容量项豁免理由（纯 Vector 单引擎、无跨引擎 GM 往返）符合 hardware-cost-model §2「轻量调研可豁免容量项、流量项必算」✓；stale 条目均带「引用条目前置重验」标注（CONST-aicore 已重验）✓。
- **同族校准锚核对**: PL-1.21 第四例 expm1 16m fp16 实测 128.14µs ✓ 条目原文；55.6µs 下界 = 67.1MB/40÷30.2GB/s 复算 ✓（128.14/55.6=2.30 ✓）。注意（建议级④）：pattern-library 记载的「2.3×」是 **erf/expm1 链长比**（294.81/128.14），设计复用的「2.3×」是 **expm1 实测/自算流量下界**——两个 2.3× 数值巧合、语义不同，建议行文加注区分，避免后续任务误引。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read 源算子全部相关层级并与 §0 逐项核对，以下每项结论均附源码证据。

- **语义理解正确性：pass**。（源码 elementwise.py L2283–2289 `op_func`：`x.dtype=="bool" → x == T.cast(0,"bool")`；`uint8 → T.bitwise_xor(x, T.cast(255,"uint8"))`；其余 `→ T.bitwise_xor(x, T.cast(-1,x.dtype))`）与 §0.1 三分支表逐字一致；类 docstring L2276–2278「Uses XOR with -1 (all-ones) because T.bitwise_not fails on vectorized int4 CUDA types」被 §0.4 S2 精确引用（含 workaround 动因辨析）。dtype 集合 `_BITWISE_DTYPES`（L140–147：bool/uint8/int8/int16/int32/int64）与 §0.1/§0.2 一致。无规约/无跨元素依赖（单语句点映射）与 §0.1「纯逐点映射」一致。边界语义：bool 存储契约 {0,1} 的依据（workloads/elementwise.py L61–62 `rand>0.5`）实核 ✓；契约外分叉被诚实披露（§9 R4）而非掩盖。`DEFAULT_STRATEGY="direct"`（L2280）✓。
- **I/O 与调用面：pass**。Op 层 flatten→kernel→reshape（_base.py L657–661 实核 `flat = input.contiguous().reshape(-1); return self.kernel(flat).reshape(orig_shape)`）✓；输出 same_as(input)（_base.py L727 `OUTPUT_DTYPE or dtype` + wrapper bitwise_not.py L220）✓；manifest 契约条目（elementwise.yaml L266–296：signature dtype 联合、shape_rules L277–278、workloads 4 组 L280–289、roofline flops=N/bytes=2N·eB L291–296）逐项与 §0.2 表一致；**落地 wrapper 调用面逐字核对**（bitwise_not.py L147 `bitwise_not_fwd_kernel(N, dtype)(block_size)(x)` 两段式、L146 torch.dtype 对象入参、L122 `_BLOCK_SIZE=256`、L199–201 ascend_mode="Developer"、L27–28 K3 签名注释）——§0.2/§3.5.2 全部吻合，且正确识别「勿照搬 GPU 抽取件 kernel(threads_arg) 单参式」（抽取件 L54 实核为 `def kernel(threads_arg)`）。
- **算法解读完整性：pass**。direct builder（L305–320）全部计算语句 = `T.Kernel(ceildiv(N,threads_arg)) + for i in T.Parallel(threads_arg): idx=bx*threads+i; y[idx]=op_func(x[idx])` 三句——§0.3 第 2/3 条逐句覆盖，无遗漏、无臆造；无 alloc_shared/fragment（L312–316 实核无分配）✓「GM→Register→Compute→Register→Global」判定成立。host 侧逻辑清单实核：SUPPORTED_DTYPES ValueError（L711–715 ✓）、bool 强制 direct 告警（L742–751，对本 op 为 no-op——DEFAULT_STRATEGY 已是 direct，判定正确 ✓）、`_wrap_fp8_accumulation`（L785 调用，L266–277 实核 `if not _is_fp8: return base_op` → 「透传」论断 ✓）、lru_cache(maxsize=32)（L304 ✓）、config 解析 threads=256/npt 不参与 direct 路径（L819–820、L878–881 实核 direct 分支只传 threads ✓）。数据流图覆盖 x/y/寄存器，无来源不明数据 ✓。
- **优化手段覆盖完整性：pass**。S1–S6 与源码显式特征一一对应；「源码无 SMEM/mma/warp shuffle/cp.async/swizzle/persistent」的**阴性清单**经全文实读确认（direct 路径仅三句计算体，无此类语句）；实现层（S1/S2）与算法/契约层（S3）、host 层（S4/S5/S6）分类无误置。
- **耦合性判定合理性：pass**（抽检全部 7 条目）。四态处置齐备且依据可查：C2 重新设计依据「T.Kernel 一维 block 数」项目约束 + PL-1.22（910B3 已两证）✓；C3 等价替换依据三件（T.vnot.md §2.2.1 dtype 矩阵实核 uint8/int8/int16/int32/int64 √、bool ×；§3 lowering hivm::VNotOp 实核 ✓；实测事实 1 ✓）✓；C4/C5 排除原生 bool/uint8 路线的依据（T.vnot.md/T.vxor.md §2.3「bool(i1) 无法 load/store GM」实核原文 ✓ + 实测事实 2/3）✓；C7 grid 逐 tile→persistent 依据 docs/开发指南.md §3.3 + 分核标准 ✓。无舍弃项、承接核对表 S1–S6→C2–C7/R1–R3 逐项闭合，**无静默丢弃** ✓。
- **NPU 重设计可行性：pass**。R1–R4 均具「源方案→NPU 新算法→语义保持论证」三要素；满足 NPU 约束：一维 Kernel ✓、UB 驻留（bs=256 ≤4KB）✓、静态循环边界（N/bs host int 常量化）✓、tail 方案与 erf/expm1 生产 kernel 同构（实核 erf_fwd_kernel.py L227–228/L269–279：`num_kernels=min(num_logical,vector_cores)`、`for i in T.serial(num_local_tasks)`、`block_id=i*num_kernels+cid`、`if block_id<num_logical`、`tail=T.min(bs,N-t0)`、零原点切片 store——与 §3.3 伪代码逐行同形）✓。语义保持论证在逐点无跨元素依赖下成立（位运算精确，无累加顺序问题）✓。原生 bool GM 路径被排除的判定与实测事实 3（挂死）一致，且生产 erf kernel 中 bool 仅作片上中间 buffer（cond_ub）旁证「i1 可片上、禁落盘」条款 ✓。
- **设计与决策一致性：pass**。§1–§7 描述的是 §0.6 决策后的 NPU 算法：§1.4 三路线/§3.1 优化后公式/§3.3 伪代码/§5 persistent 分核/§6 循环结构均无 GPU 直译残留（§0 判重设计的 S1 SIMT 结构在 §6 已消除）。
- **golden 独立性：pass**。§8.1 golden = `torch.bitwise_not`，为源测试锚 ref_program 移植（实核 test_bitwise_not.py L45–46 ✓），不含 vnot/vxor/位视图/persistent 结构知识，未复刻 §0.6 NPU 算法，验证非自证 ✓。

### 1. API 可行性 — pass

- T.copy / T.vnot / T.vxor / T.fill / T.Kernel(num_kernels, is_npu=True) / T.alloc_ub / @tilelang.jit(out_idx=[1]) 全部有佐证：T.vnot.md（§2.1 两参签名、§2.2.1 dtype 矩阵、§2.4 整 buffer 示例、§3 VNotOp——已亲自打开）；T.vxor.md（§2.2.1 int 全宽 √/bool ×/浮点 ×、§2.3、三缓冲同 shape 同 dtype 约束、§3 VXorOp——已亲自打开，§3.4 记载与原文一致）；T.fill 无专页 docs（全 16 子目录检索确认零命中），examples 双证行号精确（deepseek_v4/inference/kernel.py L381 `T.fill(scores_max, -T.infinity(FP32))`、deepseek_v32/sparse_mla_fwd.py L96 `T.fill(m_i, val)`）+ 实测事实 4 直证——「文档声称 vs 实测冲突」（T.vxor.md §2.1 标量 src1 有载但本工具链拒绝）被 §3.2/§9 R1 **显式披露并按实测设计**，处置正确（负向以实测为准 + vbrc 回退路径）。
- v-prefix 优先 ✓（AGENTS.md API Convention）；Developer 模式、无手动 set_flag/wait_flag ✓（§7 与 §2 模式一致）。
- 候选淘汰合理性：六 op 全集（逻辑操作目录实数 T.vand/vnot/vor/vshl/vshr/vxor 六文件）→「vand/vor/vshl/vshr 合成取反 ≥2 op 反向劣化」成立；T.vcast 判定「无浮点 transit 需求」与整数域语义一致。

### 2. 内存层级规划 — pass（附建议级问题①）

- §4.4 搬运路径 GM→(MTE2)→UB→v-op→UB→(MTE3)→GM 完整，无越级；纯 Vector 无 L1/L0 参与判定正确。
- §4.3 buffer 全部标注 UB 层级与复用关系（ones_ub 核级物化跨 tile 复用——与 bool 路线形态一致，生产 erf 的核级常量 buffer 同构）。
- §4.5 UB 预算：默认契约 bs=256 全部路线 ≤4KB，任何膨胀系数下裕度巨大，**阻塞判定通过**；design_calc_check ub_budget skip 项人工复核结论见「机械复核」段。**建议级问题①**：`_UB_MAX_BLOCK=8192`（全 dtype 统一）的 guard 论证使用 +12% 列（CG-2026-0008，auto-multi-buffer=**false** regime / r9 显式 alloc 实测），与本设计 auto-MB **开启**（§2.3）的 regime 不一致——1.7× 口径下 int64×8192 需 217.6KB>192KB，guard 可能放行编译失败配置。建议：cap 按 dtype 分档（1.7× 口径 int64 上限 ≈7168）或在 builder guard 处注明「宽 config 上靶前以编译探针校准」（CONST-capacity 条目原文即有此纪律）。契约路径与既有测试计划（256/2048）不受影响，不阻塞。

### 3. Tiling 策略 — pass

- §5.2 block=256 有明确契约依据（wrapper K11 `_BLOCK_SIZE=256` 实核 L122；GPU direct threads=256/npt 不参与折叠口径经 L819–820/L878–881 实核成立）；非整除 = tile 级 tail 动态方案（§0.6 R3），纯 Vector 非整除说明到位。
- 分核三要素（core-split-strategy.md §1）：① 逻辑核数计算四档数值复算全对；② 物理核数依据——设计期 ctypes 等价实查（CONST-aicore 条目记载的等价路径，含查询命令与返回 ret=0→20 记录；权威 NPUUtils 查询留工厂期、预期值与异常处理写明不猜测），**检视侧独立复跑一致（第六证）**；③ 极大规模 persistent 判定与方案，核数（40）与 block（256）取值自洽、循环边界静态（S1-CORES 要求）。
- 尾轴对齐：1-D 连续 + MTE 段搬运处理切片边界（vec_add_1d 官方模式实存 examples/elementwise/vec_add_1d.py，实核）；bs=256 小 tile 开销预判披露为未实测假设并挂 Stage 4 sweep（§9 R5）——诚实且路径完整。

### 4. 技术约束检测 — pass

- 三维 Kernel / GEMM 非整除 / L0C 均不涉及（一维、非 GEMM、纯 Vector）；ascend-constraints.md §2 强制检测清单逐项核对与 §3.5.1 声明一致（分核项已补全三要素）。项目特有限制（bool i1 禁落盘 + 实测挂死、uint8 out_idx 分配缺陷、vxor 标量拒绝、v-op 切片仅静态边界）逐项有处置方案且引用实测事实编号，无遗留。

### 5. 循环与同步 — pass

- §6 循环结构三层（核间 persistent / 核内 T.serial 静态 / tile 内无显式循环整 tile v-op）选择依据明确，与 §3.3 伪代码、§1.6.2 结论三方一致；不显式用 T.Pipelined 的理由（Developer auto-multi-buffer 覆盖 tile 间流水 + 1-op 链隐藏裕度，CONST-copy-floor-method 佐证）成立。
- §7 同步 = Developer 自动同步，与模式匹配；隐式同步点序（copy→v-op→copy RAW/WAR）描述正确；无跨核同步需求（逐点独立）判定成立。
- §6.4 尾块：stale lane 不落 GM + [0:tail] 零原点 store，动态 tail 正确性实测背书（事实 1：N=1000/tail=232，复算 ✓）。V7 guard 的「纯名常量 guard」用语与 PL-1.22 模板原文一致（模板既有表述），生产四例存活，不另计。

### 6. 验证方案 — pass（附建议级问题②）

- §8.1 golden 独立且为源测试锚移植（实核）。
- §8.3 L0 计划具体：6 dtype×N=2^20、大规模多轮核内串行、非整除尾 {1000,4097,5000}、tiny {1,7,255,256,257}、角点（torch.iinfo 取极值、禁手写溢出字面量）、persistent 边界 {39,40,41} 核（N 换算复算：39×256/40×256/41×256+1 对应 num_logical 39/40/41 ✓）、extra 项含 E2 双 bs、UB guard 拒绝、非法 dtype 拒绝（test_bitwise_not_rejects_float_dtype L56–65 实核语义一致）、**位视图 fingerprint**（uint8 dtype 回归拦截）与**原生 bool 拦截**（挂死回归）——实测事实 2/3/4 全部转化为可执行回归项，这是本设计的亮点。
- 精度标准：全整数/布尔位运算 → **精确比较 torch.equal**（源 exact_compare 实核 test_base.py L50–51 定义、引文语句出自 test_bitwise_not.py L7–8 文档串——行号与语句归属有一处混写，微瑕不计问题）；dtype 契约断言 + bool 值域断言补齐。atol/rtol 默认规则适用于浮点域，本算子整数域用精确比较**强于**默认口径，判定合规。
- **建议级问题②**：§8.3 L0-3 期望 tail 集合「{232, 1, 184}」中 N=5000 的尾块应为 5000−19×256=**136**（184 为算术笔误；1000→232 ✓、4097→1 ✓）。kernel 运行期自算 tail、测试断言走 golden 对拍，不影响有效性，建议 Stage 3 抄录测试计划时径改为 136。

### 7. 完整性与一致性 — pass

- 模板章节齐全：§0（0.1–0.7 全在，含 0.7 标杆）+ §1（1.1–1.6，1.6.0–1.6.3 四子节齐）+ §2–§10 ✓；占位符/未完成标记全文扫描零命中。
- 同类引用真实：§0.7/§3.5.3 全部 examples/ 路径 Glob 实核存在（expm1/erf/leaky_relu/elu_fwd landed kernel 与 erf DESIGN 模板、manifest/test/bench/workload 四件、vec_add_1d.py）；同族 Developer 自锁声明实核（四 landed kernel 均含 `os.environ.setdefault("TILELANG_ASCEND_MODE","Developer")`）。
- 内部一致：§1.3/§1.4/§3.1/§3.3/§4.3/§5.5/§6 与 §0.5/§0.6/§1.6 决策同口径（工厂签名 (N,dtype)、K3 丢弃项、两段式调用面、三路线、位视图包装、tail 机制、分核数值——多处交叉复算无矛盾）；§10 交付清单与 §3.5.2 Stage 5 零 wrapper 编辑契约一致（import 名单 L93/L64 实核无残留 helper）。

### 8. 算法优化分析 — pass（附建议级问题③④）

- **§1.6 存在性与结构：pass**。1.6.0/1.6.1/1.6.2/1.6.3 四子节齐；调研深度与算子复杂度匹配（单步逐元素→轻量调研口径，四问逐问有结论且候选表/四口径表完整给出）。
- **§1.6.0 调研完整性：pass**。R1 候选表含基线 B0 与 C1–C5（vnot/bool-vxor1/vcmp/标量循环/LUT），ALG-elementwise 命中行建议项（CSE/除乘倒数/cast 链合并）逐一有「无对象/不适用」结论——候选覆盖无遗漏；调研范围声明具体（docs 全 16 子目录枚举 + 逻辑操作六 op 全开 + 同族 4 landed + pattern-library 检索 + 补码恒等式自查）。R2 有明确「无在线变体」结论 + 结构依据（零边依赖图）+ 基线已单遍——与参考表「逐元素映射天然单遍」一致。R3 四口径齐、基线在表、口径写明（design_calc_check r3_metrics pass）。R4 逐候选评估六检查项齐。结论含选定算法 + 关键依据 + 「无更优替代」调研范围 ✓。
- **调研结论正确性（独立复核）：pass**。① 等价性独立推演：A1 补码恒等 `~x ≡ x⊕(2^w−1)`（任意 w 位含极值，逐位成立）✓；A2 uint8/int8 同字节表示 + 取反与符号解释无关 + view 零拷贝 ✓；A3 {0,1} 域 `b⊕1 ≡ ¬b ≡ (b==0)` ✓；契约外分叉披露与处置（不兼容、跟踪项）正确。② 负向断言复核：「无在线变体」对照参考表命中行无矛盾；「i1 禁落盘」亲验 T.vnot.md/T.vxor.md §2.3 原文 ✓（C3 淘汰前提成立）；「位运算 op 全集=六者」亲验逻辑操作目录 ✓；「无查表先例」显式标注未文档化假设且淘汰方向为负（R7 不阻塞）——符合负向断言举证规则。③ 复杂度复算：访存 2N·eB 与单入单出结构自洽、无中间 GM 往返漏计（全部候选同访存——判定改由可行性/发射主导，读法正确）；D-2 流量项 7 数字独立复算全对（见机械复核段）。④ 无 R3 并列候选，无实验裁决需求判定成立（C1 在全部口径弱支配或等于 B0，bool 路线唯一可行）；1.6.3 单轴豁免论证符合 skill 豁免条款（1-D 扁平、I/O 契约即连续尾轴、位视图非第二核内布局——三点核对项均成立）。**建议级问题③**：R3 表 B0 行「v-op 数/tile=2」与表头自述口径「fill 为核级一次性」存在张力——若 B0 直译同样采用核级一次性常量物化（与 bool 路线 ones_ub 同构），其 tile 级发射数亦为 1，A1/A2 收益应表述为「−1 常量 buffer −1 核级 fill + 操作数简化」而非「2→1」；不影响选型结论（C1 仍弱支配），建议口径修正。
- **调研→设计一致性：pass**。§1.4/§3.1/§6/§0.5–§0.6 与选定三路线算法同口径互引（§0.5 判定输入含 §1.6.0、§0.6 对照调研结论——Phase M0→R→M1 执行序声明与内容自洽）；迁移任务源优化手段承接核对表逐项闭合无静默丢弃。
- **§1.6.1 数学等价优化：pass**。三采纳项四要素齐（原式→优化后→等价论证→收益量化）；机器验证 D-1 内嵌表与重跑结果一致（8/8 EQUIV_PASS）；「无更优化空间」五项显式结论各有依据（CSE 无对象/无浮点稳定化需求/六 op 最小发射/访存信息论下界/零归约）；收益栏数字有出处（CONST-vector-launch-overhead，含问题③口径注）。**建议级问题④**：§1.6.0 校准锚「expm1 16m fp16 实测 = 流量下界 2.3×」中 2.3× 为设计自算比值，与 PL-1.21 条目记载的「2.3×」链长比（erf/expm1）数值巧合而语义不同，建议加注防止误引。
- **§1.6.2 向量化替代：pass**。V1–V7 覆盖 §3.3 伪代码与 §6 全部循环/标量点（交叉核对无遗漏：位运算链/常量物化/SIMT 结构→替代；block 索引/host 元数据/动态 tail/掩码 guard→保留），保留理由具体到本算子（O(1)/tile 非热点、零拷贝 reinterpret、动态 shape 边界 + traps-compiler「v-op 切片仅静态边界」P10 条款实核、任务掩码必需）且均属可接受理由类别，无「实现简单」类空泛理由；替代 API 佐证齐（docs 逻辑操作双页 + T.fill examples 双证行号精确 + 实测直证）；§6 无残留逐元素标量循环 ✓。
- **§1.6.3 布局决策：pass**。单轴豁免声明 + 四点核对项（广播轴不适用/GPU 轴选择是输入非结论/位视图不引入第二布局/launch 开销与位宽无关）齐；与 §3.3/§4.3/§6 三方一致；弃选前提核对：本设计弃选论断（i1 落盘禁令、vand/vor/vshl/vshr ≥2op、LUT 流量劣化）亲验文档/目录无矛盾；判定不依赖未实证常数（无并列裕度），实验裁决三件套豁免成立。
- **与下游一致：pass**。§3.1 明示「§1.6.1 优化后公式 = 本节唯一输入」且内容吻合；§6.1 循环结构与 V1–V3 全向量化结论一致。

## 结论依据汇总

- 阻塞级维度（0/1/2/3/4/6/8）全部 pass；建议级维度（5/7）pass；4 项建议级注记均不阻塞 Stage 3（契约路径 bs=256 下无编译/运行/精度风险敞口）。
- 实测事实 1–4 全部被设计承接并转化为验证/拦截项（事实 1→C3/R3/动态 tail 测试；事实 2→C5/A2/dtype fingerprint；事实 3→C4/R2/L0-extra 原生 bool 拦截；事实 4→C4/R2 路线 3 shaped buffer + T.fill 形态）。
- 机械复核双件：verify_equiv 重跑与内嵌表一致全 PASS；design_calc_check 仅 r3 机械 pass，其余 skip 项已人工复核并给出结论（1 项建议级发现）。
