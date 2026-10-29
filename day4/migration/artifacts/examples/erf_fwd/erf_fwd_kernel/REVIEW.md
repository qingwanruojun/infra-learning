# erf_fwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/erf_fwd/erf_fwd_kernel/DESIGN.md（629 行，完整读取）
- 任务类型: 迁移（migration-harness，TileOPs GPU → TileLang-NPUIR，同构仓）
- 源算子: /mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py L2118（ErfFwdKernel）+ L344（_make_unary_regcopy）+ L680–879（UnaryKernel）+ L1297（FloatUnaryKernel）；tileops/ops/elementwise/math_unary.py L207–211；tests/ops/test_unary_math.py；落地侧 wrapper examples/TileOPs/tileops/kernels/elementwise/erf_fwd/erf_fwd.py（219 行逐字）；_erf_fwd_kernels.py；NPU test/bench；manifest elementwise.yaml L225–260——**以上全部亲自 Read**
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（已内联于维度正文，不阻塞；供 Stage 3/4 参考与后续修订捎带处理）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**——`python3 examples/erf_fwd/erf_fwd_kernel/verify_equiv.py` 于本检视环境执行，exit 0，OVERALL: EQUIV_PASS。**与 §1.6.1 内嵌表逐位一致**：fp32 6,000,023 点 / 0 违反 / max|err| **5..868e-07**（7 套件明细 5.835e-07 / 5.868e-07 / 5.850e-07 / 5.621e-07 / 4.442e-08 / 1.778e-07 / 9.715e-08 全对）；fp16 5,000,023 / 0 / 2.443e-04、尾域 ulp 0.00、全域 ulp ≤3；bf16 5,000,023 / 0 / 1.951e-03、尾域 0.00；A2 标量恒等断言 PASS；REJ-1 表（0.5→9.8e-06、1.0→4.48e-03、1.5→1.451e-01、2.0→1.576、3.0→39.11）与内嵌表逐位一致且**与用户 910B3 实测 {0.5:1e-5, 1:4.5e-3, 1.5:0.15, 2:1.6, 3:39} 逐点吻合**；GOLD-DIAG fp16/bf16 位等价 200000/200000。near-zero/randn 套件 ulp=4194303 为度量伪影，DESIGN 已自我标注并以包络违反数 0 为准——复核认可该处置（erf 在 0 附近平坦、atol 主导）。**转录错误专项**：脚本 `as26_fp32_chain` 含 `m = t * m`（A&S 前导 t 因子未漏乘，Stage 1 复盘记载首版曾漏乘已由 D-1 拦截）；与 §3.3 伪代码的乘法结合序差异（脚本 (poly·e)·t vs 设计 (t·poly)·e）为 fp32 ulp 级差，两者均在 D-1 覆盖内，无语义差。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/erf_fwd/erf_fwd_kernel/DESIGN.md`）JSON checks 摘要：
  | check | status | detail | 人工复核结论（skip ≠ 免检） |
  |---|---|---|---|
  | ub_budget | skip | no per-level buffer tables parsed（设计用 B/elem 口径表，机械解析器不识别） | **人工重算 pass**：fp32 = 6×4+3×4+1 = 37 B/elem ✓；transit = 5×4+2+4+3×4+1 = 39 ✓；回退 +4 = 41/43 ✓。bs4096：151,552B(+12%=169.8KB) / 159,744B(+12%=178.9KB) < 192KB ✓；回退列 43×4096=176.1KB(+12%≈197KB) **> 192KB——设计已如实标 ✗ 并给出处置（回退时 transit 收紧 bs≤2048，§4.5）**，fp32 回退 168×1.12=188KB ✓。常量缓冲 9→3 的算术（§1.6.1 A2）有 1 缓冲偏差：移除 6 缓冲 = −49.2KB@bs2048 / −98.3KB@bs4096（6×bs×4B），设计记 −56/−112KB（对应 7 缓冲）——**建议级注记**，不影响任何决策（37 B/elem 口径与 bs4096 可行性结论不变）。 |
  | l0c_budget | skip | no affirmative Cube-path API | **合理 skip**：纯 Vector 设计（§5.1/§4.4），无 T.gemm/load_nd2nz/Scope(Cube)，L0C 不适用 ✓。 |
  | core_split | skip | insufficient parsed dims {'N': 1}; declared logical cores = 1 | **人工补位 pass**：① 逻辑核数复算 ✓（smoke-1m：1048576/1024=1024、/2048=512；16M：16384/8192；64M：32768；256M：131072——与 §5.5 全对）；② 物理核数：设计记载 2026-10-29 ctypes 实查 ret=0→20（第 4 次复跑）；**检视者独立复跑同探针 ret=0→count=20 → 纯 Vector ×2 = 40**（`rtGetAiCoreCount`，cann-9.0.0 aarch64 libruntime，2026-10-29，与本条目 910B3 数据点一致——CONST-aicore-910B2C 条目规定的 core_split skip 标准补位手段）；③ 极大规模判定 ✓：最小 workload 512 ≫ 40 → persistent `num_kernels=min(num_logical,40)`、`num_local_tasks=ceildiv` 编译期静态（N/bs Python int 折叠 IntImm）、grid-stride `block_id=i·num_kernels+cid` + 纯名 guard；抽验 smoke-1m fp16：512/40→13 任务×40 核，尾核掩码 8 空任务 ✓；落地查询 `NPUUtils.get().get_aicore_num()*2` 有 expm1 生产先例（landed kernel 实读确认）。 |
  | r3_metrics | pass | all four R3 metrics present（FLOPs/访存/扫描遍数/中间缓冲峰值，基线在表） | 口径语义亦复核 ✓（见维度 8）。 |
- **kb_stale_check**：`current stamp: tilelang 0.1.2+d7f990d0f2 + CANN ?`，**stale entries: 154**——与 §9 R4「154 条 stale」声明一致；本设计引用的 CONST-capacity-910B2C / CONST-vector-launch-overhead / CONST-mte2-degradation / CONST-copy-floor-method / CONST-aicore-910B2C / CONST-vsub-crossbuf-scalar / TRAP-expert-v-operands **全部在 stale 清单内，且设计逐条持有「引用条目前置重验」标注 + 验证路径**（aicore=已重验〔设计+检视双复跑〕；vsub/TRAP=L0 首跑即暴露；mte2/launch/copy-floor/capacity=Stage 4 copy-floor 探针 + msprof 对账）。stale 不自动否定决策，本设计满足「决策必须持有验证路径」要求 ✓。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read GPU 源码全文区段与 NPU 落地侧全部引用文件，与 §0 逐项核对。

- **语义理解正确性 pass**：§0.1 `op_func = T.erf(T.cast(x,"float32"))` 与 GPU elementwise.py **L2125–2127 逐字一致**（含类 docstring「half-precision intrinsic herf is not a valid CUDA built-in」佐证 fp32 中转是语义契约而非可选实现——DESIGN 将其定性为「中间精度契约」正确）；erf 数学定义/值域/奇函数/边界语义（±0 保号、±inf→±1、NaN→NaN、大 |x| 饱和、subnormal≈0）经独立推演全部成立；golden 锚 `ref_fn = torch.erf` 与 GPU test **L402 / NPU test L221 实读一致**；NPU test `ref_program` = `self._ref_fn(x.float()).to(x.dtype)`（**L164–168 实读**）= fp32-opmath 形态，§0.1 一致性声明 ✓。
- **I/O 与调用面 pass**：§0.2 表逐项核对 wrapper 实文：L117 `erf_fwd_kernel(N, dtype)(block_size)(x)` 两段式 curried ✓（与 GPU 抽取件 `_erf_fwd_kernels.py` L20 `kernel(threads_arg, npt_arg)` 两参式**正确地不混同**，并警示「勿照搬 GPU 两参式」）；L91–92 `_BLOCK_SIZE_FP32=1024`/`_NON_FP32=2048` ✓（与 GPU `_strategy_npt` L190–195 fp32→4/fp16→8 × threads 256 折叠自洽，源码独立复算 ✓）；L187 `output_dtype = dtype` ✓；L168 `ascend_mode="Developer"` ✓；L116 传 torch.dtype 对象 → 归一化要求 ✓；manifest elementwise.yaml L225–260 实读：shape_rules `output.shape == input.shape` ✓、3 dtype ✓、workloads smoke-1m/16m/64m/256m 与 §1.2 逐项吻合 ✓、roofline `flops: N / bytes: 2*N*elem_bytes` 与 §1.6.0 R3 口径一致 ✓。〔建议级注记：§0.2「erf_fwd.py L86–87」实指 ops 树 `ops/elementwise/erf_fwd.py` L85–87（flatten/reshape 真身，已核属实），与 kernels 树同名文件的 L186/L187 混写一行引用——内容无错，跨树行号引用建议加树前缀。〕
- **算法解读完整性 pass**：§0.3 六点与 regcopy builder **L344–363 逐句对照**：`T.Kernel(T.ceildiv(N, block_size), threads=threads_arg)` 一维 grid ✓、`alloc_fragment ×2` ✓、`T.copy → T.Parallel(threads,npt) y_reg[i*npt+j]=op_func(...) → T.copy` 三步计算原语 ✓（无遗漏：L354–359 仅这些语句；无臆造）；`block_size=threads*num_per_thread` L346 ✓；host 侧 `input.contiguous().reshape(-1) → kernel → .reshape(orig_shape)` 在 GPU `_base.py` UnaryOp（L660–661 区段）与 NPU ops erf_fwd.py L84–87 双树实证 ✓；`lru_cache(maxsize=32)` L343 ✓；fp8 helper 不生效（FloatUnaryKernel SUPPORTED_DTYPES=L1300/_FLOAT_DTYPES L154–158 仅 3 float）✓；DEFAULT_STRATEGY="register_copy" L700 ✓。数据流图覆盖全部 buffer（x/y GM + x_reg/y_reg fragment）✓。
- **优化手段覆盖 pass**：源码仅 copy/fragment/Parallel-compute/copy 四类语句，S1–S6 清单完备性成立；S2（fp32 opmath）判「算法/精度层」与 S1/S3/S6 判「实现层/硬件相关」的分类正确（herf 注释与 CUDA opmath 语义佐证）；无 SMEM/mma/warp-shuffle/cp.async 的「无特征」声明与源码一致。
- **耦合性判定合理性 pass**：C1–C7 四态处置 + 依据逐条抽检：C1 语义层无条件保留 = migration-analysis.md L182「规则 1」实文 ✓；C4 依据 = 映射表 L166「shared memory → L1/UB 等价替换」实文 ✓；C5 依据 = 映射表 L172「block × thread 两级并行 → block 级任务划分 + 向量化元素并行（重新设计，T.Kernel 只接受一维 block 数）」**逐字一致** ✓；C2 禁用 T.verf 证据链双证：docs T.verf.md §3 lowering「arith::ConstantOp + arith::DivFOp + hivm::VMulOp + hivm::VAddOp」**逐字实读吻合**（多项式分解、无硬件 erf、无 exp 支路），加 REJ-1 fp64 复现表与用户 910B3 实测逐点吻合（见机械复核段）——证据强度：强（结构证据 + 数值复现 + docs dtype 矩阵 bf16 ×）；docs §3 未写「7 阶」字样，DESIGN 的 7 阶定性由实测误差与 7 阶泰勒 fp64 误差逐点吻合反推，属实测推断，禁用结论对阶数不敏感（|x|≥1.5 即击穿），可接受 ✓；「无 vrec/verfc 补充指令」负向断言由检视者全目录 grep 独立复核（vrec/vrc/reciprocal/erfc 零命中）✓；C7 判「等价替换」与 C5「重新设计」的标签差异：两者机制同为并行结构重映射，C7 语义保持论证（逐点映射无跨元素依赖，tile 分派不触碰数值，§0.6 R3）独立推演成立，分类口径不构成阻塞〔建议级：与 C5 统一标注重审更整洁〕；C6 host 层保留 ✓。统计与「无静默丢弃」声明一致（S1–S6→C1–C7 映射闭合，§0.6 承接核对表齐）。
- **NPU 重设计可行性 pass**：R1（A&S 7.1.26）三要素齐——常数组 0.254829592/−0.284496736/1.421413741/−1.453152027/1.061405429、p=0.3275911 与 A&S 7.1.26 手册值一致；Horner 展开独立推演：t·(A1+t(A2+t(A3+t(A4+t·A5)))) ≡ ΣAi·t^(i+1)（5 步逐项展开验证 ✓）；sign 奇延拓 + |x| 用法与 A&S x≥0 形式等价 ✓；语义保持论证=容差内等价 + 边界逐一核对（±0/±inf/NaN/±max/subnormal，机器表 23 角点 0 违反，独立推演与机器双证均成立）。R2 transit：升域无损/单次舍回与源 opmath 路径一致 + GOLD-DIAG 位等价双证 ✓。R3/R4：一维 Kernel、静态边界（N/bs 编译期常量）、纯名 guard、整 buffer v-op + 零原点 `[0:tail]` store——违反检查项逐一排查无违例（一维 ✓/内存层级 ✓/容量 ✓/对齐：32B 尾轴由 T.copy MTE 处理且 vec_add_1d/expm1 官方模式先例实存 ✓）。
- **设计与决策一致性 pass**：§1.3/§1.4/§3.1/§3.3/§5/§6 描述的均是 §0.5/§0.6 决策后的 NPU 算法（A&S 链 + transit + persistent + 整 tile 向量化）；无任何章节残留源方案直译（无 T.erf/fragment/threads 消费）。
- **golden 独立性 pass**：§8.1 golden = `torch.erf(x.detach().cpu()).to(x.dtype)`——源仓测试锚移植（优先移植源仓参考实现 ✓），不含 A&S 常数/transit/分核等 §0.6 知识，无 Self-证结构 ✓。

### 1. API 可行性 — pass

§3.2 全部 API 有文档实存佐证且形态条款核对通过：T.copy/T.vcast/T.vcmp/T.vabs/T.vmul/T.vadd/T.vsub/T.vdiv/T.vexp/T.vbrc/T.vselect + 一维 T.Kernel(num_kernels, is_npu=True)/T.serial/T.ceildiv/T.min（后四者有 expm1 landed kernel 生产实文核对）。**v-prefix 优先 ✓**（无 npuir_xxx 混用）。逐条文档亲验（检视者打开）：T.vadd.md §2.1「src1 tensor或scalar」+ §2.3「仅支持src1为标量，不支持src0为标量」✓；T.vmul/vsub/vdiv.md 同构条款（vdiv「仅支持src2为标量，不支持src1」→ 分子必须张量 = ones_ub 保留依据 ✓）；「标量操作数必须在 T.Kernel 内、T.Scope 外定义」✓；T.vcmp.md L37–39（A/B 同 shape + bool 不可写 GM 需配合 vselect 片上）✓；T.vselect.md 四参全 tensor ✓（±1 缓冲保留依据成立）；T.vcast.md §2.2.1 f16→f32 rint / bf16→f32 rint / f32→f16·bf16 rint 行 ✓；T.vbrc.md 实存（shape操作/）✓；T.verf/T.vtanh/T.vpow/T.vexp2 文档实存且淘汰理由条款与文档一致（维度 8 复核）。模式匹配：Developer + 自动同步，无手动 set_flag/wait_flag ✓（§7 一致）。vbrc 标量 let 绑定位置按 TRAP-expert-v-operands 规范设计在 traced body 内 ✓。

### 2. 内存层级规划 — pass

§4.4 完整路径 GM→(MTE2)→UB→v 链→UB→(MTE3)→GM，无越级；纯 Vector 无 L1/L0 参与、无跨引擎往返 ✓。UB 预算经机械复核人工补位重算（见机械复核段 ub_budget 行）：37/39 B/elem × bs ≤ 4096 + CG-2026-0008 +12% 裕度 < 192KB（CONST-capacity-910B2C〔stale+重验路径〕口径，显式 alloc 结构 r9 实测 0.981 系数如实引用且未按 ×2 自设模型——教训条目 5 合规）；UB guard `_UB_MAX_BLOCK=4096` + ValueError（expm1 同款，landed 实文验证该形态生产可用）✓；buffer 层级标注齐全（§4.3 全部 UB）✓；L0C 不适用 ✓。复用纪律（dst∉src 或原地 dst≡src 文档条款；两路线拷出源不同、fp32 路线最终 vmul dst=x_ub 防恒等输出错误）与 §3.3 伪代码逐行一致 ✓。

### 3. Tiling 策略 — pass

§5.2 block 取值有契约来源（K11 常量）+ UB guard 上限 + 理由 ✓；非整除处理 = tile 级 tail 动态方案（§0.6 R4），纯 Vector 场景说明齐 ✓；尾轴对齐（32B 由 MTE 处理、N=1 退化用例覆盖）✓。分核策略三要素（core-split-strategy.md §1）齐全且自洽——机械 core_split skip 已由检视者人工补位（逻辑核数复算全对 + 物理核数独立复跑探针 20→40 + 极大规模方案静态边界核验，见机械复核段）；核内串行任务数不依赖动态 shape（num_local_tasks 编译期 IntImm）✓；「逻辑核数远超物理核数」的超发风险已按 persistent 消除 ✓。

### 4. 技术约束检测 — pass

三维 Kernel/GEMM 非整除/L0C 均不适用且已声明；bf16 无 v-prefix 算术（T.vabs/T.vdiv/T.vexp/T.verf dtype 矩阵 bf16 ×——检视者逐文档核对矩阵行）→ transit 强制，规避成立 ✓；v-op 切片静态边界限制（TRAP-expert-v-operands P10 实文核对）→ 整 buffer v-op 设计 ✓；bool 不可写 GM ✓；工厂契约（两段式/归一化/ValueError 防御/output_dtype 校验）与 wrapper 现状逐字对齐 ✓。

### 5. 循环与同步 — pass

§6.1 四层循环结构明确（persistent T.Kernel / T.serial 静态 grid-stride / 无显式 tile 内循环 / Horner 编译期展开），与 §3.3/§1.6.2 一致；T.Pipelined 不用的理由（Developer auto-multi-buffer + copy-floor 实证链隐藏，条目 stale 已标 + Stage 4 探针重验承诺）成立为建议级依据、无阻塞风险；§7 Developer 自动同步与模式匹配，隐式同步点清单与伪代码依赖序一致；§6.4 尾块 stale lane 分析（垃圾经 vexp 可产 inf/nan 但不落 GM）正确 ✓。

### 6. 验证方案 — pass

§8.1 golden 含 PyTorch 参考实现草案（源锚移植、独立、CPU 运行）✓；§8.2 容差逐字核对 GPU test L69–74 与 NPU test L27–32 双侧一致（fp16 1e-3/1e-3、bf16 1.6e-2/1.6e-2、fp32 1e-5/1e-5）✓，断言形态（equal_nan + 特殊位 + 零号差豁免 expm1 先例）✓，精度裕度链 17×（5.868e-07 vs 1e-5）经重跑背书 ✓；§8.3 L0 计划具体可执行：smoke 3 dtype/大规模多轮核内串行/非整除 {1000,4097,5000}/tiny {1,7,32}/finfo 角点全套（含 ±subnormal/±tiny/±max——对齐 §0.1 边界语义与 verify_equiv 角点集）/persistent 边界 {vc−1,vc,vc+1}/E2 双 bs/UB guard/负例拒绝/标量字面量形态首跑（同时履行 R1 风险重验承诺）✓。L0 门槛 shape 与 block 整除关系明确（含尾块负例）✓。

### 7. 完整性与一致性 — pass

模板章节齐全（§0 含 0.1–0.7；§1 含 1.6.0/1.6.1/1.6.2/1.6.3；§2–§10 全在）✓；无占位符/TODO/待补充（grep 复核 0 命中）✓；§3.5.3/§0.7 引用的 examples/ 路径逐一存在性核对通过：expm1 landed kernel（并实读其工厂/transit/persistent/UB guard 确认「结构模板」声明属实）、leaky_relu 目录、mish perf_opt（标量字面量 vsub 先例实存于 mish.py L160/181——〔建议级：设计注 L78 行号漂移，先例本体属实〕）、examples/elementwise/vec_add_1d.py、examples/leaky_relu/_make_leaky_relu_kernel/RETROSPECTIVE.md、双侧 test/bench、manifest ✓；API 映射↔内存规划↔Tiling↔伪代码↔循环结构交叉一致（22-op 清单三方逐 op 对齐，op 计数独立复核 =22 ✓）；交付清单与 Stage 3/5 衔接路径明确（Stage 5 零 wrapper 编辑 + L73 import 切换，与 wrapper L57–60 注释实文一致）✓。

### 8. 算法优化分析 — pass

- **§1.6 存在性与结构 pass**：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐。
- **算法调研存在性与完整性 pass**：R1 候选表 5 候选 + 基线 B0 在表（含等价性初判/收益方向/结论逐行）✓；参考表命中行（algorithm-candidates.md ALG-elementwise L98–104）已评估——R1 提示项「公共子表达式消除/除法乘倒数/cast 链合并」分别落至 §1.6.1 无优化空间项（CSE 无对象）、V5（无倒数 API，检索佐证）、R2 transit 单次舍回（链已最短）✓；R2「无在线变体」有明确结论 + 结构依据（逐点零依赖图 + 源算法已单遍）+ 参考表一致（ALG-elementwise「逐元素映射天然单遍」）✓；R3 四口径表齐（机械 r3_metrics pass + 语义复核：FLOPs 口径「exp/div/cmp/sel 各记 1」写明、访存 2N·eB、遍数、缓冲峰值，基线在表）✓；R4 逐候选评估（单元/容量/对齐/静态边界/流水/跨核 + 精度决定项）✓；调研深度与「多步超越链」定级匹配（完整调研）✓；「无更优替代」写明调研范围（R1 段四项 + C4 显式「未文档化假设 + 估算依据」）✓。
- **调研结论正确性独立复核 pass**：① 等价性：A&S 常数组/公式/Horner 展开独立推演成立 + verify_equiv 重跑 EQUIV_PASS（见机械复核）；② 复杂度独立复算：22 op/tile 与 §3.3 逐 op 清点一致；「全部候选访存/遍数相同」经核——单入单出逐点算子、中间量全驻留 UB，**无中间缓冲 GM 往返漏计**（对照 R3 表 C2 行「2N·eB 与基线相同」声明 ✓）；AI fp32=N/8N=0.125 flop/B、fp16=0.25 ✓；D-2 估算逐项复算全对（4.19/8.39/67.1/134.2/268.4MB/1.074GB 流量、105/210KB 每核、3.1/6.2/55.6/111/321µs、256M 双口径 0.85–1.46ms、11µs=22×0.5µs；CONST 条目数值 33.9/30.2/20.9/18.4GB/s + 1.26TB/s 3:1 地板 + 0.5µs/op 与 constants.md 实文逐项吻合；48→40 核外推与 1:1 vs 3:1 测量条件差异已如实注记并交 Stage 4 对账）✓；③ 负向断言逐条复核：「T.verf 非硬件 erf」docs §3 逐字 ✓ + REJ-1 与用户实测逐点吻合 ✓；「vrec/vrc 不存在」检视者全目录 grep 零命中 ✓（〔建议级：设计写「全 20 子目录」，实际 `ls docs/Tilelang.language/` = 16 子目录——检索结论本身经检视者全目录复核成立，范围数字请顺手修正〕）；「无 verfc」零命中 ✓；「bf16 无 v-prefix 算术」dtype 矩阵多文档核对 ✓；「无在线变体」对照参考表无命中 ✓；「pow(denom,−1) 不快于 vdiv」标注未文档化假设 ✓；④ 亲和性淘汰与 ascend-constraints/pattern-library 无矛盾（容量引用 CONST-capacity-910B2C〔stale+重验路径〕）✓；⑤ 无 R3 并列候选（精度单淘汰 C1/C3、C4 过冗），纸面裁决不触发（唯一存活候选，无需实验三件套）✓。
- **调研→设计一致性 pass**：§1.4 算法描述、§1.3/§3.1 公式拆解、§6 循环结构均以 §1.6.0 选定 C2 + §1.6.1 优化后公式为输入（三方交叉核对无「调研选 A 设计做 B」）；迁移一致性：§0.5 C2 处置「重新设计→A&S」即承接 §1.6.0 选定、§0.6 R1 引用 R1/R3/R4 依据，Phase R 先于 M1 的执行序声明与文档结构自洽；S6（单指令超越函数）意图承接论证（NPU 无该硬件 → 多项式链在带宽受限窗口隐藏）有逐项说明、无静默丢弃 ✓。
- **数学等价优化完整性 pass**：A1/A2 采纳项四要素（原式→优化后→等价论证→收益量化）齐全；无优化空间四类（CSE/稳定化/降代换/访存削减/归约合并）逐类有对象性论证（非空白结论）✓。
- **等价性论证正确性 pass**：A1 容差内等价 + 舍入路径分析（x→0 相消区绝对误差有界、包络 atol 主导无 rtol 风险——与 §8.2 精度标准一致）独立推演成立；A2「字面量与 vbrc 广播值为同一 fp32 数」论证经构造性断言机器验证（D-1 A2 行）✓，且诚实披露「文档√/生产实证仅 vsub」+ 回退路径（常量缓冲形态 UB 预算已含回退列）✓。〔建议级注记：A2 收益数字 −56KB/−112KB 与「9→3 = 移除 6 缓冲」的算术不符，正确值为 6×bs×4B = −49.2KB@bs2048 / −98.3KB@bs4096（数字来自 7 缓冲口径的笔误）；不影响选型与 UB 预算结论，Stage 1 下次修订可顺手更正。〕
- **向量化替代完整性 pass**：V1–V9 与 §3.3 伪代码/§6 循环结构交叉核对**无遗漏点**（伪代码全部可执行语句 = 22 v-op（V1–V5）+ block_id/t0/tail/guard（V6/V8/V9）+ host 元数据（V7）；§6 无逐元素标量循环，判定向量化点与实现形态一致）✓。
- **不可替代理由充分性 pass**：V6 block 级索引（每 tile O(1) 非热点）/V7 host 元数据/V8 动态边界（+ v-op 静态切片限制的架构性约束）/V9 掩码 guard（静态可折叠）——四类均属可接受理由清单类别且具体到本算子，无「实现简单/照源码写」类理由；TRAP-parallel2d 不适用声明与 1-D 形态一致 ✓。
- **替代 API 佐证 pass**：vabs/vmul/vadd/vsub/vdiv/vexp/vcmp/vselect/vcast/vbrc 全部有 docs 实存 + 条款亲验（见维度 1），优先 v-prefix ✓。
- **向量化轴与布局决策 pass**：1-D 扁平 I/O 契约 → 单轴豁免声明成立；核对项齐（向量轴=连续尾轴、无打包加速佐证 CONST-vector-launch-overhead「f16≈f32」✓、广播轴对齐不适用论证成立、源 GPU threads×npt 轴按「输入非结论」处置 ✓、§0.3 已解读源轴选择 ✓）；豁免后无弃选候选、无未实证常数依赖（唯一存活候选无裕度判定）→ 实验裁决三件套不适用成立 ✓。
- **布局决策与实现一致 pass**：§1.6.3（1-D (bs,) 连续）↔ §3.3（整 tile v-op 无切片）↔ §4.3（buffer 形状 (bs,)）↔ §6（无 2D 循环）四方一致 ✓。
- **与下游章节一致 pass**：§3.1 声明「§1.6.1 优化后公式 = 本节唯一输入」且实拆与 A&S+字面量化形态一致；§6 循环结构与 §1.6.2 结论一致 ✓。

**维度 8 结论**：调研四问完整、选定算法（A&S 7.1.26）是唯一满足 fp32 atol 包络且全项硬件亲和的候选，等价性经独立推演 + 机器复跑双证；阻塞级无 fail，3 处建议级注记（A2 收益数字、检索子目录计数、mish 行号漂移）均不影响任何设计决策。

---

**总评**：9 维度全部 pass（无阻塞级问题）。迁移三问回答：① 源算子读懂了——§0 与 GPU/NPU 双侧源码逐字核对无偏差；② 耦合性判定有依据——T.verf 禁用为 docs 结构证据 + 机器复现误差表双证（与用户实测逐点吻合），GPU→NPU 处置映射逐条落在映射表/常数表实文上；③ NPU 重设计可行且语义保持——容差内等价（17× 裕度）+ 边界语义逐项机器核对 + golden 独立于 NPU 算法。机械复核（verify_equiv 重跑逐位一致、design_calc_check skip 项人工补位复算全对、核数探针独立复跑 20→40 一致、154 stale 清单与引用重验路径核验通过）支持放行 Stage 3。
