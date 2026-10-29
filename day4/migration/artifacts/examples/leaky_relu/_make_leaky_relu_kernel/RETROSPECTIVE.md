# RETROSPECTIVE.md — leaky_relu / _make_leaky_relu_kernel

> Stage 1 复盘（追加式；schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`）

## Stage 1 (DESIGN) Retrospective — first_design (migration/harness) — 2026-10-28

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | 分核标准要求「NPUUtils.get().get_aicore_num() 实查并记录查询代码与返回值」，但设计环境 `import tilelang` 可能不可用（dev root 私有 TVM 包缺失），标准未给出等价实查路径，designer 需自行找到 ctypes 直调 libruntime `rtGetAiCoreCount` 的降级方案并自证等价 | DESIGN.md §5.5 要素②（查询代码 + ret=0 → 20 + npu_utils.cpp L124-137 等价性核对）；`.agents/skills/_shared/standards/core-split-strategy.md` §1 要素② | core-split-strategy.md 要素② 补一行「tilelang 不可导入时的等价实查路径：ctypes 直调 CANN libruntime.so rtGetAiCoreCount（npu_utils.cpp 同一调用）」 | R |
| Design-flow | D-1 verify_equiv 的容差比较器首版漏掉 `o == r` 精确相等分支：inf vs inf 经 `abs(inf-inf)=NaN <= env` 误判为违反，导致 6/6 采纳行假性 EQUIV_FAIL；lerp 的 `_compare`（examples/.../lerp_tensor_kernel/_make_lerp_tensor_kernel.py L319-335）已有正确形态但 verify_equiv 无模板可继承 | verify_equiv.py `violations()`（修正后含 `o == r | both_nan | env` 三分支）；本次首跑输出 6 行假 FAIL、修复后 0 | tilelang-op-design skill Phase 2 第 5 步或 algorithm-research 附近补「D-1 比较器模板：相等分支(o==r)∪both_nan∪atol/rtol 包络」，或指向 lerp `_compare` 为参考形态 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| leaky_relu 族 fp32-transit 机器验证：transit vs golden（F.leaky_relu fp32-opmath）0 ulp/0 violations，原生域基线 1 ulp——PL-1.17 双触发在单乘公式族的第三例独立确认 | D | examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py（A1 表 6 行 PASS）+ DESIGN.md §1.6.1(b) 结果表 | python3 examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py | tilelang 0.1.2+88ee8c302e（本仓 dev root）/ torch CPU 侧验证（无需设备）；设计目标设备 910B3 / CANN 9.0.0 | pattern-library traps-runtime.md TRAP-fp16-opmath-golden（补第三例） |
| vmax 条件式（y=max(x, slope·x)）slope∈[0,1] 才等价的机器证据：slope=2.0 时 65521/65548 违反——leaky/prelu 族弃选论证模板 | D | verify_equiv.py A2 段（slope=0.01 PASS / slope=2.0 FAIL）+ DESIGN.md §1.6.1(c) | python3 examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py | 同上 | tilelang-op-design references/algorithm-candidates.md ALG-elementwise（R1 候选备注） |
| tilelang 不可导入环境下的物理核数等价实查：ctypes 直调 CANN libruntime rtGetAiCoreCount（与 NPUUtils 内部 npu_utils.cpp 同一调用）；本机 910B3 实测 20（纯 Vector 40），与 910B2C 记录的 24/48 互证核数随设备变化、工厂期现查规则必要 | D | DESIGN.md §5.5 要素②（查询代码 + 返回值 + npu_utils.cpp L124-137）； ada_layer_norm `_ada_layer_norm_kernel.py` L219-222（他设备 24/48 记录） | python3 -c "import ctypes; lib=ctypes.CDLL('<cann>/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)" | 910B3 / CANN 9.0.0（libruntime 25.5.0 工具族）/ rtGetAiCoreCount ret=0 | core-split-strategy.md（要素②降级路径）或 constants.md（核数事实条目） |

### Transferable Lessons（可迁移教训）

- 接口契约「死代码」处置先查类型拒绝链再决定迁移：fp8 分支在基类 `SUPPORTED_DTYPES=(fp16,bf16,fp32)` 即被拒（builder 前置 ValueError），迁移时保留参数、运行时 ValueError 即可，勿实现不可达路径。
- D-1/精度比较器必须含 `out == ref` 精确相等分支：`abs(inf-inf)=NaN` 会穿透 atol/rtol 包络造成假违反（本任务首跑 6 行假 EQUIV_FAIL 的根因）。
- golden 锚点若为 `x.float()` opmath 形态（F.leaky_relu/torch.lerp 同族），fp16/bf16 实现一律 fp32 transit（vcast rint 升域 → fp32 域计算 → vcast rint 单次舍回），且标量参数（slope/alpha）必须以 fp32 值参与，不得 cast 到原生域。
- E2 工厂签名保持源形态 `kernel(threads_arg, npt_arg)`（wrapper `init_config` 按 `self.kernel(cfg["threads"], cfg["num_per_thread"])` 两参调用），NPU block_size 在工厂内部 `threads_arg*npt_arg` 折叠，不改接口。〔**Stage 2 检视更正**：本条不实——NPU 侧 harness wrapper（examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py）按 K3/K11 以单参 `kernel(block_size)` 调用，两参契约是 GPU 基类 L2417 的事实；见 REVIEW.md 问题 1，Stage 1 revision 时按 wrapper 现状改写。〕
- vbrc 标量常量（zero/slope）放 persistent `T.serial` 循环外、每 kernel 一次填充；TVM-script 的 dtype 分支放工厂期 Python if（if 块变量作用域限制，lerp C11 同款规避）。

## Stage 2 (REVIEW) Retrospective — 2026-10-28

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 0 检查项表无「NPU 侧 harness wrapper 调用面契约核对」条目：设计对 wrapper 的接口断言（§9.2「init_config 两参依赖」）只有逐字通读 `examples/TileOPs/.../leaky_relu.py`（K3/K11 单参 block_size + `_leaky_relu_fwd_wrapped` 调用式）才能证伪，属人工发现而非清单覆盖；本任务 Stage 1 甚至把该错误断言写进了 RETROSPECTIVE Transferable Lessons，说明设计侧同样缺这条核对 | REVIEW.md 问题 1（wrapper docstring K11 + `_leaky_relu_fwd_wrapped` L85 + lerp `_make_lerp_tensor_kernel.py` L51/L163 单参实证） | tilelang-design-review SKILL.md §3 维度 0 表增检查项「接口处置（保留/适配）与 NPU 侧已移植 wrapper/launcher 的实际调用面逐字核对，GPU 基类调用式不可当作 NPU 契约」；tilelang-op-design references/migration-analysis.md §5 同步 | R |
| Review-flow | 多 dtype 路线 buffer 别名/复用（transit 回写复用载入 buffer）时，「结果驻留 buffer → 拷出源」路径存在按路线字面错写风险，维度 2/维度 8 表均无对应机械核对点，本次靠人工数据流推演捕获（fp32 路线 ⑦ 拷陈旧输入） | REVIEW.md 问题 2（§3.3 ⑦ vs §4.3 sx32 角色矛盾） | 本 skill 维度 2 检查项补一条「逐 dtype 路线标注结果驻留 buffer 与拷贝源，交叉核对 §1.5/§3.1/§3.3/§4.4 四方一致」；tilelang-op-design references/quality-checklist.md 同步 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 910B3 物理核数实测 20（ctypes rtGetAiCoreCount ret=0；纯 Vector 翻倍 40）——与 910B2C 24/48 条目并列的跨设备核数数据点，佐证 core-split「现查现用、禁经验值」规则；检视者独立复跑设计 §5.5 同查询路径成功（等价实查路径可用性验证） | D | REVIEW.md「机械复核」段 core_split 人工复核实查记录；DESIGN.md §5.5 要素②；ada_layer_norm `_ada_layer_norm_kernel.py` L219-222（24/48 他设备记录） | python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)" | Ascend 910B3 / CANN 9.0.0 / 2026-10-28 复跑 | pattern-library/constants.md（核数条目补 910B3 数据点） |
| design_calc_check 对逐元素 persistent 设计的三项 skip（ub_budget 表格式不识别、core_split 把 host 侧 `lru_cache(maxsize=32)` 解析为「declared logical cores=32」噪声）——机械 checker 对「buffer 表以字节/元素口径」与「host 常量与核数同现」的形态覆盖缺口 | R | REVIEW.md 机械复核段（skip 明细 + 人工复核补位）；本设计 §5.5 全部三要素数字人工复算通过 | python3 .agents/tools/design_calc_check.py --design examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md | .agents/tools/design_calc_check.py 当前版 | design_calc_check.py（解析器增强）或本 skill「机械复核」段（明确此类 skip 必做的人工复核清单） |

### Transferable Lessons（可迁移教训）

- 迁移任务的接口契约核对必须查「落地侧现状」而非外推源仓：GPU 基类的调用式（threads/npt 两参）与 NPU 已移植 harness wrapper 的 K11 单参 block_size 调用式是不同事实；设计文档转述的「wrapper 依赖 X」类断言，reviewer 必须打开 wrapper 文件逐字核对（本任务据此捕获阻塞级问题 1）。
- transit 链「降域回写复用载入 buffer」形态下，每条 dtype 路线单独走一遍「结果在哪个 buffer → ⑦ 拷出哪个 buffer」；无回写步的路线（fp32 原生域 + 原地 vselect 到中间 buffer）沿用统一拷贝源即恒等输出错误——检视时对 §3.3 伪代码按路线做字面数据流推演，不轻信 buffer 角色表与伪代码互相一致。
- 机械 check skip ≠ 免检：ub_budget/core_split skip 时按「字节/元素 × block、逐 workload 逻辑核数、核数翻倍、静态边界、负载不均上界」逐项人工复算并在机械复核段留痕；本轮独立复跑物理核数实查（ret=0→20）是低成本高价值的 skip 补位手段。
- 弃选/否决类负向断言的复核捷径：条件等价候选（如 vmax）看 A2 机器证据行数字与总元素数是否自洽（65521 violations / 65548 元素与脚本 65536+12 角点结构一致即闭环），bf16 类 API 负向断言直接打开对应 docs §2.2.1 dtype 矩阵核对单元格——两处亲验均一次通过，无需重复设计推演。

## Stage 1 (DESIGN) Retrospective — revision 1 (migration/harness) — 2026-10-28

> 修订输入：Stage 2 检视不通过（REVIEW.md 2 阻塞 + 3 建议，全部采纳）。§0 三问解读、§1.6.0 调研结论、A1 等价论证、persistent 分核、developer 模式、golden 独立性维持不变；修订聚焦接口契约（阻塞 1）、数据流落点（阻塞 2）与建议级文字项。verify_equiv.py 重跑 6/6 EQUIV_PASS（修订未触及公式，结果表无需变更）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 阻塞 1（E2 两参契约断言）在 first_design 时还被写进了本文件 Transferable Lessons——说明「迁移接口契约 = 源签名保持」的先验在 skill 侧缺乏对冲材料：tilelang-op-design 迁移流程（Phase M0 步骤 1）未强制把 NPU 侧已移植 wrapper 纳入源码阅读清单并定为接口契约权威（检视侧清单缺口已在 Stage 2 章节登记）；本次修订以「落地侧现状 = 契约权威」结构性修复（§0.7 权威来源条目 + §9.2 防呆） | DESIGN.md 修订记录表调整 1（wrapper L82/L157/L64–65 行号锚定）；REVIEW.md 问题 1；本文件 Stage 1 章节 L27 更正注 | tilelang-op-design references/migration-analysis.md §2 M0 步骤 1 补「迁移任务源码阅读清单必须含 NPU 侧已移植 wrapper/launcher（harness 场景），接口契约以 wrapper 调用面为准，GPU 基类调用式仅作源侧事实」 | R |
| Design-flow | 阻塞 2（fp32 路线结果落点）的根因是「逐 dtype 路线的结果驻留 buffer → GM 拷出源」在设计侧无机械一致性自检点：Phase 2 产出 §3.3 伪代码后无步骤要求按路线做字面数据流推演（检视侧对应缺口已在 Stage 2 章节登记）；本次修订后 §1.5/§3.1/§3.3/§4.3/§4.4/§1.6.3 五处交叉一致并给出拷出源统一不变式 | DESIGN.md 修订记录表调整 2 + 「为何不会再犯」第 2 条；REVIEW.md 问题 2 | tilelang-op-design references/quality-checklist.md 增自检项「多 dtype 路线设计：逐路线标注结果驻留 buffer 与 GM 拷出源，§1.5/§3.1/§3.3/§4.3/§4.4 交叉一致后方可冻结」 | R |
| Design-flow | 原地写回 v-op（vselect Out=A/Out=B）× T.serial 多迭代循环的陷阱类（TRAP-vselect-inplace-carried-state）在设计侧无强制检索触发：首版既未查到也未防呆，本次 revision kb_search 才命中并以 §9.1#9 + §9.2 采纳（判定不构成正确性 bug：无 loop-carried 状态，lerp 原地链同形已验证） | DESIGN.md §9.1#9 / §9.2 / §3.3 注③；pattern-library traps-compiler.md TRAP-vselect-inplace-carried-state（status=verified，亲读）；`.agents/tools/kb_search.py` 命中记录（本次任务返回） | tilelang-op-design Phase 3 检索提示或 ascend-constraints.md 补「T.serial 循环体内原地写回 v-op 必须检索 TRAP-vselect-inplace-carried-state 类条目，并显式写明设计是否有 loop-carried 状态」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移任务接口契约「落地侧现状」判定规则（修订验证）：NPU harness wrapper（K3/K11 单参 `kernel(block_size)` + torch.dtype 入参 + `_BLOCK_SIZE_FP32=1024/_BLOCK_SIZE_NON_FP32=2048` 折叠）与 GPU 基类两参调用式是不同事实；按源两参实现会在 harness 单参调用处 TypeError | P | DESIGN.md 修订记录调整 1/3 + §0.7/§9.2；examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py L64–65/L82/L157；lerp 先例 _make_lerp_tensor_kernel.py L51/L163；REVIEW.md 问题 1 | none（规则类；反例复现 = 按两参 factory 实现 + wrapper 单参调用 → TypeError） | tilelang 0.1.2+88ee8c302e（本仓 dev root）；wrapper/lerp 为仓库现有验证件 | migration-analysis.md（接口契约判定规则）或 design-review SKILL.md 维度 0（对偶项已在 Stage 2 章节登记） |
| TRAP-vselect-inplace-carried-state 适用边界补充：Out=B 原地 vselect 在「无 loop-carried 状态」的 persistent T.serial 逐元素算子中可用（每 tile 全量重载+重算、输出同迭代被 ⑥/⑦ 消费；lerp 原地 vsub/vmul/vadd 链同形 bit-exact 先例）；Out=A 原地（写回 cond-true 操作数）有 gqa_prefill L875 生产先例——为该 trap 的豁免判据补两类形态佐证 | P | DESIGN.md §9.1#9 / §3.3 注③；traps-compiler.md TRAP-vselect-inplace-carried-state（触发边界 = num_full≥3 + running 类携带状态）；lerp _make_lerp_tensor_kernel.py L238–241（原地链）；gqa _gqa_prefill_fwd_kernel.py L875（Out=A） | none（判据类；trap 本体 repro = pattern-library repro/TRAP-vselect-inplace-carried-state.py） | trap 实证于 tilelang 0.1.2（2026-09-24）+ CANN 8.5.0 / 910B2C；本设计引用未上板，L0-2 多迭代用例兜底 | pattern-library traps-compiler.md TRAP-vselect-inplace-carried-state（补「无携带状态豁免判据」行） |
| 多缓冲 UB 占用口径勘误传播：朴素 ×2 模型（2×19×8192=311KB > 192KB）非平台实测口径——TRAP-UB-multibuffer-inflation 平台值（transit≈20–21 / fp32≈26–27 B/elem，系数 1.10–1.12 结构依赖、反例 0.981×manual）+ lerp 同族 resident 20B×8192 L1 bs=8192 实测通过，支撑 leaky_relu guard 8192 保留；设计文档双缓冲论证应引用平台值口径而非自设 ×2 模型 | D | DESIGN.md §4.5 / §9.1#4（修订 1）；traps-compiler.md TRAP-UB-multibuffer-inflation 第三/四证（亲读）；lerp _make_lerp_tensor_kernel.py L93–97（8192 cap 先例）+ L1 bs=8192 用例（L663） | python3 examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py --level all（L1 含 fp16/bf16 bs=8192 编译运行用例） | lerp 实证于 tilelang 0.1.2+ed787bb / CANN 8.5.0 / 910B2C；本设计目标 910B3 / CANN 9.0.0 待 Stage 3 首跑重验 | pattern-library traps-compiler.md TRAP-UB-multibuffer-inflation（补「设计文档多缓冲论证引用平台值口径」提示）或 elementwise.md PL-1.6 域 |

### Transferable Lessons（可迁移教训）

- 迁移任务的接口契约以「落地侧现状」为唯一权威：动手设计前先 Read NPU 侧已移植 harness wrapper 全文（custom_op 调用式 + Kernel 类工厂构建行 + 折叠常量定义），GPU 基类的调用式（如 `init_config` 两参）只写进源码解读章节、永不进入 NPU 设计决策行；同族已验证件（lerp 形态 `_make_X(N, dtype) -> kernel(block_size=default_block) -> main`）是最快的契约对齐模板。
- 多 dtype 路线设计冻结前，逐路线推演一遍「① 载入 → … → ⑦ 拷出」的字面 buffer 数据流：结果驻留 buffer（vselect Out= 目标 / 降域回写目标）必须与 GM 拷出源一致；「某路线无回写步」（如 fp32 原生域）时最易出现拷出源停留在陈旧载入 buffer 的恒等映射错误——§1.5/§3.1/§3.3/§4.3/§4.4 五处交叉一致后才冻结。
- 原地写回 v-op（vselect/vsub/vmul/vadd 的 Out=A/Out=B 形态）在 `T.serial` 循环体内的安全条件 = 输出仅同迭代消费（无 loop-carried 状态）；凡设计含跨迭代携带（running/accumulator 类）一律改独立 scratch + `T.copy` 回写（trap 绕法）；诊断指纹 = 输出停留 tile-0 局部值。
- 涉及 torch.dtype 入参的 kernel builder：先归一化（`dtype if isinstance(dtype, str) else dtype.name`）再与 str 集合比较——torch 对象与 str 元组 `in` 比较恒 False，会把所有 dtype 误判进同一路线（本例会把 fp16/bf16 误走 fp32 原生域）。
- 设计文档的 UB 多缓冲论证引用 pattern-library 平台值口径（TRAP-UB-multibuffer-inflation：结构类平台值 + 系数结构依赖），不自设 ×2 线性模型；guard 上限保留时给出同族实测先例（lerp bs=8192 L1 通过），并写明 Stage 4 调参前以 BishengIR `ub overflow requires` 报文复算 + 连带复验 num_kernels。

## Stage 2 (REVIEW) Retrospective — revision-1 复审 — 2026-10-28

> 复盘对象：revision 复检流程自身（9 维度全量复核 + 修复闭环逐项独立验证）。设计内容问题已载 REVIEW.md，不重复抄入。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 检视清单无「重跑 kb_stale_check 核验设计所引 pattern-library 条目在当前工具链下的 stale 状态」标准动作：设计 §9.1#8 自述「kb_stale_check 未判 stale」只能实跑该工具证伪（本轮实跑 PL-1.6/PL-1.17/TRAP-UB-multibuffer/CONST-mte2/CONST-copy-floor 均已判 [stale]，工具链 88ee8c302e/CANN 9.0.0/910B3 vs 条目 stamp ed787bb 系/8.5.0/910B2C）；本轮靠任务指令显式要求「status 须亲验非 stale」才触发核查，常规 gen 检视的维度表（仅要求核对 front-matter status=verified）会漏掉工具链漂移信号 | REVIEW.md 机械复核段 kb_stale_check 实跑记录 + 检视注记 a；AGENTS.md「Gap conclusions bind to toolchain version stamps; mechanized: kb_stale_check.py detects stale entries」 | 本 skill Phase 1 第 5 步（弃选论证查证）追加固定动作：「设计引用 pattern-library/evolution 条目作决策依据时，跑 `.agents/tools/kb_stale_check.py` 核对其在当前工具链下的 stale 状态；设计自述『未判 stale』类断言必须重跑验证」 | R |
| Review-flow | 「多缓冲 guard 上限论证」的路线覆盖范围核对无清单条目：修订版以「resident 19B ≤ lerp 同族 resident 20B×8192 L1 实测先例」统括全部 dtype，但该先例仅覆盖 transit 路线（lerp L1 bs=8192 用例亲验为 fp16/bf16；lerp fp32 仅测 12289 guard 拒绝），而 TRAP 明示 footprint 非 resident 单调（fp32 直连结构 8/16B resident 同为 26 B/elem）——按设计自己引用的平台值口径外推，fp32 路线 8192 档 ≈213KB 越限。维度 2 表只有「中间缓冲区总和 ≤ UB 容量」，未覆盖「论证先例的结构类别与被论证路线匹配」这一类错误形态 | REVIEW.md 检视注记 b（lerp L637–648 亲读 + traps-compiler.md 第三证规律 1）；上轮 REVIEW.md 问题 3（同一论证位的前身错误） | 本 skill 维度 2 检查项补「多缓冲/guard 上限论证：所引实测先例的（结构类, dtype 路线）必须与被论证路线一致；引用平台值口径时禁止 resident 跨路线统括」；tilelang-op-design quality-checklist 同步 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 修订复审（revision re-review）核对协议有效性：修订者自述的每条「亲读核对」声明必须重开原文独立复锚（本轮 wrapper L82/L64–65/L157、lerp L51/L163/L637–648、gqa L709/L875、sparse_mla L230、Op 层 numel!=N_total、kernel_base.init_config 不调用工厂——全部命中，两阻塞闭环成立；另 NPU kernel_base init_config 仅解析 config 的新事实是对「两参依赖=GPU 基类专属」判定的直接佐证，此前两轮检视均未显式打开该文件） | R | REVIEW.md 修复闭环逐项核实表；本轮全部亲读记录 | none（流程规则类） | 仓库现状（examples/TileOPs @ 2026-10-28） | 本 skill §4 Phase 1（revision 模式输入时增「修订自述逐条复锚」步骤说明） |
| TRAP-vselect-inplace-carried-state 检视侧豁免判据可操作化：核对「无 loop-carried 状态」= 逐迭代审计 vselect 输出 buffer 的所有写入点（①/②/④ 全量覆写）与消费窗口（仅同迭代 ⑥/⑦）；Out=A/Out=B 两种原地形态分别以 gqa L875（Out=A 生产）与 lerp L238–241（Out=B 别名、persistent 多迭代 bit-exact）为形态先例锚点 | P | DESIGN.md §9.1#9/§3.3 注③ + 本轮亲读 lerp/gqa；traps-compiler.md L249–（触发边界 num_full≥3 + running 携带） | none（判据类；trap 本体 repro 在 pattern-library） | trap 实证 2026-09-24/CANN 8.5.0/910B2C；先例件为仓库现有 | 本 skill 维度 0/8（原地 v-op 检视要点）——与 Stage 1 revision 复盘 VP（trap 补「无携带状态豁免判据」行）互补，不重复提 trap 侧 |

### Transferable Lessons（可迁移教训）

- 修订复审轮的两个阻塞闭环都要验证到「决策行」层级而非「文字」层级：阻塞 1 的闭环证据 = NPU 全部决策章节单参 + 两参仅存于源解读/引述位；阻塞 2 的闭环证据 = 逐路线字面数据流推演（①→⑦ 每个 buffer 读写序）而非五处「拷出源=x_ub」口号一致。
- 设计引用 pattern-library 条目做容量/代价类决策依据时，除核对 front-matter status 外必须实跑 kb_stale_check——「status=verified」与「当前工具链可复用」是两回事；stale 不自动否定决策（本例默认配置全口径安全故不阻塞），但设计侧必须持有首跑重验承诺，检视侧必须核对承诺存在。
- guard/上限类论证的「同族先例」核对要区分先例覆盖的 (结构类, dtype, 参数点) 三元组：lerp bs=8192 通过 ≠ fp32 路线 bs=8192 有先例；引用平台值口径（footprint 非 resident 单调）时，resident 比较法跨路线统括本身就是口径混用。


## Stage 3 (DEVELOP) Retrospective — first_impl attempt1 复核续跑 — 2026-10-28

> 复盘对象：上次调度在返回 verdict 前中断，本次以 first_impl attempt1 复核续跑。现有 `{op}.py` 复核发现 2 处必须修复（1 编译阻断 + 1 测试数据缺陷），修复后 L0/L1/L2/Boundary 全过、全 dtype 逐位一致（max_diff=0.0）。编程模式 Developer（`ASCEND_MODE = "Developer"`）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 中断残留文件的「已编译痕迹」（__pycache__/.pkl_memoize_py3 存在）会误导复核者以为实现已可运行：本次实测仅 transit 路线（fp16/bf16）可编译，fp32 路线在 `T.vbrc` 处编译阻断——`__pycache__` 只证明「import 过」而非「全部路线编译通过」 | `_make_leaky_relu_kernel.py` L287（修复前 fp32 vbrc rank 断言）；本任务 L0 首跑 L0-1/float32 RUNTIME-ERROR DiagnosticError | tilelang-op-develop Phase 1 复核中断文件时增加「逐 dtype 路线编译实证」动作（不能以 pycache 存在代替跑测） | R |
| Develop-flow | L0-5 角点测试数据的 fp32 max-finite 字面量 `3.4028235e38` 超出 fp32 最大有限值（`3.4028234663852886e38`），`vals[7] = max_fin` 赋 fp32 张量时 RuntimeError overflow——测试数据缺陷在 L0-5 角点构造处才暴露，fp16/bf16 的 max-finite 因 fp32 可表示未触发 | `_make_leaky_relu_kernel.py` L472（修复前 `vals[7] = max_fin` RuntimeError）；修复为 `float(torch.finfo(torch.float32).max)` | tilelang-op-develop 测试模板的角点值建议统一用 `torch.finfo` 而非手写字面量（fp32 max 手写易越界） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| vbrc 标量源 let 绑定位置细化：必须在 @T.prim_func **体内**（`with T.Kernel` 内）赋值，工厂作用域命名局部量同样触发 rank 断言——非仅「字面量」触发 | D | 本任务 fp32 路线首次编译 L0-1/float32 DiagnosticError（vbrc rank 断言）；修复（移入体内）后全路线通过；pattern-library traps-compiler.md L174 现有表述「字面量触发 rank 断言」覆盖面不足 | repro/TRAP-vbrc-scalar-let-binding.py（已转正并实跑 ASSERT PASS） | tilelang 0.1.2+88ee8c302e / CANN 9.0.0 / Ascend910B3 | pattern-library/traps-compiler.md（TRAP-vcmp-scalar-cast 条 L174 注记扩充：绑定位置 =  traced body 内，非仅「非字面量」） |
| torch 2.7.1 的 torch.dtype 无 `.name` 属性：`str(torch.float16) == "torch.float16"`（带前缀）；DESIGN §3.3 归一化公式 `dtype.name` 在 torch.dtype 输入下 AttributeError——实现层改为 `str(dtype).split(".")[-1]` 保持设计意图（wrapper L81-82/L157 传 torch.dtype 对象），str/torch.dtype 双形态与 output_dtype 校验全部通过 | D | 本任务复核期 torch.dtype 契约用例首跑 AttributeError（`_make_leaky_relu_kernel.py` L166 修复前）；修复后 torch.float16/bf16/fp32 双形态输入 max_diff=0.0；wrapper `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py` L81-82/L138/L157 | none（实证行内嵌；可用于 tilelang-op-design「dtype 入参归一化」模板修正） | torch 2.7.1+cpu + torch_npu 2.7.1.post4 | tilelang-op-design skill 的迁移 harness 模板（`dtype.name` 归一化公式改 `str(dtype).split('.')[-1]`） |
| TRAP-UB-multibuffer-inflation / PL-1.17 / PL-1.6 在 0.1.2+88ee8c302e/CANN9.0.0/910B3 重验存活：fp16/bf16 bs=8192 L1 编译运行通过（UB footprint 非 resident 单调成立）；fp16/bf16 fp32-transit 全 dtype 逐位一致（A1 等价性上板复现 0 ulp）；persistent grid-stride 结构编译运行正常 | D | 本任务 L1 14 例全过（含 fp16/bf16 bs=8192）；L0-1/2/5/6 max_diff=0.0；kb_stale_check 判 4 条目 stale（PL-1.6/PL-1.17/TRAP-UB-multibuffer/CONST-mte2），但本任务重验证明决策相关结论（transit 路径/UB guard 8192/persistent 结构）当前工具链存活 | none（重验类，已有条目） | tilelang 0.1.2+88ee8c302e / CANN 9.0.0 / Ascend910B3 | pattern-library（4 条目标注「0.1.2+88ee8c302e/910B3 重验存活」） |
| fp32 路线「常量为工厂局部、transit 路线常量为体内 let」的结构性不一致是首跑编译阻断的直接根因——同一语义两分支的 API 约束遵守必须逐分支审计而非「一处修好即复制」 | P | 本任务 fp32/transit 两分支 diff（修复前 L276-277 vs L226-227）；修复 = 把 fp32 分支常量移入体内 | none | 同上 | tilelang-op-develop 防呆清单：「vbrc 标量 let 绑定」检查覆盖**所有** dtype 分支（不要只按一条分支的形态实现） |

### Transferable Lessons（可迁移教训）

- vbrc 的标量源（如 0.0、slope 常量）必须在 @T.prim_func **体内**（`with T.Kernel(...)` 块内）用普通赋值 let 绑定；写在工厂函数作用域、即使是有名局部变量，也会在 trace 时触发 `The input vector and output vector must have same rank` 编译诊断（DiagnosticError）。本迁移任务的后续函数（elementwise 族 builder）都应把常量赋值放在 traced body 内。
- 复核中断残留的实现文件时，`__pycache__` / `.pkl_memoize_py3` 存在只说明模块被 import 过，**不能**作为「实现已可运行」的证据——必须对每条 dtype 路线实际编译跑测（本任务 fp16/bf16 可编译而 fp32 阻断，pycache 无法区分）。
- 测试数据里的 dtype 极值（max finite / subnormal）一律从 `torch.finfo(torch.<dtype>).max` 取，不要手写十进制字面量——fp32 max 的十进制近似（3.4028235e38）赋 fp32 张量会 overflow 抛 RuntimeError，且只在角点测试构造处暴露。
- 同一 kernel 内多条 dtype 分支共享 API 约束时，逐分支单独审计约束遵守情况（本任务 transit 分支正确、fp32 分支违反同一 vbrc 规则），不要凭「一条分支修好了」推断另一条也正确。
