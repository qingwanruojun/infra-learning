# RETROSPECTIVE.md — elu_fwd / _make_elu_kernel

> Stage 1 复盘（追加式；schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`）

## Stage 1 (DESIGN) Retrospective — first_design (migration/harness) — 2026-10-28

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | D-1 verify_equiv 的 verdict 段若重跑数据生成段会推进模块级 RNG，判定用的是另一份数据（本次首跑出现「打印全 PASS 但 RESULT: EQUIV_FAIL」的假失败——run_a3() 被调用两次）；且跨 run 不可复现 | verify_equiv.py 修复：每 case 派生种子（`torch.manual_seed(zlib.crc32(f"elu-equiv-{dt}-{hint}"))`）+ main() 单次执行各段、判定复用已存储行 | tilelang-op-design skill Phase 2 第 5 步补一句「D-1 脚本判定段禁止重跑生成段（RNG 推进）；每 case 派生种子保证可复现」 | P |
| Design-flow | 弃选候选的机器证据行初版漏掉 select 骨架（cand_expm1/cand_refactored 直接整式替换），把「负支舍入路径分歧」测成了「公式错误」（首跑 A2 49391 mismatches 混入正轴破坏，数字不自洽）；修正为与基线同骨架、仅替换负支后 16692 mismatches 才是真实证据 | verify_equiv.py `cand_expm1`/`cand_refactored`（含 select 形态）vs 首版；DESIGN.md §1.6.1(c) | tilelang-op-design skill Phase 2 第 5 步或 D-1 模板注记：弃选证据行必须与被比对象同结构骨架、单一变量隔离 | P |
| Constraint-doc | 角点断言规则因算子而异：leaky_relu L0-5 用 sign-bit strict（其 -0 落负支与 golden 一致），ELU 源条件方向（x>0 选正支）与 ATen（x<0 选负支）使 -0 落不同支（+0 vs -0，值相等符号不同）——照抄同族断言规则会产生假失败 | DESIGN.md §8.3 L0-5（值断言 + NaN/Inf 逐位、±0 豁免）vs leaky_relu `_assert_bits`（sign-bit strict）；§0.1 ±0 传播推演 | tilelang-op-design skill 验证方案章节提示「±0 落支按源条件方向逐支推演后定制断言规则，不可照抄同族」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ELU fp32-transit 第三例独立确认且为「结构承接」型：源负支本身是 fp32-opmath + 单次舍回（与 golden 同路径），NPU transit 链 cand≡baseline≡golden 全 0 ulp / 0 violations（6/6 行）——比 leaky_relu 的「修正」型（基线 1 ulp）等价性更强；识别信号 = 源码 cast-to-fp32 中间量 + Cast-back 单次舍回 | D | examples/elu_fwd/_make_elu_kernel/verify_equiv.py（A1 表 6 行 PASS，cand-vs-base ulp=0）+ DESIGN.md §1.6.1(b) 结果表 | python3 examples/elu_fwd/_make_elu_kernel/verify_equiv.py | tilelang 0.1.2+88ee8c302e（本仓 dev root）/ torch CPU 侧验证（无需设备）；设计目标设备 910B3 / CANN 9.0.0 | pattern-library elementwise.md PL-1.17（补第三例 + 「承接型 vs 修正型」分类注记） |
| max-form 结构性不等价机器证据：`e^x > 1+x` 对一切 x≠0 严格成立 → `max(x, α·(e^x−1))` 正半轴恒取负支表达式（32765/65548 mismatches、26990 容差违反、EQUIV_FAIL）——激活函数族「max 化简」候选的结构性反例（与 leaky_relu vmax 条件等价反例〔slope∈[0,1]〕互补：后者是条件失效，前者是恒不成立） | D | verify_equiv.py A4 行 + DESIGN.md §1.6.1(c) | python3 examples/elu_fwd/_make_elu_kernel/verify_equiv.py | 同上 | tilelang-op-design references/algorithm-candidates.md ALG-elementwise（R1 候选备注：max-form 对含超越函数分段激活的结构性陷阱） |
| α=1 算术消除的 ROI 判据应用：bitwise 安全（A3 机器验证 0 ulp）≠ 应进基线设计——copy-floor 实证计算链被 MTE2 窗口完全隐藏（−1 vmul 收益 ≤0.15µs@16M）时，采纳的代价（工厂分支维度加倍）大于收益；正确姿势 = 机器证据留存 + 不进基线 + 登记 Stage 4 杠杆 | P | DESIGN.md §1.6.1(a)/§1.6.0 R1 候选 4/§6.3 ②（A3 三行 PASS 证据留存） | python3 examples/elu_fwd/_make_elu_kernel/verify_equiv.py | 同上 | tilelang-op-design skill Phase 2（「bitwise 安全的计算链优化仍需过 copy-floor ROI 判据」决策模板） |
| 全 buffer vexp 的被屏蔽 lane 溢出语义：正轴大值 lane（x≥89，fp32 exp 溢出）在 vexp 全 buffer 计算下产生 Inf，被 vselect 屏蔽、无 FP trap、lane 独立无害——但需 L0 角点显式覆盖（±89）+ 诊断指纹（输出正轴现 Inf → 查 vselect 条件方向/Out 形态） | P | DESIGN.md §1.6.3 附加说明 / §8.3 L0-5 / §9.1#2；mish 全 buffer v-op 同形先例 | L0-5 ±89 角点（Stage 3 执行） | tilelang 0.1.2+88ee8c302e / CANN 9.0.0（文档级论证，上板待 Stage 3 实证） | tilelang-vector-skill 或 pattern-library traps-runtime.md（vexp 全 buffer 溢出 lane 条目，Stage 3 后升级 D 类） |

### Transferable Lessons（可迁移教训）

- D-1 脚本的 verdict 判定段禁止重跑数据生成段：模块级 RNG 会被推进，判定实际落在另一份数据上（本次假 FAIL 根因）；每 case 派生种子（dtype+hint → crc32）保证逐 run 可复现，判定一律复用已存储的结果行。
- 弃选/否决候选的机器证据行必须与被比对象**同结构骨架、单一变量隔离**：漏掉 select 骨架会把「负支舍入路径分歧」测成「公式错误」，mismatch 数字混入整条正轴而不自洽（首跑 49391 → 修正后 16692 才是 expm1 舍入路径的真实分歧量）。
- 源算子本身是 fp32-opmath 形态时（识别信号：cast-to-fp32 中间量 + Cast-back 单次舍回），NPU transit 是「结构承接」而非「修正」——D-1 可期待 cand≡baseline≡golden 全 0 ulp；此时 R2 采纳依据应写「承接源语义」（bf16 API 缺失仍为独立触发），不得照抄 leaky_relu 的「golden 对齐修正」话术。
- bitwise 安全的算术消除（如 α=1 跳过 ×1.0）仍需过 copy-floor ROI 判据才进基线设计：计算链被 MTE2 隐藏时收益 ≈0，而工厂分支维度加倍是真代价；正确产出 = 机器证据留存 + 弃选理由引用实测常数 + 登记 Stage 4 杠杆。
- 全 buffer 超越函数 op（vexp）会在被 vselect 屏蔽的 lane 上产生溢出 Inf/NaN——lane 独立无 trap 无害，但设计文档必须显式记录该语义并配 L0 角点（±89）与诊断指纹；「tail 陈旧 lane 无害」论证要同时覆盖「溢出值 lane」与「未初始化 lane」两类。
- ±0 落支由条件方向决定（源 x>0 选正支 vs ATen x<0 选负支）：两方向在 ±0 上落不同支时产生零符号差（+0 vs -0，IEEE 值相等）——L0-5 断言规则必须按本算子推演定制（值断言 + NaN/Inf 逐位 + ±0 豁免），不可照抄同族 sign-bit strict。
- vsub 支持标量字面量操作数（T.vsub.md §2.1）+ mish L78 `T.vadd(buf, 1.0, buf2)` 生产先例——单位常量（1.0）可免 vbrc 缓冲（省 4B/elem）；但「标量+原地」组合缺逐字同形先例时须在 §9 留回退方案（one32 via vbrc + guard 降档）。
- 迁移任务的接口契约以「落地侧现状」为唯一权威（本任务开题即 Read wrapper 全文：单参 `kernel(block_size)` L100、torch.dtype 入参 L175、折叠常量 L82–83、ascend_mode L148）——GPU 基类两参调用式只写进 §0 解读，未进入任何 NPU 设计决策行（leaky_relu 阻塞级教训的直接应用，本次零返工）。

## Stage 1 (DESIGN) Retrospective — revision r1（S1-EQUIV-EXEC 门禁修复）— 2026-10-28

> 修复范围：仅 §1.6.1/§1.6.3 文档归属重构（弃选证据行迁移），零设计决策变更。gate 1 复验 pass（0 failures / 0 warnings）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Gate-flow | D-1 弃选候选的机器证据行（含机器判定字样）与采纳项结果表同写 §1.6.1，触发 S1-EQUIV-EXEC 机械子串检查（gate_lint.py 对 §1.6.1 region 全文查禁用判定词）——弃选证据本身是合法内容，错在区域归属；gate-and-retry.md §1 已有归属规则但 skill Phase 2 产出模板未内化 | gate_lint.py L320-328（"EQUIV_FAIL" in r161）+ DESIGN.md 首版 §1.6.1(c) 表 | tilelang-op-design skill Phase 2 第 5 步模板显式分表：§1.6.1 只放采纳项结果表，弃选/否决候选机器证据（含任何失败判定行）直接产出为 §1.6.3 子节（并每行配 docs/testing/examples 佐证路径或「未文档化」标注）；另外 DESIGN.md 引用 gate 规范文字时避免「docs/examples」这类斜杠连写——REF_PATH_RE 会把连写串当作单一路径做存在性校验而误报 | P |

### Transferable Lessons（可迁移教训）

- 机器验证的「采纳项结果表」与「弃选项证据行」必须在设计期就分区落盘：前者进 §1.6.1（且全文不得混入失败判定行），后者进 §1.6.3 弃选小节（含 mismatches/violations 计数 + 佐证路径或「未文档化」标注）——这不是排版偏好而是机械门禁的 region 子串检查语义（S1-EQUIV-EXEC 检查整个 §1.6.1 region 文本）。
- 修复类任务的证据迁移用「归属指引 + 子节归档」双向锚点（§1.6.1(c) 写指向 §1.6.3.1 的归属说明、§1.6.3.1 标注「原 §1.6.1(c) 移入」+ 同脚本同一次执行声明），可避免后续检视误判「证据缺失」或「两处结果表来源不明」。
- 引用规范原文时先过一遍 REF_PATH_RE 心智模拟：`docs/`、`examples/`、`testing/`、`.agents/` 开头的斜杠连写串（如「docs/examples/testing」）会被当作单一路径做存在性校验——表头/行文改用「佐证路径（仓库相对路径）」等措辞。

## Stage 2 (REVIEW) Retrospective — 2026-10-28

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 机械复核条款定义了「skip 项转人工复核」但未定义「机械 FALSE FAIL 的定位与判定」流程：design_calc_check 的 l0c_budget 正则对否定语境不识别（§8.2 否定句字面含 `Scope("Cube")` 即通过 Cube 门槛，随后邻接数字正则把 §4.5 的 UB 算术「19B×8192」按 L0C 累加器公式复算出 608KB fail）——本轮靠自行定位触发行（grep token + 读 checker L163–L195 正则）才排除误判；同构 leaky_relu 设计此前因措辞未含该 token 而判 skip，同一实质两种机械表观 | REVIEW.md「机械复核」段 l0c_budget 行（触发链定位 + 人工判定过程）；`.agents/tools/design_calc_check.py` L163–L195；对比 examples/leaky_relu/_make_leaky_relu_kernel/REVIEW.md L34（skip 表观） | 本 skill §4 Phase 1 第 7 步补一句「l0c/ub 机械 FAIL 须先定位触发行与 checker 正则，甄别否定语境 token/口径噪声后再判维度，留痕于机械复核段」；design_calc_check.py Cube-token 正则宜排除反引号包裹的否定句提及 | R |
| Review-flow | 维度 0 检查项无「提取件 vs GPU 原件双源归因核对」条目：设计对源侧事实的陈述（「源 builder 无 lru_cache」）对 verbatim 提取件为真、对 GPU 原件为假（原件 elementwise.py L2496 有 `@functools.lru_cache(maxsize=32)`，装饰器被 extract_tl_kernel.py 剥离）——「算法解读完整性」现口径只要求覆盖计算语句，host 侧装饰器/imports 类版本信息丢失形态未被清单覆盖（本例零决策影响，同类偏差可能承载真实语义） | REVIEW.md 检视注记 a（两版源文件亲读对比）；提取件 L8（import functools 未使用）vs 原件 L2496 | 本 skill §3 维度 0 表「算法解读完整性」行补注「源侧事实归因陈述须对 GPU 原件单独核对——提取件可能剥离装饰器/imports，非原件全貌」；tilelang-op-design references/migration-analysis.md §2 同步 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| TRAP-vselect-inplace-carried-state「无 loop-carried 状态」豁免判据第二独立证据（检视侧）：Out=B/Out=A 原地 vselect 在 persistent T.serial 纯逐元素算子中安全——触发边界对表（argmax running 类携带 + num_full≥3）+ 每 tile 全量重载重算 + 输出仅同迭代消费 + leaky_relu 910B3 多迭代实测先例（≤205 任务/核同形态）+ lerp 原地 vsub/vmul/vadd 链（perf_opt L262–L264）；reviewer 复核方法 = trap 条目实文与实现逐迭代消费链对表，不轻信设计 §9 自述 | P | REVIEW.md 维度 5 原地 v-op 审计段；traps-compiler.md TRAP-vselect-inplace-carried-state（亲读）；examples/leaky_relu/.../\_make_leaky_relu_kernel.py L267/L306；lerp perf_opt L262–264 | none（判据类；trap 本体 repro = pattern-library repro/TRAP-vselect-inplace-carried-state.py） | 检视轮 tilelang 0.1.2+f388fab185 环境（文档级对表 + 已验证先例引用，未新增上板） | pattern-library traps-compiler.md TRAP-vselect-inplace-carried-state（补「无携带状态豁免」判据行——与 leaky_relu Stage 1 同题提案合并计为两任务证据） |
| 910B3 物理核数实查第三独立复跑（设计 §5.5 记录→leaky_relu Stage 2→本轮 elu Stage 2）：ctypes rtGetAiCoreCount ret=0→20→翻倍 40；并核证 npu_utils.cpp getAiCoreNum 即封装 rtGetAiCoreCount（等价实查路径成立性）、core-split-strategy.md L14 翻倍条款——「检视者独立复跑低成本核数探针」作为 core_split skip 的标准补位手段已跨两任务复用 | D | REVIEW.md「机械复核」段 core_split 行（本轮复跑输出）；tilelang/utils/npu_utils.cpp L124–L137；core-split-strategy.md §1 要素② | python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)" | Ascend 910B3 / CANN 9.0.0 / 2026-10-28 复跑（checker stamp f388fab185） | pattern-library constants.md（910B3 核数数据点 occurrences+1，佐证现查现用规则） |
| kb_stale_check stamp 漂移下的设计决策判定规则实例：设计写作时点（88ee8c302e）「PL-1.6/PL-1.17/TRAP-UB 非 stale」的自述在本轮复跑（f388fab185）已变 STALE——判定依「status:verified 亲验 + 条目 910B3 上板重验存活注记 + 设计持首跑重验承诺」维持决策不否定，登记建议级注记转 Stage 3/4；该三步判定式在 conductor 检视指令有依据但 skill 机械复核段无落笔 | R | REVIEW.md 机械复核段 kb_stale_check 行 + 检视注记 c；pattern-library elementwise.md PL-1.17 origin_task 重验注记（亲读） | python3 .agents/tools/kb_stale_check.py --json（对比设计标注 stamp） | .agents/tools/kb_stale_check.py 当前版 | tilelang-design-review SKILL.md §4 Phase 1（机械复核第 7 步补「stamp 漂移 stale 的三步判定式：status 亲验→存活注记→设计重验承诺，齐备则不否定、登记转后续 Stage」） |

### Transferable Lessons（可迁移教训）

- 机械复核「skip ≠ 免检、FAIL ≠ 有罪」：对机械 FAIL 同样要定位触发链（grep 设计中的 token 命中行 + 读 checker 正则）再下判定——本轮 l0c_budget 608KB「溢出」实为 §8.2 否定句 `Scope("Cube")` + §4.5 邻接 UB 算术的两级解析噪声；纯 Vector 设计的 L0C=0 直接核对即可排除，判定过程留痕而非只留结论。
- 迁移任务的源侧事实陈述要区分「提取件」与「原件」两个真值源：Pattern-B 提取器会剥离 `@functools.lru_cache` 等装饰器，设计引用提取件行做归因时可能「对提取件真、对原件假」——检视者两边都要读并对比（本例恰为动作与原件一致、仅措辞归因不实，按注记处理）。
- 接口契约逐字核对的锚点清单化（wrapper 单参调用行/折叠常量行/ascend_mode 行/SUPPORTED_DTYPES 行/工厂构建行）一次 grep 即可全覆盖，成本低且是防「源侧调用式渗入 NPU 决策行」的决定性检查——leaky_relu 阻塞教训在本任务验证生效（两参依赖仅在 §0 叙述 + §9.2 禁令中）。
- 同族已验证实现的 buffer 清单与字节口径（leaky_relu 17/19B、bs=8192 L1 实测）是维度 2 UB 预算「人工补位」的最强旁证；引用前先亲读实现逐 buffer 对数，防止「同族名号」掩盖结构差异。

## Stage 3 (DEVELOP) Retrospective — first_impl — 2026-10-28

> 范围：`_make_elu_kernel.py`（kernel + golden + L0/L1/L2/Boundary + main）首次实现与全量实测。最终 verdict `[PRECISION_PASS]`。工具链：tilelang 0.1.2（dev root `/mnt/workspace/learn/tilelang-ascend/build`，PYTHONPATH 首位）/ Ascend 910B3 / CANN 9.0.0 / torch 2.7.1。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | L0-5 角点用例首跑失败根因 = 值断言未配 `equal_nan=True`：角点张量含 NaN lane，`_compare`/`assert_close` 以 equal_nan=False 时 NaN lane 被计入 diff → max_diff=nan、violations=0（诊断指纹 = 「max_diff=nan 且 violations=0 + assert_close Tensor-likes not close」），而 `_assert_special_bits`（NaN/Inf 逐位）在 assert_close 之前已通过——证明传播正确、纯为断言参数缺漏；修正（角点 `_try_case(..., equal_nan=True)`）后 6/6 角点 PASS 且 fp16/bf16 角点 max_diff=0.0（逐位） | `_make_elu_kernel.py` `_run_L0_corners`（首跑 6 FAIL → 修正后 ALL PASS）；设计 §8.3 L0-5 断言规则（值断言 + NaN/Inf 逐位 + ±0 豁免） | tilelang-op-develop skill Phase 4 或模板注记：角点/特殊值用例的值断言必须显式 `equal_nan=True`，并记录「max_diff=nan + violations=0」指纹 = 断言参数缺漏而非传播错误 | P |
| Develop-flow | 设计 §9.1#10 回退条款（vsub 标量+原地三要素无逐字先例 → 失败则退 one32 vbrc 形态）**首跑未触发**：`T.vsub(w32, 1.0, w32)` 在当前工具链直接编译运行通过（全 dtype 路线），回退未用、guard 保持 8192 | `_make_elu_kernel.py` L275/L321（vsub 标量字面量 + 原地形态）+ L0/L1 全量 PASS（含 bs=8192 fp16/bf16 L1） | pattern-library traps-compiler.md 或 elementwise.md 补「vsub/vadd 标量字面量 + 原地组合当前工具链可用（ELU 生产实证）」正例条目 | D |
| Develop-flow | 设计期 STALE 条目的首跑重验承诺全部兑现：TRAP-UB-multibuffer-inflation 口径（bs=8192 fp16/bf16 L1 编译运行通过，footprint 非驻留单调）、PL-1.17 transit 0 ulp（角点逐位）、PL-1.6 copy-floor 结构（persistent grid-stride 编译运行）——三条目在本工具链**存活**；唯一「设计期 vs 上板」差异 = fp32 角点 max_diff=5.96e-08（设计 §9.1#1 vexp ULP 附注如期） | `_make_elu_kernel.py` L1 bs=8192 两档 PASS + L0-5 fp16/bf16 0.0/fp32 5.96e-08 + verify_equiv.py 复跑 EQUIV_PASS 6/6 | 无（设计已按「首跑重验承诺」处置，结论现可升级为「本工具链实测存活」） | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ELU 全 dtype 上板实测：transit 路线（fp16/bf16）角点与 randn 全档 **0 ulp 级一致**（L0-5 角点 max_diff=0.0、L1/L2/Boundary max_diff ≤7.6e-6@fp16 / 3.1e-5@bf16，均 ≪ 源仓容差）；fp32 原生域 max_diff=5.96e-08 = vexp 硬件 ULP（设计 §9.1#1 预期）——「vcast rint 升域 → fp32 域链 → vcast rint 单次舍回」在 910B3/CANN 9.0.0 上确证与 golden（F.elu(x.float()).to(dtype)）逐位同路径 | D | `_make_elu_kernel.py` L0-5/L1 输出（角点 0.0 / 汇总 max_diff 表）；verify_equiv.py A1 表（0 ulp 0 viol） | python3 examples/elu_fwd/_make_elu_kernel/_make_elu_kernel.py --level all | tilelang 0.1.2（dev root /mnt/workspace/learn/tilelang-ascend）/ 910B3 / CANN 9.0.0 / 2026-10-28 | pattern-library elementwise.md PL-1.17（补 ELU 第四例 + 910B3 上板 0 ulp 实证，含 fp32 vexp ULP 上限 5.96e-08 数据点） |
| vsub 标量字面量 + 原地三要素组合当前工具链生产可用：`T.vsub(w32, 1.0, w32)` 直接编译运行（无需 one32 vbrc 回退），与 mish L78 `T.vadd(buf, 1.0, buf2)` 标量操作数先例同族扩展——「src1 标量 + dst≡src0」形态从「文档支持无逐字先例」升级为「生产实证」 | D | `_make_elu_kernel.py` L275/L321（transit/fp32 双路线 vsub 原地）+ 全量 PASS（含 alpha=0.5/0.0 用例） | python3 examples/elu_fwd/_make_elu_kernel/_make_elu_kernel.py --level all | 同上 | tilelang-vector-skill 或 pattern-library traps-compiler.md（vsub 标量原地正例条目） |
| 角点用例断言指纹：「max_diff=nan + violations=0 + assert_close not close」= 值断言缺 `equal_nan=True`（NaN lane 未豁免）而非 NaN 传播错误——传播正确性由独立 `_assert_special_bits`（NaN/Inf 逐位）先行验证；设计 §8.3 的「值断言 + NaN/Inf 逐位」双轨须两参数齐配（equal_nan=True + special_bits=True） | P | `_make_elu_kernel.py` `_run_case`/`_run_L0_corners`（首跑 6 FAIL → 修正 6 PASS）；`_compare` diff 计算 | python3 examples/elu_fwd/_make_elu_kernel/_make_elu_kernel.py --level L0 | 同上 | tilelang-op-develop skill Phase 4 测试模板（角点用例断言参数注记） |
| 本仓 dev root（refs/tilelang-mlir-ascend/tilelang）私有 TVM 包缺失不可导入；可用 tilelang 在 `/mnt/workspace/learn/tilelang-ascend`（PYTHONPATH 首位）——`python3 -c` 于 refs 根目录会 import 到损坏 dev root（ImportError private TVM），`python3 /abs/path/{op}.py`（脚本目录在 sys.path[0]）则正常；测试执行须从 refs 根之外目录（如 /tmp）发起 | D | 本任务执行：refs 根 `python3 -c "import tilelang"` → ImportError；/tmp `cd` 后 `python3 /abs/.../op.py --level all` → 全量 PASS | cd /tmp && python3 /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/elu_fwd/_make_elu_kernel/_make_elu_kernel.py --level all | 同上 | docs/安装指南.md 或开发指南.md（dev root 双份时 PYTHONPATH 首位生效规则 + 跑测 CWD 纪律） |

### Transferable Lessons（可迁移教训）

- 角点/特殊值用例的「值断言」必须显式 `equal_nan=True`（NaN lane 在 diff 中产生 nan）；「max_diff=nan 且 violations=0」是断言参数缺漏的指纹，先查 equal_nan 再怀疑 NaN 传播——传播正确性应已由独立的 NaN/Inf 逐位检查先行覆盖。
- 设计 §9 的「无逐字先例 → 留回退」条款（如 vsub 标量+原地）在本工具链上首跑很可能直接可用——按设计实现即可，回退只在上板报错时启用；同时把「回退未触发」作为首跑实证记录进复盘（比「按设计写了回退」更有价值）。
- 迁移任务逐函数跑测时注意 dev root 双份：PYTHONPATH 首位是可用 tilelang，refs 仓库根内自带一份缺私有 TVM 的 dev root——凡 `import tilelang` 的脚本一律从 refs 根之外的 CWD（/tmp）以绝对路径运行，避免 CWD 相对导入踩到损坏 dev root。
- fp32 原生域超越算子（vexp）与 golden 的 exp 存在 ≤1 ulp 硬件差（实测上界 5.96e-08@fp32 角点），属设计预期（§9.1#1），不应据此判失败；fp16/bf16 经单次舍回后实测 0 ulp（角点逐位一致）。
