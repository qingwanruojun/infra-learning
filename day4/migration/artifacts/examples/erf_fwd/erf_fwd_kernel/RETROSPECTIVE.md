# RETROSPECTIVE.md — erf_fwd / erf_fwd_kernel

> Stage 1 复盘（追加式；schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`）

## Stage 1 (DESIGN) Retrospective — first_design (migration/harness) — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | D-1 verify_equiv.py 每任务从零手写（本次公式转录首版漏掉 A&S 7.1.26 的前导 t 因子，首跑 99.75% 违反才定位）；lerp `_compare` 形态参考在 leaky_relu 复盘已建议过模板化，仍未落地，poly/Horner 类公式转录是高频错误面 | examples/erf_fwd/erf_fwd_kernel/verify_equiv.py（as26_fp32_chain 首版 vs 修正版 diff：`m = t * m` 一行）；DESIGN.md §1.6.1 D-1 表 | tilelang-op-design skill Phase 2 第 5 步补「D-1 脚本最小模板 + 公式转录 checklist（逐项：多项式前导因子/符号函数分支/分母分组）」 | R |
| Info-source | negative-claim 举证要求「docs/Tilelang.language/ 全部子目录检索」，但 kb_search 不索引 docs 语料，实际只能手工 grep——无命令模板，每次自行拼 | 本任务 V5 弃选（vrec/vrc）用 `grep -rln "vrec\|vrc\|reciprocal" docs/Tilelang.language/` 零命中；DESIGN.md §1.6.2 V5 | algorithm-research.md §6 或 negative-claim-evidence.md 补一行标准检索命令模板（grep -rln + 目录展开）或扩 kb_search 语料 | R |
| Design-flow | 本仓双树并存（conductor 编排树 examples/{project}/{op}/ 与 TileOPs 集成树 examples/TileOPs/tileops/...）同名算子目录同名文件，跨树引用（尤其 §3.5.3 同类实现与教训文件）极易混淆——本次 leaky_relu 教训文件实际在 examples/leaky_relu/_make_leaky_relu_kernel/ 而非 TileOPs 树，首版引用即错 | gate_lint 首跑 S1-REF-PATHS 两次 FAIL 输出（examples/pattern-library / leaky_relu 路径）；修正后 0 fail | info-sources.md 或 migration-analysis.md §1 补「双树说明」小节（编排树 vs 集成树职责与交叉引用规则） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| T.verf=7 阶泰勒的独立机器复现：fp64 误差表 0.5→9.8e-6 / 1→4.5e-3 / 1.5→0.145 / 2→1.58 / 3→39.1 与 910B3 用户实测逐点吻合，+ docs T.verf.md §3 lowering（ConstantOp+DivFOp+VMulOp+VAddOp，无硬件 erf）双证——erf/erf 类算子禁用 T.verf 的可复用证据链 | D | examples/erf_fwd/erf_fwd_kernel/verify_equiv.py（REJ-1 段输出）+ DESIGN.md §1.6.0 C1 / §1.6.1 REJ-1 表 | python3 examples/erf_fwd/erf_fwd_kernel/verify_equiv.py | torch CPU fp64 侧验证（无需设备）；设计目标 910B3 / CANN 9.0.0 / 2026-10-29 | tilelang-op-design references/algorithm-candidates.md（erf 条目 R1 备注）或 pattern-library traps-compiler.md（T.verf 精度陷阱） |
| A&S 7.1.26 fp32 op-by-op 链全域最大绝对误差 5.868e-07（6M 随机 + 23 角点 × 3 dtype，包络 0 违反）——transcendental 多项式替代的精度基准数据点 | D | examples/erf_fwd/erf_fwd_kernel/verify_equiv.py（A1 表）+ DESIGN.md §1.6.1 D-1 结果表 | python3 examples/erf_fwd/erf_fwd_kernel/verify_equiv.py | 同上 | tilelang-op-design references/algorithm-candidates.md（R1 候选） |
| 910B3 物理核数第四次独立复跑（ctypes rtGetAiCoreCount ret=0→20，纯 Vector 40）：CONST-aicore-910B2C 910B3 数据点 2026-10-29 再确认；本设计环境 tilelang 不可导入（dev-root TVM 缺失），ctypes 等价路径再次可用 | D | DESIGN.md §5.5 要素②（查询代码 + ret=0 + 20）+ 本任务执行记录 | python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)" | Ascend 910B3 / CANN 9.0.0 / npu-smi 25.5.0 / 2026-10-29 | pattern-library/constants.md CONST-aicore-910B2C（追加复跑数据点） |
| D-1 拦截公式转录错误实证：多项式前导 t 因子漏乘 → 首跑 3 dtype 全体 ~10^6 级违反（max err 7.9e-2）→ 单行修正后 5.868e-07/0 违反——「散文论证正确、脚本才能抓住抄录错误」的正面案例（对照 lerp GOLD-DIAG 抓 opmath 分歧案例） | P | verify_equiv.py 首跑/修正两次输出（本文档对话记录）+ DESIGN.md §1.6.1 | python3 examples/erf_fwd/erf_fwd_kernel/verify_equiv.py（修正后版本） | torch CPU / 2026-10-29 | tilelang-op-design SKILL.md Phase 2 第 5 步（动机案例引用） |
| 逐元素 transcendental 链的 NPU 形态基线：22-op v-prefix 链（vabs/vmul/vadd/vdiv/vexp/vcmp/vselect）+ 3 常量缓冲（仅 src0-位常量需 vbrc：vdiv 分子 / vcmp B / vselect A·B）+ 标量字面量第二操作数——多常量多项式族的 UB 预算核算方法（37–41 B/elem） | P | DESIGN.md §3.3 伪代码 + §4.5 预算表 + §1.6.1 A2 | Stage 3 实现后 L0（本设计期无上板 repro） | 设计期（910B3 / CANN 9.0.0 目标）；实测 stamp 待 Stage 3/4 | pattern-library elementwise.md（transcendental 链模式条目） |

### Transferable Lessons（可迁移教训）

- 多项式/Horner 类公式转录为 op-by-op 链前，先做单点 fp64 手算对照（如 x=1 应得 erf(1)=0.8427）再跑 D-1——首跑大范围（>50%）违反 = 公式转录错误第一信号（本次：A&S 7.1.26 的 erfc 型结构是 `t·poly₅(t)·e^(−x²)`，poly 前有前导 t 因子，漏乘即 0.08 量级系统偏差）。
- 多常量多项式链的常量落位规则：vadd/vmul/vsub/vdiv 的第二操作数（src1/src2）标量字面量文档均支持（各 md §2.1/§2.3.4），优先字面量；仅「src0 位」常量（vdiv 被除数、vcmp 的 B、vselect 的 A/B）必须 vbrc 常量缓冲——UB 预算常量项 = src0 位常量数 × 4B/elem，勿按常量总数预算。
- D-1 结果表报 max ulp 时必须限定判读域：golden≈1e-9 的 near-zero 套件，5e-7 的 atol 域绝对误差除以亚正规 ulp 会给出 10^6 级伪 ulp——pass 判据是包络违反数（0），ulp 只在中高值域（如 |x|≥3）有判读意义。
- erf/erfc 类 ±1 饱和域无需显式 clamp：大 \|x\|（含 ±inf、±max_finite）由 exp(−x²) 下溢 → m=0 → r=1 自然饱和，角点断言可直接断 ±1 精确值（vcmp(NaN<0)=False 语义下 NaN 亦正确传播，equal_nan + both-NaN 位断言覆盖）。
- 跨树引用先 ls 确认归属树：本仓 conductor 编排树（examples/{project}/{op}/）与 TileOPs 集成树（examples/TileOPs/tileops/...）同名算子并存，S1-REF-PATHS 对两树都机械核对，教训/案例文件（RETROSPECTIVE.md、verify_equiv.py 等）多在编排树，wrapper/kernel 产品在集成树。
- golden 锚点为 opmath 形态的第三例确认（expm1/elu/erf）：fp16/bf16 一律 fp32 transit + 单次舍回；CPU GOLD-DIAG（原生 dtype torch.erf vs .float() transit 位等价）可先行判定 golden 双形态是否可互换，避免设备侧反复。

## Stage 2 (REVIEW) Retrospective — review — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | design_calc_check 的 ub_budget 不识别「B/elem 口径」预算表（逐元素族的标准写法）→ 机械 skip 后全靠检视者人工重算；本次人工重算发现 A2 收益字段 −56/−112KB 与「9→3=移除 6 缓冲」自述矛盾（正确值 6×bs×4B=49.2/98.3KB）——收益字段与 B/elem 表都在同一机械盲区 | design_calc_check JSON（ub_budget skip）+ REVIEW.md 机械复核段 + DESIGN.md §1.6.1 A2/§4.5 | .agents/tools/design_calc_check.py 增补「B/elem 合计行 × bs ≤ 192KB」解析分支；或 tilelang-design-review SKILL.md 维度 2 固定「B/elem 表人工重算 + 收益字段 = Δ缓冲数×bs×字节宽 交叉核对（与发射省数声明一致性）」检查项 | R |
| Review-flow | 双树**同名不同文件**（kernels/.../erf_fwd.py 与 ops/.../erf_fwd.py 各 219/105 行）的行号引用需检视者两树都打开核对——§0.2「erf_fwd.py L86–87, L186」跨两树各取行号，单树核对会误判引用失真 | DESIGN.md §0.2 行引用 vs 两树实文（ops L85–87 flatten/reshape、kernels L187 output_dtype，均属实）；REVIEW.md 维度 0 注记 | tilelang-design-review SKILL.md 维度 0 检查项补「同名文件行号引用先 find 双树定位归属文件」一句 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 检视者独立核数探针第 5 次复跑：ctypes rtGetAiCoreCount ret=0→20（纯 Vector ×2=40），910B3 数据点在 4 次任务侧复跑后再获检视侧确认；「core_split skip → 检视复跑探针」补位路径第 3 次任务复用成立 | D | REVIEW.md 机械复核段 core_split 行 + .agents/skills/tilelang-op-optimize/references/pattern-library/constants.md CONST-aicore-910B2C | python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)" | Ascend 910B3 / CANN 9.0.0 libruntime / 2026-10-29 | pattern-library/constants.md CONST-aicore-910B2C（复跑数据点追加） |
| 逐元素族 D-1 重跑复核的标准形态 = 逐套件明细逐位比对（非仅 OVERALL 行）：本次 7 套件 max\|err\|（5.835e-07/5.868e-07/5.850e-07/5.621e-07/4.442e-08/1.778e-07/9.715e-08）+ REJ-1 五行表 + GOLD-DIAG 计数全部逐位对上，伪 ulp 度量伪影（near-zero 4194303）须按设计自我声明的判读域核对而非误报 | P | REVIEW.md 机械复核段 verify_equiv 重跑行 | python3 examples/erf_fwd/erf_fwd_kernel/verify_equiv.py | torch CPU / 2026-10-29 | tilelang-design-review SKILL.md Phase 1 第 7 步（D-1 重跑比对口径一句补充） |
| docs/Tilelang.language 负向断言复核实测基线：全目录 vrec/vrc/reciprocal/erfc 检索零命中；子目录实际 16 个（本任务设计误写「全 20 子目录」）——负向断言的检索范围数字也应纳入检视核对 | D | REVIEW.md 维度 8 注记 + 检视 grep 执行记录 | ls docs/Tilelang.language/ \| wc -l; grep -rn "vrec\|erfc" docs/Tilelang.language/ | CANN 9.0.0 仓 / 2026-10-29 | .agents/skills/_shared/standards/negative-claim-evidence.md（补「检索目录数以 ls 实测为准」提示） |

### Transferable Lessons（可迁移教训）

- A&S/erfc 型公式的检视三段核对：① Horner 展开恒等推演 + 前导因子逐项查（t·poly₅(t)·e^(−x²) 的 t 因子在 §3.3 与 verify_equiv 脚本两处都要落位）；② 脚本链与伪代码乘法结合序差异（(poly·e)·t vs (t·poly)·e）判定为 fp32 ulp 级等价而非转录错误；③ ±1 饱和域机制核对（exp(−x²) 下溢自然饱和、无需显式 clamp）——三项缺一则「容差内等价」自述不可信。
- 收益量化字段的复核捷径：把「省 N 个缓冲」与「发射省 N 次 vbrc」两处自述交叉对算（6 缓冲 × bs × 4B），数字互斥即笔误——本任务 −56/−112KB 即 7 缓冲口径残留，与其自身「9→3、省 6 次」声明矛盾。
- KB 单位混用检测：同一文档内 §4.5 用十进制 KB（151.6=37×4096/1000）、A2 用 KiB（9×16KB=144），复算一律回到字节数，差异 >10% 先查单位口径再下结论。
- REJ 类「硬件入口禁用」证据的检视复核 = 双证对齐：docs lowering op 类型 → 公式形态推断（ConstantOp+DivFOp+VMulOp+VAddOp 无 exp 支路 → 多项式）+ 机器复现误差表与用户设备实测**逐点**（非量级）吻合；docs 未写明展开阶数时，「阶数定性由实测反推」是可接受的推断型证据，但须确认禁用结论对阶数不敏感（|x|≥1.5 已击穿）。

## Stage 3 (DEVELOP) Retrospective — first_impl — 2026-10-29

> 复盘对象：erf_fwd_kernel 首次实现（mode=first_impl, attempt=1）。L0/L1/L2/Boundary 全量一次通过（`python erf_fwd_kernel.py --level all` exit 0，无任何 case FAIL/WARN 以外的失败），四出口判定 [PRECISION_PASS]。编程模式 Developer（`ASCEND_MODE = "Developer"`）。实现文件 `examples/erf_fwd/erf_fwd_kernel/erf_fwd_kernel.py`（1054 行）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 本环境 kernel 测试的导入前提未文档化：site-packages 的 tilelang wheel（0.1.1.10+cann900）**无 `tilelang.utils.npu_utils`**（utils 目录仅 deprecated/language/target/tensor），dev-root tilelang-ascend（0.1.2+bd7c60c）才有且需 PYTHONPATH 补 3rdparty/tvm/python——首跑 `from tilelang.utils.npu_utils import NPUUtils` ModuleNotFoundError 才定位 | erf_fwd_kernel.py 首跑 stderr（ModuleNotFoundError）；修复命令 = `PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:/mnt/workspace/learn/tilelang-ascend/3rdparty/tvm/python:...examples/TileOPs python3 ...` | docs/安装指南.md 或 docs/developer/npu runtime.md 补「kernel 测试运行前提：PYTHONPATH 含 dev-root 与 3rdparty/tvm/python；先 `python3 -c "import tilelang, tilelang.utils.npu_utils; print(tilelang.__file__)"` 确认导入 dev root（site-packages wheel 缺 npu_utils）」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| erf A&S 22-op 链上板精度实证：fp32 max_diff 8.028e-07（atol 1e-5 下 12.5× 裕度；D-1 CPU 预测 5.868e-07 + NPU vexp ≤1 ulp 域差传播后仍 ≪ atol）；fp16 4.883e-04（atol 1e-3 下 2×）；bf16 3.052e-05（1.6e-2 下 524×）——transcendental 多项式替代链的 910B3 实测精度基线 | D | erf_fwd_kernel.py 全量跑 `Summary max_diff per dtype: bfloat16=3.052e-05, float16=4.883e-04, float32=8.028e-07`（L1/large/fp32 2^24 用例）；DESIGN.md §1.6.1 D-1 表 | python3 examples/erf_fwd/erf_fwd_kernel/erf_fwd_kernel.py --level all（Summary 行） | tilelang 0.1.2+bd7c60c（dev root tilelang-ascend）/ CANN 9.0.0 / Ascend910B3 / 2026-10-29 | pattern-library elementwise.md（transcendental 链精度基准行） |
| vmul 标量字面量 + 原地 dst≡src + 跨缓冲 dst=载入 buffer 三形态在 3 dtype × bs≤4096 实测全过——CONST-vsub-crossbuf-scalar 的 vsub 单形态证据扩展为 vmul 变体；设计 §9 R1 风险项（「vmul 变体首跑重验，回退 dedicated out_ub」）无回退直接关闭 | D | erf_fwd_kernel.py fp32 路线（L0-1/fp32、L1 bs=4096 全程 A2 标量字面量 + 最终 `T.vmul(e_ub, h_ub, x_ub)` 跨缓冲写回载入 buffer）；无恒等输出指纹（max_diff ≠ 0） | python3 examples/erf_fwd/erf_fwd_kernel/erf_fwd_kernel.py --level all | 同上 | pattern-library elementwise.md CONST-vsub-crossbuf-scalar（补 vmul 变体实证行） |
| erf 逐元素 persistent 模板（本族第 4 例）：工厂 `erf_fwd_kernel(N, dtype, block_size=None)`（K3 去 output_dtype/threads/npt + block_size 仅预设 curried 默认）→ 内层 `@tilelang.jit(out_idx=[1])` kernel(bs) → transit/fp32 双路线；vbrc 体内 let × 3 常量缓冲 + vcmp tensor-tensor + 22-op 链 + 尾块 [0:tail]——migration-harness 逐元素族可直接复制的完整骨架 | P | erf_fwd_kernel.py（factory/kernel 两分支/测试全套）；wrapper erf_fwd.py L100–117 逐字契约；DESIGN.md §3.3/§3.5.2 | python3 examples/erf_fwd/erf_fwd_kernel/erf_fwd_kernel.py --level L0 | 同上 | tilelang-op-develop templates/op_template.py（transit 逐元素族模板候选） |

### Transferable Lessons（可迁移教训）

- 环境前提自检先行：kernel 测试前先确认导入的 tilelang 来自 dev-root 而非 site-packages wheel——`python3 -c "import tilelang, tilelang.utils.npu_utils; print(tilelang.__file__)"`；site-packages 旧 wheel 缺 `npu_utils`，运行 PYTHONPATH 必须含 dev-root 与其 `3rdparty/tvm/python`（否则首跑 ModuleNotFoundError 空耗一轮）。
- vbrc 标量 let 绑定必须在 `with T.Kernel` 体内，且**逐 dtype 分支单独审计**（fp32 与 transit 两路线都要体内 let；本任务按 leaky_relu 教训一次写对，但该陷阱跨函数高发，勿凭「一条分支修好」推断另一条）。
- 常量落位规则（跨函数通用）：`vadd/vmul/vsub/vdiv` 的标量字面量第二操作数 + 原地 dst≡src 组合文档合法且实测可用，**只有 vbrc 有 let 绑定触发面**；需 vbrc 的仅 src0 位常量（vdiv 分子 / vcmp B / vselect A·B），UB 预算常量项按 src0 位常量数计。
- `T.vcmp` 只做 tensor-tensor（域输入 vs zeros_ub 常量缓冲），标量经 cast 后喂 vcmp 会被 codegen 拒绝（tir.Cast 操作数）。
- 多 dtype 路线「结果驻留 buffer → GM 拷出源」逐路线核对：fp32 路线拷出源 = x_ub（载入 buffer，最终 vmul 跨缓冲写回其中）、transit 路线拷出源 = x16_ub（vcast 舍回复用载入 buffer）——两路线拷出源不同，沿用统一拷贝源即恒等输出错误。
- erf/erfc 类 ±1 饱和无需显式 clamp：|x|→∞（含 ±inf、±max_finite）由 x² 溢出 → exp(−x²) 下溢 → m=0 → r=1 自然饱和；角点测试可直接断 ±1 精确值（golden 同步饱和，双端正值位等价）。
- 两段式工厂契约的测试语义：factory 返回值恒为 curried builder，`block_size` 工厂 kwarg 只预设默认不改变返回形态；单段 `(x)` 调用在 build 期抛错是**契约正确行为**，测试应断言「被拒绝」而非当作失败。
