# RETROSPECTIVE.md — expm1 / expm1_fwd_kernel

> Stage 1 复盘（追加式；schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`）

## Stage 1 (DESIGN) Retrospective — first_design (migration/harness) — 2026-10-28

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 调度 prompt「已知事实」预注入把 fp32 1e-5 容差风险框定为「相对误差远超 rtol=1e-5」（引向 rtol 侧），但实际判定口径是 atol + rtol·\|ref\| 联合包络——小 \|x\| 域由 atol 主导（abs err ≤6e-8 ≪ atol 1e-5，裕度 67–10738×），机器验证 0 违反把「核心风险」收口为低；若设计直接顺着 prompt 的 rtol 框架找解法会误入 Taylor 分段采纳（+19 op 零收益） | verify_equiv.py [1] band 表（abs/rel err 双列 × 11 区间）+ DESIGN.md §1.6.1(a) 风险收口段 | conductor 调度 prompt 的精度风险预注入统一标注「判定口径 = atol + rtol·\|ref\| 联合包络，abs/rel 分域判定」；tilelang-op-design skill Phase 2 第 1 步提示「相消族（expm1/log1p 类）先算绝对误差上界（Sterbenz/ulp 分析）再判定，相对误差发散不自动构成违反」 | R |
| Design-flow | 同族已验证先例（leaky_relu/elu）与本案 wrapper 调用面三处不同（工厂级 block_size kwarg vs curried 层传参；`(x, N)` 双实参 vs `(x)`；dtype_str 直传 vs getattr 转 dtype）——「同族 wrapper 同构」假设不成立，任何一处照抄模板都会引入契约错配；本次靠逐字 Read wrapper 全文才捕获 `(x, N)` 非法尾参 | expm1.py L81/L155 verbatim vs leaky_relu.py L98 / elu_fwd.py L100；DESIGN.md §3.5.2 契约分析 + Stage 5 适配清单 | tilelang-op-design references/migration-analysis.md §2.1「I/O 契约」行补注「NPU wrapper 调用面属契约要素：同族先例存在也必须逐字 Read 本案 wrapper，禁止以同族形态外推」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 相消族（expm1 = exp(x)−1）容差判定的 band 表方法论与量化结论：|x|≤0.1 域绝对误差被 ulp(1.0)/2 ≈ 6.0e-8 封顶（exp(x)∈[0.5,2] 减法由 Sterbenz 律精确，误差全部来自 exp 舍入），相对误差虽发散至 1.0 但被 atol 项吸收——fp32 1e-5/1e-5 包络下朴素式 0 违反（54/54 行，最窄裕度 67×）；Taylor 分段（T=0.5, k=2..9）把 x=1e-7 rel err 0.192→5e-8 但 viol 0→0 零测试可观测收益，+19 op 纯负 ROI | D | examples/expm1/expm1_fwd_kernel/verify_equiv.py（[1] band 表 / [2] 54 行判定 / [3] Taylor 量化）+ DESIGN.md §1.6.1(a)(b) / §1.6.3.1 | python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py | tilelang 0.1.2+88ee8c302e（dev root）/ torch 2.7.1 CPU（无需设备）；目标设备 910B3 / CANN 9.0.0 | tilelang-op-design references/algorithm-candidates.md ALG-elementwise（补 expm1 族 R1 候选行：Taylor/fp64 候选的 ROI 判定数据与「绝对误差上界先行」判定顺序） |
| 无 select 逐元素算子的「统一拷出源」第二形态：elu 靠 vselect Out=A 原地写回统一；无 select 算子（expm1）靠末端 v-op 的 dst 直指载入 buffer（`T.vsub(w32, 1.0, x_ub)` 跨缓冲三缓冲形态，vsub.md §2.4 示例一）实现——两路线拷出源统一为 x_ub；误写原地+拷旧 buffer 即恒等映射错误（fp32 路线输出≡输入） | P | DESIGN.md §1.5（逐路线字面推演 + 与 elu 差异注记）/ §3.3 注② / §9.2 防呆条款 | Stage 3 L0（fp32 路线 randn 对拍即拦截恒等输出） | 文档级论证（上板待 Stage 3 实证）；design 期五处交叉一致推演完成 | tilelang-vector-skill（vsub dst 指向载入 buffer 的结果落位形态）或 tilelang-op-design templates/design-template.md 数据流图小节注记 |
| npuir fp64 向量路径缺失的三重文档负向证据（expm1 候选 C 弃选）：T.vexp.md §2.2.1 fp64 ×、T.vcast.md §2.2.1 规格表无任何 f64 行、T.vbrc.md §2.2.1 dtype 列无 fp64——sub-fp32 超越函数算子的「更高精度中间量」候选在 npuir 的系统性与论证模板 | R | DESIGN.md §1.6.0 R1 候选 2（三处亲读引用）+ §1.6.3.1 C 行 | 打开三份 doc 核对 dtype 矩阵单元格 | docs 当前版本（tilelang-mlir-ascend 本仓 docs/） | pattern-library capability-gaps 或 algorithm-candidates.md ALG-elementwise（fp64 transit 路线缺失条目，若未来工具链补 fp64 再翻案） |
| 同族 wrapper 调用面三态差异与「工厂级参数兼容 + curried 双形态」适配模式：expm1 wrapper 工厂级 block_size kwarg 传入时成为返回 kernel 的默认 bs（wrapper 现状调用面 1:1 兼容），同时保留 `kernel(bs)` 单参 curried（E2 家族契约 + L0 双配置复用）——适配收敛为 1 token（删 `(x, N)` 尾参） | P | DESIGN.md §3.5.2（现状 verbatim + 契约分析 + 适配清单）+ expm1.py L81/L155 | Stage 5 集成 + L0「工厂级 block_size 兼容面」用例（DESIGN.md §8.3 附加） | 文档级论证（Stage 5 实证） | tilelang-op-design references/migration-analysis.md §2（wrapper 契约要素）或 harness 集成文档（K11 适配变体注记） |

### Transferable Lessons（可迁移教训）

- 相消族逐元素算子（expm1/log1p 类）的精度判定顺序：先做绝对误差上界分析（exp(x)∈[0.5,2] 时减法由 Sterbenz 律精确 → 误差 = exp 舍入 ≤ ulp(1.0)/2），再对 atol+rtol 联合包络分域判定（小 |x| 看 abs vs atol、大 |x| 看 rel vs rtol）——「相对误差发散」不自动构成违反，band 表（|x| 区间 × max abs/rel err 双列 + 裕度倍数）是设计期收口的标准产出形态。
- 同族已验证先例不能替代本案 wrapper 逐字 Read：expm1 与 leaky_relu/elu 同为 elementwise harness 迁移，但调用面三处不同（工厂级 block_size、`(x, N)` 双实参、dtype_str 直传）——`(x, N)` 在 `@tilelang.jit(out_idx=[1])` + main(x,y) 契约下是必然 TypeError 的非法尾参，只有逐字读才能捕获；「接口契约以落地侧现状为唯一权威」必须落到「每案全文亲读」而非「查过同族」。
- 无 select 的逐元素算子统一拷出源：末端 v-op 的 dst 直接指向载入 buffer（跨缓冲形态）——比 vselect Out=A 原地写回更普适；设计期必须按路线推演「结果落位 buffer == 拷出源」，恒等映射错误的指纹是「fp32 路线输出≡输入」。
- 提取件与 GPU 原件逐件核对：expm1 提取件自带 `@lru_cache(32)`（elu 提取件该装饰器被剥离）——提取管线对不同算子的保真度不一致，「提取件 = 原件全貌」不成立，host 侧装饰器/imports 逐件核对后再写进设计（NPU 侧 lru_cache 直接沿用即可）。
- 轻量调研的「无更优替代」结论也要给足枚举面：本案 R1 覆盖 Taylor/fp64/exp2 降代换/two-sum 补偿/CSE/原生域六族 + 数学操作目录 19 文件全枚举（FMA 缺失证据），弃选证据含机器验证行（band 表/54 行判定/Taylor 量化表）与三重 doc 负向证据——轻量 ≠ 减配举证。

## Stage 2 (REVIEW) Retrospective — full review (migration, 9 dims) — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 对纯 Vector 算子的 `l0c_budget` 稳定误报：把 §4.5「10 B/elem × block_size 8192」的 **UB 字节/元素行**解析为 Cube accum tile（"10x8192 fp32 accum = 320KB"），每次 Vector 迁移检视都需人工判定机械误报并留痕（expm1 与 leaky/elu 同型） | 本检视机械复核段：l0c_budget fail + 人工判定误报（§3.5.1 L0C n/a、§5.1 纯 Vector）；.agents/tools/design_calc_check.py 解析逻辑 | checker 在遇到设计文本含「纯 Vector/无 L0C/不涉及」明示或 buffer 表列头为「字节/元素」时输出 skip(n/a) 而非 fail；review skill 机械复核段模板补「l0c fail × 纯 Vector → 误报白名单判定语句」示例 | R |
| Review-flow | 维度 2 现有检查表未拦截「多缓冲平台值 vs 基础分配差系数」口径混用——该错误形态系 leaky_relu Stage 2 问题 3 已修订过的同类错误在本案复现（×1.10–1.12 被当多缓冲口径使用并声称「一切口径安全」），检视靠亲读 TRAP-UB 原文「不是 resident × 系数」才发现 | DESIGN.md §4.5（×1.12 作平台口径）vs pattern-library/traps-compiler.md TRAP-UB-multibuffer-inflation 第三证（transit≈20.01/fp32≈26.00 B/elem 结构类平台值）与第四证（1.10–1.12 系 multi-buffer 关闭下的分配差）；REVIEW.md 问题 3 | tilelang-design-review SKILL.md 维度 2 增补机检要点：「UB 多缓冲主张必须按结构类平台值 B/elem（20/26 档）核对 guard 上限，引用 ×系数者核对系数出处现象」；tilelang-op-design 模板 §4.5 注记同款 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| curried 工厂契约的「应用段数」核对法：wrapper 调用面兼容性断言须按工厂返回形态逐段数应用（factory → kernel(bs) → executable(x) 为两段；已验证家族 leaky L98/elu L100 均 `(block_size)(x)`），「删一个 token 即 1:1 兼容」类主张在返回 curried builder 的实现下必然失败（tensor 绑定到 bs 参数） | R | REVIEW.md 问题 2（§3.3 `return kernel` vs §3.5.2 `(x)` 适配断言）；elu_fwd.py L100 / leaky_relu.py L98 亲读 | 静态核对 DESIGN §3.3 return 形态与 §3.5.2 适配表调用段数，无需上板 | 文档级（Stage 5/3 实证待验证） | tilelang-design-review SKILL.md 维度 0 接口契约行 + tilelang-op-design references/migration-analysis.md §2.1 |
| verify_equiv 内嵌表「判定列一致、描述列失配」型证据腐化模式：脚本 compare 语义迭代（如 `~eq` 掩码新增/修订）会使旧表 ulp 列不可复现（本案 4/9 行失配：32768* 不可复现、281691 被记作 2），mismatch/violation 列却仍逐值命中——机械复核必须逐格含描述列比对，并把「ulp/位距类描述值 + 掩码语义」列为高漂移风险字段 | R | REVIEW.md 问题 1（重跑 vs §1.6.1(b) 表）；verify_equiv.py compare() `ulp = ulp[finite_pair & ~eq]` 掩码语义 L119 | python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py 与 DESIGN §1.6.1(b) 表逐格 diff | torch 2.7.1 CPU（判定列/种子全复现）；检视环境 tilelang 0.1.2+ef4078a697（≠设计 stamp 88ee8c302e） | tilelang-design-review SKILL.md Phase 1 第 7 步机械复核口径注记 |

### Transferable Lessons（可迁移教训）

- 机械复核的「一致性」按**全部列**执行：判定列（viol/mism）命中不等于表可复现——描述性列（max_ulp 等）是脚本版本漂移的高敏指示器；重跑时同步记录脚本的掩码/判定函数语义，便于定位「表出自旧版脚本」类失配。
- 弃选/保留类决策引用 pattern-library 条目时，除 stale 状态外还要**核对所引数字属于条目内哪一个现象口径**（同一 TRAP 条目内并存多个口径：平台 B/elem 膨胀值、分配差系数、随 num_local_tasks 变化规律）；「引用了条目名」不等于「用对了口径」——guard 上限类安全主张必须用与所主张现象（多缓冲开启）同口径的数值评估。
- 迁移任务的 wrapper 契约断言按「每段应用逐层绑定」核对：工厂级 kwargs 兼容（构建期）与返回 callable 形参兼容（调用期）是两层独立事实；家族先例的调用面要数**括号段数**而非只看首段形态。
- 工具链 stamp 漂移（本检视时点 ef4078a697）会使设计期「非 stale」复核结论全部失效——检视统一以实跑 kb_stale_check 为准，并检查设计是否对每个被引用条目持有首跑/Stage 4 重验承诺（本案仅 CONST-* 有，PL-1.21/PL-1.17/PL-1.6 的 stale 措辞需按问题 4 收敛）。

## Stage 1 (DESIGN) Retrospective — revision 1 (migration/harness, Stage 2 检视不通过 → 修订路径 A) — 2026-10-29

### Revision Summary

- 修订输入：REVIEW.md 3 阻塞（问题 1 内嵌表不可复现 / 问题 2 wrapper 契约断裂 / 问题 3 UB 口径误用）+ 4 建议（问题 4–7），全量采纳；基线 history_version/design_v1.md；检视 pass 项（§0 三问、分核三要素、persistent、Developer、A1 采纳、golden 独立性、逐 dtype buffer 数据流）零回退。
- 契约形态决策：问题 2 选**方案 (a) 两段式**——§3.3 `return kernel`（curried）不变，工厂级 block_size kwarg 仅预设 default_block；wrapper 适配 = 删 `, N` + 增 `(block_size)` → `expm1_fwd_kernel(N, dtype_str, block_size=block_size)(block_size)(x)`（家族标准，leaky_relu.py L98 / elu_fwd.py L100 同形亲读确认）。方案 (b)「返回形态随 kwarg 切换」否决：双形态偏离家族契约 + Stage 3/L0 复杂度。
- UB 口径决策：问题 3 修定为**分 dtype guard**（fp32 4096 / fp16·bf16 8192）+ TRAP-UB 结构类平台值口径（transit 20.01 / fp32 直连 26.00 B/elem）；×1.10–1.12 归还第四证「基础分配差（multi-buffer 关闭）」语境；探针条款挂 Stage 3 首跑（harness 流程无独立 Stage 4）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | verify_equiv.py 的随机数据流依赖 main() 执行顺序（band_table 先消费 RNG），「部分执行」取数（单独调 randn_sets）会得到与完整脚本不同的 mismatches/max_ulp（本修订实测：fp32 randn 496101/143978 vs 完整跑 496251/281691）——内嵌表必须以「完整脚本执行输出」为唯一誊录来源，补充分析须复刻 RNG 消费顺序 | 本修订重跑 + 复刻验证（band_table() 先行后 randn_sets：81.7% 失配 lane ≤2 ulp）；verify_equiv.py main() L215–228 | tilelang-op-design skill Phase 2 第 5 步补注「内嵌表誊录自完整脚本 main() 输出；D-1 门禁的『执行输出』指完整 main() 而非任意子集执行」 | R |
| Design-flow | 「guard 上限安全主张」的口径核对清单缺项：TRAP-UB 同一条目内并存三个口径（第三证平台 B/elem 值 / 第四证分配差系数 / nl·extents 浮动规律），v1 引用条目名却用了错误现象的数字——建议 skill §4.5 模板强制「引用平台值条目时逐现象标注出处证号」 | DESIGN.md v1 §4.5（×1.12 作平台口径）→ 修订版（20.01/26.00 平台值 + ×1.10–1.12 归位第四证 + 分 dtype guard 4096/8192）；traps-compiler.md TRAP-UB 第三/四/五证 | tilelang-op-design templates/design-template.md §4.5 增加必填行「multi-buffer 口径 = 结构类平台值 B/elem（transit≈20/fp32≈26），×系数仅用于 multi-buffer 关闭语境」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| fp32 直连（无 transit vcast 链）elementwise 的 UB guard 上限实测律：TRAP-UB 平台值 26.00 B/elem 下 UB 192KB 的 fp32 guard 上限 = 4096（26×4096≈106.5KB 含 44% 浮动余量），8192 档 ≈213KB 超限；transit（vcast 链）20.01 B/elem 下 8192 档 163.9KB 可行（leaky 第五证 910B3 实测先例 + +10% 浮动仍 180.2KB 安全）——同族 fp32 直连设计应默认分 dtype guard 而非统一 8192 | D | DESIGN.md revision 1 §4.5 分 dtype guard 表 + §9.1#4；traps-compiler.md TRAP-UB 第三证规律 1（26.00 B/elem 结构类）+ 第五证（leaky bs=8192 transit 编译运行通过） | BishengIR `ub overflow requires` 报文编译探针（Stage 3 首跑复算；fp32 bs=8192 预期报 requires > 196608 bits） | 设计期估算（TRAP-UB 第三证平台值 + 算术复算；上板探针待 Stage 3） | tilelang-op-design templates/design-template.md §4.5 或 pattern-library/INDEX.md（fp32 直连 guard=4096 档位条目） |
| 相消族 expm1 的实测精度包络数据（revision 1 复跑固化）：fp32 randn 判定集 496251/2²⁰ mismatches、81.7% 失配 lane ≤2 ulp、>2 ulp lane 绝对差 ≤8.941e-08 ≪ atol（最窄裕度 67×@band (0.1,1]）；band 表 11 区间 abs/rel 双列——「绝大多数 lane ≤2 ulp、小 |ref| lane 位距大但绝对差极小」的两层表述模板可防 Stage 3 误写 bitwise 断言 | D | DESIGN.md revision 1 §1.6.1(b) 表 + 位距分布脚注；verify_equiv.py 完整跑输出 | python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py（模块级种子 20261028） | tilelang 0.1.2+ef4078a697 / torch 2.7.1 CPU | algorithm-candidates.md ALG-elementwise（expm1 族行，revision 1 数字替换 v1 失效数字） |

### Transferable Lessons（可迁移教训）

- 内嵌证据表的可复现性以「完整脚本执行」为准、补充分析须复刻 RNG 消费顺序——「判定列一致 + 描述列失配」的表是脚本版本/执行方式漂移的高敏信号（Stage 2 复盘已录 R 类，本次修订落地：§1.6.1(b) 表逐格 = 完整 main() 输出）。
- wrapper 契约断言以「工厂返回形态（恒 curried / 随参切换）+ 调用段数」双要素核对，并对照家族至少两例的调用面 verbatim——适配清单给完整调用式而非 token 差异描述（「删 1 token 即兼容」类主张在 curried 返回下必然断裂）。
- 引用 pattern-library 条目做 guard 主张前先「对现象、对证号」：平台值（第三证）/ 分配差系数（第四证）/ nl·extents 浮动规律（第三证规律 2/3）三者不可互代——v1 把第四证系数当第三证平台值用，导致 fp32 guard 8192 的安全主张被推翻（修订后分 dtype 4096/8192 + 浮动余量显式计算）。
- 工具链 stamp 漂移后，设计期一切「非 stale」复核结论作废重跑；修订版头部 stamp 行应写明「写作时点 stamp → 修订时点实跑 stamp」双记录，被引条目统一 stale 口径（本次 ef4078a697 实跑六条目全 stale）。

## Stage 2 (REVIEW) Retrospective — revision-1 复审 (migration, 9 dims) — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 设计内嵌「复刻 RNG 消费顺序」类补充分布断言（81.7%/90908/8.941e-08）在随附脚本 stdout 中无输出，检视侧只能手写复刻代码验证——证据可核验性依赖检视方自行重建取数路径；本次复刻成功（81.68%/90908/8.9407e-08 命中），但该形态天然脆弱（RNG 顺序一变即失效）。建议门禁口径补充：D-1 采纳表若含脚本未直接输出的补充断言，脚本本身应带可选分布打印（或断言降级为脚本直出数字） | 本复审复刻验证代码（band_table→randn_sets 顺序 + `~eq&finite` 掩码逐 lane 统计）vs DESIGN §1.6.1(b) 脚注行；verify_equiv.py main() 无分布输出段 | tilelang-design-review SKILL.md Phase 1 第 7 步机械复核口径补注：「补充实测断言须可由随附脚本直接或经记录序复刻产出；检视侧复刻过程写入机械复核段留痕」 | R |
| Review-flow | `design_calc_check.py` l0c_budget 对 UB「B/elem × block_size」表行的同型误读第三次复现（expm1 v1 "10x8192→320KB"、rev1 "26x8192→832KB"，此前 leaky/elu 检视亦有）——纯 Vector 设计每轮都要人工白名单判一次并留误报签名注记，属重复机械成本 | 本复审机械复核段 l0c_budget fail 判定 + 首轮 REVIEW 机械复核段（同型）；.agents/tools/design_calc_check.py 解析逻辑 | checker 增加 n/a 短路（设计文本含「纯 Vector/无 L0C」明示或表头「字节/元素」→ skip）；已在首轮复盘登记 R 类，本次为第二/三次独立证据，满足 Tier 1 升级条件 | R |
| Review-flow | 家族先例「行号锚点」随文件编辑漂移（elu 两段式 return 设计引 L100、实读 L99；leaky L98 精确）——内容 verbatim 一致而行号 off-by-one 时，检视只能按「引用内容 + 邻近行号」双查判定，单靠行号 grep 会假阴性 | 本复审亲读 elu_fwd.py L99（`return _make_elu_kernel(N, dtype, alpha)(block_size)(x)`）vs DESIGN 多处「L100」；leaky_relu.py L98 精确命中 | tilelang-design-review SKILL.md 维度 7 行号引用抽查要点补注：「先例引用按 verbatim 语句为主锚、行号为辅；off-by-one 且内容命中记建议级不阻塞」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 修订复审「闭环三件套」验证模板：证据类修订=重跑脚本逐格比对 + 文字断言复刻取数（RNG 序）；契约类修订=工厂返回形态/最终调用式段数一致 + 家族 verbatim 主锚 + 全文规范位置残留扫描（批判性引用豁免）；资源类修订=条目原文口径亲读 + 独立算术复算 + 实测先例独立复跑——三类各自收敛判据明确，复审结论「通过」可全部机械背书 | P | 本复审修订闭环核实表（问题 1/2/3 + 建议 4–7 全 PASS 的取证过程）；REVIEW.md revision-1 机械复核段 | 对照 REVIEW.md「修订闭环核实表」逐行复核取证命令 | 检视环境 torch 2.7.1 CPU / tilelang ef4078a697 stamp | tilelang-design-review SKILL.md（§4 工作流程可增补「revision 复审模式」小节：闭环核实表为通过时的必产出形态） |
| expm1 修订版精度包络数据经复审二次独立复现（9/9 表行逐格 + 54/54 0 违反 + GOLD-DIAG 全 0 + 分布 81.68%/78.32%/fp16 max 1）——D-1 证据在修订时点完整可复现，支持「内嵌表=完整脚本输出」誊录纪律的收敛效果 | D | 本复审 verify_equiv 重跑输出 + 复刻分布统计 vs DESIGN revision 1 §1.6.1(b)/表脚注 | python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py | torch 2.7.1 CPU（模块级种子 20261028）/检视时点 tilelang 0.1.2+ef4078a697 | algorithm-candidates.md ALG-elementwise expm1 行（Stage 2 复审存活戳注记，随任务终态蒸馏） |

### Transferable Lessons（可迁移教训）

- 修订复审必须「验证到决策行、复锚到实测」：修订自述（「已按实跑重填」「已改两段式」）只是索引，每条闭环判定的成立来自复审方独立重跑（verify_equiv）、独立复算（UB 四档/逻辑核数表/余项 2.7e-10）、独立原文亲读（TRAP-UB 证号口径/PL-1.21 msprof 口径/家族 return 行）——文字断言与数字断言同等对待，理想化推导值（81.7% 类）同样可复刻验证。
- 契约闭环的判定粒度 = 「规范位置零残留」：除主锚点（§3.3 vs §3.5.2）逐字一致外，须全文 grep 旧断言关键词（1:1/单段/直调）并区分「被推翻主张的批判性引用」（允许，存于修订记录）与「规范性引用残留」（不允许）——只看主章节会漏 §8.3/§9/§10 的旧形态残留。
- 资源口径闭环的判定顺序 = 条目原文（对证号/对现象）→ 独立算术复算 → 实测先例独立复跑（第五证查 TRAP-UB origin_toolchain 行而非设计转述）——三步齐备才给维度 2 pass。
- 机械检查误报（l0c_budget）按「误读签名一致性」处置：同型解析错位在新数字下复现即沿用白名单判定并在复核段留原文数字（"26x8192→832KB"），保证后续复审可识别同源误报而非真实越限。

## Stage 3 (DEVELOP) Retrospective — first_impl/attempt 1 (migration/harness) — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 契约分析对「单段直调失败形态」的预测（build 期 TypeError，DESIGN §3.5.2 契约分析 2）与实际工具链不符：带 host guard 的 kernel 在 `if bs <= 0` 对 tensor 求值处抛 RuntimeError（"Boolean value of Tensor with more than one value is ambiguous"）、不带 guard 的最小形态在 alloc_ub 处抛 DiagnosticError——失败类型不同但「必须拒绝」契约行为均满足；契约验证断言若锁死 TypeError 会假失败 | expm1_fwd_kernel.py L0-extra/L1 单段直调用例（RuntimeError）+ repro/TRAP-curried-single-stage-reject.py（DiagnosticError）；DESIGN §3.5.2 契约分析 2（"build 期 TypeError"） | 契约类「必须失败」用例的断言口径统一为「必须拒绝（任意异常）」，文档预测失败形态时加「实际异常类型以上板实证为准」后缀 | R |
| Develop-flow | 探针转正时最小 repro 的 kernel 结构若覆盖不全（单核 T.Kernel(1) 只写前 bs 元素），剩余 lane 是未初始化 garbage，断言失败信号与 API 问题无法区分（本次首跑 75% mismatch 实为 repro 结构错误、非 vsub 问题）——最小化 ≠ 丢弃覆盖度 | repro/CONST-vsub-crossbuf-scalar.py 首跑（T.Kernel(1)+N=4096 只写前 1024 元素）vs 修正后（T.Kernel(ceildiv(N,bs)) 全覆盖 PASS） | repro README（ED-B）补一条：最小 repro 的 kernel 分核必须覆盖全部输出 shape（N 全域），否则 garbage lane 会伪装成 API 行为异常 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| fp32 路线跨缓冲 vsub+标量字面量（`T.vsub(w32, 1.0, x_ub)`，dst=载入 buffer、dst≠src0，T.vsub.md §2.4 示例一三缓冲形态）在 ef4078a697 编译运行通过且数值正确（max_diff 9.537e-07 ≪ fp32 atol 1e-5；非恒等输出）——设计 §9.1#8「无逐字同形生产先例 → 回退 one32 缓冲」的风险条目在本工具链消除，无 select 逐元素算子可直接用 dst 指载入 buffer 实现「结果落位 buffer = 拷出源」统一 | D | expm1_fwd_kernel.py fp32 路线（L0-1/L1 全过）+ repro/CONST-vsub-crossbuf-scalar.py（自包含断言 + 非恒等指纹检查） | python3 examples/expm1/expm1_fwd_kernel/repro/CONST-vsub-crossbuf-scalar.py | tilelang 0.1.2+ef4078a697 / Ascend 910B3 / CANN 9.0.0 / torch 2.7.1+torch_npu | pattern-library traps-compiler.md 或 elementwise.md（vsub 三缓冲形态可用实证；设计 §9.1#8 回退需求降级） |
| transit 8192 / fp32 直连 4096 分 dtype UB guard 的 Stage 3 首跑复验存活：fp16/bf16 bs=8192（平台值 20.01B/elem ≈163.9KB）与 fp32 bs=4096（26.00B/elem ≈106.5KB）在 910B3 全部编译运行通过、无 BishengIR `ub overflow requires` 报文——TRAP-UB 平台值口径（设计 §4.5，REVIEW 问题 3 修订）上板实证存活；fp32 8192 档未试（guard 拒绝 4097+，按平台值 213KB 超限排除合理） | D | expm1_fwd_kernel.py L1 sweep（fp16/bf16 bs=8192 + fp32 bs=4096 全 PASS）+ L0-extra UB guard 分 dtype 拒绝用例 | python3 examples/expm1/expm1_fwd_kernel/expm1_fwd_kernel.py --level all（L1 block_size sweep 段） | tilelang 0.1.2+ef4078a697 / Ascend 910B3 / CANN 9.0.0 | pattern-library traps-compiler.md TRAP-UB-multibuffer-inflation（910B3 ef4078a697 重验戳刷新，stale → verified 候选） |
| curried 工厂单段直调拒绝的实际失败形态实证：tensor 绑定 bs 参数后失败点 = kernel 体 host guard（`if bs <= 0` 对多元素 tensor 求值 → RuntimeError）或 trace 期 alloc_ub（→ DiagnosticError）——与设计预测的 TypeError 不同；「必须拒绝」契约成立，「具体异常类型」随 kernel 体结构漂移 | D | expm1_fwd_kernel.py L0-extra/L1 单段直调用例（fp16/fp32 双路线）+ repro/TRAP-curried-single-stage-reject.py | python3 examples/expm1/expm1_fwd_kernel/repro/TRAP-curried-single-stage-reject.py | tilelang 0.1.2+ef4078a697 / Ascend 910B3 / CANN 9.0.0 | pattern-library traps-compiler.md（curried 契约失败形态条目）或 tilelang-op-design migration-analysis §2.1 |
| 三条 dtype 路线在 910B3 的实际编译运行 + 精度包络数据：fp16/bf16 transit（5 步链）与 fp32 原生域（3 步链）各自独立编译（非互推），max_diff 与 verify_equiv 设计期预测逐位一致（fp16 randn 6.104e-05 / bf16 3.052e-05 / fp32 7.629e-06，均 ≪ atol 1e-3/1.6e-2/1e-5）；2^24 大规模多迭代（205 任务/核）兜底原地/跨缓冲纪律通过 | D | expm1_fwd_kernel.py L0-1/L0-2/L1 输出（每条路线独立 "AscendNPU IR compile success"）+ 特殊值专项验证（88.8→+Inf 双侧相等、-88.8→-1.0、±0 值相等、1e-7 abs err 1.92e-8 ≪ atol） | python3 examples/expm1/expm1_fwd_kernel/expm1_fwd_kernel.py --level all | tilelang 0.1.2+ef4078a697 / Ascend 910B3 / CANN 9.0.0 / torch 2.7.1 | algorithm-candidates.md ALG-elementwise expm1 行（A1 设计期机器验证 → 上板实证闭环） |

### Transferable Lessons（可迁移教训）

- 同 kernel 多条 dtype 分支共享 API 约束时，逐分支实际编译跑测（fp16/bf16 transit 通过 ≠ fp32 原生域正确；本次三条路线各自独立 "AscendNPU IR compile success" + 独立精度对比才收口）——「一条分支修好了」推断另一条是 false economy。
- 无 select 的逐元素算子 fp32 路线直接让末端 v-op 的 dst 指向载入 buffer（跨缓冲三缓冲形态 `T.vsub(w32, 1.0, x_ub)`）即可统一「结果落位 buffer = 拷出源」——本工具链实测可用，不需要 one32/vbrc 回退；恒等映射错误的指纹是 fp32 路线输出≡输入，randn 对拍即拦截。
- 契约「必须失败」类用例断言「必须拒绝（任意异常）」而非锁死具体异常类型——单段直调 tensor 绑定 bs 的失败形态随 kernel 体结构漂移（host guard → RuntimeError、无 guard → trace 期 DiagnosticError），锁 TypeError 会假失败。
- 最小 repro 的 kernel 分核必须覆盖全部输出 shape（单核 T.Kernel(1) + N>bs 只写部分元素 → 未初始化 garbage lane 伪装成 API 异常信号；断言失败先查 kernel 覆盖度）。
- 特殊值断言用 `out == ref` 精确相等分支先行（Inf==Inf、±0 IEEE 值相等）+ equal_nan（NaN 双方相等），对 expm1 不用 sign-bit strict——88.8 双侧 +Inf、-88.8 → -1.0、±0 零符号差豁免、1e-7 相消区 abs err 1.92e-8 ≪ atol（rel err 19.2% 由 atol 吸收）全部按设计预期通过。
