# RETROSPECTIVE.md — bitwise_not_fwd_kernel (migration-harness, Stage 1 first_design)

> 追加式复盘日志（canonical schema: .agents/skills/tilelang-skill-evolution/references/retrospective-schema.md）。本文件为 Stage 1 设计阶段复盘（2026-10-29，first_design 一次通过，无 revision）。

## Stage 1 (design) Retrospective — attempt1-completed — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| docs | T.vxor.md §2.1 声称 src1 支持「Python 标量 int/float（提升为常量并广播）」，但 910B3 本工具链实测拒绝标量操作数（调度 prompt 实测事实 4，同一工具链族 leaky_relu/erf 的 TRAP 记录同口径：标量必须 let 绑定 vbrc）——文档规格与实测行为冲突，设计期只能以实测为准 | docs/Tilelang.language/逻辑操作/T.vxor.md §2.1 vs 本任务 DESIGN.md §9 R1 | docs T.vxor.md §2.1 增加实测限制注（标量 src1 在 CANN 9.0.0 工具链未验证/被拒，shaped buffer 为已验证形态），或在 pattern-library traps-compiler.md TRAP-expert-v-operands 补 vxor 条目 | P |
| docs | T.fill 无专页文档（docs/Tilelang.language/ 全 16 子目录无 fill 条目），仅 examples 两处使用（deepseek_v4/inference/kernel.py L381、deepseek_v32/sparse_mla_fwd.py L96）+ 本任务实测事实 4 直证——按 AGENTS.md 负向断言规则「全目录检索后再断言」，fill 属「有使用先例、无规格文档」状态 | `grep -r "T.fill" docs/` 零命中；examples 双命中；本任务 DESIGN.md §3.2/§9 R1 | docs/Tilelang.language/ 补 T.fill.md（或创建操作/内存操作 下挂规格页：buffer+scalar 签名、dtype 矩阵、核级物化复用语义） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| bitwise 族 NPU 迁移三路线模板：int 域原生 T.vnot（docs 逻辑操作/T.vnot.md int 全宽 √）+ uint8→int8 位视图 + bool→int8 位视图 vxor(1)——原生 bool 域被 docs 禁止 GM 落盘且实测挂死、uint8 out_idx 分配缺陷，位视图是「零 wrapper 编辑 + 实测通过」唯一无缝路线；等价性为逐位精确（D-1 8 行全 EQUIV_PASS，0 违反） | P | 本任务 DESIGN.md §0.6 R2 / §1.6.1 A1–A3 / examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py | python3 examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py（exit 0） | tilelang 0.1.2 dev-root / CANN 9.0.0 / Ascend 910B3 / 2026-10-29 | pattern-library（后续 bitwise and/or/xor/shift 族迁移直接复用） |
| 位运算等价性机器验证无需 fp64 参照：bitwise op 全程精确，D-1 参照取原生 dtype 位语义（torch.bitwise_not）逐位相等，强于 fp64 口径且无舍入歧义 | P | examples/bitwise_not/bitwise_not_fwd_kernel/verify_equiv.py（参照口径注释 + 8 行结果表） | 同上 | 同上 | tilelang-op-design skill algorithm-research.md §3（D-1 执行口径补注：精确整数域算子参照口径） |
| 910B3 物理核数第五/六次独立复跑一致（ret=0→20→纯 Vector 40）：本任务设计期 ctypes rtGetAiCoreCount 复跑，与 CONST-aicore-910B2C 910B3 数据点（leaky/elu/erf 五次）一致 | D | 本任务 DESIGN.md §1.6.0 D-2 / §5.5 要素②（查询命令 + 返回值记录） | `python3 -c "import ctypes; lib=ctypes.CDLL('<cann>/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)"` | tilelang 0.1.2 dev-root / CANN 9.0.0 / libruntime 25.5.0 / 2026-10-29 | pattern-library/constants.md CONST-aicore-910B2C（追加复跑计数，Tier 0） |

### Transferable Lessons（可迁移教训）

- **GPU workaround 不等于 NPU 缺陷**：源算子用 xor 全 1 替代 T.bitwise_not 是 CUDA int4 向量化缺陷的 workaround（elementwise.py L2276–2278 docstring 明示）——迁移时须读 docstring/注释辨析「workaround 动因」，NPU 有原生 vnot 时 workaround 直接等价替换（op 2→1），但 bool/uint8 的 NPU 侧自有约束（bool i1 禁落盘、uint8 out_idx 分配缺陷）要求新的等价路线（int8 位视图），不能反向照抄「xor 直译」了事（§1.6.0 C1 vs C2/C4 的区分即此）。
- **文档声称 ≠ 实测行为**：T.vxor.md §2.1 标量 src1 与实测冲突——同族先例（CONST-vsub-crossbuf-scalar：vsub 标量形态实测可用）不能外推到其他 v-op；凡设计依赖「标量操作数」形态，逐 op 须有本工具链实测或回退形态（本设计：fill shaped buffer 为主、vbrc 为回退）。
- **wrapper 契约是迁移签名的唯一权威**：bitwise_not wrapper 的 K3 注释明确 factory 丢 output_dtype/threads（`(N, dtype)`），与 erf wrapper 保留 7 参的契约不同——同族不等于同签名，每个 op 的工厂签名以**该 op wrapper 注释逐字核对**为准（本设计 §3.5.2 显式记录差异，防止 Stage 3 照抄 erf）。

## Stage 2 (review) Retrospective — completed（结论: 通过）— 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | 检视维度 2 与 design_calc_check 的 ub_budget 检查项对「auto-multi-buffer 开/关 × 膨胀系数」的**口径 regime 一致性**无显式核对位：CONST-capacity-910B2C 携带三个系数（auto-MB 开启 ~1.7× / auto-MB 关闭 1.10–1.12（CG-2026-0008）/ 关闭+显式 alloc 结构 0.981（r9 反例）），本任务设计在 §2.3 声明 auto-multi-buffer 开启、却在 §4.5 用 +12%（关闭态口径）论证全 dtype 统一 cap=8192——该不一致在机械检查 skip（表未解析）后仅靠人工复核暴露（REVIEW.md 建议级①）；checker 默认 ub_inflation=1.7 即开启态口径，两套口径的适用条件分散在条目正文 | examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md §2.3 vs §4.5；pattern-library/constants.md CONST-capacity-910B2C 三个系数的 regime 标注；REVIEW.md 机械复核段 ub_budget skip 人工结论 | tilelang-design-review skill 维度 2 表补一行「膨胀系数口径与模式声明一致（auto-MB 开启须按 ~1.7× 评估大 block 上限）」；或 hardware-cost-model.md §1 Phase 4 行加 regime 对照注 | R |
| Info-source | pattern-library 锚点语义歧义增加复核成本：PL-1.21 条目记载的「2.3×」是 **erf/expm1 链长比**（294.81/128.14µs，同文件同 block_size 对照），本设计复用的「2.3×」是 **expm1 实测/自算流量下界比**（128.14/55.6µs，55.6µs 非条目记载值）——两比值数值巧合（均 ≈2.3）语义不同，检视侧须独立重算才能区分继承还是自生，误引风险留给后续任务 | .agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md PL-1.21「第四例 erf」段；DESIGN.md §1.6.0 D-2 校准锚行；REVIEW.md 建议级④ | elementwise.md PL-1.21 条目补注 expm1 16m fp16 的流量下界口径值（40 核/30.2GB/s → ≈55.6µs）并区分「链长比 2.3×」与「实测/下界比」两种倍率语义 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 910B3 物理核数第七次独立复跑一致（Stage 2 检视侧，ret=0→20→纯 Vector 翻倍 40）——core_split 机械 skip 的标准补位手段跨四任务存续 | D | examples/bitwise_not/bitwise_not_fwd_kernel/REVIEW.md 机械复核段 core_split 人工结论 | `python3 -c "import ctypes; lib=ctypes.CDLL('/opt/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64/libruntime.so'); c=ctypes.c_uint32(); print(lib.rtGetAiCoreCount(ctypes.byref(c)), c.value)"` | tilelang dev-root（不可导入环境）/ CANN 9.0.0 / libruntime 25.5.0 / Ascend 910B3 / 2026-10-29 | pattern-library/constants.md CONST-aicore-910B2C（追加第七次复跑计数，Tier 0） |
| 迁移任务维度 0 重设计可行性的最强证据 = §3.3 伪代码与同族 landed kernel 逐行同构对照（erf/expm1 的 num_kernels=min/serial grid-stride/guard/tail T.min/零原点切片 store 六要点逐一比对），比散文论证可核验 | P | REVIEW.md 维度 0「NPU 重设计可行性」段（erf_fwd_kernel.py L227–228/L269–279 vs DESIGN.md §3.3）；PL-1.22 模板原文 | none（检视方法类，操作 = grep 两文件结构行对照） | tilelang 0.1.2 dev-root / CANN 9.0.0 / Ascend 910B3 / 2026-10-29 | tilelang-design-review skill SKILL.md 维度 0 表「NPU 重设计可行性」行通过标准补「与 landed 同族模板逐行同构对照」为核对手段 |
| 精确整数域算子的维度 6 检视口径：atol/rtol 默认规则不适用，torch.equal 精确比较（源 exact_compare 锚）+ dtype fingerprint + 值域断言是更强合规形态；bitwise 迁移的实测事实应逐项转化为 L0-extra 回归拦截（uint8 float32 分配缺陷→dtype 断言、bool 挂死→拦截项） | P | REVIEW.md 维度 6 段；DESIGN.md §8.2/§8.3 L0-extra（位视图 fingerprint、原生 bool 拦截） | python3 examples/TileOPs/tests/ops/test_bitwise_not.py（集成后由 Stage 3 承接） | Ascend 910B3 / torch_npu 2.7.1 / CANN 9.0.0 / 2026-10-29 | tilelang-design-review skill 维度 6 精度标准行补「整数/布尔域用精确比较并说明依据」 |

### Transferable Lessons（可迁移教训）

- **UB 膨胀系数先问 regime 再问数字**：CONST-capacity 三系数（1.7×/1.12×/0.981）的适用前提互斥（auto-multi-buffer 开/关、结构形态），检视 §4.5 类预算表时第一步核对设计 §2.3/§7.3 的 auto-MB 声明与所用系数是否同 regime，再看 cap/默认档数字——checker 的 ub_inflation 默认 1.7 即开启态口径，与 skip 后人工复核须同口径。
- **复用 pattern-library 锚点先核语义再核数值**：数值巧合（两个 2.3×）会掩盖语义漂移；引用条目记载比值时保持原语义，自算比值另行标注计算式，检视侧对「≈X×」类锚点一律要求给出分子分母定位。

## Stage 3 (develop) Retrospective — first_impl|attempt1-completed — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 运行环境 PYTHONPATH 含两棵 tilelang 树：cwd=repo（refs/tilelang-mlir-ascend）导入时命中本仓 dev root 的 build/（缺 tvm-python）报 "TileLang's private TVM Python package is missing"；PYTHONPATH 首项 tilelang-ascend 的 build 含完整 tvm，cwd=/tmp 或任何非 refs 目录即可正常导入。凡 kernel 测试报该 ImportError 应先查 PYTHONPATH 归属与 dev root 内容，勿直接判环境故障 | 本任务执行日志（refs cwd ImportError vs /tmp cwd tilelang 0.1.2 导入成功、NPUUtils 20 核）；bash `echo $PYTHONPATH` | tilelang-op-develop skill Phase 4 前置加「运行环境自检一行：print(tilelang.__file__) 确认 dev root 归属，ImportError 先查 PYTHONPATH」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| bitwise 族三路线 NPU 实现全 dtype 全 shape 实测通过（int 域 T.vnot / uint8 int8 位视图 vnot+view 回 / bool int8 位视图 fill(1)+vxor+view 回），含 manifest 最大档 N=2^28 int32（256m）、masking 边界 N=10241、动态 tail {232,1,136}、L0-extra 位视图 fingerprint——DESIGN.md §0.6 R2 三路线的可执行落地形态，后续 bitwise and/or/xor/shift 族直接复用 | examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py（route 分派 + _run_case 断言）；examples/bitwise_not/bitwise_not_fwd_kernel/DESIGN.md §0.6 R2 | python3 examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py --level all（exit 0，L0/L1 全过，0 WARN/FAIL） | tilelang 0.1.2 dev-root（tilelang-ascend）/ CANN 9.0.0 / Ascend 910B3 / 2026-10-29 | pattern-library（bitwise 族迁移模板条目，Tier 1 需第二证） |
| 910B3 物理核数第八次独立实查一致（工厂期权威 API NPUUtils.get().get_aicore_num()=20 → 纯 Vector 翻倍 40，与设计期 ctypes 等价路径 7 次复跑全部一致）——设计期不可导入环境的 ctypes 等价路径与运行期权威 API 双口径闭环 | 本 kernel L184（vector_cores 查询）；本文件 Stage 1/2 复跑记录 | python3 -c "from tilelang.utils.npu_utils import NPUUtils; print(NPUUtils.get().get_aicore_num())"（返回 20） | tilelang 0.1.2 dev-root / CANN 9.0.0 / libruntime 25.5.0 / Ascend 910B3 / 2026-10-29 | pattern-library/constants.md CONST-aicore-910B2C（追加第八次计数，Tier 0） |
| 精确整数域算子的嵌入式测试断言形态：torch.equal（无容差）+ 输出 dtype fingerprint（uint8 out_idx float32 分配缺陷的回归拦截）+ bool 存储值域 {0,1} 断言——把设计期实测缺陷（事实 2/3）固化为逐 case 可执行回归，强于浮点容差口径 | examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py `_run_case`（dtype/equal/bool 值域三段断言）+ L0-extra #7 | python3 examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py --level L0（含 Extra 7 uint8 fingerprint） | Ascend 910B3 / CANN 9.0.0 / torch_npu 2.7.1 / 2026-10-29 | tilelang-op-develop skill templates/op_template.py 补「整数/布尔域精确比较 + dtype fingerprint」断言形态节 |

### Transferable Lessons（可迁移教训）

- **两段式 curried 工厂对需要 host 位视图的 dtype（bool/uint8）**：view 包装放 builder 层（factory 返回的 `kernel(bs)` 内包 `def call(x)`），不改 @tilelang.jit kernel 的返回值形态——wrapper 的 `factory(N,dtype)(bs)(x)` 调用面完全不变，后续 bitwise 族函数直接复用该结构（int 路线返回 jit callable，view 路线返回 Python callable，测试侧统一 `kernel(x)` 调用不受影响）。
- **dtype 归一化必须先 str 化再进集合判断**：`dtype_key = dtype if isinstance(dtype, str) else str(dtype).split(".")[-1]`，torch.dtype 对象与 str 元组 `in` 比较恒 False——凡工厂接受 wrapper 传 torch.dtype 对象（本任务 bitwise_not.py L146）都必须先归一化。
- **UB cap 按 dtype 分档而非全 dtype 统一**：auto-multi-buffer 开启 regime 用 ~1.7× 膨胀口径核算 per-dtype 上限（本任务 int64=7168、int8/16/32=8192），int64@8192 在 1.7× 口径下 217.6KB>192KB 会放行编译失败配置——REVIEW 建议级① 的落地形态。
- **精确整数域 kernel 测试不用 assert_close 容差**：torch.equal 精确比较 + 输出 dtype 断言 +（bool 路线）存储值域断言三件套；输出 dtype 断言同时拦截 out_idx 分配缺陷类回归。
- **运行 kernel 测试若报 "TileLang's private TVM Python package is missing"**：先查 PYTHONPATH 首项与 dev root build/ 内容（本环境须 cwd=/tmp 或确保 PYTHONPATH 指向含 tvm-python 的 tilelang-ascend 树），勿判环境不可用。
