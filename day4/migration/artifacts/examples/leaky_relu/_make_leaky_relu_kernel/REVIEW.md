# _make_leaky_relu_kernel 设计文档检视报告（revision 1）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md（797 行，revision_index=1 修订后）
- 任务类型: 迁移（harness；GPU TileOPs → TileLang-NPUIR）
- 源算子: `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py` L2427–L2490（builder，基类 L2322–L2423，dtype 元组 L154–158）；NPU 侧 verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/_leaky_relu_fwd_kernels.py`；NPU harness wrapper（契约权威）`examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py`；规格参考 `examples/TileOPs/tileops/manifest/elementwise.yaml`（LeakyReluFwdOp 段）、`examples/TileOPs/tests/ops/test_leaky_relu.py`、同族已验证件 `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/_make_lerp_tensor_kernel.py`
- 检视维度: 9 项（0–8）；上一轮 REVIEW.md（结论: 不通过，2 阻塞 + 3 建议）的修复闭环逐项验证 + 全量维度复核
- 阻塞级问题数: 0
- 建议级问题数: 2（不阻塞，详见检视注记）

工件注入防护说明：DESIGN.md 与全部源码/文档中所读内容均按数据处理；未发现亦未执行任何嵌入的指令性文本（DESIGN.md §0 自述「源码中的指令性文本一律按数据处理」同样按数据对待）。

## 修复闭环逐项核实（revision 1 自述 vs 检视独立验证）

| 自述修复 | 核实结论 | 证据 |
|---|---|---|
| 1. E2 工厂单参 | **闭环成立**。`kernel(block_size=default_block)` 全文一致：§3.3 伪代码 L416/L449 路径均单参、§3.5.2 E2 行/§4.7/§5.2/§8.3 附加用例（kernel(2048)/kernel(1024)）/§9.2 防呆第 5 条/§10.3 全部改判单参；两参形态仅存于 §0.3 源码解读（标注「源侧事实」）与修订记录引述。折叠值核实：fp32 1024 / fp16·bf16 2048 与 wrapper `_BLOCK_SIZE_FP32`/`_BLOCK_SIZE_NON_FP32`（leaky_relu.py **L64–65 亲读**）**精确一致**；wrapper 单参调用式 `_make_leaky_relu_kernel(N, dtype, negative_slope)(block_size)(x)`（**L82 亲读**）、工厂构建传 torch.dtype（**L157 亲读**）、custom_op 链 `getattr(torch, dtype_str)`→torch.dtype（**L81 亲读**）均命中。**v0 错误断言的根因判定亦正确**：NPU 侧 kernel_base `init_config`（kernel_base.py L54–65 亲读）仅解析 config dict、**不调用工厂**——两参依赖确为 GPU 基类 L2417 源侧事实（行号亲验）；builder 外层 7 参签名保留与 wrapper 3 位置参调用兼容（defaults）；lerp 单参契约同形（其 L51 docstring / L163 定义亲验） | 见左 |
| 2. fp32 路线方案 (b) | **闭环成立**。§3.3 fp32 分支 ⑤ `T.vselect(cond, x_ub, sx32, x_ub)`（Out=A）+ ⑦ `T.copy(x_ub[0:tail], y[...])`；逐路线字面数据流独立推演：fp32——① 写 x_ub→③/④ 读→⑤ 写 x_ub→⑦ 读 x_ub ✓ 无陈旧拷贝；transit——⑤ 写 sx32（Out=B 同迭代）→⑥ 写 x_ub→⑦ 读 x_ub ✓。gqa_prefill 先例**亲读核对精确命中 L875**：`T.vselect(ub_cond, ub_f32_ND, ub_neg, ub_f32_ND)`（Out=A，第 2/4 参同 buffer）。五处交叉一致核实：§1.4 步骤 2、§1.5 两路线分列 + 拷出源一致性注、§3.1 ⑤/⑦ 按路线分列、§4.3 x_ub「选择输出与结果驻留（fp32）」、§4.4 双路径图、§1.6.3 结论行「结果统一写回载入 buffer」——**无残留旧表述**（全文 grep：sx32 不再出现于任何 fp32 拷贝源语境）。fp32 路线不再分配独立 x32：§3.3 L451–455 buffer 清单 = x_ub/sx32/cond/zero32/slope32（17B/elem，与 §4.5 fp32 行一致）| 见左 |
| 3. dtype 入参归一化 | **闭环成立**。`dtype_key = dtype if isinstance(dtype, str) else dtype.name`（§3.3/§3.5.2/§9.2 三处一致）；torch.float16.name==\"float16\" 归一化正确；误判后果描述准确（torch 对象 ∈ str 元组恒 False → 误走 fp32 路线） | 见左 |
| 4. UB 双缓冲口径（建议 3） | **实质闭环，残留 1 项（见建议 a/b）**。v0 的「×2 已留裕量」自相矛盾表述已删（全文 grep「裕量」无残留）；TRAP-UB-multibuffer-inflation **亲读全文**：front-matter status=verified ✓、平台值口径复述精确（transit≈20–21 / fp32≈26–27 B/elem、系数 1.10–1.12 结构依赖、反例 0.981×manual、规律 2「需求随 num_local_tasks 变化」）；lerp L1 套件 bs=8192 先例**亲验存在**（其 L637–648：fp16/bf16 bs=8192 编译运行用例）；默认 1024/2048 双缓冲 34.8/77.8KB 在 ×2、CONST-capacity 1.7×、平台值一切口径下均安全 ✓；Stage 4 编译探针 + 连带复验 num_kernels 条款 ✓ | 见左 |
| 5. N=0 措辞（建议 4） | **闭环成立**。§0.1/§9.3 新表述与源码一致：Op 层 `_validate_input` 实为 `input.numel() != self.N_total`（ops/elementwise/leaky_relu.py L184 亲读——N_total=0 空张量 numel=0 可穿透 ✓）；GPU grid=0 空启动 ✓；NPU 拦截 = builder 防御 ValueError（标注「源无此项，属适配」✓）；§8.3 附加用例 N=0/N<0 → ValueError ✓ | 见左 |
| 6. 文字项（建议 5） | **闭环成立**。「埸」错字全文 grep 无残留 ✓；lane 宽度含糊数字已删（§1.6.0 R4 / §1.6.3 改为「lane 宽度/lane 组大小为编译器实现细节，不作设计依赖」、尾浪费 O(1) 结论改依 `[0:tail]` 拷出屏蔽 ✓）；§1.6.0 D-2 canonical 估算下界行已补，格式与 hardware-cost-model.md §2 L25 协议一致（`估算下界: <值> us（流量+发射+容量，引用 CONST-*）`；轻量调研容量项豁免为协议 L25 明文允许）✓ | 见左 |
| 新增 §9.1#9 TRAP-vselect-inplace-carried-state 采纳边界 | **边界正确**。trap 本体亲读（traps-compiler.md L249–，status=verified）：触发条件 = **Out=B 形态 + 输出作为跨 `T.serial` 迭代携带状态**（argmax running 类，num_full≥3 暴露；诊断指纹「输出停留 tile-0 局部值」）。§3.3 伪代码核对：两路线 ⑤ 输出均**仅同迭代**被 ⑥/⑦ 消费、下一迭代 ①/②/④ 全量覆写——无 loop-carried 状态，论证成立 ✓；同形先例亲验：lerp 原地链（`vsub(b32,a32,b32)/vmul(w32,b32,b32)/vadd(a32,b32,b32)` Out=B 别名、persistent T.serial 多迭代、bit-exact 已验证——其 L238–241 亲读）、gqa L875 Out=A；L0-2（205 任务/核多迭代）兜底 + §9.2 禁令固化 ✓ | 见左 |

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**。`python3 examples/leaky_relu/_make_leaky_relu_kernel/verify_equiv.py` 本检视环境重跑：A1 采纳项 **6/6 行 EQUIV_PASS**，各行数值（cand-gold ulp 0/viol 0；base-gold ulp fp16·bf16=1、fp32=0/viol 0；cand-base ulp 1/1/1/1/0/0）与 §1.6.1(b) 内嵌表**逐格一致**。A2 弃选证据：slope=0.01 → 0 mismatch；slope=2.0 → mismatch 65543 / violations 65521 = §1.6.1(c) 所引「65521/65548 违反」自洽（65536 随机 + 12 角点）。脚本语义独立核验：golden=`F.leaky_relu(x.float(),slope).to(dtype)`（源仓测试锚点），candidate 以 `np.float32(slope)` 参与 fp32 域计算——与 §9.1#2「fp64→fp32 最近舍入」前提一致。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/leaky_relu/_make_leaky_relu_kernel/DESIGN.md`）JSON checks 摘要：
  - `ub_budget`: **skip**（buffer 表未被机械解析）→ 人工复核 pass：fp32 17B（4+4+1+4+4）、fp16/bf16 19B（2+4+4+1+4+4）与 §4.3/§3.3 伪代码逐 buffer 复算一致；cap 196608//17=11565、//19=10347 ✓；默认 block 双缓冲峰值 2×17×1024=34.8KB、2×19×2048=77.8KB < 192KB（含 ×2 朴素模型在内全口径安全）✓。**skip 项人工判定 pass**（8192 guard 上限口径见建议 b）。
  - `l0c_budget`: **skip**（无 Cube 路径 API）→ 人工复核：纯 Vector，n/a 正确。
  - `core_split`: **skip**（dims 解析不足；「declared logical cores = 32」为 host 侧 `lru_cache(maxsize=32)` 字样解析噪声，非设计声明）→ 人工复核 pass：要素①复算（2²⁰/1024=1024、2²⁰/2048=512、2²⁴/2048=8192、2²³/2048=4096 ✓）；要素②实查独立复跑（ctypes `rtGetAiCoreCount` ret=0 → **aicore=20**，与 §5.5 记录一致；翻倍规则出自 core-split-strategy.md §1 要素② L14「纯 Vector 算子核数翻倍」亲读 → 40 ✓）；要素③规模判定（512–8192 ≫ 40，block 增至 UB cap 亦无法 ≤40 ✓）、静态边界（工厂特化常量 ✓）、负载不均 ≤1 任务/核（8192/40=204.8→≤205、512/40=12.8→≤13 ✓）。**skip 项人工判定 pass**。
  - `r3_metrics`: **pass**（四口径 FLOPs/访存/遍数/缓冲峰值齐备）。
- **kb_stale_check（本轮新增复核项，检视者实跑）**：`python3 .agents/tools/kb_stale_check.py` → **PL-1.6-copy-floor、PL-1.17-subfp32-fp32-transit、TRAP-UB-multibuffer-inflation、CONST-mte2-degradation、CONST-copy-floor-method 当前均判 [stale]**（条目 stamp tilelang 0.1.2+ed787bb~1990aa9fe4 / CANN 8.5.0 / 910B2C vs 当前工具链 0.1.2+88ee8c302e / CANN 9.0.0 / 910B3）。与 DESIGN §9.1#8「kb_stale_check 未判 stale」表述矛盾（见检视注记 a）。条目 front-matter `status: verified` 本身亲验无误（stale 判定是工具链 stamp 漂移信号，非条目失效）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

> 检视方式：亲自 Read GPU 原件（builder L2427–L2490 + `ParametricUnaryKernel` 基类 L2322–L2423 + dtype 元组 L154–158）、NPU verbatim 提取件全文、NPU harness wrapper 全文、Op 层（`_validate_input`/`_eager_forward`/`_validate_scalar_param_repr`）、test、manifest、lerp 同族件。修订未动 §0 结论，本轮做抽样核对 + 修订触及项全核。

- **语义理解正确性：pass**。严格 `>`（源码 `T.if_then_else(val > zero, val, slope*val)`）、NaN/±Inf/±0 传播、无规约、dtype 域逐 op 舍入（L2471–2472 `T.cast(0/negative_slope, val.dtype)` 亲验）、host 展平/inplace copy-back 均与源码一致；§0.1 边界语义措辞已按上轮问题 4 更正并与 Op 层实文核对成立（`numel != N_total` 对空张量穿透亲读确认）。§0.2 I/O 与签名一致（`out_dtype = output_dtype or dtype` L2437；manifest dtype 三选/shape_rules/flops=2N/bytes=2N·elem 亲读一致）。
- **算法解读完整性：pass**。§0.3 逐语句：入口签名/`lru_cache(32)`/`out_idx=[1]`/一维 grid/非 fp8 fragment 三语句（S1–S3 全覆盖）/fp8 guard 分支/host `default_config npt=4|8`（L2405–2412 亲验）/`_compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])` **L2417 行号亲验**；新增澄清「该两参依赖是 GPU 侧契约，不得作为 NPU 接口约束」——与 NPU kernel_base 实文（init_config 不调用工厂）核对成立。
- **优化手段覆盖：pass**。O1–O6 与源码逐项对应；清单法「源无 SMEM/mma/shuffle/流水/online」与亲读一致；算法层 vs 实现层区分正确。
- **耦合性判定合理性：pass**。八条目四态处置齐备、统计一致；#5 fp8 舍弃依据亲验（基类 `SUPPORTED_DTYPES=_FLOAT_DTYPES` L2342/L154–158 ValueError 前置拒绝）；#2/#4/#8 映射表条目引用上轮已亲验。
- **NPU 重设计可行性：pass**。R1（persistent 一维 + vcmp/vmul/vselect 链）满足一维 Kernel/UB/静态边界约束；R2 transit 双触发亲验（bf16 无 v-算术 + golden fp32-opmath 锚点）；语义保持论证四要素与 A1 机器验证互证（本轮重跑复现）。R1 原地写回形态的陷阱边界论证成立（修复闭环表 #6）。
- **设计与决策一致性：pass（上轮 fail 项已闭环）**。§1–§7 与 §0.5/§0.6 一致；接口契约断言改判后经逐字核对与 NPU wrapper 现状完全对齐（修复闭环表 #1）；§0.7 契约权威来源（文件 + L82/L157/L64–65 行号锚定）准确。
- **golden 独立性：pass**。§8.1 golden 移植自源仓测试锚点（test L40 `F.leaky_relu(x.float(), 0.01).to(x.dtype)` 亲读一致），不含 transit/vcmp/persistent 结构知识。

### 1. API 可行性 — pass

§3.2/§3.4 全部 API 文档 + 先例双佐证本轮复核：`T.vmul` §2.2.1 dtype 行亲读（fp32 √、bf16 ×——transit 后仅 fp32 域使用，绕开正确）；`T.vselect` §2.2.1（fp16/fp32/bf16 √）与语义 `T.vselect(Cond, A, B, Out)` 文档 L8 亲读，cond 真取 A 与公式 ⑤ 一致；`T.vcmp`「gt」合法（§2.3 cmp_mod 列表 L20/L36）+ bool 不可写 GM 条款（L39）亲读；`T.vcast` §2.2.1 亲读（f16→f32 **仅 rint** L36、bf16→f32 **仅 rint** L42、f32→f16/f32→bf16 含 rint L30/L35）；`T.vbrc` §2.2.2 标量广播 + §2.3 i1 尾轴 16 对齐条款（L48，本设计 vbrc 仅 fp32 不触发，§9.1#1 判定正确）；`T.copy` 零起点切片（vec_add_1d/lerp 先例）。先例行号精确：gqa_prefill **L709**（`ub_cond = T.alloc_ub([half, legacy_w], "bool")`）与 **L875**（Out=A vselect）、sparse_mla `examples/deepseek_v32/sparse_mla_fwd_dynamic_shape.py` **L230**（`T.alloc_ub([1, block], "bool")`）亲验命中。全 v-prefix 优先；Developer 模式与自动同步 API 匹配。

### 2. 内存层级规划 — pass

- 搬运路径 GM↔UB（MTE2）纯 Vector 无越级 ✓；buffer 层级标注齐 ✓；§4.4 两路线路径图与 §3.3 伪代码逐 op 一致 ✓。
- **上轮阻塞（fp32 拷出源）已闭环**：两路线 ⑦ 统一拷 x_ub，fp32 ⑤ Out=A 写回 x_ub，独立推演无陈旧拷贝路径（修复闭环表 #2）。
- UB 预算：17/19B/elem 逐 buffer 复算 ✓；cap 11565/10347 ✓；默认 block 双缓冲 34.8/77.8KB < 192KB 全口径安全 ✓；§4.3「复用」列与 fp32 无独立 x32 的口径一致 ✓。
- guard 8192 上限论证：transit 路线有 lerp 同族实测先例支撑（L1 fp16/bf16 bs=8192 用例亲验）；**fp32 路线 8192 档无实测先例且与所引平台值口径存在张力**（检视注记 b）；Stage 4 编译探针 + 连带复验 num_kernels 条款 ✓。不阻塞：交付管线（默认 block、harness 1024/2048、L0 用例）无任何路径触及 8192，且越限失败模式为编译期报错非错误结果。
- L0C：不涉及 ✓。

### 3. Tiling 策略 — pass

§5.2 单参注入 + 默认 1024/2048（与 wrapper 折叠常量同源核对 ✓，GPU default_config threads=256×npt=4|8 折叠积亲验 ✓）；UB cap 与核数约束明确；分核三要素齐且经机械 skip→人工复核 + 检视者独立复跑核数实查（ret=0→20）；静态边界 + grid-stride + `if block_id < num_logical` 掩蔽，负载不均 ≤1 任务/核——满足 core-split-strategy.md §1/§2.2 全部检视要点。非整除走 tail 切片（§6.4），纯 Vector 无需 padding。

### 4. 技术约束检测 — pass

三维 Kernel/GEMM/L0C 不涉及；§3.5.1 对 ascend-constraints 强制检测项逐条给结果，关键负向项本轮亲验：bool 不写 GM（vcmp.md 条款 L39）、bf16 v-算术缺失绕开（vmul.md dtype 矩阵 bf16 ×）、UB 有界、in-place 契约保持、fp8 ValueError 处置与源基类拒绝语义一致。

### 5. 循环与同步 — pass

§6.1 循环结构表（persistent 一维 + 静态 serial + tile 内 v-prefix 隐式并行）与 §3.3 伪代码一致；无 T.Pipelined 理由（Developer 自动多缓冲）与模式匹配；§7.2 同步点表本轮更新后与两路线 vselect 原地写回一致（同迭代 UB 内 RAW 自动栅栏；跨核零同步——tile 两两不相交）；尾块 §6.4 完整（陈旧 lane 无污染论证：v-op lane 独立 + `[0:tail]` 屏蔽）。

### 6. 验证方案 — pass

§8.1 golden 草案独立且与源仓锚点逐字一致（test L40 亲验）；§8.2 精度标准与测试仓 `_get_tolerances` **逐值亲验一致**（fp32 1e-5/1e-5、fp16 1e-3/1e-3、bf16 1.6e-2/1.6e-2）；§8.3 L0 六项 + 附加覆盖：manifest 4 workloads（smoke-1m 三 dtype / 2²⁴ / 2²³ 结构等价 + harness 全量）、非整除/微型/角点（±Inf/NaN/±0/次正规/65504/slope=0.2）/分核边界（39/40/41 与 N 换算复算正确）、E2 附加用例已改单参双配置（2048/1024）、UB guard 8193 三档拒绝 + **N=0/负 N 用例本轮增补** ✓。

### 7. 完整性与一致性 — pass（附检视注记 a）

模板 11 章节齐全（§0 含 0.1–0.7）✓；占位符扫描：全文无未填记号残留（§10.1/§10.2 的 op 文件名模板记号属交付物命名约定且状态列注明 Stage 3 产出，沿上轮裁定非缺失）✓；§3.5.3 六条引用路径全部实测存在（lerp base/perf_opt、vec_add_1d、gqa_prefill、ada_layer_norm、mish/、verbatim 件）✓；行号引用本轮抽查全部精确（GPU L2417/L2437/L2438/L2390/L154–158/L2342；wrapper L64–65/L82/L157；lerp L51/L163；gqa L709/L875；sparse_mla L230；Op L184）；修订触及章节交叉一致（§1.4/§1.5/§3.1/§3.3/§4.3/§4.4/§1.6.3/§7.2 数据流五处交叉 + 契约描述 §0.3/§0.7/§3.5.2/§4.7/§5.2/§8.3/§9.2/§10.3 全文一致，无矛盾残留）。**检视注记 a（建议级，不阻塞，随 Stage 3 首跑自然消解）**：§9.1#8「kb_stale_check 未判 stale」与本轮实跑结果不符——PL-1.6/PL-1.17 等条目在当前工具链（88ee8c302e/CANN 9.0.0/910B3）下已被判 [stale]；该条缓解栏已含「Stage 3/4 首跑即重验」承诺，满足 AGENTS.md 对 stale 条目的 re-verify 要求，故不阻塞，但建议措辞更正为「kb_stale_check 已判 stale（工具链 stamp 漂移），首跑重验前不得当作已实证」。

### 8. 算法优化分析 — pass

- **§1.6 结构存在性：pass**。1.6.0/1.6.1/1.6.2/1.6.3 四子节齐备；轻量调研判定符合 algorithm-research §2（单步逐元素），实际深度超出轻量下限；修订触及的 §1.6.3 结论行已与方案 (b) 对齐。
- **算法调研完整性：pass**。R1 候选表含基线且覆盖 algorithm-candidates.md **ALG-elementwise 命中行全部 R1 手段**（本轮对照亲读候选表 L104：CSE/除法乘倒数/cast 链合并——R1 表 #4/#5/§0.6 R2 逐项对应）+ 额外三族；R2「无在线变体」有结构判据（逐元素天然单遍——候选表 L104 同文佐证）；R3 四口径齐（`r3_metrics` 机械 pass）；R4 逐项评估含 dtype 路径行。
- **调研结论正确性（独立复核）：pass**。① 负向断言复核：「公式级无降算空间」——候选 2 vmax −1 op 收益被条件等价性否决，A2 机器证据本轮重跑复现（slope=2.0 → violations 65521/65548，与 65536 随机 + 12 角点结构自洽）且契约「任意 finite slope」（Op `_validate_scalar_param_repr` 亲验）成立；候选 1 UB 劣势复算（需 zero32+one32+slope32 三常量 = 21B > 基线 17B ✓）；「bf16 无 v-算术」负向断言本轮亲开 T.vmul.md §2.2.1 核对 bf16 单元格 = × ✓；② 复杂度复算：2N·B = I/O 下界、无中间 GM 往返（本算子无中间 GM buffer，漏计往返检查项 n/a）✓；③ 亲和性引用 constants.md/PL-1.6 实测条目（数值 33.9/30.2 GB/s、+3.2%@256M、delta ≤1.1µs 本轮亲读复验一致；跨设备 40/48 缩放与 910B3 曲线均已标注未实测假设 + Stage 4 校准回写 ✓）；④ 并列候选裁决：基线 vs 候选 2 非并列（等价性否决 + 计算链隐藏 ROI≈0），无纸面拍板 ✓；⑤ 「无更优替代」调研范围五条声明齐，联网豁免引 algorithm-research §6 条款 ✓。
- **调研→设计一致性：pass**。§1.4/§3.1/§6/§1.5 与选定算法（基线公式 + A1 transit + R1 重设计形态）一致；§0.5/§0.6 与选定族一致；源优化意图承接表逐项（O1/O2/O3/O6）无静默丢弃。
- **数学等价优化（1.6.1）：pass**。A1 四要素齐；**独立推演复核**：fp16/bf16⊂fp32 升域无损保序→比较严格一致 ✓；slope 以 `np.float32(slope)` 参与与 golden fp32-opmath 标量语义一致 ✓；fp32 乘积单次 rint 舍回 = golden 舍入路径 ✓；与源基线容差内等价（≤1 ulp ≪ 测试容差）且严格更近 golden（0 vs 1 ulp）✓；机器验证重跑 6/6 与内嵌表逐格一致 ✓。公式级「无优化空间」结论有 R1 全表 + D-1 豁免边界说明。
- **向量化替代分析（1.6.2）：pass**。V1–V7 与 §3.3 修订后伪代码交叉核对无遗漏（vselect/vcast 改道后 V1/V4 替代对象不变）；V5/V6/V7 保留理由具体（任务映射/block 级索引/host 元数据），无「实现简单」式理由；§6.1「tile 内无显式循环」与结论一致。
- **向量化轴与布局决策（1.6.3）：pass**。I/O 契约与核内布局区分明确；单候选轴豁免论证成立（纯 1-D 逐元素、4D→1D 在 host 完成）；候选 ② 弃选引 PL-1.1 实测 µs 级代价；lane 表述修订后可溯源 ✓；实验裁决不触发判定成立（带宽引用已实测条目 + 假设已标注）。
- **布局/决策与 §3.3/§4/§6 一致：pass**。方案 (b) 落地后「结果统一写回载入 buffer」与 buffer 表、UB 预算、伪代码三方一致（fp32 17B 口径同步更新 ✓）。
- **检视注记 b（建议级，不阻塞，随 Stage 3 首跑自然消解）**：§4.5/§9.1#4 的 guard-8192 论证以「resident 19B ≤ lerp 同族 resident 20B 实测先例」统括全部 dtype——该先例仅覆盖 **transit 路线**（lerp L1 bs=8192 用例为 fp16/bf16；lerp fp32 仅验证 12289 guard 拒绝，无大 bs 通过用例）；而 TRAP 平台值口径明示 footprint **非 resident 单调**（ada 实测 fp32 直连 8/16B resident 同为 26.00 B/elem）——按同口径外推，fp32 路线 8192 档 ≈26×8192≈213KB > 192KB，存在编译不可行可能。不阻塞（默认配置全口径安全、8192 仅为契约上限、越限失败模式为编译报错、Stage 4 探针条款已覆盖调参路径），建议修订：guard 上限按路线分置（transit 8192 / fp32 ≈196608//26≈7500 或明示「fp32 大 block 以编译探针为准」）。
