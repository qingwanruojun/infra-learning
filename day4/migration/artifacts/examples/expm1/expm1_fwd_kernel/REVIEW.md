# expm1_fwd_kernel 设计文档检视报告（revision 1 复审）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/expm1/expm1_fwd_kernel/DESIGN.md（revision 1，857 行，含修订记录表；基线 history_version/design_v1.md）
- 任务类型: 迁移（TileOPs GPU `Expm1FwdKernel`/`expm1_fwd_kernel` → TileLang-NPUIR）
- 源算子: `/mnt/workspace/learn/refs/TileOPs/tileops/kernels/elementwise.py`（`Expm1FwdKernel.op_func` L2142–2147 本次复审重新亲读）；NPU 侧 verbatim 提取件 `examples/TileOPs/tileops/kernels/elementwise/expm1/_expm1_fwd_kernels.py`（全 34 行重新亲读）；NPU harness wrapper（契约权威）`examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py`（全 181 行重新亲读）；家族两段式锚点 `examples/TileOPs/tileops/kernels/elementwise/leaky_relu/leaky_relu.py` L98 / `examples/TileOPs/tileops/kernels/elementwise/elu_fwd/elu_fwd.py`（本次亲读，见 F-1）
- 检视维度: 9 项（0–8；维度 0 源算子理解与迁移分析、维度 8 算法优化分析均含）
- 阻塞级问题数: 0
- 建议级问题数: 3（F-1/F-2/F-3，记于维度详情内，不阻塞）
- 工件指令注入扫描：DESIGN.md / 提取件 / GPU 原件 / wrapper / pattern-library 条目中未发现要求改写检视流程的指令性文本；设计 §0 「源码指令性文本按数据处理」自述经复审确认执行。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `python3 examples/expm1/expm1_fwd_kernel/verify_equiv.py`（本复审环境，torch 2.7.1 CPU，模块级种子 20261028）——**PASS，且与 §1.6.1(b) 内嵌表逐格逐值一致（revision 1 修复达成 D-1 门禁）**：
  - 判定列 9/9 代表行逐值命中：fp16 randn（mism 440 / max_abs 6.104e-05 / viol 0 / ulp 1）、fp16 ×0.001（125745 / 1.907e-06 / 0 / **1**）、fp16 角点（1 / 5.960e-08 / 0 / **1**）、bf16 randn（7 / 1.526e-05 / 0 / 4）、bf16 ×0.001（6351 / 1.526e-05 / 0 / **20054**）、bf16 角点（5 / 5.960e-08 / 0 / **32640**）、fp32 randn（496251 / 7.629e-06 / 0 / **281691**）、fp32 ×0.001（1047656 / 5.960e-08 / 0 / 1314243271）、fp32 角点（10 / 5.960e-08 / 0 / 2147483647）——**问题 1 四处 max-ulp 修正（1/1/32640/281691）全部与实跑一致**；新增 max abs diff 列 9/9 与实跑 max_abs 列一致；判定列全 EQUIV_PASS。
  - 全量 54/54 行 viol=0；SUMMARY「0 violations: 54; EQUIV_FAIL rows: 0」与 A1 verdict PASS 与设计 §1.6.1(b) 汇总行一致 ✓；GOLD-DIAG 全 dtype×全判定集 mism=0 ✓（§0.1 golden 域语义机器佐证存活）。
  - **补充位距分布断言独立复刻验证**（复刻 main() RNG 消费顺序：band_table 先行 → randn_sets）：fp32 randn mism=496251、≤2 ulp 405343（**81.68%**，设计记 81.7% ✓）、>2 ulp **90908** lane abs max **8.9407e-08**（设计记 ≤8.941e-08 ✓）；bf16 ×0.001 ≤2 ulp **78.32%**（设计 78.3% ✓）；fp16 五判定集 max ulp=1 ✓——三处「≤2 ulp」文字断言修订（§0.6 R2 / §1.6.1(b) / §8.2）与实跑一致化达成。
  - band 表关键数复现：(0.1,1.0] abs 1.501e-07/裕度 67× ✓、(1e-2,1e-1] **6.300e-08** ✓（问题 6 修订引用值）、(1e-4,1e-2] rel 5.828e-04 ✓（设计 5.83e-4）、|x|≥0.1 rel 峰值 5.793e-07 ≤ 设计所记 5.85e-7 上界 ✓（纯 rel 裕度 ≥17× ✓）。
  - §1.6.3.1 弃选证据行复现：Taylor fp32 randn mism 496251→**306906** ✓、fp64 C 行 mism 2737 / max 1 ulp ✓、[3] 段 0.192@1e-7→5e-8 ✓；脚本 L45 注释余项已为 **2.7e-10**（问题 5 同步达成；独立复算 0.5¹⁰/10! = 9.766e-4/3628800 = 2.691e-10 ✓）。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design examples/expm1/expm1_fwd_kernel/DESIGN.md`）JSON checks 摘要：
  - `ub_budget`: **skip**（无法机械解析分级 buffer 表）→ **人工复算（机械跳过，人工已核）**：§4.5 平台值口径逐档重算全部正确——26.00×4096=106,496 B（≈106.5KB，<196,608 B ✓）、20.01×8192=163,922 B（≈163.9KB ✓）、22.00×8192=180,224 B（浮动上界 ✓）、26.00×8192=212,992 B（≈213.0KB >196,608 B，**fp32 8192 档超限被 guard 正确排除** ✓）、默认档 26×1024=26,624 / 20.01×2048=40,980 ✓；与 §3.3 `ub_cap=4096/8192`、§3.5.1、§1.6.0 R4、§6.3①（16384 档 transit 20.01×16384≈327.8KB 超限拒用 ✓）全部一致。
  - `l0c_budget`: **fail**（"block 26x8192 x fp32 accum = 832.0 KB vs L0C 128 KB"）→ **机械误报白名单判定**：解析器把 §4.5「26.00 B/elem × guard 8192」UB 字节/元素表行误读为 Cube accum tile；本算子纯 Vector、无 L0C/L1 路径（§3.5.1 L0C 行明示不涉及、§5.1 一致），与首轮（10×8192 误读）同型签名，不计维度 fail。
  - `core_split`: **skip**（"declared logical cores = 32" 为误解析值）→ **人工复算（机械跳过，人工已核）**：① §5.5 逻辑核数表 6 行独立复算全对（2²⁰/1024=1024、2²⁰/2048=512、2²⁴/1024=16384、2²⁴/2048=8192、2²⁶/2048=32768、2²⁸/2048=131072）；② 物理核数**独立复跑** ctypes `rtGetAiCoreCount`：ret=0 → cnt=20 → 纯 Vector ×2=40，与设计 §5.5 所记一致；③ num_local_tasks 复算（1M fp16=13、2²⁴ fp16=205、2²⁸=3277）+ 静态边界/掩蔽条款 ✓。人工复核结论：pass。
  - `r3_metrics`: **pass**（四口径齐备、基线在表）。
- **kb_stale_check 实跑**（`--stamp "tilelang 0.1.2+ef4078a697"`）：CONST-mte2-degradation / CONST-copy-floor-method / PL-1.6-copy-floor / PL-1.17-subfp32-fp32-transit / PL-1.21-910b3-elementwise-baseline / TRAP-UB-multibuffer-inflation **六条目全部 [stale]**——与设计修订后「六条目统一 stale 口径」自述一致（问题 4 闭环的机械佐证）；各条目 front-matter 的 2026-10-28 重验存活注记亲读属实。
- **torch 契约复核**：torch 2.7.1 下 `str(torch.float16).split(".")[-1]=="float16"`、无 `.name`——§3.3 dtype 归一化前提成立（维持项，与首轮一致）。

## 修订闭环核实表（逐项验证到决策行）

| 项 | 判定 | 核实证据（复审独立执行） |
|----|------|------------------------|
| 问题 1（内嵌表/文字断言 vs 实跑） | **PASS** | verify_equiv 重跑：9/9 行判定列+max_abs 列+max-ulp 列（含 4 处修正 1/1/32640/281691）逐格一致；54/54 viol=0；三处「≤2 ulp」断言改为实跑口径并经 RNG 序复刻独立验证（81.68%/90908/8.941e-08、bf16 78.32%）；星号脚注改「位距伪影，以 max abs diff 与 viol=0 为准」与 compare 掩码语义（L119）自洽；连带修正 §1.6.1(a) band 域句（见 F-2 残留措辞注记，不阻塞） |
| 问题 2（wrapper 契约两段式） | **PASS** | §3.3 `return kernel`（恒 curried）与 §3.5.2 适配清单 #1 最终调用式 `expm1_fwd_kernel(N, dtype_str, block_size=block_size)(block_size)(x)` 段数逐字一致；§8.3 附加用例两段式改写 + 单段直调 TypeError 防呆对照；§9.1#7「2 处修改」、§9.2 末条防呆、§10.2/§10.3/§10.4、§4.7/§5.2/差异表 E2 行全部同步，规范位置零单段直调残留（「1:1 兼容」仅存于修订记录作为被推翻主张的批判性引用）；家族亲读：leaky L98 verbatim ✓、elu 两段式 return verbatim ✓（行号见 F-1）；wrapper L81 现状 `(x, N)` grep verbatim 核对 ✓ |
| 问题 3（UB 平台值口径 + 分 dtype guard） | **PASS** | §4.5 口径 = TRAP-UB 第三证规律 1（20.01/26.00 B/elem、与 staging 解耦）亲读原文命中；×1.10–1.12 归还第四证「基础分配差（multi-buffer=false）」语境（原文命中）；算术复算 4 档全对（106.5/163.9/213.0/180.2 KB）；fp32 8192 超限已排除（guard 4096）；transit 8192 保留依据=163.9KB<192KB + leaky 第五证（TRAP-UB origin_task/toolchain 行「fp16/bf16 bs=8192 910B3 编译运行通过」亲读命中）+ 浮动上界 180.2KB；探针条款挂 **Stage 3 首跑**（BishengIR `ub overflow requires` 报文 + 调 block/nl 连带复验，harness 无独立 Stage 4 的适配正确）；§1.6.0 R4/§3.3/§3.5.1/§5.2/§5.3/§6.3①/§8.3（4097/8193 分 dtype 用例）/§9.1#4/#8 口径一致；「一切口径安全/朴素 ×2」规范位置无残留；默认档 1024/2048（26.6/41KB）不受影响 |
| 建议 4（stale 口径统一） | **PASS** | §0.5#3 与 §1.6.0④/§9.1#6 统一为「ef4078a697 判 [stale]→设计期佐证+Stage 3 首跑重验」（diff 确认 v1「非 stale」删除）；kb_stale_check 实跑六条目全 stale；PL-1.21 改「设备侧实测、跨小版本量级锚」口径 |
| 建议 5（余项 2.7e-10） | **PASS** | §1.6.0 R1 候选 1 / §1.6.3.1 / verify_equiv.py L45 注释三处一致；独立复算 2.691e-10；结论不变（≪2⁻²⁴≪atol） |
| 建议 6（half-ulp 封顶措辞） | **PASS** | §1.6.1(a) 改「≤1 ulp(exp(x))≈1.2e-7 封顶（正确舍入时 half-ulp≈6.0e-8；实跑 (1e-2,1e-1] 带峰值 6.30e-8）」——实跑 6.300e-08 命中，与 §9.1#1 vexp 硬件 ULP 风险论证自洽 |
| 建议 7（PL-1.21 归因） | **PASS** | D-2 段两处锚注改「msprof op 口径实测（设备时长，含流水建立/启动损耗）……口径差非矛盾，Stage 3 集成后分解校准」；PL-1.21 条目原文亲读（「msprof 口径，--prof-mode msprof，6 profile case 全 profiler_valid」）命中；「〔含 harness 开销〕」无残留 |
| 维持项抽查（§0 三问 / 分核三要素 / A1 / 逐路线数据流 / golden / L0 六项） | **无回归** | diff 范围核对：239 变更行集中于修订记录声明的锚定集合；提取件 L13/L16/L23–L30、wrapper L65–66/L81/L130/L133/L142–146/L155/L159–168、§5.5 三要素（仅 UB cap 措辞随问题 3 同步，分核逻辑不变）、§1.5/§3.1/§3.3/§4.3/§4.4 逐路线数据流（fp32 `vsub(w32,1.0,x_ub)` 跨缓冲 dst=载入 buffer / transit 原地+⑤ 回写、两路线统一拷出 x_ub）、§8.1 golden=`torch.expm1` 独立、§8.3 L0-1–L0-6 六项——与首轮 pass 结论逐项一致 |

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：重新 Read GPU 原件 `Expm1FwdKernel.op_func`（L2142–2147）、verbatim 提取件全 34 行、NPU wrapper 全 181 行、家族两段式先例调用面，与 §0 逐项核对。

- **语义理解正确性：pass**。提取件 L29 `T.exp(T.cast(x_reg[i·npt+j], 'float32')) - T.cast(1.0, 'float32')` = GPU 原件 L2146–47 逐字一致 → §0.1「计算全程 fp32 域 + 赋值隐式单次舍回」与 L25–26（`y_reg` 为 out_dtype fragment、RHS fp32 表达式）源码结构一致；「结构承接型」定性（源实现本身即 fp32-opmath，异于 leaky 修正型）经 GOLD-DIAG 全 dtype mism=0 机器佐证存活 ✓；溢出（x≥88.73→+Inf，ln(3.4e38)=88.72 复算 ✓）/下溢（≤−103.97→−1.0）/FTZ 不敏感带/±0 零符号差/NaN 传播逐项复核与首轮一致 ✓；host 展平、dtype 白名单（wrapper L133/L142–146 亲验）、源无 N=0 拦截 → builder 防御适配 ✓。
- **算法解读完整性：pass**。§0.3 S1–S3 覆盖提取件 L27–L30 全部计算语句（载入→fp32 exp−1→写回）无遗漏无臆造 ✓；数据流图覆盖 x_reg/y_reg/常量 1.0 内联、无其他 buffer ✓；host 链（lru_cache 提取件 L13 自带、「与 elu 提取件不同」事实注 ✓、K11 常量 wrapper L65–66 ✓、custom_op L74–86 ✓、Kernel 类 L95–181 ✓）全部命中；`threads×npt` 两参标注「源侧事实不进 NPU 决策」纪律保持 ✓。
- **优化手段识别：pass**。O1–O5 与源码逐项对应；「源码无 SMEM/mma/shuffle/流水/online/persistent/fp8 分支」对照声明经全文重读复核成立；O4 算法层定性 + §0.6 R2 显式承接不静默丢弃 ✓。
- **耦合性判定合理性：pass**。§0.5 八条目四态处置 + 依据齐备（fragment→UB 等价替换、二级并行→重新设计与映射表一致；0 舍弃与本案最简接口面相符）；#3 的 PL-1.6 引用措辞修订后与实跑 stale 状态一致（闭环见修订核实表）。
- **NPU 重设计可行性：pass**。R1（一维 persistent + v 链）、R2（transit 单次舍回逐位同构论证：升域无损保序→fp32 域同序→末端唯一 rint，独立推演成立）、R3（tail 切片严格强于源整块依赖）满足 NPU 约束；差异注记（expm1 vs exp(x)−1 舍入路径位型差异、验收=容差包络非 bitwise）修订后与实跑数字一致（81.7%/≤8.941e-08，非 v1「47% lane ≤2 ulp」）。
- **接口契约与 NPU wrapper 现状核对（首轮 fail 项，本次 pass）**：L81 verbatim 引用逐字准确（grep 核对 `(x, N)` 现状 ✓）；契约分析 1（`(x, N)` 在 `@tilelang.jit(out_idx=[1])` + main(x,y) 下第二实参无法绑定——调用期 TypeError）独立复核成立；契约分析 2 修订为**两段式**：工厂返回恒 curried（§3.3 `return kernel` 无条件）→ 单段 `(x)` 把 tensor 绑定 bs（build 期 TypeError）→ 适配目标 `(...)(block_size)(block_size 同值)(x)`，与家族调用面（leaky L98 / elu return verbatim 亲读，均 `(block_size)(x)`）同形一致；L155 两位置参构建「缺省合法、不变」与 §3.3 签名默认值一致 ✓。**问题 2 契约断裂闭环。**
- **设计与决策一致性：pass**。§1–§7 为决策后 NPU 算法；数据流五处交叉一致；契约段（§3.5.2/§4.7/§5.2/§8.3/§9/§10）同步无残留矛盾（首轮两处内部不一致①②均已消除）。
- **golden 独立性：pass**。§8.1 `torch.expm1` 以 §0.1 语义为唯一依据，无 NPU 算法复刻（无 transit/向量链/persistent 知识）。

### 1. API 可行性 — pass

修订未改变 API 面；抽查复验：T.vexp.md §2.2.1 A2/A3 行 fp16 √/fp32 √/fp64 ×/bf16 ×（亲读表行 ✓，支撑 R2 双触发与候选 2 负向证据）；T.vsub.md §2.3 条 2「标量须在 T.Kernel 内」、条 4「仅 src1 可标量」、§2.4 示例一三缓冲形态（亲读 ✓，fp32 路线 `vsub(w32,1.0,x_ub)` dst≠src0 合法性依据）；T.vcast.md rint 路径与无 f64 行（首轮亲读，修订未触及）；mish L77 `T.vexp(x_f32_ub, buf1_ub)` / L78 `T.vadd(buf1_ub, 1.0, buf2_ub)` 行号本次 grep 亲验 ✓。v-prefix 优先、Developer 自动同步一致，无手动 set/wait。§9.1#8 无逐字同形先例的如实声明 + one32 回退方案维持（UB 代价改平台值口径表述，见维度 2）。

### 2. 内存层级规划 — pass（首轮 fail 项，本次 pass）

- **搬运路径完整：pass**。两路线 GM↔UB 直连（MTE2/MTE3）纯 Vector 无越级；逐路线字面数据流复推：transit ⑤ vcast 回写 x_ub→⑥ 拷 x_ub ✓；fp32 ④ vsub 跨缓冲 dst=x_ub（原值死于 ③ vexp 全量消费，WAR 由自动栅栏保证）→⑥ 拷 x_ub ✓；恒等映射错误防呆（§9.2）保持。
- **UB 预算：pass**。§4.5 平台值口径经条目原文亲读 + 四档算术独立复算全对（见机械复核段）；guard 分 dtype（fp32 4096 / fp16·bf16 8192）使设计允许配置上界全部 < 196,608 B；transit 8192 保留有实测先例（leaky 第五证）与浮动上界（180.2KB）双重支撑；探针条款挂 Stage 3 首跑并明确「调 block/nl 连带复验」（TRAP-UB 规律 2 同源）——**问题 3 闭环，维度 2 通过标准（中间缓冲不超平台 UB 容量）在设计全部允许配置上成立**。默认档不受影响。
- **Buffer 层级标注：pass**。§4.3 全 buffer 标 UB + 角色/复用；x32 仅 transit 路线、fp32 无独立 x32 与 §3.3 代码一致。

### 3. Tiling 策略 — pass

§5.2 默认 block 1024/2048（= wrapper `_BLOCK_SIZE_*` L65–66 grep 核对 ✓）+ UB cap 分 dtype（修订同步）；§5.5 分核三要素齐且本次独立复跑要素②（ret=0→20→40 ✓）、复算要素①表 6/6 ✓、要素③静态边界 ✓（core-split-strategy.md §1 核对标准）；非整除 tail 切片（官方 vec_add_1d 模式）；尾轴对齐由 GM 连续 1-D + [0:tail] 消化；2 的幂建议保留。`design_calc_check` core_split skip → 人工复算 pass（见机械复核段）。

### 4. 技术约束检测 — pass

§3.5.1 十项齐：一维 Kernel ✓ / L0C·GEMM·L1 分形 n/a（纯 Vector 属实）✓ / bool-i1 无（无 vcmp）✓ / bf16 绕开（docs 亲验）✓ / fp64 三重负向证据 ✓ / 核数超发 persistent ✓ / 无 inplace ✓；**UB 容量行已改分 dtype 平台值口径**（首轮受问题 3 影响的行随修订同步 ✓）。

### 5. 循环与同步 — pass

§6.1 循环表与 §3.3 伪代码逐行一致（核间 `T.Kernel(num_kernels)` / 核内 `T.serial(num_local_tasks)` 静态边界 / tile 内无显式循环）；无 Pipelined/Parallel/Persistent 理由与同族一致；§7 同步表逐边与 Developer 自动栅栏模式一致（含 fp32 路线 vsub 跨缓冲 WAR 论断复验成立）；尾块 §6.4 与代码一致；§9.2 loop-carried 禁令保持。

### 6. 验证方案 — pass（首轮 warn 项随问题 1/2/3 修订闭环）

§8.1 golden 独立且与源仓锚点逐字一致 ✓；§8.2 容差表与 `_get_tolerances`（L24–29）一致，「容差包络 + 裕度、不主张 bitwise」口径声明及其依据（81.7% 分布 + 绝对差 ≤7.629e-06）修订后与实跑一致 ✓；§8.3 L0 六项覆盖面维持（角点/分核边界 39-41/N≤0/output_dtype 防御复算一致），**附加用例按两段式契约改写 + 分 dtype guard 用例（8193/4097/8193）与 §4.5 一致**——首轮两处「按错误契约书写」项已修正。

### 7. 完整性与一致性 — pass（两处建议级引用精度注记）

模板 §0（0.1–0.7 齐）–§10 + 修订记录齐 ✓；占位符/未完成记号扫描无命中（「待 Stage 3/5」为交付状态列合法标注）✓；§3.5.3 七条 examples 引用路径本次逐一实测存在 ✓（含 manifest/测试/基准共 10 路径）；行号引用抽查：GPU L2142–2147、提取件 L13/L16/L23–L30、wrapper L65–66/L81/L130/L133/L142–146/L155/L159–168/L170–181、mish L77–L78、leaky_relu.py L98 全部精确命中；**F-1（建议级）**：elu 两段式 return 行实读为 **L99**（`return _make_elu_kernel(N, dtype, alpha)(block_size)(x)`），设计多处引作「L100」——引用内容 verbatim 正确、仅行号 off-by-one（家族两段式实质经亲读成立），建议 Stage 3 前顺手更正或改以 verbatim 为主锚；**F-2**：§1.6.1(a) 修订句中「|x| ∈ [1e-4, 0.1] 域绝对误差 ≤6.0e-8 / 联合包络裕度 ≥167×」按 (1e-4,1e-2] 带读与实跑一致（5.999e-08/166.7），但按整域读 (1e-2,1e-1] 带实跑 6.300e-08/纯 atol 159×——括注「(1e-4, 1e-2] 带」仅限定 rel 值，建议同步限定 abs 值出处带；判定锚（最窄裕度 67×、viol=0）不受影响，不阻塞。首轮两处内部不一致（契约/ stale）均已消除，无新增矛盾。

### 8. 算法优化分析 — pass（首轮 fail 项，本次 pass）

- **§1.6 结构：pass**。1.6.0/1.6.1/1.6.2/1.6.3 + 1.6.3.1 齐备；轻量调研深度超出下限。
- **§1.6.0 调研完整性：pass**（维持项，修订未回退）。R1 候选表含基线 + 6 族；R2 结构判据明确；R3 四口径基线在表（机械 r3_metrics pass）；R4 亲和评估（容量行已改分 dtype 平台值口径）；调研结论 + 6 面调研范围声明齐。
- **调研结论正确性（独立复核）：pass**。① 候选覆盖对照 algorithm-candidates.md ALG-elementwise 命中行（CSE/除转乘/cast 合并/恒等变形/R2 单遍/R3 融合/R4 dtype + max 化简两例模板）逐项有回应，max 模板不适用判定（无 select 结构）复核成立；② 负向断言维持首轮亲读证据（fp64 三重 doc ×、bf16 vexp/vsub ×、FMA 目录枚举缺失），本次 vexp 表行复验 ✓；③ 复杂度独立复算维持（2N·B 下界、8/10B/elem resident、Taylor +19 op 口径自洽）；④ 机器证据自洽：B 306906<A 496251（重跑命中，「减少 40%」实算 38.2%——**F-3 建议级**，约数表述，建议写「≈38%」或注明口径）；C 2737/1ulp 命中；⑤ 无纸面拍板并列候选（基线四口径唯一最优 + Taylor 双判）；⑥ D-2 估算行锚点归因修订后与 PL-1.21 原文（msprof op 口径）一致（问题 7 闭环）。
- **调研→设计一致性：pass**。§1.4/§3.1/§3.3/§6 = 选定基线 + transit 落地形态；O4 承接表 5 行逐项；§0.5/§0.6 与调研同族。
- **§1.6.1 数学等价优化：pass**。A1 四要素齐；等价性独立推演维持首轮结论（升域无损保序→同序同域→唯一舍回）；**机器复核一致性本次达成**：内嵌表 = 完整脚本执行输出逐格（D-1 门禁），星号位距伪影脚注与 compare 掩码语义（`ulp[finite_pair & ~eq]`）自洽，验收口径声明（47% lane mismatch / 81.7% ≤2 ulp / 绝对差 ≪ atol / viol=0 / 67× 最窄裕度）全部实跑复现——首轮「等价性机器验证证据完整性失效」已闭环，且未改动判定结论（54/54 维持）。
- **§1.6.2 向量化替代：pass**。7 点覆盖与 §3.3/§6.1 交叉核对无遗漏；V1–V4 替代 API docs+先例双佐证；V5–V7 保留理由（任务映射/block 级索引/host 元数据）维持充分；与 §6「tile 内无显式逐元素循环」一致。
- **§1.6.3 轴与布局决策：pass**。单轴豁免成立；弃选②量化理由（repack 负收益，layout.md 实测条目）维持；溢出 lane/陈旧 lane 语义与 §6.4 一致；实验裁决不触发前提成立（带宽常数为已实测条目、按 stale 口径引用留首跑重验，非判定裕度依赖未实证常数）。
- **与下游一致：pass**。§3.1 以 A1 形态为输入；§6 与 1.6.2 一致；§9.1#8 回退方案的 UB 代价表述改平台值口径后与 TRAP-UB 规律 1（结构类与 staging 数解耦）自洽。

## 结论说明

三个阻塞级问题（§1.6.1 证据一致性、wrapper 契约形态、UB 多缓冲口径与 guard）全部闭环且经独立复跑/复算/原文亲读验证到决策行层级；四个建议级问题全部闭环；首轮 pass 维持项抽查无回归；阻塞级维度 0–8 全 pass。三条建议级注记（F-1 行号 off-by-one、F-2 精度表述带限定、F-3 约数 40%→≈38%）均为证据措辞精度问题，不影响任何设计决策与 Stage 3 可实现性，随 Stage 3/5 顺手处理即可。设计可进入 Stage 3。
