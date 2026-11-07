# rmsnorm 算子设计文档

> 平台：Ascend910（evidence: `full_soc=Ascend910B3`, `npu_arch=2201`，来源 asys/ini）
> 目标：`tilelang.compile(func, target="npuir")`，`TILELANG_ASCEND_MODE=Developer`
> 本文档为 Stage 1 产物，仅含设计与测试计划；kernel / 测试代码由 Stage 2 实现。

## 1. 概述

### 1.1 算子名称

`rmsnorm`

### 1.2 功能描述

对二维张量 `X` 的最后一维做均方根归一化（RMSNorm），再乘以逐列仿射权重 `W`。纯 Vector（element-wise + 行归约）算子，不含矩阵乘，无 Cube 核参与。

### 1.3 数学公式

$$
Y_{i,j} = X_{i,j} \cdot \mathrm{rsqrt}\!\left(\frac{1}{N}\sum_{k=0}^{N-1} X_{i,k}^{2} + \varepsilon\right) \cdot W_{j},
\qquad i\in[0,M),\ j\in[0,N)
$$

默认 `eps = 1e-5`，`rsqrt(z) = 1/sqrt(z)`。

### 1.4 算法描述

1. 每个 block 负责 `block_m` 个完整行；`N` 个隐藏维元素一次性驻留 UB（不切 N）。
2. 搬入 `X[m0:m0+rows, 0:N]` 与 `W[0:N]`。
3. fp16 路径把输入 cast 到 fp32；fp32 路径直接复用输入 tile。
4. 逐元素平方，沿 `dim=1` 做行归约得到 `(block_m, 1)`。
5. `mean = sum * (1/N) + eps`，再 `rstd = 1/sqrt(mean)`（`T.vsqrt` + `T.vrec`，fp32）。
6. `Y = X_f32 * rstd * W`；fp16 路径最后 cast 回 fp16。
7. 只把有效行 `[0:rows)` 写回 `Y[m0:m0+rows, 0:N]`。

### 1.5 数据流图

```
GM[X] --T.copy--> UB[x_sh] --(fp16: T.Parallel+T.cast)--> UB[x_f32]
GM[W] --T.copy--> UB[w_sh] --(fp16: T.Parallel+T.cast)--> UB[w_f32]
UB[x_f32|x_sh] --T.Parallel 平方--> UB[xsq] --T.reduce_sum(dim=1)--> UB[acc]
UB[acc] --T.Parallel: *1/N+eps--> UB[acc] --T.vsqrt--> UB[rstd] --T.vrec--> UB[rstd]
UB[x_f32]*UB[rstd]*UB[w_f32] --T.Parallel--> UB[xsq] --(fp16: T.vcast)--> UB[x_sh]
UB[x_sh] --T.copy(valid rows)--> GM[Y]
```

### 1.6 支持性声明（不得静默缩小）

| 需求附加验证项 | 结论 | 设计依据 |
|---|---|---|
| 边界 `X [1023,1024] float32`、`W [1024]`、`Y [1023,1024]` | **支持** | `block_m=16` → `ceildiv(1023,16)=64` 个 block，最后一个 block 只 `15` 行有效；输入/输出两侧都用显式 valid extent BufferRegion，不做 host padding/crop。见 §5.4/§6.4 |
| float16 路径 `[1024,1024]` / `[1024]` | **支持** | fp16 存储 + fp32 累加/归一化，最后 `T.vcast(round_mode="rint")` 回 fp16；精度标准见 §9.3（atol=rtol=1e-2） |
| `X/W/Y` 形状 `[1024,1024]` / `[1024]` float32（L0） | **支持** | §9.2 L0 用例 |

> Host 侧 Buffer 约束：host 仅做张量创建（测试侧）、kernel 调用与结果取回验证；无 `.contiguous()`、无 host padding、无 `F.pad/cat`、无 `.to(dtype)`。`X/W/Y` 原生连续 2D/1D，无需 reshape。见 §4.7 与 [HOST-METADATA-AUDIT]。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式：Developer**（`TILELANG_ASCEND_MODE=Developer`，`tilelang.compile(func, target="npuir")` 或 `@tilelang.jit(target="npuir")`）。

### 2.2 选型理由

| 依据 | 说明 |
|---|---|
| 需求指定 | `op_requirements.programming_mode = developer` |
| 计算类型 | 纯 Vector：只有 element-wise 与行归约，无 `T.gemm`，不需要 Cube/L1/L0A/L0B/L0C |
| 无 CV 融合 | 不存在 Cube↔Vector 数据交互，不需要 workspace、`workspace_idx`、`vid` |
| 框架同型示例 | `examples/norm/rms_norm.py` 正是 Developer 写法：`alloc_shared` + `alloc_fragment` + `T.Parallel` + `T.reduce_sum` |
| 编译后端 | `tilelang/engine/lower.py:169-179`：`TILELANG_ASCEND_MODE` 不是 `expert/exp/e` 时走 `target.build.tilelang_npuir_dev`，`"Developer"` 与示例里的 `"Dev"` 均命中该分支 |

### 2.3 模式影响

| 维度 | 本算子的选择 |
|---|---|
| 内存分配 | `T.alloc_shared` 承载 GM↔UB tile；`T.alloc_fragment` 承载计算中间量与行归约结果（Developer 下均由编译器规划到 UB） |
| 计算方式 | `T.Parallel` 逐元素 + `T.reduce_sum` 内置归约 + `T.vsqrt/T.vrec/T.vcast` v-prefix 向量指令 |
| 作用域 | 无显式 `T.Scope`；编译器自动分离核内操作 |
| 同步 | 自动同步（`npuir.disable_hivm_auto_inject_sync` 默认 False，即自动注入）；无手动 `T.barrier` |
| 线程 | 不传 `threads`（纯 Vector，`T.Kernel(..., is_npu=True)` 的 1D block 语义） |
| workspace | 无（Developer 默认消除 workspace/vid） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|---|---|---|
| 1 | `t = X[m0:m0+rows, :]` | 行分块搬入（尾块 valid extent） |
| 2 | `x = fp32(t)` | 仅 fp16 路径需要；fp32 路径无 cast |
| 3 | `s = x ⊙ x` | 逐元素平方 |
| 4 | `a_i = Σ_j s_{i,j}` | 沿隐藏维行归约，`(block_m,N) → (block_m,1)` |
| 5 | `z_i = a_i / N + eps` | 均值 + 数值稳定项 |
| 6 | `r_i = 1 / sqrt(z_i)` | RMS 的倒数 |
| 7 | `y_{i,j} = x_{i,j} · r_i · w_j` | 行广播 × 列广播 |
| 8 | `Y = cast(y)` | 仅 fp16 路径需要 |

### 3.2 TileLang API 映射（均已在本 repo 验证存在）

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 存在性证据 |
|---|---|---|---|---|---|
| 1 | 搬入行 tile | `T.copy(src_region, dst_region)` | `X[m0:m0+rows, 0:N] → x_sh[0:rows, 0:N]`；两侧均 BufferRegion | Developer | `tilelang/language/copy.py:98`；动态 region 见 `examples/flash_attn_npuir_dev.py:88,89`、`examples/vectorization_in_parallel.py:37-43` |
| 1b | 搬入权重 | `T.copy(W, w_sh)` | 1D GM → UB | Developer | `testing/npuir/norm_ops/test_group_norm_dev.py:63` |
| 2 | fp16→fp32 | `T.Parallel(bm,N)` + `T.cast(..., "float32")` | 逐元素 | Developer | `examples/norm/rms_norm.py:37-40`；`T.cast` 可用性实测 True |
| 3 | `x ⊙ x` | `T.Parallel(bm,N)` 标量乘 | 同下标 `[i,j]`，不做下标变换 | Developer | `examples/norm/rms_norm.py:37-40`；下标规则见 `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3 |
| 4 | 行归约 | `T.reduce_sum(xsq, acc, dim=1, clear=True)` | `(bm,N) → (bm,1)`，fp32 输入受支持 | Developer | `tilelang/language/customize_npuir.py:950`（最终导出）；`docs/Tilelang.language/规约操作/T.reduce_sum.md`（fp32 √）；`examples/norm/rms_norm.py:43` |
| 5 | 均值+eps | `T.Parallel(bm)` 标量乘加 | `acc[i,0]*(1/N)+eps` | Developer | `examples/norm/rms_norm.py:46-47`；`own_ops/rmsnorm.py:144-145` |
| 6 | `1/sqrt(z)` | `T.vsqrt(acc, rstd)`；`T.vrec(rstd, rstd)` | fp32；in-place 非原地二选一 | Developer | `tilelang/language/customize_npuir.py:347`（sqrt）、`:386`（rec→1/A）；导出 `tilelang/language/__init__.py:144,150`；用例 `own_ops/rmsnorm.py:149-151` |
| 6' | 可选快速路径 | `T.vrsqrt(z, rstd)` | **仅 fp16 且实测达标后启用**；fp32 禁用（硬件近似 ~3e-3 相对误差，过不了 1e-4） | Developer | `tilelang/language/customize_npuir.py:360`；导出 `__init__.py:146`；精度数据 `own_ops/rmsnorm.py:15-20,318-334` |
| 7 | 广播乘 | `T.Parallel(bm,N)`：`x* rstd[i,0] * w[j]` | `(bm,N)×(bm,1)×(N,)` 广播 | Developer | `examples/norm/rms_norm.py:51-56`；`test_group_norm_dev.py:87-91` |
| 8 | fp32→fp16 | `T.vcast(xsq, x_sh, round_mode="rint")` | fp32 结果 → 输出 dtype | Developer | `tilelang/language/customize_npuir.py:723`；导出 `__init__.py:119`；用例 `test_group_norm_dev.py:88`、`own_ops/rmsnorm.py:171` |
| 9 | 搬出有效行 | `T.copy(x_sh[0:rows, 0:N], Y[m0:m0+rows, 0:N])` | 显式 valid extent | Developer | 同步骤 1；`examples/vectorization_in_parallel.py:44` |

> 注：`T.reduce_sum` 在 `tilelang/language/__init__.py` 中先由 `.reduce` 导入（第 51-57 行），随后被 `.customize_npuir` 的同名符号覆盖（第 119-126 行），最终绑定的是 `customize_npuir.reduce_sum`（签名 `dim=-1, size=None, clear=True`）。两者语义一致，本文按最终绑定书写。

### 3.3 计算伪代码

```python
@tilelang.jit(out_idx=[2], target="npuir")
def _rmsnorm(M, N, block_m, dtype, eps):
    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        W: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (cid, _):
            x_sh  = T.alloc_shared((block_m, N), dtype)     # GM 入口/出口 tile
            w_sh  = T.alloc_shared((N,), dtype)
            xsq   = T.alloc_fragment((block_m, N), "float32")
            acc   = T.alloc_fragment((block_m, 1), "float32")
            rstd  = T.alloc_fragment((block_m, 1), "float32")

            x_f32 = T.alloc_fragment((block_m, N), "float32")  # 仅 fp16 路径实际使用
            w_f32 = T.alloc_fragment((N,), "float32")          # 仅 fp16 路径实际使用

            m0   = cid * block_m
            rows = T.min(block_m, M - m0)

            T.copy(X[m0 : m0 + rows, 0:N], x_sh[0:rows, 0:N])
            T.copy(W, w_sh)

            if dtype == "float16":                 # Python 编译期分支
                for i, j in T.Parallel(block_m, N):
                    x_f32[i, j] = T.cast(x_sh[i, j], "float32")
                for j in T.Parallel(N):
                    w_f32[j] = T.cast(w_sh[j], "float32")
                for i, j in T.Parallel(block_m, N):
                    xsq[i, j] = x_f32[i, j] * x_f32[i, j]
            else:
                for i, j in T.Parallel(block_m, N):
                    xsq[i, j] = x_sh[i, j] * x_sh[i, j]

            T.reduce_sum(xsq, acc, dim=1, clear=True)

            for i in T.Parallel(block_m):
                acc[i, 0] = acc[i, 0] * (1.0 / float(N)) + eps
            T.vsqrt(acc, rstd)
            T.vrec(rstd, rstd)

            if dtype == "float16":
                for i, j in T.Parallel(block_m, N):
                    xsq[i, j] = x_f32[i, j] * rstd[i, 0] * w_f32[j]
                T.vcast(xsq, x_sh, round_mode="rint")
            else:
                for i, j in T.Parallel(block_m, N):
                    x_sh[i, j] = x_sh[i, j] * rstd[i, 0] * w_sh[j]

            T.copy(x_sh[0:rows, 0:N], Y[m0 : m0 + rows, 0:N])

    return main
```

要点：
- `if dtype == "..."` 是 **Python 字符串分支**，在 trace 期展开，不会生成运行时分支（`own_ops/rmsnorm.py:121-134` 同型用法）。
- fp32 路径原地覆写 `x_sh`，省掉独立的输出 buffer，这是 UB 预算能容纳 `block_m=16` 的关键（§4.5）。
- fp16 路径用 `x_f32` 保留原始输入供最终乘；`T.vcast` 覆写 `x_sh` 时输入已消费完，安全。
- 没有 `T.if_then_else`、没有动态循环边界、没有 `<i+1>` 形式的下标变换。

### 3.4 API 可行性确认

| API | 结论 | 确认方式 |
|---|---|---|
| `T.Kernel(blocks, is_npu=True)` | 存在；NPU 必须且仅能 1 个 block 维 | 源码 `tilelang/language/kernel.py:218-253`；文档 `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3；运行时 `hasattr` 实测 True |
| `T.ceildiv` | 存在 | `hasattr(T, "ceildiv") == True`；示例 `examples/norm/rms_norm.py:25` |
| `T.alloc_shared` | `scope="shared.dyn"` | `tilelang/language/allocate.py:22` |
| `T.alloc_fragment` | `scope="local.fragment"` | `tilelang/language/allocate.py:54` |
| `T.Parallel` | 存在；循环体内只能访问 UB Buffer，下标不可变换 | `docs/Tilelang.language/编译器提示操作/T.Parallel.md` §2.3 |
| `T.reduce_sum` | 存在，fp32/fp16 输入支持，`dim`/`clear` | `tilelang/language/customize_npuir.py:950`；`docs/Tilelang.language/规约操作/T.reduce_sum.md` §2.2.1 |
| `T.vsqrt` / `T.vrec` / `T.vrsqrt` | 存在 | `customize_npuir.py:347/386/360`；`__init__.py:144/150/146` |
| `T.vcast` | 存在，`round_mode="rint"` | `customize_npuir.py:723`；`__init__.py:119` |
| `T.cast` | 存在（TVMScript T 命名空间） | `hasattr(T,"cast")==True`；框架示例 `examples/norm/rms_norm.py:38`；仓内测试 `test_group_norm_dev.py:68` |
| `T.min` | 存在 | `hasattr(T,"min")==True`；`examples/flash_attn_npuir_dev.py:88` |
| `T.serial` / `T.Pipelined` | 存在但本设计**不使用**（N 全量入 UB，无 K 循环） | `hasattr` 实测 True |
| `T.clear` / `T.fill` | 存在（可选防御手段，见 §6.4） | `tilelang/language/fill.py:10,25` |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|---|---|---|
| 不支持三维 Kernel | 否 | 仅 1D grid：`T.Kernel(T.ceildiv(M, block_m), is_npu=True)` |
| threads 参数限制（仅 1 或 2） | 否 | 纯 Vector，不传 `threads` |
| 动态循环边界不支持 | 否 | 无 `T.serial`/`T.Pipelined` 循环；只有动态 **region extent**（前端支持） |
| 流水线不支持动态边界 | 否 | 不使用流水线；多 buffer 由 `npuir.enable_auto_multi_buffer`（默认 True）自动处理 |
| `T.copy` 列方向 strided 切片 | 否 | 2D 连续 tile，最内维 `N` 连续、无列向 gap |
| Host 侧 padding / `reshape` 物理化 | 否 | 无 host padding；`N=1024` 直接按真实 shape 建 kernel；无 reshape |
| `T.if_then_else` 逐元素分支 | 否 | 数值稳定只依赖 `+eps`，无分支 |
| L0C 溢出 | 否 | 无 Cube |
| GEMM 非整除 | 否 | 无 GEMM；仅 M 方向尾块，按 §5.4 处理 |

#### 3.5.2 参考实现差异说明

无 GPU 版外部参考实现；对照的是本仓 NPUIR 示例。差异：

| 差异项 | 仓内示例做法 | 本设计 | 原因 |
|---|---|---|---|
| `examples/norm/rms_norm.py` | host 侧把 N 对齐到 256 并 padding 张量 | 不 padding，直接用 `N=1024` | 约束 §5 禁止 host padding；且真实 N 已 256 对齐 |
| 同例 | `T.vrsqrt` | fp32 用 `vsqrt+vrec`，fp16 默认也用 `vsqrt+vrec` | `vrsqrt` 硬件近似 ~3e-3 相对误差，过不了 fp32 1e-4 |
| 同例 | 权重直接在 `T.Parallel` 里读 GM `weight[j]` | 先 `T.copy(W, w_sh)` 到 UB 再读 | `T.Parallel` 循环体只允许访问 UB Buffer（Parallel 文档 §2.3） |
| 同例 | 无尾块处理 | `T.min` + 双侧 BufferRegion valid extent | M 非整除需求（1023） |
| `examples/norm/example_rms_norm.py` | `T.clear` + split-K 双 pass | 单 pass 全 N 入 UB | N=1024 可全量驻留，避免二次 GM 读 |

#### 3.5.3 本项目同类实现参考（强制步骤 0 结果）

| 文件路径 | 相似度 | 关键参考点 |
|---|---|---|
| `/mnt/workspace/learn/tilelang-ascend/examples/norm/rms_norm.py` | 高度相似 | Developer 模式 rmsnorm 主结构：`alloc_shared`+`alloc_fragment`、`T.Parallel` 平方、`T.reduce_sum(dim=1)`、`T.vrsqrt`、最终乘 W |
| `/mnt/workspace/learn/tilelang-ascend/examples/norm/example_rms_norm.py` | 高 | `T.reduce_sum` + `T.npuir_mul/div/add` 的 split-K 变体；`"Dev"` 模式运行方式 |
| `/mnt/workspace/learn/tilelang-ascend/testing/npuir/norm_ops/test_group_norm_dev.py` | 高 | Developer 模式归一化算子的完整测试骨架：`alloc_shared` 全流程、fp16→fp32 cast、`T.vcast` 输出、fp32/fp16 参数化 |
| `/mnt/workspace/learn/tilelang-ascend/examples/flash_attn_npuir_dev.py` | 中（尾块参考） | Developer 模式下 `real_m = T.min(block_m, seq_len - cid*block_m)` + 动态 region 写回 |
| `/mnt/workspace/learn/tilelang-ascend/examples/vectorization_in_parallel.py` | 中（尾块参考） | `T.copy(A[start:start+tail], A_VEC[0:tail])` 双侧动态 extent |
| `/mnt/workspace/learn/tilelang-ascend/own_ops/rmsnorm.py` | 参考（Expert 方言，不复制） | 证明 `T.vrsqrt/vsqrt/vrec/reduce/vmul/vadd/vcast` 与动态行尾切片在本 revision 可用；其精度/UB 实测数据用于本设计选型 |

#### 3.5.4 多路径分派审计

```text
[DISPATCH-COVERAGE]
supported_domain: rank-2 X(M,N) with M>=1, 1<=N<=8192; W(N,); X/W/Y 同 dtype ∈ {float32, float16}; eps float
generic_fallback: rmsnorm.py 的单一 Developer kernel（无特化分支；dtype 仅在 trace 期决定缓冲布局）
specializations:
  - predicate: dtype == "float16"
    fallback_on_miss: float32 branch
    equivalence_evidence: 同一公式；仅多一次 cast 与一次回 cast（§3.3），数学等价
unsupported_inputs: N>8192; dtype ∉ {float32,float16}; rank!=2; W.shape[0]!=N; M==0; X/W dtype 不一致
result: pass
```

说明：`block_m` 是**编译期 tiling 参数**，由 host 侧纯整数运算按 `N/dtype/UB` 决定（§5.2），不读取 tensor 数据，不构成数据相关分派。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `X` | `(M, N)`，L0 为 `(1024,1024)`，边界 `(1023,1024)` | `float32` / `float16` | 行归一化输入；`N` 为归约轴 |
| `W` | `(N,)`，`(1024,)` | 与 `X` 相同 | 逐列仿射权重 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `Y` | `(M, N)`，与 `X` 同形 | 与 `X` 相同 | 归一化结果 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 | 是否两路径共用 |
|---|---|---|---|---|---|
| `x_sh` | `(block_m, N)` | 输入 dtype | UB (`alloc_shared`) | 搬入 tile；fp32 原地写结果；fp16 用 `vcast` 写回 | 是 |
| `w_sh` | `(N,)` | 输入 dtype | UB (`alloc_shared`) | 权重 tile | 是 |
| `xsq` | `(block_m, N)` | float32 | UB (`alloc_fragment`) | 平方 / 最终 fp32 结果 | 是 |
| `acc` | `(block_m, 1)` | float32 | UB (`alloc_fragment`) | 行和 → 均值+eps（原地） | 是 |
| `rstd` | `(block_m, 1)` | float32 | UB (`alloc_fragment`) | `1/sqrt(mean)` | 是 |
| `x_f32` | `(block_m, N)` | float32 | UB (`alloc_fragment`) | fp16 输入的 fp32 副本 | 仅 fp16 |
| `w_f32` | `(N,)` | float32 | UB (`alloc_fragment`) | fp16 权重的 fp32 副本 | 仅 fp16 |

**复用方案**：`acc` 在 `T.vsqrt(acc, rstd)` 后不再使用；`x_sh` 在 fp32 路径同时承担输入与输出；`xsq` 同时承担平方中间量与最终 fp32 结果；fp16 路径不额外申请输出 buffer，直接 `T.vcast` 覆盖 `x_sh`。因此每路径只有 5 个（fp32）/ 7 个（fp16）buffer。

### 4.4 内存搬运路径

```
纯 Vector，无 L1 / L0A / L0B / L0C：

GM[X] --T.copy(valid rows)--> UB[x_sh]
                                 ├─ fp32: -------------------------→ UB[xsq]（平方）
                                 └─ fp16: --T.Parallel cast--> UB[x_f32] --平方--> UB[xsq]
GM[W] --T.copy--> UB[w_sh] --(fp16: cast)--> UB[w_f32]

UB[xsq] --T.reduce_sum(dim=1)--> UB[acc] --(1/N, +eps)--> UB[acc] --vsqrt--> UB[rstd] --vrec--> UB[rstd]

UB[x_f32|x_sh] * UB[rstd] * UB[w_f32|w_sh] --T.Parallel--> UB[xsq]
                                 ├─ fp32: 原地写回 UB[x_sh]
                                 └─ fp16: T.vcast(xsq -> x_sh)
UB[x_sh] --T.copy(valid rows)--> GM[Y]
```

### 4.5 UB 内存预算（Ascend910 单核 UB = 192 KiB = 196,608 B，见 `docs/开发指南.md:126`）

**fp32 路径，block_m = 16，N = 1024：**

| Buffer | Shape | dtype | 大小 (Bytes) |
|---|---|---|---|
| `x_sh` | (16, 1024) | float32 | 65,536 |
| `xsq` | (16, 1024) | float32 | 65,536 |
| `w_sh` | (1024,) | float32 | 4,096 |
| `acc` | (16, 1) | float32 | 64 |
| `rstd` | (16, 1) | float32 | 64 |
| **总计** | | | **135,296 B = 132.1 KiB（68.8%）** |

**fp16 路径，block_m = 16，N = 1024：**

| Buffer | Shape | dtype | 大小 (Bytes) |
|---|---|---|---|
| `x_sh` | (16, 1024) | float16 | 32,768 |
| `x_f32` | (16, 1024) | float32 | 65,536 |
| `xsq` | (16, 1024) | float32 | 65,536 |
| `w_sh` | (1024,) | float16 | 2,048 |
| `w_f32` | (1024,) | float32 | 4,096 |
| `acc` | (16, 1) | float32 | 64 |
| `rstd` | (16, 1) | float32 | 64 |
| **总计** | | | **170,112 B = 166.1 KiB（86.5%）** |

剩余空间给编译器生成的归约临时量（`reduce_sum(clear=True)` 会为 `out` 建同形 temp，64 B 级）与自动 multi-buffer 规划，均远小于余量。两路径均 < 192 KiB。

**UB 利用率说明（准则 1）**：`block_m` 取 16 的硬约束是"必须整除 1024 以满足 L0 规则 shape"；下一个可整除候选 32 在 fp32 下需 `2×32×1024×4 + 4 KiB = 260 KiB > 192 KiB`，不可行。若放宽整除，fp32 可取 `block_m=20`（164 KiB，85%）或 `block_m=22`（180 KiB，92%），但会破坏 L0 的 block 整除前提，故作为性能阶段的可选调优点记录，不作为主设计。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|---|---|---|
| `M` | `T.Tensor((M, N), dtype)` 的 Python 编译期常量 | `1 ≤ M ≤ 2^31-1`（本任务 1024 / 1023） |
| `N` | 同上 | `1 ≤ N ≤ 8192`（本任务 1024） |

本设计的 `M`/`N` 都是 **JIT 编译期常量**（每个 shape 组合单独编译），不做 `T.dyn` 符号化。尾块由 `rows = T.min(block_m, M - m0)` 在 kernel 内计算。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="npuir")
def _rmsnorm(M: int, N: int, block_m: int, dtype: str, eps: float): ...
# 调用：program = _rmsnorm(M, N, block_m, dtype, eps); y = program(x, w)
```

- `out_idx=[2]`：由 wrapper 分配 `Y`，与 `examples/norm/rms_norm.py:16` 一致。若 Stage 2 需要显式输出 buffer（benchmark/多流场景），可去掉 `out_idx` 并传 `y`。
- `pass_configs`：**不设置，使用默认值**。本 revision 中真实存在的相关 key 是：
  - `tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER`（默认 True）
  - `tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC`（默认 False，即自动同步）
  - `tilelang.PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`（默认 True）
  - `tilelang.PassConfigKey.TL_DISABLE_DYNAMIC_TAIL_SPLIT`（默认 False）
  纯 Vector 单 pass 无流水可调，默认配置即可。

```text
[HOST-METADATA-AUDIT]
operation: y = kernel(x, w)（可选 out_idx 分配 Y）；验证时 y.detach().cpu()
input_stride -> output_stride: X 连续 (N,1) 不变；W 连续 (1,) 不变；Y 为内核直写
shares_storage / same_data_ptr: true（host 不创建/改写任何输入 buffer）
aclnn_or_physical_copy: false（无 contiguous/pad/cat/to/clone；.cpu() 属结果验证）
result: allow
```

`block_m` 的选择（§5.2）只读 shape/dtype，不读 tensor 数据，是纯 metadata 的 host 整数运算。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：纯 Vector。**

**判定依据**：公式只含 element-wise（平方、乘）与沿最后一维的归约，无 `@`/matmul，不使用 Cube 单元；因此只需 UB，不需要 L1/L0 分块。

### 5.2 Block 划分

```python
# host 侧纯整数 tiling 选择（编译期常量，不读数据）
def pick_block_m(N, dtype, ub_bytes=196608):
    if dtype == "float32":
        # 每行元素字节: x_sh(4) + xsq(4)；权重区 4*N
        return max(1, min(16, (ub_bytes - 4 * N) // (8 * N)))
    else:
        # x_sh(2) + x_f32(4) + xsq(4)；权重区 (2+4)*N
        return max(1, min(16, (ub_bytes - 6 * N) // (10 * N)))

block_m  = pick_block_m(N, dtype)      # N=1024 → 16（fp32/fp16 同）
block_num = T.ceildiv(M, block_m)      # M=1024 → 64；M=1023 → 64
```

| 参数 | 值 | 选择理由 |
|---|---|---|
| `block_m` | 16（N=1024） | 1024 的最大可行整除因子；使 L0 为规则 shape；fp32 132.1 KiB / fp16 166.1 KiB 均 < 192 KiB |
| `block_n` | `N`（不切） | 归约轴必须完整驻留才能一次 `reduce_sum`；N=1024 的 fp32 行仅 4 KiB |
| grid | `ceildiv(M, 16)` | M=1024 → 64；M=1023 → 64（末块 15 行） |
| 通用性 | `pick_block_m` 随 N 自动缩小 | N 增大时保持 UB 不溢出：`N=8192` fp32 → `block_m=2`；fp16 → `block_m=1` |

### 5.3 约束分析

- **对齐约束**：`N=1024`，fp32 行 4096 B、fp16 行 2048 B，均满足 32 B 向量对齐；`block_m` 为 2 的幂，tile 起始地址天然对齐。
- **UB 容量**：见 §4.5，两路径均有余量。
- **L0C 容量**：不适用（无 Cube）。
- **L1/L0A/L0B**：不使用。
- **核数匹配**：64 个 block。910B3 为 20 AI Core（文档 `docs/开发指南.md:166` 以 20/40/60 作为物理核整数倍示例，对应约 40 AIV），64 个 Vector 任务约 1.6 个 AIV 波次，存在轻微尾波不均衡；受"block 必须整除 1024 以保 L0 规则 shape"与 UB 上限双重约束，16 已是可行最优，记录为性能阶段调优点（例如 fp16 可试 `block_m=20`，grid=52）。

### 5.4 注意事项（非整除必须显式设计）

`M=1023` 时 `1023 = 63 × 16 + 15`：

1. 网格按 `ceildiv` 取 64 个 block，最后一个 block 的 `rows = T.min(16, 1023 - 63*16) = 15`。
2. **输入侧**：`T.copy(X[m0:m0+rows, 0:N], x_sh[0:rows, 0:N])` —— 源、目的都写显式 valid extent（BufferRegion）。
3. **输出侧**：`T.copy(x_sh[0:rows, 0:N], Y[m0:m0+rows, 0:N])` —— 仅写 15 行，不越界写第 16 行。
4. compute 仍按静态 `(block_m, N)` 展开：末 block 的第 16 行是未初始化 UB 垃圾。`T.reduce_sum(dim=1)` 按行独立归约，垃圾行只产生垃圾的 `acc[15,0]/rstd[15,0]`，既不写回 GM，也不影响前 15 行（行间无归约耦合）。因此不需要 host padding，也不需要预清零。
5. 若 L1 尾块用例观测到非预期数值污染（例如底层实现并非严格行独立），退路是在搬入前加一次 `T.clear(x_sh)` 全 tile 清零（`tilelang/language/fill.py:25`，64 KiB 片上写，成本远低于 8 MiB GM 流量）。该退路不改接口、不改精度标准。
6. 禁止方案：host 侧 `torch.zeros` padding + 切片 crop、`F.pad`、对齐到 256 后 reshape —— 均违反 host Buffer 约束。

### 5.5 数据搬运性能可行性

本算子不做数据重排（无 permute/transpose/索引重排），仍按约束给出成本审计：

```text
[REORDER-COST]
path: generic_fallback (Developer, 全形状统一)
gm_passes: 1 读 X + 1 写 Y；W 每 block 读一次（4 KiB）
dma_transactions: ~64 个 X tile DMA + 64 个 Y tile DMA + 64 个 W DMA ≈ 192 次
average_dma_bytes: X/Y tile 64 KiB；W 4 KiB（加权平均 ≈ 42.7 KiB）
gm_scalar_accesses: 0（无逐元素 GM 标量读写；所有标量在 UB）
address_div_mod_per_element: 0（无逐元素地址解码）
active_cores / serial_tasks_per_core: 64 个 Vector 任务 / 约 40 AIV → 1~2 个任务/核
largest_case_timeout_gate: X[1024,1024] float32，120 s（含 JIT 编译；稳态目标见 §9.4）
result: pass
```

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|---|---|---|---|
| M（行分块） | block 级并行 | `T.Kernel(T.ceildiv(M, block_m), is_npu=True)` | 每个 block 独立处理 `block_m` 行，无跨 block 依赖 |
| tile 元素 | 向量化并行 | `T.Parallel(block_m, N)` | 平方、cast、最终乘；下标只做原样 `[i,j]` |
| 行标量 | 向量化并行 | `T.Parallel(block_m)` | 均值、eps、rstd 的逐行标量 |
| N（隐藏维） | 不循环 | `T.reduce_sum(xsq, acc, dim=1)` | N 全部驻留 UB，一次内置归约 |
| K（若切 N） | 不适用 | — | 本设计不切 N，无 `T.serial` |

### 6.2 循环伪代码

```python
with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (cid, _):
    # ... alloc / copy（见 §3.3）
    for i, j in T.Parallel(block_m, N):   # 向量化
        xsq[i, j] = ...
    T.reduce_sum(xsq, acc, dim=1)          # 内置行归约
    for i in T.Parallel(block_m):          # 行标量
        acc[i, 0] = ...
    T.vsqrt(acc, rstd); T.vrec(rstd, rstd)
    for i, j in T.Parallel(block_m, N):   # 向量化
        ... 写回 x_sh
    T.copy(x_sh[0:rows, 0:N], Y[m0:m0+rows, 0:N])
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**：每 block 只有一次搬入、一次搬出，没有 K 方向迭代；动态边界也不允许流水化。DMA 与计算的重叠交给默认开启的 `npuir.enable_auto_multi_buffer`（True）与自动同步注入；后续若把每 block 的行数改成多轮 `T.serial`，再评估多 buffer。

### 6.4 尾块处理

见 §5.4。核心是"网格 `ceildiv` + 输入/输出双侧显式 valid extent + compute 按静态 tile 展开"；不使用 host padding。`T.Parallel` 的静态 extent 保证向量化不因动态行数退化。

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步（Developer）。**

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|---|---|---|
| 全部 DMA↔Vector 交接 | 编译器自动注入（HivmAutoInjectSync） | Developer 模式由 `--disable-hivm-auto-inject-sync=false`（默认）控制；无需手写 barrier |

无手动 `T.barrier_all` / `T.sync_block_*`；无跨核（Cube↔Vector）同步，因为无 Cube 参与。

### 7.3 pass_configs 配置

不覆盖默认。若调试阶段需要显式关闭自动同步（不推荐），对应本 revision 的真实 key 为 `tilelang.PassConfigKey.NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC`。

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果：否。**

**判定依据**：公式不含 GEMM/matmul，属于纯 Vector 多步归一化（Normalization 类），不存在 Cube 输出交给 Vector 后处理的情形；因此没有 workspace、没有 `workspace_idx`、没有 `vid` 偏移。

### 8.2 CV 交互设计

不适用。Developer 单核类型 kernel：`T.Kernel(block_num, is_npu=True) as (cid, _)`，无 workspace 参数。

---

## 9. 验证方案

### 9.1 Golden 函数

```python
import torch

def golden_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5,
                   out_dtype=None) -> torch.Tensor:
    """CPU 参考实现（与 NPU torch 算子解耦）。

    全程 float64 计算后转回输出 dtype：既避免参考端自身误差在 fp32 1e-4 门限下
    占比过高，也保证 Boundary 大值域用例的 x^2 累加不在 float32 溢出。
    """
    xf = x.detach().cpu().to(torch.float64)
    wf = w.detach().cpu().to(torch.float64)
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = xf * rstd * wf
    return y.to(out_dtype or x.dtype)
```

说明：默认就用 float64 golden（成本仅测试侧，1024×1024 可接受），彻底消除参考实现精度争议；`out_dtype` 指定按 kernel 输出 dtype 落回。特殊值（inf/nan）用例同样走此函数，比较时先做 mask 结构比对（§9.3 契约）。

### 9.2 L0 门槛测试计划（Stage 1 仅生成 L0）

> L0 只做规则 shape（block 整除）的快速精度收敛；完整 L1/L2/Boundary 由 Stage 2 在 L0 通过后按 `tilelang-op-test-design` 场景 B 扩展。

| 用例名 | 级别 | Shape | dtype | block | 说明 / 精度标准 |
|---|---|---|---|---|---|
| `l0_basic` | L0 | `X (1024,1024)`, `W (1024,)`, `Y (1024,1024)` | float32 | `block_m=16`（1024/16=64，整除） | 需求指定 L0；`eps=1e-5`；`X~N(0,1)`、`W~N(0,1)`；golden §9.1；标准 §9.3 float32 行 |

L0 通过条件：`[PRECISION_PASS]`（`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2`，逐元素 `|a-g| ≤ 1e-4 + 1e-4·|g|`）。

### 9.3 精度标准

> 采用**混合容差 + 双门限**：逐元素 `|actual-golden| ≤ atol + rtol·|golden|`；整体要求 `matched_ratio ≥ required_matched_ratio` **且** `max_abs_error ≤ max_abs_error_limit`。
> 阈值按 dtype，与算子类别无关。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|---|---|---|---|---|
| float32 | 1e-4 | 1e-4 | 1e-2 | 0.99 |
| float16 | 1e-2 | 1e-2 | 1e-1 | 0.99 |

**与 skill 默认表的差异（显式声明，不静默缩小）**：
- 本算子的 `atol/rtol` 取自 `op_requirements.tolerance`（float32 `1e-4/1e-4`，float16 `1e-2/1e-2`），这是用户验收契约。
- `max_abs_error_limit` 与 `required_matched_ratio` 取自 `tilelang-op-test-design/references/precision-standard.md` 的同 dtype 硬帽（fp32 `1e-2`，fp16 `1e-1`）与通过率（0.99）。
- 相对 skill 默认 fp32 行（atol 2⁻¹⁶、rtol 2⁻¹⁰），本设计 **rtol 更严（1e-4 < 9.77e-4）、atol 更松（1e-4 > 1.53e-5）**；相对默认 fp16 行，**atol/rtol 均更松**。两者取并集后仍由双门限约束。
- **实现约束**：Stage 2 的 `get_precision(dtype)` 必须把本表数值硬编码（不能用 skill 的默认 `fp_table`），否则 fp32 门限会被放大到 rtol 9.77e-4。
- 特殊值位置契约：若输入含 NaN/Inf，先严格比较 `isnan` / 正 Inf / 负 Inf mask，再只对双方有限值位置套用上表容差；禁止用全 NaN/全 Inf 作为唯一特殊值用例。
- 该标准可达性证据：`own_ops/rmsnorm.py:302-306` 在 M=N=1024、fp32、`vsqrt+vrec` 下实测 `max_abs < 1e-4`；fp16 + `vsqrt+vrec` 实测 `max_abs 9.8e-4 ~ 3.9e-3`（`own_ops/rmsnorm.py:344-354`），均满足本表。

### 9.4 分层测试计划（L0/L1/L2/Boundary）

> Stage 1 只强制产出 L0；以下 L1/L2/Boundary 为设计级计划，Stage 2 据此落地并在 L0 通过后跑覆盖门禁。覆盖维度 ID 对应 `coverage-matrix.md`。

**L1（功能，阻塞，规则 + 不规则 shape）**

| 用例 | Shape | dtype | eps | 值域 | tags | 说明 |
|---|---|---|---|---|---|---|
| `l1_fp32_base` | (1024,1024)/(1024,) | float32 | 1e-5 | [-1,1] | D-DTYPE-fp32, D-SHAPE-ALIGNED, D-VALRANGE-S | 与 L0 同形的 L1 回归 |
| `l1_fp16_base` | (1024,1024)/(1024,) | float16 | 1e-5 | [-1,1] | D-DTYPE-fp16, D-SHAPE-ALIGNED, D-VALRANGE-S | **需求指定 fp16 路径** |
| `l1_tail_1023` | (1023,1024)/(1024,) | float32 | 1e-5 | [-1,1] | D-SHAPE-TAIL-MID, D-SHAPE-TAIL-1023 | **需求指定边界**；末块 15 行 |
| `l1_tail_1025` | (1025,1024)/(1024,) | float32 | 1e-5 | [-1,1] | D-SHAPE-TAIL-1 | 余数 1（最易暴露 off-by-one） |
| `l1_tail_1032` | (1032,1024)/(1024,) | float32 | 1e-5 | [-1,1] | D-SHAPE-TAIL-MID | 余数 8（block 中点） |
| `l1_prime_1009` | (1009,1024)/(1024,) | float32 | 1e-5 | [-1,1] | D-SHAPE-PRIME | 质数 M，完全非对齐 |
| `l1_edge_1xN` | (1,1024)/(1024,) | float32 | 1e-5 | [-1,1] | D-SHAPE-EDGE | 单行 |
| `l1_edge_Nx1` | (1024,1)/(1,) | float32 | 1e-5 | [-1,1] | D-SHAPE-EDGE | 单列：`y=sign_with_w`，验证 N=1 归约 |
| `l1_valrange_M` | (1024,1024) | float32 | 1e-5 | [-10,10] | D-VALRANGE-M | 中值域 |
| `l1_valrange_L` | (1024,1024) | float32 | 1e-5 | [-100,100] | D-VALRANGE-L | 大值域（平方 1e4，行和 1e7，fp32 安全） |
| `l1_valrange_ASYM` | (1024,1024) | float32 | 1e-5 | [-5,10] | D-VALRANGE-ASYM | 非对称 |
| `l1_eps_1e-3` | (1024,1024) | float32 | 1e-3 | [-1,1] | D-PARAM-eps | eps 非默认值，影响计算路径 |
| `l1_eps_1e-8` | (1024,1024) | float16 | 1e-8 | [-1,1] | D-PARAM-eps, D-DTYPE-fp16 | 极小 eps（fp32 累加下仍稳定） |
| `l1_n_1000` | (1024,1000)/(1000,) | float32 | 1e-5 | [-1,1] | —（额外） | N 非 2 的幂/非 32B 对齐，验证 `DataCopyPad` 与全 N 向量运算 |

覆盖豁免/说明：
- `D-SHAPE-RANK-*`：**N/A**，proto 仅声明 rank-2 算子（理由写入 `COVERAGE_NA`）。
- L1 中规则 shape 与尾块 shape 均已覆盖；GEMM 的 K 轴非对齐不适用（无 GEMM）。

**L2（异常，非阻塞）**

| 用例 | 非法输入 | 期望 |
|---|---|---|
| `l2_dtype_bf16` | `X/W/Y` 用 bfloat16 | 在 kernel 构建入口 `raise ValueError` |
| `l2_rank3` | `X (2,1024,1024)` | 拒绝（仅 rank-2） |
| `l2_w_len` | `W (512,)` 配 `X (1024,1024)` | 拒绝（`W.shape[0] != N`） |
| `l2_n_too_large` | `N=16384 > 8192` | 拒绝（超出声明域） |
| `l2_m_zero` | `X (0,1024)` | 拒绝（空 tensor 无定义） |
| `l2_mixed_dtype` | `X float32, W float16` | 拒绝（要求同 dtype） |

L2 判定：抛异常 = `[BOUNDARY_PASS]`；静默接受 = `[BOUNDARY_WARN]`，不阻塞退出码。

**Boundary（合法特殊值，精度比对，非阻塞）**

| 用例 | 输入 | dtype | 说明 |
|---|---|---|---|
| `bnd_inf` | 有限值 + 稀疏 `+inf/-inf`（至少一个有限值） | float32 | 先 mask 结构比对，再有限值容差 |
| `bnd_nan` | 有限值 + 稀疏 `nan` | float32 | 同上 |
| `bnd_zero` | 全 0 | float32 | golden `rstd = 1/sqrt(eps)`，`y = 0` |
| `bnd_db_fp16` | 含 `±65504` 的稀疏极值 + 有限值 | float16 | dtype 上界；golden 用 float64 |
| `bnd_db_fp32` | `±1e17` 稀疏极值 + 有限值 | float32 | 保持 `x^2` 行和 < fp32 上限（1e34×1024≈1e37 < 3.4e38） |

全 NaN / 全 Inf 不作为唯一特殊值用例。空 tensor（M=0）归入 L2 拒绝，不做 Boundary 精度比对。

### 9.5 性能可行性哨兵（强制执行）

| 用例名 | Shape | dtype/属性 | 覆盖路径 | 单 case timeout | 选择理由 |
|---|---|---|---|---|---|
| `perf_main_fp32` | (1024,1024)/(1024,) | float32, eps=1e-5 | generic_fallback | 120 s（含 JIT 编译） | 需求最大/主 case；GM 流量 8.25 MiB |
| `perf_main_fp16` | (1024,1024)/(1024,) | float16, eps=1e-5 | generic_fallback（fp16 分支） | 120 s（含 JIT 编译） | 最坏第二 dtype；UB 占用最高的路径（166.1 KiB） |
| `perf_tail_fp32` | (1023,1024)/(1024,) | float32, eps=1e-5 | generic_fallback + 尾块 | 120 s（含 JIT 编译） | 尾块 path；验证 64 block 中 1 个 15 行任务不异常退化 |

**性能可行性估算（标注为估算，Stage 2 用实测替换）**：
- GM 流量：`X 4 MiB + Y 4 MiB + W 64×4 KiB = 8.25 MiB`。若有效 HBM 带宽 1.0–1.2 TB/s，理论 7–9 µs；考虑 DMA 启动、UB 往返与 1.6 波次，稳态 kernel 预计 **15–40 µs**。
- Host/框架开销：本仓库既有实测（记忆条目 `npu-benchmark-host-overhead`）显示 host 侧每次调用开销约 **250–450 µs**，远大于 8.25 MiB 的 kernel 时间；因此 e2e 计时主要反映 host 开销。性能结论必须以 **CANN `op_summary` 的 kernel 时间**为准，benchmark 采用 **10 轮 ABBA、≥256 MiB 规模**降低噪声；本算子 4 MiB 规模仅用于功能回归哨兵。
- 权重重复搬运：64 次 × 4 KiB = 256 KiB，约占 GM 流量 3%，可接受；若后续 N 增大导致 block 数变多，可评估把 W 留在 L1/UB 的持久化方案（当前无必要）。
- 无 `T.if_then_else`，所有 `T.Parallel` 均可向量化；无逐元素 GM 标量访问。
- 测试数据、随机数、特殊值与 golden 全部在 CPU 生成；运行阶段仅 H2D → kernel → D2H，不引入 aclnn。

---

## 10. 风险点与注意事项

### 10.1 已知约束

1. **N 必须整体驻留 UB**：这是单 pass 方案的前提。声明域 `N ≤ 8192`，超出后需要 split-N 两遍 kernel（第一遍只累加平方和，第二遍重算并输出），本设计不支持。
2. **`block_m` 受 UB 与"整除 1024"双重约束**：N=1024 时只能是 16；N 增大时 `pick_block_m` 自动缩小到 1，此时 grid = M，block 数暴增，性能退化，但功能仍正确。
3. **fp32 绝对禁止 `T.vrsqrt`**：硬件近似 ~3e-3 相对误差，必然超过 rtol 1e-4；必须 `vsqrt` + `vrec`（或 vrsqrt + 一次 Newton 迭代，`own_ops/rmsnorm.py:152-159` 有实现，但本设计优选 sqrt+rec，无额外误差项）。
4. **`T.Parallel` 循环体只能读写 UB Buffer**：因此 W 必须先 `T.copy` 到 `w_sh`，不能在循环里直接读 GM 的 `W[j]`。这是与 `examples/norm/rms_norm.py` 的一处刻意偏离。
5. **NPU Kernel 只能 1 个 grid 维**：`T.Kernel(..., is_npu=True)` 传 2 维会触发 `AssertionError`（`docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.3）。
6. **尾块未初始化 UB 行**：按行归约语义不会污染有效行；若 Stage 2 实测异常，启用 `T.clear(x_sh)` 退路（§5.4）。

### 10.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|---|---|---|---|
| UB 溢出 | N 增大而 `block_m` 仍固定 16；或擅自加大 `block_m` | 编译失败/运行错误 | 用 §5.2 `pick_block_m`；N=1024 固定 16 |
| fp32 精度不达标 | 使用 `T.vrsqrt` 作为 fp32 rstd | `[PRECISION_FAIL]`（rtol 1e-4） | `T.vsqrt` + `T.vrec` |
| 读 GM 标量 | `T.Parallel` 内直接 `W[j]` | 解析/编译错误或非预期 lowering | 先搬 `w_sh` |
| 下标变换 | `T.Parallel` 内写 `A[i+1]` | 编译拒绝/向量化失败 | 只用原样 `[i,j]` |
| 非整除越界写 | 用标量起点 + 完整 tile 写 `Y` | 覆写相邻行/越界 | 双侧 BufferRegion valid extent |
| host padding | 为对齐 N 在 host 建新 buffer | 违反约束，Stage 2 `[DESIGN_ERROR]` | 禁止；直接按真实 N 建 kernel |
| 2D grid | 想按 (M-block, N-block) 二维并行 | `AssertionError` | 1D grid，N 不切 |
| 手工 barrier | 在 Developer 模式加 `T.barrier_all` | 与自动同步重复/错误 | 删除 |

### 10.3 特殊场景处理

- **M=1023 尾块**：§5.4。
- **fp16 累加**：平方与归约一律在 fp32 buffer 中进行（`x_f32`/`xsq`），只有存储是 fp16；这是 fp16 达到 1e-2 的关键。
- **全零行/全零 X**：`mean=0 → rstd=1/sqrt(eps)≈316.2`，结果有限，无 NaN。
- **inf 行**：`mean=inf → rstd=0`，`y = inf*0 = nan`，与 PyTorch golden 一致，按 mask 结构比对。
- **eps 极小（1e-8）**：全零行 `rstd≈1e4`，仍在 fp32 动态范围内。
- **rank/形状异常**：在 kernel 构建入口做纯 Python 校验并抛错（L2）。

### 10.4 文档/仓库差异（doc gaps，均已在设计中规避）

1. `tilelang-op-design` 引用的 `examples/softmax/`、`examples/normalization/`、`examples/developer_mode/`、`examples/chunk_gated_delta_rule/`、`examples/grouped_gemm/` 在本仓库**不存在**（已逐个 `ls` 确认）。本设计改用 `examples/norm/`、`examples/flash_attn_npuir_dev.py`、`examples/vectorization_in_parallel.py`、`testing/npuir/norm_ops/`。
2. design-template §8.5 列出的 `TL_ASCEND_AUTO_CV_COMBINE` / `TL_ASCEND_AUTO_CV_SYNC` / `TL_ASCEND_AUTO_SYNC` / `TL_ASCEND_MEMORY_PLANNING` 在本 revision 的 `tilelang/transform/pass_config.py` 中**不存在**；真实 key 是 `NPUIR_ENABLE_AUTO_MULTI_BUFFER`、`NPUIR_DISABLE_HIVM_AUTO_INJECT_SYNC`、`TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`、`TL_DISABLE_DYNAMIC_TAIL_SPLIT`。本设计不依赖这些 key（用默认值）。
3. `docs/Tilelang.language/编译器提示操作/T.Kernel.md` §2.2 写"blocks 1～3 维"，但同文 §2.3 与源码要求 NPU 模式恰好 1 维；以 §2.3/源码为准。
4. `T.vrec` 在 `docs/Tilelang.language/数学操作/` 下**没有独立文档页**；其语义由源码 `customize_npuir.py:386`（`npuir_rec` = `1/A`）确认，导出名见 `__init__.py:150`。`T.rsqrt.md` 只有 `vrsqrt`。
5. `examples/norm/rms_norm.py` 的 host 侧把 N 对齐到 256 并做 padding，且用 `TILELANG_ASCEND_MODE="Dev"`；本设计按约束禁止 host padding，模式用 `"Developer"`（lowerer 只把 `expert/exp/e` 判为 Expert，其余均走 Dev 后端，`tilelang/engine/lower.py:169-179`）。
6. 本 revision **没有 `set_env.sh`**；运行要求 `PYTHONPATH=/mnt/workspace/learn/tilelang-ascend`，且依赖从 user site 加载（`python3 -I` 会因缺 `tqdm` 失败）。
7. skill 模板使用小写 `design.md`，而本工作流与 orchestrator 契约要求 `DESIGN.md`；已按后者产出。
8. `T.reduce_sum` 的导出被 `customize_npuir` 覆盖（`__init__.py` 导入顺序），文档 `T.reduce_sum.md` 的签名不含 `size` 参数、源码含；本设计只用 `buffer/out/dim/clear`，两者兼容。

---

## 11. 交付清单

### 11.1 目录结构

```
custom/rmsnorm/
├── DESIGN.md          # 本文档（Stage 1）
├── proto.yaml         # 算子接口规格（Stage 1）
├── rmsnorm.py         # 纯 kernel（Stage 2）
├── test_rmsnorm.py    # golden + L0/L1/L2/Boundary + main（Stage 2）
└── history_version/   # 回退备份（orchestrator 管理）
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `DESIGN.md` | ✅ 已完成 | 设计文档 + L0 计划 + L1/L2/Boundary 设计级计划 |
| `proto.yaml` | ✅ 已完成 | dtype 全集 `[float32, float16]`；attr `eps`；供 `coverage_check.py --proto` 派生 D-DTYPE-*/D-PARAM-* |
| `rmsnorm.py` | ⬜ 待实现（Stage 2） | 纯 `@tilelang.jit(target="npuir")`；Developer；按 §3.3 |
| `test_rmsnorm.py` | ⬜ 待实现（Stage 2） | `from rmsnorm import rmsnorm`；golden §9.1；L0 §9.2；L1/L2/Boundary §9.4 |

### 11.3 命名规范

- 目录：`custom/rmsnorm/`
- kernel：`rmsnorm.py`，对外入口建议 `rmsnorm(M, N, dtype="float32", eps=1e-5, block_m=None)`
- 测试：`test_rmsnorm.py`，`sys.path.insert(...)` + `from rmsnorm import rmsnorm`，`--level {l0,l1,l2,boundary,all}`

### 11.4 实现顺序

1. ✅ 设计文档 + `proto.yaml` + L0 计划（本 Stage）
2. ⬜ kernel `rmsnorm.py`
3. ⬜ `test_rmsnorm.py`：golden + L0 + main
4. ⬜ L0 精度收敛（1024×1024 fp32，§9.3）
5. ⬜ 扩展 L1/L2/Boundary（`tilelang-op-test-design` 场景 B），跑 `coverage_check.py --proto proto.yaml`
6. ⬜ 全量套件（L0/L1 阻塞；L2/Boundary 记录不阻塞）+ 性能哨兵（§9.5）

### 11.5 proto.yaml（Stage 1 产出）

见同目录 `proto.yaml`。dtype 全集与 §9.3 精度表一致（float32、float16）；关键属性为 `eps`（Float，默认 1e-5）。
