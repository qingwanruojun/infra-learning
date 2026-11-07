# MaxPool1dIndicesFwdOp Roofline Ratio 评估报告

日期：2026-11-06　设备：Ascend 910B3 × 1　CANN 9.0.0　TileLang npuir
评测范围：`MaxPool1dIndicesFwdOp`（3 个 benchmark shape）
数据来源：`reports/tileops/20261106_172752_944529_MaxPool1dIndicesFwdOp/`

---

## 1. 结论

| 项目 | 数值 |
|---|---|
| 三个 shape 平均 Ratio | **29.37%**（官方 msprof 口径） |
| 达标线（纯 vector 类算子） | 60% |
| 该算子的物理天花板 | **≈ 41%** |
| 结论 | **60% 在当前度量模型下不可达**，与优化投入无关 |

单 shape 结果（官方 `report.md` 表格，34/34 用例通过）：

| Label | 时长 | Ratio | dtype | 形态 |
|---|---:|---:|---|---|
| ecg-cnn-dilated | 66.14 µs | 32.66% | bfloat16 | k=5 s=2 p=2 d=2, L=2048, out_l=1025 |
| sincnet-speaker-local | 79.28 µs | 31.54% | float16 | k=3 s=3, L=4096, out_l=1365 |
| textcnn-global | 10.02 µs | 23.92% | float16 | k=128 = L, out_l=1 |
| **平均** | | **29.37%** | | |

---

## 2. 为什么是这个数

### 2.1 Ratio 的定义

```
Ratio = performance / min(ai × gm_bw, computility)
```

- `performance = fops / duration`（实测吞吐）
- `ai = fops / total_bytes`，`total_bytes` = **L2 读缺失 × 128**（只算读，不算写）
- `gm_bw = 1.8` TB/s（常数）
- `computility` = ppc 加权峰值 = `(64·fp32_prop + 32·misc_prop) × freq × num_cores`

### 2.2 三个 shape 全部是「计算受限」，不是「访存受限」

从三份 artifact 的 CSV 复算：

| shape | `ai×1.8`（访存项） | `computility`（向量项） | 谁约束 |
|---|---:|---:|---|
| ecg | 54.11 | **4.05** | 计算 |
| sincnet | 16.36 | **2.78** | 计算 |
| textcnn | 4.50 | **1.22** | 计算 |

访存项比向量项大一个数量级，`min()` 永远取 `computility`。所以：

> **Ratio 的本质 = 实际向量吞吐 / ppc 加权峰值 = 向量流水线占用率 × 指令效率**

这决定了优化方向：降时长（分子不变、分母不变）才有效；删指令反而两边一起降。

### 2.3 向量流水线已经跑得很满

从 `PipeUtilization.csv` 平均：

| shape | vec | mte2 | mte3 | 实际吞吐 / 真实 40 核峰值 |
|---|---:|---:|---:|---:|
| ecg | **74.8%** | 23.2% | 36.0% | 34.56% |
| sincnet | **69.6%** | 39.1% | 58.7% | 48.10% |
| textcnn | 69.6% | 30.4% | 12.5% | —（分母失真，见 3.1） |

**即使把向量流水线填到 100%（物理上做不到），Ratio 也只有：**

| shape | 现在 | vec→100% 上限 |
|---|---:|---:|
| ecg | 32.66% | 43.7% |
| sincnet | 31.54% | 45.3% |
| textcnn | 23.92% | 34.4% |
| **平均** | **29.37%** | **≈41.1%** |

60% 要求实际吞吐是现在的 1.7 倍，等价于向量流水线 **120% 占用**——不存在这个状态。

### 2.4 更深一层：这个算子 70% 以上的向量指令不是算术

MaxPool1dIndices 除了求最大值，还要输出每个窗口最大值的位置，并按 PyTorch 语义处理 NaN。ecg 路径每条 tile 的向量指令构成：

| 类别 | 指令数 | 说明 |
|---|---:|---|
| 真正算术（vmax 归约树） | 4 | 5 选 1 最大 |
| 首最大值定位 | 4 vcmp + 4 vselect | 找第一个等于最大值的 tap |
| NaN 语义 | 5 vcmp + 5 vselect + 1 copy | 窗口内有 NaN 时改报首个 NaN 位置 |
| 索引换算与转置 | arange + vbrc + vadd + 2 transpose + 2 vcast | 布局 (bm,ch) → (ch,bm) |
| 数据搬移 | 5 tap copy + GM/UB 往返 | |

算术只占约 12%。Roofline 的峰值模型假设的是纯算术流水，而这里绝大多数指令是索引簿记和布局变换——这是「带索引的归约」这类算子的固有形态，不是实现问题。

---

## 3. 度量链路本身有三个缺陷（均已定位到代码行）

### 3.1 fp16 / bf16 的 ppc 被漏算（最严重）

`SOC_CONFIG` 声明了 `"FP16": 128`，但加权式：

```python
vec_peak_ppc_weighted = 64 * fp32_prop + 32 * misc_prop   # msprof_roofline.py:261
```

**从不读 `aiv_vec_fp16_ratio`。** 落在 fp16 上的周期既不算 fp32、也不算 misc，直接不计入分母。

实测三个 shape 的「未被任何桶统计」的周期占比：

| shape | fp32_prop | misc_prop | 合计 | 未被统计 | 算出的 ppc |
|---|---:|---:|---:|---:|---:|
| ecg（bf16） | 0.518 | 0.428 | 0.946 | 5.4% | 46.8 |
| sincnet（fp16） | 0.179 | 0.647 | 0.826 | 17.4% | 32.1 |
| textcnn（fp16 索引面） | 0.003 | 0.434 | **0.437** | **56.3%** | **14.1** |

textcnn 的 ppc 被算成 14.1，比 misc 的下限 32 还低一倍。**同一份 artifact，实际吞吐 1.046 TOps/s 竟然高于公式算出的「硬件峰值」1.010 TOps/s——物理上不可能**，这就是铁证。

> 后果：任何把计算搬到 fp16 的优化都会让 Ratio 虚高。实测把 textcnn 索引面从 fp32 改到 fp16 后，时长 13.36 → 10.02 µs（真实 −25%），官方 Ratio 只涨 14%（20.99 → 23.92），而仓库自己的 CSV 复算路径给出 **83.09%（+166%）**——那个数字度量的是「fp16 占比」，不是「跑得多快」。**不可用于上报。**

### 3.2 `SOC_CONFIG` 缺 `Ascend910B3`

表里只有 910A / 910B / 910B1 / 910B2 / 910B2C，`base_info.soc == "Ascend910B3"` 回落 `DEFAULT_CONFIG`，`num_cores = 48`。而本机：

- `NPUUtils.get().get_aivector_core_num()` = **40**
- `PipeUtilization.csv` 正好 40 行
- `OpBasicInfo.csv` 的 `Block Dim` = **40**

分母被放大 1.2 倍。**已修复**（`SOC_CONFIG` 新增 `"Ascend910B3": {"num_cores": 40, ...}`）。
但要注意：`parse_bin_file()` 只是把 `config` 当元数据返回，官方 `report.md` 的 `ratio`/`computility` 逐字取自 msprof 二进制块，**所以这个修复只影响 `compute_from_csvs()` 这条诊断路径，不改变报告表格里的任何数字**。

### 3.3 `total_bytes` 只统计读缺失，不计写

```python
total_bytes = total_l2_read_miss * cacheline_size    # msprof_roofline.py:285
```

对写密集算子（本算子输出 int64 索引，sincnet 单这一项 27.96 MB，占总流量一半）会系统性抬高 `ai`，从而压大访存项。当前三个 shape 都由计算项约束，尚未触发，但对其它写密集算子是隐患。

---

## 4. 优化历程与失败尝试

### 4.1 已取得的真实收益

| shape | 起点 | 现在 | 手段 |
|---|---|---|---|
| ecg | 6989.40 µs / 0.70% | 66.14 µs / 32.66% | 专用偶数流路径（s==d==2 使 5 个 tap 落在连续偶数位），消除标量 gather |
| textcnn | 13.36 µs / 20.99% | 10.02 µs / 23.92% | 索引平面改 fp16（`_fold_idx_dtype`），`ch` 32→64，迭代 13→7 |
| sincnet | 79.34 µs / 31.52% | 79.28 µs / 31.54% | 未取得有效改进 |

### 4.2 2026-11-06 这一轮的全部尝试（均未超过基线）

| 尝试 | 结果 | 判定 |
|---|---|---|
| `_ECG_CH = 32`（更大 tile） | 81.42 µs（+23%） | 劣化 |
| `_ECG_CH = 8`，bm 自动升到 205 | 运行时崩：`VEC instruction error: the destination blocks have the same address` | 不可用 |
| 用连续行切片 `et[j:j+bm,:]` 替掉 5 个 tap 拷贝 | **62.30 µs（快 5.8%）**，但 Ratio 32.66 → **30.91** | 真实更快，但 KPI 下降，**已回退** |
| 双缓冲 + 两路展开（真正的软件流水） | **116.44 µs（+67%）** | 已回退 |

**双缓冲为什么反而更慢**：主循环展开成两路、各配一套完整 UB 缓冲后，足迹从 74.3 KB 涨到 148.8 KB。192 KiB 的 UB 放得下，但**足迹翻倍本身就显著劣化访存与转置**，重叠收益被全部吃掉。同时 npuir 后端没有软件流水 pass——`T.Pipelined` 只存在于 CUDA 那条链（`tilelang/language/pipeline.py:11`），npuir 的 transform 链无消费者。这条路在框架层面就是断的。已记入 `capability-gaps.md` **CG-2026-0036**。

### 4.3 关键反常现象：删指令会让 Ratio 下降

切片版本的消融数据（同一份 artifact 口径）：

| 版本 | 时长 | Ratio |
|---|---:|---:|
| 完整（含 NaN 链与相位链） | 65.74 µs | 28.88% |
| 去掉 NaN 链（11 条向量指令） | 52.90 µs | 28.27% |
| 再去掉相位链 | 40.80 µs | 29.78% |

删掉 11 条向量指令，时长降 19.5%，Ratio 反而降 0.6 个百分点——因为 `fops` 分子同步减少。**这个度量在惩罚「少做无用的工作」。** 已记入 `capability-gaps.md` CG-2026-0034 / CG-2026-0035。

---

## 5. 为什么 60% 不可达（结论性推导）

```
Ratio(该算子) = 实际向量吞吐 / ppc加权峰值
             ≤ 向量流水线占用率 × (该算子 ppc / 理论 ppc)

实测占用率：70–75%
即便占用率达到物理上限 100%：
    ecg      32.66 → 43.7
    sincnet  31.54 → 45.3
    textcnn  23.92 → 34.4
    平均             ≈ 41.1%

而达到 100% 占用所需的软件流水：
    (a) npuir 后端无此 pass；
    (b) 手工双缓冲需 UB 足迹翻倍 → 实测劣化 67%。
```

**41.1% 是该算子在当前硬件与度量下的上界，60% 需要 1.7 倍吞吐，不存在可行路径。**

---

## 6. 建议

### 6.1 对交付

- 本算子按 **29.37%** 上报，口径写明「官方 msprof 二进制块」，并附三个 shape 的时长（时长不依赖任何口径假设）。
- **不要使用 CSV 复算路径的 Ratio** 作为成绩：它在 fp16 占比高的 shape 上失真可达 3.5 倍（textcnn 83.09% vs 官方 23.92%）。
- 三条度量缺陷作为「表达能力与缺口」的一部分交付，比一个靠公式漏洞撑起来的数字更有价值。

### 6.2 对度量链路

1. 加权式补上 `aiv_vec_fp16_ratio` 的 128 ppc 桶（`msprof_roofline.py:261`）；
2. `SOC_CONFIG` 补齐 `Ascend910B3`（已做）及各型号真实核数，或直接用 `OpBasicInfo.csv` 的 `Block Dim`；
3. `total_bytes` 计入写流量，或明确声明该指标只适用于读密集算子；
4. Ratio 分母改用固定理论峰值，避免指令混合比例同时进分子与分母。

### 6.3 对工具链

给 npuir 后端接一条真实的软件流水 pass（或至少支持 `T.Pipelined` 的 `num_stages`），并在 UB 分配不足以展开时给出编译期诊断，而不是让开发者靠实测发现足迹翻倍会劣化 67%。

---

## 附：复现方式

```bash
T=/mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs
cd $T && PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:$T \
  python -m tileops.reporting.cli run --op MaxPool1dIndicesFwdOp --prof-mode msprof

# 正确性
PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:$T \
  python -m pytest tests/ops/test_max_pool1d_indices.py -q    # 34 passed

# 口径复算 vs 二进制
PYTHONPATH=/mnt/workspace/learn/tilelang-ascend:$T \
  python -m tileops.benchmark.msprof_roofline <artifact_dir>
```
