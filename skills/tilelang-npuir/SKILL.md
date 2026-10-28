---
name: tilelang-npuir
description: 写/编译/调试 TileLang Ascend NPU kernel（target="npuir"）的固定套路：UB 预算与自动多缓冲、Expert/Developer 两种模式、动态 shape 尾部处理、环境与缓存陷阱、launcher 的 host 开销与 NPUGraph 限制。当要在 910B 上新增或修改 tilelang 算子时使用。
---

# TileLang npuir 算子开发

## 1. 从现成例子起步，不要从零写

- 官方：`examples/elementwise/vec_add_1d.py`（结构最干净）。
- 本项目：`own_ops/abs.py`（对拍 torch、含两种模式）、`own_ops/rmsnorm.py`、`day4/code/own_ops/gemm.py`。
- 约定见仓库 `AGENTS.md`：优先 `T.vabs`/v 前缀向量 API，保留 `npuir_xxx` 兼容名，改动后 `bash format.sh --files <files>`。

## 2. 最小可跑 kernel（固定 grid + 串行 tile）

```python
import tilelang, tilelang.language as T

def fixed_grid_kernel(n, block_N, ncore, dtype="float32"):
    tiles = T.ceildiv(n, block_N)
    per_core = T.ceildiv(tiles, ncore)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), shape: T.int32):
        with T.Kernel(ncore, is_npu=True) as (cid, _):
            a_buf = T.alloc_ub((block_N,), dtype)
            b_buf = T.alloc_ub((block_N,), dtype)
            for i in T.serial(per_core):
                t = cid + i * ncore
                if t * block_N < shape:              # 越过末尾的块直接跳过
                    off = t * block_N
                    tail = T.min(block_N, shape - off)   # 只搬真实存在的元素
                    T.copy(A[off:off + tail], a_buf[0:tail])
                    T.vabs(a_buf, b_buf)
                    T.copy(b_buf[0:tail], B[off:off + tail])
    return main
```

要点：
- grid 用 `T.ceildiv(N, block_N)`，**不是** `N // block_N`（官方例子会静默丢尾）。
- 1D `T.Kernel(ncore, is_npu=True) as (cid, _)` → `cid` 就是 `blockIdx.x`。
- 动态 shape 靠标量参数 + `tail` clamp；编译期把 N 写死可以省掉分支，但图捕获会冻结标量。
- 编译：`tilelang.compile(f, target="npuir")`；模式用 `TILELANG_ASCEND_MODE`（见下），`own_ops.abs.ascend_mode("Expert")` 就是临时改这个环境变量。

## 3. 两种模式

| 模式 | 分配 | 环境变量 |
|---|---|---|
| Expert | `T.alloc_ub` | `TILELANG_ASCEND_MODE=Expert` |
| Developer | `T.alloc_shared` | `TILELANG_ASCEND_MODE=Developer` |

同一个 kernel 两套 buffer 都实现，方便对拍（见 `own_ops/abs.py`）。

## 4. UB 预算：192 KiB/core，`T.serial` 会自动翻倍

`T.serial` 内层循环触发**自动多缓冲**，每个 buffer ×2：

- `block_N=8192` f32：64 KiB × 2 buffer × 2 = 128 KiB，OK。
- `block_N=16384` f32：128 KiB × 2 = **256 KiB > 192 KiB**，编译直接失败：
  `ub overflow, requires 2097152 bits while 1572864 bits available!`

关掉：

```python
pass_configs={tilelang.PassConfigKey.NPUIR_ENABLE_AUTO_MULTI_BUFFER: False}
```

实测开关对带宽型 kernel 影响 <1%（217.7 vs 218.7 µs @128 MiB）——**别指望双缓冲救性能**。
先按 `2 × block_N × itemsize × 2` 算预算，再决定 block_N。

## 5. 环境陷阱（都真实踩过）

- **PATH 顺序**：仓库的 `3rdparty/AscendNPU-IR/build/bin` 必须在 CANN 自带 `bishengir-compile` **之后**。否则报
  `AscendNPU IR compile reported success but output object was not generated`——它的 bishengir 会给 CANN 9.0.0 的 hivmc 传 `--enable-lir-compile` 等不认识的 flag。
- **环境变量烧进缓存 `.so`**：`TILELANG_ENABLE_TASKQUEUE` 等是在生成 launcher 时读的，改完必须换 `TILELANG_CACHE_DIR` 触发重新编译，否则毫无效果。
- **CANN 导出目录**：`export_only_prof_dir/` 会长到 127 MB，跑完删。

## 6. Launcher 事实（`tilelang/jit/jit_npu.py:575-665`）

- 每次调用 **一次** `rtKernelLaunch`；`workspace_size > 0` 才每块 `rtMalloc`；`lock_num > 0` 才 `syncBlockLock`。
- host 开销：216–232 µs @1–256 块，252 @1024，**1085 @4096**（≥1024 块后每块 +0.26 µs）；固定 grid 串行 kernel 250–450 µs（进程相关，原因未查明）。
- 默认把 launch 包进 `at_npu::native::OpCommand::RunOpApi(name, call, true)`（会同步流）——这是 NPUGraph capture 失败的直接原因（line ~623）。
- `TILELANG_ENABLE_TASKQUEUE=0` 后可以捕获，但 **replay 是静默 no-op**（裸 `rtKernelLaunch` 没被 ACL 图记录）；同进程 torch 图 replay 正常。**本 build 上 NPUGraph 不可用**，且 device 时长本身就是 214 µs 量级的下界。

## 7. 正确性

每次 `compile` 后立刻：

```python
kern(a, b, N); torch.npu.synchronize()
assert torch.equal(b, torch.abs(a))
```

- 先验正确性，再测性能。
- 尾部处理是最大错误源：`T.min(block_N, shape - off)`，且 `t * block_N < shape` 的块要跳过。
- 需要的话用 `TILELANG_PROBE_CAPTURE` 之类的开关把危险实验挡住（见 `notes/day2/experiments/probe_npugraph.py`）。

## 8. 已知无收益的旋钮

- block_N ∈ {4096, 8192, 16384}：带宽型 kernel 差 <1%。
- 核数 40 → 20：只慢 5–7%。
- 自动多缓冲：<1%。
- 结论：HBM 带宽型算子，先把尺寸选对（≥256 MiB 对拍），再谈调参。详见 `notes/day2/optimization-knobs.md`。
