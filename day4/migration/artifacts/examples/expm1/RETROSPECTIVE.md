# RETROSPECTIVE.md — expm1 (op 级)

> Stage 5 集成复盘（追加式；schema: `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`）。
> 函数级 Stage 1 复盘见 `examples/expm1/expm1_fwd_kernel/RETROSPECTIVE.md`；本文件为 op 级单份（集成问题不归属单个函数）。

## Stage 5 (integrate) Retrospective — harness — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | integrate_kernel.py 的 smoke test 仅做 import 冒烟（hasattr 检查），不覆盖 wrapper 调用面；两段式 kernel 配旧单段 wrapper 调用 `(x, N)` 时 import 不会触发 `_expm1_fwd_wrapped` 函数体，静默通过 smoke，错误延迟到 report correctness 才暴露（且那时才消耗调试 attempt） | examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#smoke_test（仅 `hasattr(m, n)` + class 检查）；examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py#L95（集成前残留旧调用面 `(x, N)`，本次由 conductor Stage 3 冻结结论预先指出，未实际踩坑） | integrate_kernel.py 在 smoke 中对每个函数追加一个最小 wrapper forward 调用（或 [warn] 提示核对 wrapper 调用面与 kernel 工厂返回形状一致性），把调用面错误从 report 阶段前移到集成阶段 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| elementwise 家族 wrapper 两段式调用面适配模式第三次验证（factory(N, dtype_str, block_size=bs)(bs)(x)，删尾参 `, N`） | P | examples/TileOPs/tileops/kernels/elementwise/expm1/expm1.py#L88-L104（本次适配 + 注释）；同形先例 leaky_relu.py#L98 / elu_fwd.py#L99；集成 + 全量 report 一次通过（expm1_kernel/integration_log.md） | `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/expm1/.migration_meta.json` 后按上式适配 wrapper L95，跑 `python -m tileops.reporting.cli run --op Expm1FwdOp --prof-mode msprof` | tilelang 0.1.2+ef4078a697b36c575c8c258179be3fac293325e9（report 环境显示 0.1.1.10+cann900 同 commit），Ascend910B3，CANN 9.0.0 | pattern-library |
| expm1 NPU 集成基准数值：7 个 msprof profile case 全有效，1.75 TB/s 模型带宽下 fp16/bf16 [16384,16384] 达 roofline 27.08%/26.60%，fp32 [4096,4096] 219.70us 达 17.45% | D | examples/TileOPs/reports/tileops/20261029_012203_068316_Expm1FwdOp/run.json#performance.cases（block_size fp32=1024/fp16.bf16=2048，全部 profiler_valid=true） | `cd examples/TileOPs && python -m tileops.reporting.cli run --op Expm1FwdOp --prof-mode msprof` | tilelang 0.1.1.10+ubuntu.20.4.cann900，Ascend910B3，CANN 9.0.0，torch 2.7.1+cpu / torch_npu 2.7.1.post4 | pattern-library / bottleneck-patterns |

### Transferable Lessons（可迁移教训）

- elementwise 单输入单输出 kernel（out_idx=[1]、main(x) 契约）集成时 wrapper 必须两段式：`factory(N, dtype_str, block_size=bs)(bs)(x)`；单段 `(x)` 会把张量绑到 bs 参数在 build 期 TypeError，尾参 `, N` 无法绑定 jit callable 契约。
- 集成脚本 smoke 只验证 import 不验证调用面：集成完成后、跑全量 report 前，先用 3 dtype × N=4096 的最小 forward 冒烟（构造 Op class 调 forward 对拍 torch golden），可把调用面/dtype 归一化问题提前定位，避免消耗 report 级调试 attempt。
- kernel 工厂内做 dtype 归一化（`str(dtype).split(".")[-1]`，torch 2.7.1 无 `torch.dtype.name`）时，wrapper 对 `dtype_str` 直传 str 与传 torch.dtype 均合法，无需胶水转换。
