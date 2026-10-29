# Integration Log -- BitwiseNotFwdOp

- meta: examples/TileOPs/tileops/kernels/elementwise/bitwise_not/.migration_meta.json
- integrated: bitwise_not_fwd_kernel -> tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/bitwise_not_fwd_kernel.py
- design_docs: bitwise_not_fwd_kernel -> bitwise_not_fwd_kernel_DESIGN.md（无缺失）
- attempts: 0（调试闭环未触发；集成与 report 一次通过）

## 集成步骤

1. 前置核对（leaky_relu 教训 #1）：
   - wrapper `from ._bitwise_not_fwd_kernels import <names>` 名单 = `["bitwise_not_fwd_kernel"]`，与 meta `extracted_functions` 一致；conductor 产物 `__all__ = ["bitwise_not_fwd_kernel", "golden_bitwise_not"]` 定义了该名字。GPU 侧 manually-appended helper `bitwise_not_op_func` 未被 wrapper import（死代码，不进集成名单），无 `[warn] not found` 风险。
   - 产物模式声明：`ASCEND_MODE = "Developer"` 字面量 + `os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")`；wrapper 类级 `ascend_mode = "Developer"` 一致。
   - 工厂签名匹配：产物 `bitwise_not_fwd_kernel(N, dtype)` 返回 `kernel(block_size)` 返回 `main(x)`（bool/uint8 路线带 int8 bit-view 零拷回 view 对）；wrapper 两段式调用 `bitwise_not_fwd_kernel(N, dtype)(block_size)(x)`（K3/K11 契约）匹配。
2. 运行 `python .agents/skills/add-npu-op/scripts/integrate_kernel.py --meta tileops/kernels/elementwise/bitwise_not/.migration_meta.json`（cwd=examples/TileOPs）：
   - kernel + DESIGN.md 拷贝至 `bitwise_not_kernel/`；聚合 `__init__.py` 生成（re-export `bitwise_not_fwd_kernel`，无 warn）；wrapper 改写为 baseline/perf_opt 切换块（baseline 激活 `from .bitwise_not_kernel import bitwise_not_fwd_kernel`）；ascend_mode Developer（unchanged）；smoke pass。
3. 胶水注释清理（非调试修复，备份于 history_version/bitwise_not.py_s5_attempt1.py）：scaffold 预埋的过时注释块（声称 active import 仍指向 `._bitwise_not_fwd_kernels` 的 "Stage 5 integration note" + 重复 selection block）与改写后实际状态矛盾，按 leaky_relu 已集成形态（leaky_relu.py#46-65）改写为与实际一致的说明注释（仅注释，无行为变更）。
4. 幂等复跑 integrate_kernel.py：wrapper 判定 "already integrated"、ascend_mode 重新核对（Developer unchanged）、聚合 `__init__.py` 重新生成一致、smoke pass —— 幂等性确认（leaky_relu 教训 #2）。
5. 激活源验证：`bitwise_not_fwd_kernel.__wrapped__.__module__` 解析到 `tileops.kernels.elementwise.bitwise_not.bitwise_not_kernel.bitwise_not_fwd_kernel`（conductor 产物，baseline 激活）。注：`inspect.getsourcefile` 对 `functools.lru_cache` 包装的工厂抛 TypeError，须用 `__wrapped__` 检查。

## 验证结果

- report_run_json: examples/TileOPs/reports/tileops/20261029_055054_204932_BitwiseNotFwdOp/run.json
- report_md: examples/TileOPs/reports/tileops/20261029_055054_204932_BitwiseNotFwdOp/report.md
- report_html: examples/TileOPs/reports/tileops/20261029_055054_204932_BitwiseNotFwdOp/report.html
- correctness: pass，9/9（6 个 dtype 用例 N=1048576：bool/uint8/int8/int16/int32/int64，max_abs_err 全部 0.00e+00，位级精确匹配；3 个 float dtype 拒绝用例通过）—— 来自 run.json summary（correctness_passed=true, pass_rate=1.0）
- benchmark: pass，8/8 bench 测试通过；5 个 msprof 性能用例全部 profiler_valid=true，invalid_profile_count=0，profiler_fallback_count=0，warnings=[]，CLI exit code 0/0。run.json status="passed"（非 partial）。
- benchmark 数值（自 run.json 引用，msprof 设备时长，tileops kernel config block_size=256）：
  - smoke-1m-bool: 29.2406 us；smoke-1m-uint8: 26.3605 us；smoke-1m-int32: 31.0806 us（bench XML 记录）
  - elementwise-16m int32 (4096x4096): 455.509125 us，roofline util 8.4187%
  - elementwise-16m int64 (4096x4096): 501.970032 us，roofline util 19.0980%
  - elementwise-64m int16 (8192x8192): 1699.754028 us，roofline util 5.5697%
  - elementwise-64m int32 (8192x8192): 1801.015991 us，roofline util 13.7977%
  - elementwise-256m int32 (16384x16384): 7220.364746 us，roofline util 16.1979%
  - msprof 采集 op 名 "main"，sample_count=1（各用例）；环境：Ascend910B3 x 1，CANN 9.0.0，tilelang 0.1.1.10+ubuntu.20.4.cann900，git 2a94a298

## 调试历史

无（attempts=0，调试闭环未触发；无精度/编译/shape/wrapper 胶水失败）。
