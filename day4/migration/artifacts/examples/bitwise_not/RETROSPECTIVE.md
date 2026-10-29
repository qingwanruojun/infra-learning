# RETROSPECTIVE.md — bitwise_not（op 级）

> migration-harness 场景，op 级集成复盘（Stage 5 integrator 追加）。函数级 Stage 1/3 复盘见 `bitwise_not_fwd_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt 1 — 2026-10-29

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | integrate_kernel.py 改写 wrapper import 后，scaffold 预埋的过时注释块（声称 active import 仍指向 `._bitwise_not_fwd_kernels` 的 "Stage 5 integration note" + 重复 selection block）原样残留，与实际激活路径矛盾，需 integrator 手工注释级清理 | TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/history_version/bitwise_not.py_s5_attempt1.py（改前快照 85–105 行）对照清理后 bitwise_not.py#80-91；已集成参照 leaky_relu.py#46-65 | add-npu-op SKILL.md S3 Part B：scaffold 不预埋 selection block 注释（留给 integrate_kernel.py 生成）；或 integrate_kernel.py rewrite 时同步替换预埋块 | R |
| Integrate-flow | integrate_kernel.py 打印 `[smoke]` 摘要取 `out.splitlines()[-1]`，当 stderr 警告（如 torch_npu Permission mismatch）晚于 stdout 的 `smoke-ok` 输出时，摘要行显示的是警告文本而非 smoke 结论（本例 smoke 实际 pass，exit code 0） | 本任务集成运行输出：`[smoke]   warnings.warn(f"Permission mismatch: The owner of ... does not match.")`（integrate_kernel.py:511 `out.splitlines()[-1]`） | integrate_kernel.py：[smoke] 行改为过滤 `smoke-ok` 前缀行（未命中再回退末行） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| bitwise_not 集成 baseline（persistent split + vnot/vxor 路线，block_size=256）msprof 实测数据 | D | examples/TileOPs/reports/tileops/20261029_055054_204932_BitwiseNotFwdOp/run.json#performance | `cd examples/TileOPs && python -m tileops.reporting.cli run --op BitwiseNotFwdOp --prof-mode msprof` | tilelang 0.1.1.10+ubuntu.20.4.cann900（git 2a94a298），Ascend910B3 x 1，CANN 9.0.0 | pattern-library |
| 验证 wrapper 激活 kernel 源解析：lru_cache 包装工厂用 `func.__wrapped__.__module__`（inspect.getsourcefile 对 lru_cache 对象抛 TypeError） | P | TileOPs/tileops/kernels/elementwise/bitwise_not/bitwise_not_kernel/integration_log.md#集成步骤-5；产物 examples/bitwise_not/bitwise_not_fwd_kernel/bitwise_not_fwd_kernel.py#146（@functools.lru_cache(maxsize=32)） | `python -c "import tileops.kernels.elementwise.bitwise_not.bitwise_not as w; print(w.bitwise_not_fwd_kernel.__wrapped__.__module__)"` | tilelang 0.1.1.10+ubuntu.20.4.cann900，Ascend910B3，CANN 9.0.0 | add-npu-op SKILL.md Tier 2 |

### Transferable Lessons（可迁移教训）

- 集成前先核对 wrapper `from ._{op_slug}_fwd_kernels import <names>` 名单只含 conductor 产物实际定义的名字（产物 `__all__` 为准）；GPU 侧 manually-appended helper（如本 op 的 `bitwise_not_op_func`）不被 NPU wrapper import 即无须处理——本 op 据此一次通过，无 `[warn] not found`。
- wrapper 中 scaffold 预埋的 selection block 注释（含注释形式的 perf_opt import 行）在 integrate_kernel.py 改写后成为过时矛盾信息；集成完成后做一次注释级清理（先备份 history_version/），使激活的 baseline 路径唯一可读。
- `integrate_kernel.py` 幂等：改 wrapper 胶水注释后重跑，wrapper 判定 "already integrated"、聚合 `__init__.py` 重新生成、ascend_mode 重新核对，安全。
