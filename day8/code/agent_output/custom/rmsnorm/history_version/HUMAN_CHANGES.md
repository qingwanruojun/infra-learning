# 人工修改记录：L2/Boundary 改为阻塞门禁

时间：2026-11-04 10:0x（UTC+8）
修改人：主代理（orchestrator），非 Stage 2 Agent。

## 动机

原 `test_rmsnorm.py` 只有 L0/L1 写回 `blocking_ok`；L2 与 Boundary 只打印
`[BOUNDARY_PASS]` / `[BOUNDARY_WARN]`，不影响退出码。这样即使非法输入被
静默接受、或特殊值精度超差，程序仍然退出 0，无法作为验收门禁。

## 改动

- `_run_exception` 返回 bool：正确拒绝 -> True，静默接受 -> False。
- `test_rmsnorm_l2` 聚合所有 L2 结果并返回。
- `_run_boundary` 返回 `m["passed"]`；异常分支（原本只 WARN）返回 False。
- `test_rmsnorm_boundary` 聚合所有 Boundary 结果并返回。
- `main` 中四层全部 `blocking_ok &= ...`；只有全部通过才打印
  `Test Passed!` 并退出 0。
- 注释从 “non-blocking / never block” 改为 “blocking”。

## 文件

- 改前：`test_rmsnorm_pre_blocking_gate.py`
- 改后：`test_rmsnorm_post_blocking_gate.py`
- diff：`human_blocking_gate.diff`

## 未改动

- `rmsnorm.py`（kernel）一行未动。
- 容差、测试数据、覆盖维度未改。
