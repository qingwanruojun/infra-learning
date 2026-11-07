# Day 8｜让 Agent 生成 RMSNorm，再把验收做扎实

1. **Agent 生成的 RMSNorm 已在 910B3 上跑通。**我按 CANNBot 的 Ascend910 工作流材料组织设计和开发，最后用独立的 CPU float64 参考值检查了 fp32 常规形状、fp32 尾块和 fp16 三组输入，全部逐元素通过。新的验收脚本还会检查负向对照：三组数值全对、三次故意注入的错误都被检出，程序才以 0 退出。
2. **把两次 UB 故障和测试门禁补齐了证据。**初版 fp32 编译后上卡报 MTE 地址越界；fp16 的 16 行 tile 在编译期报 UB overflow。Agent 先去掉常驻的 fp32 副本，再把 fp16 的 `block_m` 降为 8。这次新包补回了两段故障输出和探针脚本。人工又把原先只打印结果的 L2/Boundary 测试改成阻塞门禁，并实测强制失败时会退出 1。

实验在 11 月 4 日的 Ascend 910B3、CANN 9.0.0、TileLang NPUIR Developer 环境完成。我依据新交付包的代码、日志和结果整理，**没有在本机重新上卡**。下文的 `code/` 路径均相对 Day 8 根目录。

## 先让 Agent 做同一道题

任务是带权重 RMSNorm：`Y = X × rsqrt(mean(X², dim=-1) + 1e-5) × W`。我选它，是因为 Day 3 已有手写版，能对照同一算子的实现与验证方式。Agent 先用 `[1024,1024]` 的 fp32 输入完成设计和 L0，再覆盖 `[1023,1024]` 尾块与 fp16。

机器上的 CANNBot 没有注册成自动运行的插件。这次由主代理阅读 Ascend910 workflow，手工调度 analyst 和 developer 两个角色：前者写 `DESIGN.md`、`proto.yaml`，后者生成 kernel、测试并修复编译问题。`rmsnorm.py` 的修复由 Agent 完成；独立验收器、环境修复，以及本轮测试门禁的修改由人工完成。状态文件记录 Stage 1、2 完成，Stage 3 性能调优没有执行。过程见 `code/process/env-and-orchestration.md`、`stage2-agent-timeline.md`。

一开始还碰到工具链问题：CANNBot 的准备脚本把当前 build 换成非 NPUIR 配置，导致 `libtilelangir.so` 找不到。重建后，Python JIT 和仓库的 VecAdd smoke 可用；完整 `make` 仍因 `tilelangir-opt` 的 pthread 链接错误以 2 退出。因此我只把本次需要的 JIT 路径记为恢复可用，详见 `code/logs/00-rebuild-npuir.log`。

## 两次 UB 问题是怎么暴露的

第一次在 fp32。设计伪代码把输入、权重及其 fp32 副本都留在片上；按静态大小相加，`block_m=16` 需 204,928 B，超过 192 KiB。编译虽然通过，首次执行却报 `The DDR address of the MTE instruction is out of range`，进程退出 134。新版的 `code/process/probes/01-fp32-mte-incident.log` 从子代理 transcript 留下了原命令和输出尾段；初版与修复版也在 `history_version/`。删掉两块常驻副本、改在使用处转换后，fp32 L0 上卡和数值检查通过。超额预算与修复前后结果支持这个定位，但单靠 MTE 错误本身不能唯一确定根因。

第二次在 fp16。`block_m=16` 编译时明确报 `ub overflow`：需要 264,192 B，可用 196,608 B。逻辑 buffer 的大小漏算了类型转换等产生的临时占用。补回的编译探针记录显示，默认配置下 16、12 行失败，10、8 行可编译；关闭自动多缓冲后，16、12 行仍失败（`code/process/probes/02-fp16-blockm-probe.log`）。Agent 据此把 fp16 tile 上限设成 8 行；最终 fp16 全量测试通过。探针是 **compile-only**，通过编译还要靠后面的上卡检查确认数值。

## 跑通以后，验收能不能拦住错误

Agent 的原版测试分 L0、L1、L2、Boundary 四层，但后两层即使打印 WARN，也不影响退出码。新版由人工修改 `test_rmsnorm.py` 的结果汇总，让四层都参与最终判定，没有改 kernel、容差或测试数据。`code/logs/09-test-suite-gate-selftest.log` 记录了强制 Boundary 失败和强制 L2 失败均退出 1，恢复后退出 0。正式全量日志 `code/agent_output/custom/rmsnorm/logs/all_blocking_gate.log` 为 16 个精度用例、11 个边界用例通过，退出 0；coverage 为 18 PASS、0 MISS。修改前后文件及 diff 保存在 `history_version/`，并有 `HUMAN_CHANGES.md` 标明人工改动。

独立验收没有调用 Agent 测试里的 golden，而是用 CPU float64 重算参考值，按 `|输出−参考| ≤ atol + rtol × |参考|` 检查每个元素：

| 输入 | atol / rtol | 最大绝对误差 | 超差元素 |
|---|---:|---:|---:|
| fp32，1024×1024 | 1e-4 / 1e-4 | 1.353e-6 | 0 / 1,048,576 |
| fp32，1023×1024 | 1e-4 / 1e-4 | 1.149e-6 | 0 / 1,047,552 |
| fp16，1024×1024 | 1e-2 / 1e-2 | 3.901e-3 | 0 / 1,048,576 |

三组负向对照各改动一个输出元素，独立检查均返回 `pass=False、n_mismatch=1`。新版 `agent_acceptance.py` 把“三组数值都 PASS、三组负例都被抓到”设为退出 0 的条件；上卡重跑日志写明 `ACCEPTANCE_RESULT: PASS` 和 `EXIT_CODE=0`（`code/logs/07-independent-acceptance-blocking.log`）。`gate_selftest.py` 还单独验证了数值失败或漏检负例都会返回 1（`code/logs/08-acceptance-gate-selftest.log`）。这比只看一句 `Test Passed!` 更能说明验收确实在工作。

## 和 Day 3 手写版对照后，学到了什么

两版在相同的 shape、dtype 和计算公式下都满足精度要求。Agent 版把 kernel 与分层测试分开，过程和边界覆盖更容易查；Day 3 手写版保留了不同 rstd 实现、权重开关等实验选项，适合继续研究数值与调优。两轮输入种子和参考精度不完全相同，最大误差不能拿来排精度名次。数据见 `code/reference/day3_baseline.json` 和 `code/process/code_metrics.json`。

本轮 profiler 里，Agent 版 device 中位数约 12–17 µs，与 Day 3 手写版处在同一量级；重复测量约有 2 µs 波动，日志还提示 profiler 停止时仍处于 RECORD 状态，因此暂不声称哪版更快。Agent 版 host 入队和 e2e 更高，`out_idx=[2]` 每次分配输出可能是原因之一，但没有做复用输出的 A/B 实验，不能把全部差额归给它。三时钟结果在 `code/reference/agent_acceptance.json`、`day3_three_clocks.json`。
