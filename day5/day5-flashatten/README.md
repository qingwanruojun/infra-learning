# Day 5 交付包：FlashAttention / MixCV

日期 2026-10-29 · Ascend910B3（20 AIC / 40 AIV）· CANN 9.0.0 ·
tilelang-ascend npuir 分支 · AscendNPU-IR 19.1.7（仓库 3rdparty 构建）

## 包内容

```
day5-flashattn/
  note/
    day5学习指标.md      主笔记：运行结果、mixcv skill 摘要、四项机制、口述版、未验项
    本周周报.md           第一周汇报 v2：链路图 + 性能表 + 3 个卡点 + 下周计划
    worklog.md           时间线、错误与修复、结论翻转序列
    learning-report.md   当天学习报告（结论 + 依据 + 我错在哪 + 还没验的）
  code/
    bench_fa_dev.py      Developer 模式 FA 的 seq_len sweep + 正确性 + host/device/e2e
    profile_fa_dev.py    用 CANN op_summary 取设备时间（event 含 host，不可当设备时间）
    bishengir-compile-wrap  expert IR 的早期兼容 shim（仅用于定位，未作为最终修复）
    tilelang-jit_npu.patch  expert 最终修复：属性改写 + 编译器 SIGSEGV 重试
  logs/
    run_default.log                      expert 例子：CANN 编译器解析失败
    run_dev_example.log                  Developer 例子：跑通
    run_expert_repo_compiler.log         repo 编译器：解析过、hivmc 失败
    run_default_wrapped.log              strip 属性后：pass crash
    run_default_repocc_hivmcwrap.log     repo 编译器 + 过滤 hivmc flag：仍不认属性
    run_default_repocc_hivmcwrap2.log    strip hivmc 输入：nd2nz legalize 失败
    bench_fa_dev.log / bench_fa_dev.csv  Developer 模式性能 sweep
    profile_fa_dev*.log                  op_summary 设备时间（512/1024/2048/4096）
    profiler/op_summary/                 扁平化的 op_summary CSV（短文件名 + manifest.csv）
    fa_expert_ir.npuir                   expert 例子的 TileLang/HIVM IR（_mix_aic/_mix_aiv）
    expert_seq512_fixed.log              expert 修复后：seq=512 通过（1 次重试）
    expert_seq1024_fixed.log             expert 修复后：seq=1024 通过（1 次重试）
    expert_seq4096_fixed.log             expert 修复后：seq=4096 通过（1 次重试）
  STATUS.md
```

## 复现

```bash
cd /mnt/workspace/learn/tilelang-ascend

# 1) 官方 expert 例子（修复后跑通；首次编译可能因 CANN 偶发 SIGSEGV 自动重试）
python examples/flash_attn_npuir.py --seq_len 512
python examples/flash_attn_npuir.py

# 2) Developer 例子（本机可跑通）
python examples/flash_attn_npuir_dev.py

# 3) 性能 sweep（10 轮事件计时 + 正确性断言）
TILELANG_CACHE_DIR=/tmp/day5_fa_bench_cache \
python day5-flashattn/code/bench_fa_dev.py \
  --seq-lens 512 1024 2048 4096 \
  --csv day5-flashattn/logs/bench_fa_dev.csv

# 4) 设备时间（op_summary；event 计时含 host 入队，不能当设备时间）
python day5-flashattn/code/profile_fa_dev.py \
  --seq-lens 512 1024 2048 4096 --log-dir day5-flashattn/logs/profiler
```

> 注意：`bench_fa_dev.csv` 的 `device_ms_med` 列是 **event 计时**（含 host 入队），
> 不是设备时间；设备时间见 `profile_fa_dev*.log` / op_summary。

环境变量：

```bash
export TILELANG_ASCEND_MODE=Developer   # 仅对 bench_fa_dev.py 需要；脚本内已 setdefault
```

## 一键结论

- **expert 路径已跑通**：JIT 按编译器版本把 `syn_instr_mode` 改写为 CANN 9.0.0 接受的
  `sync_instr_mode`，并对后端偶发 SIGSEGV 自动重试；512/1024/4096 均 `All check passed!`。
  本轮未测 expert 设备时间，性能表仍全部来自 Developer 模式。详见 `note/worklog.md`。
- **Developer 路径已跑通**：`flash_attn_npuir_dev.py` 在 seq_len=512 通过
  `assert_close(rtol=atol=1e-2)`；`bench_fa_dev.py` 给出 512/1024/2048/4096 的形状与耗时。
- **机制结论**：Cube 负责 QK^T 与 PV 两个 gemm，Vector 负责 online softmax；
  expert 模式按 `T.Scope`/`T.rs` 显式拆成 `_mix_aic` / `_mix_aiv` 两个函数，
  Developer 模式自动判 MIX 后交给 HIVM 的 CV pipelining 拆；
  910B3 是 20 Cube : 40 Vector（1:2），FA 用 `subid` 把每块的 M 行分给两个 AIV；
  代码请求 `T.Pipelined(num_stages=2)` + HIVM workspace multi-buffer；文档默认配置为 2 槽，本轮未确认后端实际生效。
