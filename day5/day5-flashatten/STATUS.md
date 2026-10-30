# Day 5 STATUS：FlashAttention / MixCV

更新：2026-10-29 16:12 · Ascend910B3 · CANN 9.0.0 · tilelang npuir

## 交付状态

| 交付物 | 状态 | 位置 |
|--------|------|------|
| 跑 FlashAttention，记录 shape 与耗时 | ✅ | expert 512/1024/4096 正确性通过；Developer 4 形状通过并有性能数据 |
| 读 mixcv skill（4 份） | ✅ | 摘要见 `note/day5学习指标.md` §2 |
| 讲清机制（分工 / 自动拆分 / 1:2 / double buffer） | ✅ | `note/day5学习指标.md` §3（含口述版） |
| 周报 v2（链路图 + 性能表 + 3 卡点 + 下周计划） | ✅ | `note/本周周报.md` |
| 当天日志 + worklog + 学习报告 | ✅ | `logs/`、`note/worklog.md`、`note/learning-report.md` |

## 关键数字

- 正确性：Developer 512/1024/2048/4096 全过，`max_abs ≤ 3.66e-4`；expert 512/1024/4096 `All check passed!`（均为 `rtol=atol=1e-2`）。
- 设备时间：27.33 / 56.60 / 208.61 / 824.57 µs（512 → 4096）。
- 对 torch 手写三段式：设备侧 **0.62x / 1.42x / 2.22x / 3.76x**；
  e2e 侧 1.40x / 1.45x / 1.91x / 3.60x。
- 交叉点：512 我们快，1024 起 torch 快；ours 的 TFLOPs 在 ~10 饱和，torch 到 39.14；
  64×64 tile 是首要假设，尚未由 sweep 证实。

## 阻塞

1. ~~**expert 后端编译阻塞**~~ **已解除**：JIT 把 `syn_instr_mode` 改写为 CANN 9.0.0 接受的 `sync_instr_mode`，
   并对 `ConvertHFusionToHIVM` 的偶发 SIGSEGV 最多重试 8 次。遗留：expert 设备时间未测，重试仍是临时兜底。
2. host 入队地板 230–450 µs，小 kernel 的 e2e 主要反映 host 路径；kernel 时间以 op_summary 为准。
3. 未做 tile sweep / 多缓冲 A/B，当前不能把小 tile 定为唯一根因。

## 复现

```bash
cd /mnt/workspace/learn/tilelang-ascend
python examples/flash_attn_npuir_dev.py                     # 期望 All check passed.
python examples/flash_attn_npuir.py --seq_len 512           # 期望 All check passed!；首次编译可能自动重试
python day5-flashattn/code/bench_fa_dev.py --seq-lens 512 1024 2048 4096
python day5-flashattn/code/profile_fa_dev.py --seq-lens 512 1024 2048 4096
```

## 下一步

见 `note/本周周报.md` §4（P0：工具链 + tile sweep；P1：多缓冲 A/B、GEMM 流水）。
