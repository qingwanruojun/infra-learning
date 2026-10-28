# profiler_raw — 原始设备时长

从 `export_only_prof_dir/*/PROF_*/mindstudio_profiler_output/op_summary_*.csv`
拷出的 `Task Duration(us)` 原始数据（每份 20 次调用），共 61 份 = 16:28–16:38
的 21 份 + 16:55 ABBA 十轮的 40 份。CANN 的临时目录（13–30 MB，含 host/device
原始 trace）每次用完即删，重跑对应探针即可再生成。

| 文件 | 进程批次 | op | 中位数 | 对应探针 / 用例 |
|------|---------|----|-------:|-----------------|
| `op_summary_20261028162856.csv` | 16:28 #1 | `main` | 13.68 µs | `probe_profiler_kernel_time.py`，tilelang N=1M，4096 块 |
| `op_summary_20261028162901.csv` | 16:28 #2 | `main` | 28.02 µs | 同上，tilelang N=16M，40 块串行 |
| `op_summary_20261028162905.csv` | 16:28 #3 | `aclnnAbs_AbsAiCore_Abs` | 3.57 µs | 同上，torch N=1M |
| `op_summary_20261028162909.csv` | 16:28 #4 | `aclnnAbs_AbsAiCore_Abs` | 22.17 µs | 同上，torch N=16M |
| `op_summary_20261028163253.csv` | 16:32 #1 | `main` | 12.99 µs | 复跑（`logs/profiler_summary.log` 采用值） |
| `op_summary_20261028163258.csv` | 16:32 #2 | `main` | 26.86 µs | 复跑 |
| `op_summary_20261028163302.csv` | 16:32 #3 | `aclnnAbs_AbsAiCore_Abs` | 3.55 µs | 复跑 |
| `op_summary_20261028163306.csv` | 16:32 #4 | `aclnnAbs_AbsAiCore_Abs` | 22.13 µs | 复跑 |
| `op_summary_20261028163407.csv` | 16:34 #1 | `main` | 212.14 µs | `probe_hbm_compare.py`，N=32M，40 块 × block_N=8192 |
| `op_summary_20261028163411.csv` | 16:34 #2 | `aclnnAbs_AbsAiCore_Abs` | 166.93 µs | 同上 torch（**孤点**，见 worklog 错误 12） |
| `op_summary_20261028163518.csv` | 16:35 #1 | `aclnnAbs_AbsAiCore_Abs` | 214.37 µs | `probe_hbm_peak.py`，128 MiB 张量 |
| `op_summary_20261028163522.csv` | 16:35 #2 | `aclnnAbs_AbsAiCore_Abs` | 421.90 µs | 同上，256 MiB |
| `op_summary_20261028163526.csv` | 16:35 #3 | `aclnnAbs_AbsAiCore_Abs` | 841.76 µs | 同上，512 MiB |
| `op_summary_20261028163612.csv` | 16:36 #1 | `aclnnAbs_AbsAiCore_Abs` | 217.26 µs | `probe_hbm_interleaved.py`，round0 torch |
| `op_summary_20261028163615.csv` | 16:36 #2 | `main` | 203.09 µs | round0 tilelang |
| `op_summary_20261028163619.csv` | 16:36 #3 | `aclnnAbs_AbsAiCore_Abs` | 217.30 µs | round1 torch |
| `op_summary_20261028083623.csv` | 16:36 #4 | `main` | 204.36 µs | round1 tilelang（CANN 给的文件名时间戳是 08:36，按批次归到此轮） |
| `op_summary_20261028163821.csv` | 16:38 #1 | `main` | 158.73 µs | `probe_blockcount_16M.py`，N=16M，4096 块 round0 |
| `op_summary_20261028163825.csv` | 16:38 #2 | `main` | 27.25 µs | 同上，40 块串行 round0 |
| `op_summary_20261028163829.csv` | 16:38 #3 | `main` | 155.96 µs | 同上，4096 块 round1 |
| `op_summary_20261028163832.csv` | 16:38 #4 | `main` | 26.98 µs | 同上，40 块串行 round1 |

注意 `Block Num` 列全是 0，不能用它核对 grid 大小；块数要以探针代码为准。

### ABBA 10 轮（`probe_abba_hbm.py`，16:55 批次）

每轮 4 次设备时长测量，顺序 torch → ours → ours → torch；每次重复调用 20 次取中位数。
（文件名时间戳有一处 CANN 异常：`op_summary_20261028085506.csv` 属于 16:55 批次 #1。）

| 文件 | 批次 | op | 中位数 |
|------|------|----|-------:|
| `op_summary_20261028085506.csv` | round0 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 218.21 µs |
| `op_summary_20261028165511.csv` | round0 #2 tilelang A | `main` | 203.95 µs |
| `op_summary_20261028165514.csv` | round0 #3 tilelang B | `main` | 204.11 µs |
| `op_summary_20261028165518.csv` | round0 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 193.64 µs |
| `op_summary_20261028165522.csv` | round1 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 196.68 µs |
| `op_summary_20261028165526.csv` | round1 #2 tilelang A | `main` | 202.84 µs |
| `op_summary_20261028165530.csv` | round1 #3 tilelang B | `main` | 203.22 µs |
| `op_summary_20261028165534.csv` | round1 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 188.90 µs |
| `op_summary_20261028165538.csv` | round2 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 193.84 µs |
| `op_summary_20261028165541.csv` | round2 #2 tilelang A | `main` | 203.06 µs |
| `op_summary_20261028165545.csv` | round2 #3 tilelang B | `main` | 203.55 µs |
| `op_summary_20261028165549.csv` | round2 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 186.32 µs |
| `op_summary_20261028165553.csv` | round3 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 187.65 µs |
| `op_summary_20261028165558.csv` | round3 #2 tilelang A | `main` | 202.35 µs |
| `op_summary_20261028165601.csv` | round3 #3 tilelang B | `main` | 202.73 µs |
| `op_summary_20261028165605.csv` | round3 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 184.51 µs |
| `op_summary_20261028165609.csv` | round4 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 184.99 µs |
| `op_summary_20261028165613.csv` | round4 #2 tilelang A | `main` | 200.66 µs |
| `op_summary_20261028165617.csv` | round4 #3 tilelang B | `main` | 200.59 µs |
| `op_summary_20261028165621.csv` | round4 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 184.48 µs |
| `op_summary_20261028165625.csv` | round5 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 185.54 µs |
| `op_summary_20261028165629.csv` | round5 #2 tilelang A | `main` | 199.13 µs |
| `op_summary_20261028165634.csv` | round5 #3 tilelang B | `main` | 199.70 µs |
| `op_summary_20261028165637.csv` | round5 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 185.91 µs |
| `op_summary_20261028165641.csv` | round6 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 186.86 µs |
| `op_summary_20261028165645.csv` | round6 #2 tilelang A | `main` | 198.04 µs |
| `op_summary_20261028165649.csv` | round6 #3 tilelang B | `main` | 197.91 µs |
| `op_summary_20261028165652.csv` | round6 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 184.34 µs |
| `op_summary_20261028165656.csv` | round7 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 184.17 µs |
| `op_summary_20261028165700.csv` | round7 #2 tilelang A | `main` | 196.75 µs |
| `op_summary_20261028165704.csv` | round7 #3 tilelang B | `main` | 197.21 µs |
| `op_summary_20261028165708.csv` | round7 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 184.23 µs |
| `op_summary_20261028165711.csv` | round8 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 184.04 µs |
| `op_summary_20261028165715.csv` | round8 #2 tilelang A | `main` | 198.60 µs |
| `op_summary_20261028165719.csv` | round8 #3 tilelang B | `main` | 198.72 µs |
| `op_summary_20261028165723.csv` | round8 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 184.77 µs |
| `op_summary_20261028165727.csv` | round9 #1 torch | `aclnnAbs_AbsAiCore_Abs` | 185.80 µs |
| `op_summary_20261028165730.csv` | round9 #2 tilelang A | `main` | 197.94 µs |
| `op_summary_20261028165734.csv` | round9 #3 tilelang B | `main` | 198.25 µs |
| `op_summary_20261028165738.csv` | round9 #4 torch | `aclnnAbs_AbsAiCore_Abs` | 184.51 µs |
