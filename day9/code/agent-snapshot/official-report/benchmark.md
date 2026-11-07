# NPU Benchmark Report
Generated: 2026-11-05 19:04:00

## Environment

- **Torch version**: 2.7.1+cpu
- **torch_npu version**: 2.7.1.post4
- **NPU model**: Ascend910B3
- **NPU count**: 1
- **Profiling mode**: msprof

## AminFwdOp

### tileops

| Label | Latency (us) | Ratio (%) | AI(Ops/Byte) | Bandwidth(TB/s) | Computility(TOps/s) | Perf(TOps/s) | artifact_dir | msprof | prof_mode | op_params | config | dtype | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-amin | 1.5400 | 0.6174 | 0.5231 | 1.7500 | 7.8336 | 0.0057 | /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261105_190051_728147_AminFwdOp/artifacts/msprof/main_swi3m31k | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1.54, 'max_latency_us': 1.54, 'min_latency_us': 1.54, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 8, 'tile_n': 0} | float16 | [32, 256] |
| smoke-amin | 1.5800 | 0.6018 | 1.0462 | 1.7500 | 4.4133 | 0.0110 | /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261105_190051_728147_AminFwdOp/artifacts/msprof/main_ycpad1sb | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1.58, 'max_latency_us': 1.58, 'min_latency_us': 1.58, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 8, 'tile_n': 0} | bfloat16 | [32, 256] |
| smoke-amin | 1.4800 | 1.2751 | 0.2636 | 1.7500 | 4.3520 | 0.0059 | /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261105_190051_728147_AminFwdOp/artifacts/msprof/main_do05s3_m | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 1.48, 'max_latency_us': 1.48, 'min_latency_us': 1.48, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 8, 'tile_n': 0} | float32 | [32, 256] |
| hidden-state-reduce | 14.0600 | 68.1867 | 0.5060 | 1.7500 | 8.8988 | 0.6038 | /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261105_190051_728147_AminFwdOp/artifacts/msprof/main_lzhx5v4z | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 14.06, 'max_latency_us': 14.06, 'min_latency_us': 14.06, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 4, 'tile_n': 0} | float16 | [2048, 4096] |
| hidden-state-reduce | 15.0200 | 63.8286 | 1.0152 | 1.7500 | 4.5512 | 1.1340 | /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261105_190051_728147_AminFwdOp/artifacts/msprof/main_1qdbgi_q | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 15.02, 'max_latency_us': 15.02, 'min_latency_us': 15.02, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 4, 'tile_n': 0} | bfloat16 | [2048, 4096] |
| long-seq-reduce | 7.1200 | 33.6632 | 1.0103 | 1.7500 | 4.5743 | 0.5952 | /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261105_190051_728147_AminFwdOp/artifacts/msprof/main_tnku73t6 | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': [None], 'kernel_name_requested': None, 'kernel_name_resolved': None, 'latency_us': 7.12, 'max_latency_us': 7.12, 'min_latency_us': 7.12, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': -1} | {'block_m': 1, 'tile_n': 16384} | bfloat16 | [64, 32768] |
| 3d-multidim-reduce | 5.6800 | 42.1975 | 0.5084 | 1.7500 | 8.7797 | 0.3754 | /mnt/workspace/learn/refs/tilelang-mlir-ascend/examples/TileOPs/reports/tileops/20261105_190051_728147_AminFwdOp/artifacts/msprof/main_i5s_8l5i | {'captured_op_count': 1, 'captured_op_names': ['main'], 'kernel_name_attempts': ['main'], 'kernel_name_requested': 'main', 'kernel_name_resolved': 'main', 'latency_us': 5.68, 'max_latency_us': 5.68, 'min_latency_us': 5.68, 'op_name': 'main', 'sample_count': 1} | msprof | {'dim': [0, 2]} | {'block_m': 1, 'tile_n': 0} | float16 | [4, 128, 4096] |
