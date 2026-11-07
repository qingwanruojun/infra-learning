import sys, torch
import tilelang
from rmsnorm import rmsnorm

k = rmsnorm(1024, 1024, "float32", 1e-5, block_m=16)
print("COMPILED_FP32_OK")
x = torch.randn(1024, 1024, dtype=torch.float32)
w = torch.randn(1024, dtype=torch.float32)
y = k(x, w)
print("RAN_FP32_OK", y.shape, y.dtype)
