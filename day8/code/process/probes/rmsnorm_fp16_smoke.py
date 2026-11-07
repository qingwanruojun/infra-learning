import sys
import torch
sys.path.insert(0, "/mnt/workspace/learn/infra-learning/day8/code/agent_output/custom/rmsnorm")
from rmsnorm import rmsnorm, pick_block_m

print("pick_block_m(N=1024, fp16) =", pick_block_m(1024, "float16"))
print("pick_block_m(N=1024, fp32) =", pick_block_m(1024, "float32"))

M, N = 1024, 1024
gen = torch.Generator().manual_seed(0)
x = ((torch.rand((M, N), generator=gen, dtype=torch.float32) * 2) - 1).to(torch.float16)
w = ((torch.rand((N,), generator=gen, dtype=torch.float32) * 2) - 1).to(torch.float16)
ref = (x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-5) * w.double()).to(torch.float16)
k = rmsnorm(M, N, "float16", 1e-5)
y = k(x.npu(), w.npu()).detach().cpu()
a, g = y.double(), ref.double()
err = (a - g).abs()
tol = 1e-2 + 1e-2 * g.abs()
print("fp16 smoke: max_abs=%.3e n_exceed=%d/%d" % (err.max().item(), int((err > tol).sum()), g.numel()))
