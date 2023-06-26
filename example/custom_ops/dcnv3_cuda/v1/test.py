import torch 
import dcn 
import numpy as np 
import torch.nn.functional as F 
import time 
from tqdm import trange

# v: [B, G, C, H, W]
# g: [B, Ho, Wo, G, P, 2]
v = torch.randn(128, 4, 16, 64, 64).cuda().contiguous()
g = torch.rand(128, 64, 64, 4, 9, 2).cuda().contiguous()
m = torch.rand(128, 64, 64, 4, 9).cuda().contiguous()
# print(torch.cuda.memory_summary())
# g = torch.zeros(1,1,1,2).cuda()
# g[..., 1] *= 2

repeat = 100
t1 = time.time()
for i in range(repeat):
	res = dcn.dcn(v, g, m)
torch.cuda.synchronize()
t2 = time.time()
print('Kernel implementation:', (t2-t1)/repeat)
print(res.shape)
# print(res)
print(torch.cuda.memory_summary())

# g = g * 2 - 1 
# # print(g)
# t1 = time.time()
# for i in range(1000):
# 	res = F.grid_sample(v, g, align_corners=True, mode='bilinear')
# torch.cuda.synchronize()
# t2 = time.time()
# print('torch implementation:', t2-t1)
# print(torch.cuda.memory_summary())
## print(res)
