import torch 
import gridsample 
import numpy as np 
import torch.nn.functional as F 
import time 

v = torch.from_numpy(np.float32(list(range(10000)))).reshape(1, 1, 100, 100).cuda().expand(-1, 32, -1, -1).contiguous()
g = torch.rand(1, 16, 16, 2).cuda()
# g = torch.zeros(1,1,1,2).cuda()
# g[..., 1] *= 2

t1 = time.time()
for i in range(10000):
	res = gridsample.grid_sample(v, g)
torch.cuda.synchronize()
t2 = time.time()
print('Kernel implementation:', t2-t1)
# print(res.shape)
# print(res)


g = g * 2 - 1 
# print(g)
t1 = time.time()
for i in range(10000):
	res = F.grid_sample(v, g, align_corners=True, mode='bilinear')
torch.cuda.synchronize()
t2 = time.time()
print('torch implementation:', t2-t1)

# print(res)
