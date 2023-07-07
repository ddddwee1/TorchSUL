import torch 
import gridsample 
import numpy as np 
import torch.nn.functional as F 
import time 
from torch.autograd import Function 


class GS(Function):
	@staticmethod 
	def forward(ctx, value, grid):
		ctx.save_for_backward(value, grid)
		return gridsample.grid_sample(value, grid)

	@staticmethod
	def backward(ctx, grad_out):
		value, grid = ctx.saved_tensors
		g_v, g_g = gridsample.grid_sample_backward(value, grid, grad_out)
		return g_v, g_g


# v = torch.from_numpy(np.float32(list(range(10000)))).reshape(1, 1, 100, 100).cuda().expand(-1, 32, -1, -1).contiguous()
# g = torch.rand(1, 8, 8, 2).cuda() 

# v_back = v.clone()
# g_back = g.clone()

# v.requires_grad_(True)
# g.requires_grad_(True)

# res = GS.apply(v, g)
# r = res.mean()

# r.backward()
# print(g.grad[0,0])

# grad1 = g.grad.clone()


# v = v_back.clone()
# g = g_back.clone()

# v.requires_grad_(True)
# g.requires_grad_(True)

# g2 = g*2-1
# res2 = F.grid_sample(v, g2, align_corners=True, mode='bilinear')
# r2 = res2.mean()
# r2.backward()
# print(g.grad[0,0])

# grad2 = g.grad.clone()

# print('Grad diff:', torch.abs(grad1 - grad2).sum())





v = torch.from_numpy(np.float32(list(range(10000)))).reshape(1, 1, 100, 100).cuda().expand(-1, 32, -1, -1).contiguous()
g = torch.rand(1, 8, 8, 2).cuda() 

res = gridsample.grid_sample(v, g)
g = g*2-1
res2 = F.grid_sample(v, g, align_corners=True, mode='bilinear')
print(res[0,0])
print(res.shape)
print(res2[0,0])
print(res2.shape)

diff = torch.abs(res - res2)
print(torch.allclose(res, res2))

# print('GridSample diff:',.max())
