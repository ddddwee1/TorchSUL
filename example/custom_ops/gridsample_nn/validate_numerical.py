import torch 
import gridsample_nn
import numpy as np 
import torch.nn.functional as F 
import time 
from torch.autograd import Function 


class grid_sample_nn(Function):
	@staticmethod 
	def forward(ctx, value, grid):
		ctx.save_for_backward(value, grid)
		return gridsample_nn.grid_sample(value, grid)

	@staticmethod
	def backward(ctx, grad_out):
		value, grid = ctx.saved_tensors
		g_v, g_g = gridsample_nn.grid_sample_back(value, grid, grad_out)
		return g_v, g_g


def test_ours(v, g):
	v.requires_grad_(True)
	g.requires_grad_(True)

	res = grid_sample_nn.apply(v, g)
	# res.retain_grad()
	r = res.mean()
	r.backward()
	# print(res.grad)

	return res, v.grad, g.grad

def test_torch(v, g):
	v.requires_grad_(True)
	g.requires_grad_(True)

	gg = g * 2 - 1 
	res = F.grid_sample(v, gg, align_corners=True, mode='nearest')
	# res.retain_grad()
	r = res.mean()
	r.backward()
	# print(res.grad)

	return res, v.grad, g.grad

def test_ours1(v, g):
	res = gridsample_nn.grid_sample(v, g)
	return res 

def test_torch_1(v, g):
	v.requires_grad_(True)
	g.requires_grad_(True)

	gg = g * 2 - 1 
	res = F.grid_sample(v, gg, align_corners=True, mode='nearest')
	return res 

def test_torch_0(v, g):
	v.requires_grad_(True)
	g.requires_grad_(True)

	gg = g * 2 - 1 
	res = F.grid_sample(v, gg, align_corners=False, mode='nearest')
	return res 



v = torch.from_numpy(np.float32(list(range(16)))).reshape(1, 1, 4, 4).cuda().expand(-1, 64, -1, -1).contiguous()
g = torch.rand(1, 1, 1, 2).cuda() 


# r1 = test_ours1(v.clone(), g.clone())
# r2 = test_torch_1(v.clone(), g.clone())

# print(torch.allclose(r1, r2, rtol=1e-2, atol=1e-3))



r1 = test_ours(v.clone(), g.clone())
r2 = test_torch(v.clone(), g.clone())

# print(r2[1].max(), r2[1].min(), r1[1].max(), r1[1].min())
print(r2[0][0,0])
print(r1[0][0,0])
print(r2[1][0,0])
print(r1[1][0,0])
print(r1[2][0,0])

for a1,a2 in zip(r1, r2):
	print(torch.allclose(a1, a2, rtol=1e-2, atol=1e-3))
