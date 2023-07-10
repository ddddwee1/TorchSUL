import torch 
import dcnv3 
import numpy as np 
import torch.nn.functional as F 
import time 
from torch.autograd import Function 


class DCNv3(Function):
	@staticmethod 
	def forward(ctx, value, grid, weight):
		ctx.save_for_backward(value, grid, weight)
		return dcnv3.grid_sample(value, grid, weight)

	@staticmethod
	def backward(ctx, grad_out):
		value, grid, weight = ctx.saved_tensors
		g_v, g_g, w_g = dcnv3.grid_sample_backward(value, grid, weight, grad_out)
		return g_v, g_g, w_g


def test_ours(v, g, w):
	v.requires_grad_(True)
	g.requires_grad_(True)
	w.requires_grad_(True)

	res = DCNv3.apply(v, g, w)
	r = res.mean()
	r.backward()

	return res, v.grad, g.grad, w.grad


def test_torch(v, g, w):
	v.requires_grad_(True)
	g.requires_grad_(True)
	w.requires_grad_(True)

	gp = v.shape[1]
	ho = g.shape[2]
	wo = g.shape[3]
	pmax = g.shape[4]
	b = v.shape[0]
	c = v.shape[2]

	res = []
	for i in range(gp):
		ww = w[:,i].reshape(b, 1, ho, wo*pmax)
		gg = g[:,i].reshape(b, ho, wo*pmax, 2) * 2 - 1
		vv = v[:,i]
		r = F.grid_sample(vv, gg, align_corners=True, mode='bilinear') * ww
		r = r.reshape(b, c, ho, wo, pmax).sum(dim=-1)
		res.append(r)
	res = torch.stack(res, dim=1)
	r = res.mean()
	r.backward()
	return res, v.grad, g.grad, w.grad



v = torch.from_numpy(np.float32(list(range(10000)))).reshape(1, 1, 1, 100, 100).cuda().expand(-1, 4, 128, -1, -1).contiguous()
g = torch.rand(1, 4, 32, 32, 9, 2).cuda() 
w = torch.rand(1, 4, 32, 32, 9).cuda()


r1 = test_ours(v.clone(), g.clone(), w.clone())
r2 = test_torch(v.clone(), g.clone(), w.clone())

# print(r1[2])
# print(r2[2])

# print(r1[0][0,:,0])
# print(r2[0][0,:,0])

for a1,a2 in zip(r1, r2):
	print(torch.allclose(a1, a2, rtol=1e-2, atol=1e-3))
