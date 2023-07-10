import torch 
import dcnv3 
import numpy as np 
import torch.nn.functional as F 
import time 
from torch.autograd import Function 
from tqdm import trange

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

def test_ours_forward(v, g, w):
	v.requires_grad_(True)
	g.requires_grad_(True)
	w.requires_grad_(True)

	res = DCNv3.apply(v, g, w)

	return res


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


def test_torch_forward(v, g, w):
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
	return res


v = torch.from_numpy(np.float32(list(range(64*64)))).reshape(1, 1, 64, 64).cuda().expand(512, 64, -1, -1).reshape(512, 4, 16, 64, 64).contiguous()
g = torch.rand(512, 4, 64, 64, 9, 2).cuda()
w = torch.rand(512, 4, 64, 64, 9).cuda()


# t1 = time.time()
# for i in trange(100):
# 	dcnv3.grid_sample(v, g, w)
# torch.cuda.synchronize()
# t2 = time.time()
# print('Kernel implementation:', t2-t1)



t1 = time.time()
for i in range(40):
	# test_ours(v.clone(), g.clone(), w.clone())
	with torch.no_grad():
		test_ours_forward(v.clone(), g.clone(), w.clone())
torch.cuda.synchronize()
t2 = time.time()
print('Kernel implementation:', t2-t1)
# print(torch.cuda.memory_summary())


t1 = time.time()
for i in range(40):
	# test_torch(v.clone(), g.clone(), w.clone())
	with torch.no_grad():
		test_torch_forward(v.clone(), g.clone(), w.clone())
torch.cuda.synchronize()
t2 = time.time()
print('torch implementation:', t2-t1)
# print(torch.cuda.memory_summary())
