import os 
from TorchSUL import Model as M 
import network 
import torch 
import numpy as np 
import data_reader
import random 
from tqdm import tqdm 

class modelBundle(M.Model):
	def initialize(self):
		self.refine2D = network.Refine2dNet(17*2)
		self.Depth3D = network.Refine2dNet(17)
	def forward(self, x):
		x2d = self.refine2D(x)
		x3d = self.Depth3D(x)
		x2dx = x2d[:,0::2]
		x2dy = x2d[:,1::2]
		x = torch.stack([x2dx, x2dy, x3d], dim=1)
		return x 

def generate_mask(x):
	shape = x.shape
	choice = random.random() * 4
	if choice<1:
		outshape = [x.shape[0], x.shape[1], 17, 1]
		mask = np.random.choice(2, size=outshape, p=[0.2, 0.8]) # edit probability here
		mask = np.concatenate([mask]*2, axis=-1)
		mask = mask.reshape([x.shape[0], x.shape[1], 34])
		return np.float32(mask) 
	elif choice<2:
		outshape = [x.shape[0], x.shape[1], 1]
		mask = np.random.choice(2, size=outshape, p=[0.2, 0.8]) # edit probability here
		# mask = np.concatenate([mask]*2, axis=-1)
		# mask = mask.reshape([x.shape[0], x.shape[1], 1])
		return np.float32(mask) 
	elif choice<3:
		outshape = [x.shape[0], x.shape[1], 1]
		mask = np.ones(outshape)
		length = random.randint(0,20)
		start_point = random.randint(0, x.shape[1]-length)
		mask[:,start_point:start_point+length] = 0
		return np.float32(mask) 
	else:
		outshape = [x.shape[0], 1, 17, 1]
		mask = np.random.choice(2, size=outshape, p=[0.1, 0.9]) # edit probability here
		mask = np.concatenate([mask]*2, axis=-1)
		mask = mask.reshape([x.shape[0], 1, 34])
		return np.float32(mask) 

def generate_noise(x):
	noise = np.random.random(x.shape) * 0.15 - 0.075
	noise = np.float32(noise)
	noisemask = generate_mask(x)
	noise = noise * noisemask
	return noise

def lossfunc(x, model, init=False):
	data, label = x 
	mask = generate_mask(data)
	noise = generate_noise(data)
	data = (data + noise) * mask
	data = np.transpose(data, [0,2,1])
	label = np.transpose(label, [0,2,1])

	data = torch.from_numpy(data)
	label = torch.from_numpy(label)

	if not init:
		data = data.cuda()
		label = label.cuda()
	
	out = model(data)
	out = out.squeeze(-1)

	loss = torch.mean(torch.pow(out - label, 2))
	with torch.no_grad():
		loss2d = torch.pow(out[:,:2] - label[:,:2], 2)
		loss2d = torch.mean(torch.sqrt(torch.sum(loss2d, dim=1)))
		loss3d = torch.mean(torch.abs(out[:,2] - label[:,2]))
	return loss, loss2d, loss3d

nets = modelBundle()

reader = data_reader.data_reader(64, 64)
x = reader.get_next()
loss, loss2d, loss3d = lossfunc(x, nets, True)
saver = M.Saver(nets)
saver.restore('./model/')
nets.cuda()
nets.train()

optim = torch.optim.Adam(nets.parameters(), lr=0.001, betas=(0.5, 0.999))
MAXITER = 45001
bar = tqdm(range(MAXITER))
for i in bar:
	x = reader.get_next()
	optim.zero_grad()
	loss, loss2d, loss3d = lossfunc(x, nets)
	loss.backward()
	optim.step()

	lr = optim.param_groups[0]['lr']
	outstr = 'Loss:%.4f L2D:%.4f L3D:%.4f LR:%.1e'%(loss, loss2d ,loss3d, lr)
	bar.set_description(outstr)

	if i%3000==0 and i>0:
		saver.save('./model/%d.ckpt'%i)

	if i%15000==0 and i>0:
		newlr = lr * 0.1 
		for param_group in optim.param_groups:
			param_group['lr'] = newlr
