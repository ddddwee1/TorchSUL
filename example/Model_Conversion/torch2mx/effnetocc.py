from TorchSUL import Model as M 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MBBlock(M.Model):
	def initialize(self, ksize, filters, stride, expand):
		self.outchn = filters
		self.expand = expand
		self.stride = stride
		outchn = filters * expand
		self.bn0 = M.BatchNorm()
		self.c0 = M.ConvLayer(1, outchn, usebias=False, batch_norm=True, activation=M.PARAM_PRELU)
		self.c1 = M.DWConvLayer(ksize, 1, stride=stride, usebias=False, batch_norm=True, activation=M.PARAM_PRELU)

		# se 
		self.se1 = M.ConvLayer(1, outchn//8, activation=M.PARAM_PRELU)
		self.se2 = M.ConvLayer(1, outchn, activation=M.PARAM_SIGMOID)

		self.c2 = M.ConvLayer(1, filters, batch_norm=True, usebias=False)

		self.sc = M.ConvLayer(1, filters, stride=stride, batch_norm=True, usebias=False)

	def build(self, *inputs):
		self.inchn = inputs[0].shape[1]

	def forward(self, x):
		inp = x 
		x = self.bn0(x)
		x = self.c0(x)
		# print(x.shape)
		x = self.c1(x)

		se = M.GlobalAvgPool(x)
		se = self.se1(se)
		se = self.se2(se)

		# print(x.shape, se.shape)
		x = x * se 
		x = self.c2(x)

		if self.outchn==self.inchn and self.stride==1:
			sc = inp
		else:
			sc = self.sc(inp)
		x = sc + x 
		return x 

class LMF(M.Model):
	def initialize(self, ratio=0.9):
		self.ratio = ratio 
	def forward(self, x):
		# x: [N, c, h, w]
		N,c,h,w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		x = x.view(N, c, h*w)
		x, idx = torch.sort(x, dim=2)
		keep = int(h*w*self.ratio)
		# x[:,:,:keep] = 0
		x = x[:,:,-keep:]
		return x 

class Phi(M.Model):
	def initialize(self, chn):
		self.c1 = M.ConvLayer(3, chn)
		self.u1 = M.BilinearUpSample(2)
		self.u2 = M.BilinearUpSample(2)

	def forward(self, x):
		res = []
		target_size = x[0].shape[-2:]

		for i,a in enumerate(x):
			if i>0:
				# print('SIZEA', a.shape, 'TARGETSIZE',target_size)
				# a = F.interpolate(a, scale_factor=2, mode='bilinear', align_corners=False)
				if i==1:
					a = self.u1(a)
				else:
					a = self.u2(a)
			res.append(a)
		res = torch.cat(res, dim=1)
		res = self.c1(res)
		return res 

class EffNet(M.Model):
	def initialize(self, ksizes, channels, strides, expansions, repeats, finalType='E'):
		self.finalType = finalType

		self.c0 = M.ConvLayer(3, 32, batch_norm=True, usebias=False, activation=M.PARAM_PRELU)

		self.body = nn.ModuleList()
		for i,(k,c,s,e,r) in enumerate(zip(ksizes,channels,strides,expansions,repeats)):
			stage = nn.ModuleList()
			for j in range(r):
				stage.append(MBBlock(k,c,s if j==0 else 1, e))
			self.body.append(stage)

		self.phi = Phi(512)
		self.lmf = LMF(ratio=0.9)

	def forward(self, x):
		x = self.c0(x)
		feats = []
		for stage in self.body:
			for unit in stage:
				x = unit(x)
			feats.append(x)

		fmap = x = self.phi(feats[-3:])
		# x = self.lmf(x)
		# x = torch.mean(x, dim=-1)
		return x

def effnet():
	repeats = [1,2,2,4,3,4,3]
	channels = [16, 24, 40, 80, 112, 192, 320]
	ksizes = [3,3,5,3,5,5,3]
	strides = [1,2,2,1,2,2,1]
	expansions = [1,6,6,6,6,6,6]
	return EffNet(ksizes, channels, strides, expansions, repeats, finalType='E')

if __name__=='__main__':
	import numpy as np 

	eff = effnet()
	eff.eval()

	# intialize 
	x = torch.from_numpy(np.ones([1,3,112,112], dtype=np.float32))
	y = eff(x)

	print(y.shape)
