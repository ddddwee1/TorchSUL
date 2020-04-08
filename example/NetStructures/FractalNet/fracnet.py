import torch 
import random 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 

FRAC = 3

class Merge(M.Model):
	def initialize(self, prob):
		self.prob = prob
	def forward(self, x):
		if self.training:
			res = []
			for i in range(len(x)):
				if random.random()<self.prob:
					res.append(x[i])
			if len(res)==0:
				res = random.choice(x)
			else:
				res = torch.stack(res, dim=0)
				res = torch.mean(res, dim=0)
			return res 
		else:
			res = torch.stack(x, dim=0)
			res = torch.mean(res, dim=0)
			return res 

class FracUnit(M.Model):
	def initialize(self, chn, frac, prob):
		self.c1 = M.ConvLayer(3, chn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)
		self.frac = frac
		if frac==0:
			self.frac1 = M.ConvLayer(3, chn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)
			self.frac2 = M.ConvLayer(3, chn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)
		else:
			self.frac1 = FracUnit(chn, frac-1, prob)
			self.merge = Merge(prob)
			self.frac2 = FracUnit(chn, frac-1, prob)

	def forward(self, x):
		x1 = self.c1(x)
		if self.frac==0:
			x2 = self.frac1(x)
			x2 = self.frac2(x2)
			res = [x1, x2]
		else:
			x2 = self.frac1(x)
			x2 = self.merge(x2)
			x2 = self.frac2(x2)
			res = [x1] + x2
		# return value is a list 
		return res 

class FracBlock(M.Model):
	def initialize(self, chn, frac, prob):
		self.frac = FracUnit(chn, frac, prob)
		self.merge = Merge(prob)
	def forward(self, x):
		x = self.frac(x)
		x = self.merge(x)
		return x 

class ResBlock_v1(M.Model):
	def initialize(self, outchn, stride):
		self.stride = stride
		self.outchn = outchn
		self.bn0 = M.BatchNorm()
		self.c1 = M.ConvLayer(3, outchn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)

		# shortcut 
		self.sc = M.ConvLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)

	def build(self, *inputs):
		self.inchn = inputs[0].shape[1]

	def forward(self, x):
		res = self.bn0(x)
		res = self.c1(res)
		res = self.c2(res)
		# shortcut 
		if self.inchn==self.outchn and self.stride==1:
			sc = x 
		else:
			sc = self.sc(x)
		res = res + sc 
		return res 

class Stage(M.Model):
	def initialize(self, outchn, num_blocks, drop_prob):
		self.r1 = ResBlock_v1(outchn, 2)
		self.units = nn.ModuleList()
		for i in range(num_blocks):
			self.units.append(FracBlock(outchn, FRAC, drop_prob))
	def forward(self, x):
		x = self.r1(x)
		for i in self.units:
			x = i(x)
		return x 

class FracNet(M.Model):
	def initialize(self, channel_list, blocknum_list, drop_prob):
		self.c1 = M.ConvLayer(3, channel_list[0], usebias=False, batch_norm=True, activation=M.PARAM_PRELU)
		self.stage1 = Stage(channel_list[1], blocknum_list[0], drop_prob)
		self.stage2 = Stage(channel_list[2], blocknum_list[1], drop_prob)
		self.stage3 = Stage(channel_list[3], blocknum_list[2], drop_prob)
		self.stage4 = Stage(channel_list[4], blocknum_list[3], drop_prob)

		self.bn1 = M.BatchNorm()
		self.fc1 = M.Dense(512, usebias=False, batch_norm=True)

	def forward(self, x):
		x = self.c1(x)
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.bn1(x)
		x = M.flatten(x)
		x = F.dropout(x, 0.4, self.training, False)
		x = self.fc1(x)
		return x 

def Frac100():
	return FracNet([64,32,64,128,256],[1,1,7,3], 0.5)

if __name__=='__main__':
	net = Frac100()
	x = np.zeros([2,3,112,112]).astype(np.float32)
	x = torch.from_numpy(x)
	y = net(x)
	M.Saver(net).save('./model/abc.pth')
