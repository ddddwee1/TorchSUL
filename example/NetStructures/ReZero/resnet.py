from TorchSUL import Model as M 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import pickle 
from torch.nn.parameter import Parameter

class R0ConvLayer(M.Model):
	def initialize(self, ksize, outchn, pad='SAME_LEFT', stride=1, activation=-1, usebias=True, batch_norm=False):
		self.conv = M.ConvLayer(ksize, outchn, pad=pad, stride=stride, activation=activation)

	def build(self, *inputs):
		self.alpha = Parameter(torch.zeros(1), requires_grad=True)

	def forward(self, x):
		res = self.conv(x)
		if x.shape[1]!=res.shape[1] or x.shape[2]!=res.shape[2] or x.shape[3]!=res.shape[3]:
			res = res  
		else:
			res = x + self.alpha * res 
		return res 

class ResBlock_v1(M.Model):
	def initialize(self, outchn, stride):
		self.stride = stride
		self.outchn = outchn
		self.bn0 = M.BatchNorm()
		self.c1 = R0ConvLayer(3, outchn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)
		self.c2 = R0ConvLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)

		# se module 
		#self.c3 = R0ConvLayer(1, outchn//16, activation=M.PARAM_PRELU)
		#self.c4 = M.ConvLayer(1, outchn, activation=M.PARAM_SIGMOID)

		# shortcut 
		self.sc = R0ConvLayer(1, outchn, stride=stride, usebias=False, batch_norm=True)

	def build(self, *inputs):
		self.inchn = inputs[0].shape[1]

	def forward(self, x):
		res = self.bn0(x)
		res = self.c1(res)
		res = self.c2(res)
		# print(res.shape)
		# se
		#se = M.GlobalAvgPool(res)
		#se = self.c3(se)
		#se = self.c4(se)
		#res = res * se 
		# shortcut 
		if self.inchn==self.outchn and self.stride==1:
			sc = x 
		else:
			sc = self.sc(x)
		res = res + sc 
		return res 

class Stage(M.Model):
	def initialize(self, outchn, blocknum):
		self.units = nn.ModuleList()
		for i in range(blocknum):
			self.units.append(ResBlock_v1(outchn, stride=2 if i==0 else 1))
	def forward(self, x):
		for i in self.units:
			x = i(x)
		return x 

class ResNet(M.Model):
	def initialize(self, channel_list, blocknum_list, embedding_size, embedding_bn=True):
		self.c1 = R0ConvLayer(3, channel_list[0], 1, usebias=False, activation=M.PARAM_PRELU, batch_norm=True)
		# self.u1 = ResBlock_v1(channel_list[1], stride=2)
		self.stage1 = Stage(channel_list[1], blocknum_list[0])
		self.stage2 = Stage(channel_list[2], blocknum_list[1])
		self.stage3 = Stage(channel_list[3], blocknum_list[2])
		self.stage4 = Stage(channel_list[4], blocknum_list[3])
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

def Res50():
	return ResNet([64,64,128,256,512],[3,4,14,3],512)
	
def Res100():
	return ResNet([64,64,128,256,512],[3,13,30,3],512)
