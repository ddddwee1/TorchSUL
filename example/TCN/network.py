from TorchSUL import Model as M 
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class ResBlock1D(M.Model):
	def initialize(self, outchn=512, dilation=1, k=3):
		self.bn = M.BatchNorm()
		self.c1 = M.ConvLayer1D(k, outchn, dilation_rate=dilation, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		self.c2 = M.ConvLayer1D(3, outchn, pad='VALID')

	def forward(self, x):
		short = x

		# residual branch
		branch = self.bn(x)
		branch = M.activation(branch, M.PARAM_LRELU)
		branch = self.c1(branch)
		branch = self.c2(branch)

		# slicing & shortcut
		branch_shape = branch.shape[-1]
		short_shape = short.shape[-1]
		start = (short_shape - branch_shape) // 2
		short = short[:, :, start:start+branch_shape]
		res = short + branch
		res = F.dropout(res, 0.4, self.training, False)
		return res

class Refine2dNet(M.Model):
	def initialize(self, outchn):
		self.c1 = M.ConvLayer1D(5, 512, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		self.r1 = ResBlock1D(k=3, dilation=2)
		self.r2 = ResBlock1D(k=3, dilation=4)
		self.r3 = ResBlock1D(k=5, dilation=8)
		self.r4 = ResBlock1D(k=5, dilation=16)
		self.c5 = M.ConvLayer1D(9, 512, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		self.c4 = M.ConvLayer1D(1, outchn)

	def forward(self, x, drop=True):
		x = self.c1(x)
		# print(x.shape)
		x = self.r1(x)
		# print(x.shape)
		x = self.r2(x)
		# print(x.shape)
		x = self.r3(x)
		# print(x.shape)
		x = self.r4(x)
		# print(x.shape)
		# x = self.r5(x)
		# print(x.shape)
		x = self.c5(x)
		# print(x.shape)
		x = self.c4(x)
		# print(x.shape)
		return x 

