from TorchSUL import Model as M 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
import torch.nn.init as init 

class Zero(M.Model):
	def initialize(self, stride):
		self.stride = stride
	def forward(self, x):
		if self.stride==1:
			return x * 0
		else:
			x = x[:,:, ::self.stride, ::self.stride] * 0
			return x 

class MP33(M.Model):
	def initialize(self, stride, bn=False):
		self.bn = bn 
		self.p = M.MaxPool2D(3, stride)
		if self.bn:
			self.batch_norm = M.BatchNorm()
	def forward(self, x):
		# print(x)
		x = self.p(x)
		if self.bn:
			x = self.batch_norm(x)
		return x

class AP33(M.Model):
	def initialize(self, stride, bn=False):
		self.bn = bn 
		self.p = M.AvgPool2D(3, stride)
		if self.bn:
			self.batch_norm = M.BatchNorm()
	def forward(self, x):
		x = self.p(x)
		if self.bn:
			x = self.batch_norm(x)
		return x

class Skip(M.Model):
	def initialize(self, out, stride):
		self.stride = stride
		if stride!=1:
			self.rd = FactReduce(out)
	def forward(self, x):
		if self.stride==1:
			return x 
		else:
			return self.rd(x)

class SPConv33(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.DWConvLayer(3, 1, stride=stride, usebias=False)
		self.c2 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
	def forward(self, x):
		x = M.activation(x, M.PARAM_RELU)
		x = self.c1(x)
		x = self.c2(x)
		return x 

class SPConv55(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.DWConvLayer(5, 1, stride=stride, usebias=False)
		self.c2 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
	def forward(self, x):
		x = M.activation(x, M.PARAM_RELU)
		x = self.c1(x)
		x = self.c2(x)
		return x 

class DilConv33(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.ConvLayer(3, out, stride=stride, dilation_rate=2, batch_norm=True, usebias=False)
	def forward(self, x):
		x = M.activation(x, M.PARAM_RELU)
		x = self.c1(x)
		return x 

class DilConv55(M.Model):
	def initialize(self, out, stride):
		self.c1 = M.ConvLayer(5, out, stride=stride, dilation_rate=2, batch_norm=True, usebias=False)
	def forward(self, x):
		x = M.activation(x, M.PARAM_RELU)
		x = self.c1(x)
		return x 

# class ASPP(M.Model):
# 	def initialize(self, out, dilation):
# 		self.c11 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
# 		self.c33 = M.ConvLayer(3, out, usebias=False, batch_norm=True, dilation_rate=dilation)
# 		self.convp = M.ConvLayer(1, out, usebias=False, batch_norm=True)
# 		self.concat_conv = M.ConvLayer(1, out, usebias=False, batch_norm=True)
# 		self.GP = M.GlobalAvgPool()
# 	def forward(self, x):
# 		x11 = self.c11(x)
# 		x33 = self.c33(x)
# 		pool = self.GP(x)
# 		up = tf.ones_like(x) * pool 
# 		concat = tf.concat([x11, x33, up], axis=-1)
# 		out = self.concat_conv(concat)
# 		return out 

class FuseDown(M.Model):
	def initialize(self, steps, inp, o):
		self.mods = nn.ModuleList()
		for i in range(steps):
			if i==(steps-1):
				self.mods.append(M.ConvLayer(3, o, stride=2, batch_norm=True, usebias=False))
			else:
				self.mods.append(M.ConvLayer(3, o, stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for m in self.mods:
			x = m(x)
		return x 

class FactReduce(M.Model):
	def initialize(self, out):
		self.c1 = M.ConvLayer(1, out//2, stride=2, usebias=False)
		self.c2 = M.ConvLayer(1, out//2, stride=2, usebias=False)
		self.bn = M.BatchNorm()
	def forward(self, x):
		x = M.activation(x, M.PARAM_RELU)
		x1 = self.c1(x)
		x2 = self.c2(x[:,1:,1:,:])
		x = torch.cat([x1, x2], axis=1)
		x = self.bn(x)
		return x 

class FactIncrease(M.Model):
	def initialize(self, out):
		self.c1 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
	def forward(self, x):
		x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear', align_corners=False)
		x = M.activation(x, M.PARAM_RELU)
		x = self.c1(x)
		return x 

class MixedOp(M.Model):
	def initialize(self, out, stride):
		ops = nn.ModuleList()
		ops.append(Zero(stride))
		ops.append(MP33(stride, True))
		ops.append(AP33(stride, True))
		ops.append(Skip(out, stride))
		ops.append(SPConv33(out, stride))
		ops.append(SPConv55(out, stride))
		ops.append(DilConv33(out, stride))
		ops.append(DilConv55(out, stride))
		self.ops = ops 
	def forward(self, x, weights):
		# print(x.shape)
		res = sum(w*op(x) for w,op in zip(weights,self.ops))
		# print(res.shape)
		return res 

class CellBuilder(M.Model):
	def initialize(self, step, multiplier, c):
		self.multiplier = multiplier
		self.step = step

		self._ops = nn.ModuleList()
		self.step = step
		for i in range(step):
			for j in range(1+i):
				self._ops.append(MixedOp(c, 1))
		self.conv_last = M.ConvLayer(1, c, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

	def forward(self, x, w):
		states = [x]
		offset = 0
		for i in range(self.step):
			buff = []
			for j in range(1+i):
				op = self._ops[offset]
				buff.append(op(states[j], w[offset]))
				offset += 1
			buff = sum(buff)
			states.append(buff)
		concat_feat = torch.cat(states[-self.multiplier:], dim=1)
		out = self.conv_last(concat_feat)
		return out 

class Stage(M.Model):
	def initialize(self, num_unit, chn, multiplier, step, reduce_size=None):
		self.units = nn.ModuleList()
		for n in range(num_unit):
			self.units.append(CellBuilder(step, multiplier, chn))
		self.reduce_size = reduce_size
		if not reduce_size is None:
			self.reduce = FactReduce(reduce_size)
	def forward(self, x, w):
		for u in self.units:
			x = u(x, w)
		if not self.reduce_size is None:
			x = self.reduce(x)
		return x 

class Body(M.Model):
	def initialize(self, unit_list, chn_list, multiplier, step):
		self.step = step 

		self.c1 = M.ConvLayer(5, 64, stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c2 = M.ConvLayer(3, 64, stride=1, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		self.stage1 = Stage(unit_list[0], chn_list[0], multiplier, step, reduce_size=chn_list[1])
		self.stage2 = Stage(unit_list[1], chn_list[1], multiplier, step, reduce_size=chn_list[2])
		self.stage3 = Stage(unit_list[2], chn_list[2], multiplier, step, reduce_size=chn_list[3])
		self.stage4 = Stage(unit_list[3], chn_list[3], multiplier, step, reduce_size=None)

	def build(self, *inputs):
		k = sum(1 for i in range(self.step) for n in range(1+i))
		self.alphas_cell = Parameter(torch.Tensor(k, 8))
		init.zeros_(self.alphas_cell)

	def forward(self, x):
		# hardcode: there are 4 different scales 
		x = self.c1(x)
		x = self.c2(x)

		w_cell = F.softmax(self.alphas_cell, dim=-1)

		x = self.stage1(x, w_cell)
		x = self.stage2(x, w_cell)
		x = self.stage3(x, w_cell)
		x = self.stage4(x, w_cell)

		return x 

class AutoFaceNet(M.Model):
	def initialize(self):
		self.body = Body([2,2,10,2], [64,128,256,512], 3, 3)
		self.fc1 = M.Dense(512, usebias=False)
	def forward(self, x):
		x = self.body(x)
		x = M.flatten(x)
		feat = self.fc1(x)
		return feat 

if __name__=='__main__':
	import numpy as np 
	x = torch.from_numpy(np.float32(np.zeros([1, 3, 112, 112])))
	# t = M.ConvLayer(3, 5, dilation_rate=5)
	# y = t(x)
	# print(y.shape)
	body = Body([2,2,10,2], [64,128,256,512], 3, 3)

	y = body(x)
	print(y.shape)
