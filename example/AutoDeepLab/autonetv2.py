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
		self.p = M.MaxPool2D(3, stride)
	def forward(self, x):
		# print(x)
		x = self.p(x)
		return x

class AP33(M.Model):
	def initialize(self, stride, bn=False):
		self.p = M.AvgPool2D(3, stride)
	def forward(self, x):
		x = self.p(x)
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


## pp means previous-of-previous 
## p means previous
class CellBuilder(M.Model):
	def initialize(self, step, multiplier, c, rate):
		self.multiplier = multiplier
		self.step = step
		if rate==2:
			self.preprocess1 = FactReduce(c)
		elif rate==0:
			self.preprocess1 = FactIncrease(c)
		else:
			self.preprocess1 = M.ConvLayer(1, c, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		self._ops = nn.ModuleList()
		self.step = step
		for i in range(step):
			for j in range(1+i):
				self._ops.append(MixedOp(c, 1))
		self.conv_last = M.ConvLayer(1, c, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

	def forward(self, x, w):
		s1 = self.preprocess1(x)
		states = [s1]

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

def build_cells(step, multiplier, num_chn, in_size, out_size):
	cells = nn.ModuleList()
	connections = []
	for i in range(in_size):
		for j in range(max(0, i-1), min(i+2, out_size)):
			out = num_chn * 2**j
			if i==j:
				rate = 1 
			elif i<j:
				rate = 2
			else:
				rate = 0

			# build the cell 
			c = CellBuilder(step, multiplier, out, rate)
			cells.append(c)
			connections.append([i, j])
	return cells, connections

class Body(M.Model):
	def initialize(self, num_layer, num_chn, multiplier, step):
		# step means how many mixed ops you want in one layer
		# the num_layer here is not number of conv layers, but number of block of layers
		self.num_layer = num_layer
		self.step = step 

		self.c1 = M.ConvLayer(5, 64, stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		cells = nn.ModuleList()
		connections = []
		for i in range(num_layer):
			in_size = min(i+1, 4)
			out_size = min(i+2, 4)

			cell, conn = build_cells(step, multiplier, num_chn, in_size, out_size)
			cells.append(cell)
			connections.append(conn)

		self.cells = cells
		self.connections = connections

		self.down1 = FuseDown(3, 64, 512)
		self.down2 = FuseDown(2, 128, 512)
		self.down3 = FuseDown(1, 256, 512)
		self.down4 = M.ConvLayer(1, 512, batch_norm=True, usebias=False)

		self.final_conv = M.ConvLayer(1, 512, batch_norm=True, usebias=False)

	def build(self, *inputs):
		k = sum(1 for i in range(self.step) for n in range(1+i))
		self.alphas_cell = Parameter(torch.Tensor(k, 8))
		self.alphas_net = Parameter(torch.Tensor(self.num_layer, 4, 3))
		init.zeros_(self.alphas_net)
		init.zeros_(self.alphas_cell)

	def forward(self, x):
		# hardcode: there are 4 different scales 
		results = [[] for i in range(4)]
		x = self.c1(x)
		results[0].append(x)

		w_cell = F.softmax(self.alphas_cell, dim=-1)
		w_net = F.softmax(self.alphas_net, dim=-1)

		for l,(cell,conn) in enumerate(zip(self.cells, self.connections)):
			if not self.is_built:
				print('Layer %d'%l)
			buff = [[] for i in range(4)]
			for ce,con in zip(cell, conn):
				p = results[con[0]][-1]
				buf = ce(p, w_cell)
				buff[con[1]].append(buf)
			for i,b in enumerate(buff):
				if len(b)>0:
					w_layer = w_net[l, i]
					scale_sum = sum(w_layer[j]*b[j] for j in range(len(b)))
					results[i].append(scale_sum)

		res64 = self.down1(results[0][-1])
		res32 = self.down2(results[1][-1])
		res16 = self.down3(results[2][-1])
		res8 = self.down4(results[3][-1])

		feat_sum = res64 + res32 + res16 + res8
		feat_last = self.final_conv(feat_sum)
		return feat_last 

class AutoFaceNet(M.Model):
	def initialize(self):
		self.body = Body(5, 32, 3, 3)
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
	body = Body(10, 64, 3, 3)

	y = body(x)
	print(y.shape)
