from TorchSUL.Layers import activation
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import numpy as np 

class ResUnit(M.Model):
	def initialize(self, out, stride, shortcut=False):
		self.shortcut = shortcut
		self.c1 = M.ConvLayer(1, out//4, usebias=False, activation=M.PARAM_RELU, batch_norm=True)
		self.c2 = M.ConvLayer(3, out//4, usebias=False, activation=M.PARAM_RELU, pad='SAME_LEFT', stride=stride, batch_norm=True)
		self.c3 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
		if shortcut:
			self.sc = M.ConvLayer(1, out, usebias=False, stride=stride, batch_norm=True)

	def forward(self, x):
		branch = self.c1(x)
		branch = self.c2(branch)
		branch = self.c3(branch)
		if self.shortcut:
			sc = self.sc(x)
		else:
			sc = x 
		res = branch + sc
		res = M.activation(res, M.PARAM_RELU)
		return res 

class ResBlock(M.Model):
	def initialize(self, out, stride, num_units):
		self.units = nn.ModuleList()
		for i in range(num_units):
			self.units.append(ResUnit(out, stride if i==0 else 1, True if i==0 else False))
	def forward(self, x):
		for unit in self.units:
			x = unit(x)
		return x 

class BasicUnit(M.Model):
	def initialize(self, out, stride, shortcut=False):
		self.shortcut = shortcut
		self.c1 = M.ConvLayer(3, out, pad='SAME_LEFT', usebias=False, activation=M.PARAM_RELU, batch_norm=True)
		self.c2 = M.ConvLayer(3, out, pad='SAME_LEFT', usebias=False, batch_norm=True)
		if shortcut:
			self.sc = M.ConvLayer(1, out, usebias=False, stride=stride, batch_norm=True)

	def forward(self, x):
		branch = self.c1(x)
		branch = self.c2(branch)
		if self.shortcut:
			sc = self.sc(x)
		else:
			sc = x 
		res = branch + sc
		res = M.activation(res, M.PARAM_RELU)
		return res 

class ResBasicBlock(M.Model):
	def initialize(self, out, num_units):
		self.units = nn.ModuleList()
		for i in range(num_units):
			self.units.append(BasicUnit(out, 1))
	def forward(self, x):
		for unit in self.units:
			x = unit(x)
		return x 

class Transition(M.Model):
	def initialize(self, outchns, strides):
		self.trans = nn.ModuleList()
		for i,(o,s) in enumerate(zip(outchns,strides)):
			if o is None or s is None:
				self.trans.append(None)
			elif s==1:
				self.trans.append(M.ConvLayer(3,o, stride=s, pad='SAME_LEFT', activation=M.PARAM_RELU, usebias=False, batch_norm=True))
			else:
				self.trans.append(M.ConvLayer(3,o, stride=s, pad='SAME_LEFT', activation=M.PARAM_RELU, usebias=False, batch_norm=True))

	def forward(self, x):
		results = []
		for i,t in enumerate(self.trans):
			if t is None:
				results.append(x[i])
			else:
				results.append(t(x[-1]))
		return results

class FuseDown(M.Model):
	def initialize(self, steps, inp, o):
		self.mods = nn.ModuleList()
		for i in range(steps):
			if i==(steps-1):
				self.mods.append(M.ConvLayer(3, o, stride=2, pad='SAME_LEFT', batch_norm=True, usebias=False))
			else:
				self.mods.append(M.ConvLayer(3, inp, stride=2, pad='SAME_LEFT', activation=M.PARAM_RELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for m in self.mods:
			x = m(x)
		return x 

class FuseUp(M.Model):
	def initialize(self, o):
		self.c1 = M.ConvLayer(1, o, batch_norm=True, usebias=False)
	def forward(self, x, target_shape):
		x = F.interpolate(x, size=target_shape, mode='nearest')
		x = self.c1(x)
		return x 

class Fuse(M.Model):
	def initialize(self,outchns):
		branches = nn.ModuleList()
		for i in range(len(outchns)): # target
			branch = nn.ModuleList()
			for j in range(len(outchns)): # source
				if i==j:
					branch.append(None)
				elif i<j:
					branch.append(FuseUp(outchns[i]))
				else:
					branch.append(FuseDown(i-j, outchns[j], outchns[i]))
			branches.append(branch)
		self.branches = branches
	def forward(self, x):
		out = []
		for i in range(len(self.branches)): # target
			branch_out = []
			for j in range(len(self.branches)): # source
				if i==j:
					branch_out.append(x[i])
				elif i<j:
					branch_out.append(self.branches[i][j](x[j] , target_shape=x[i].shape[2:4]))
				else:
					branch_out.append(self.branches[i][j](x[j]))
			branch_out = sum(branch_out)
			out.append(M.activation(branch_out, M.PARAM_RELU))
		return out 

class FuseLast(M.Model):
	def initialize(self, outchns):
		self.c1 = FuseUp(outchns[0])
		self.c2 = FuseUp(outchns[0])
		self.c3 = FuseUp(outchns[0])
		self.c_all = M.ConvLayer(3, outchns[0]*4, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
	def forward(self, x):
		out = [x[0]]
		out.append(self.c1(x[1], x[0].shape[2:4]))
		out.append(self.c2(x[2], x[0].shape[2:4]))
		out.append(self.c3(x[3], x[0].shape[2:4]))
		# out = sum(out)
		# out = M.activation(out, M.PARAM_RELU)
		out = torch.cat(out, dim=1)
		return out 

class Stage(M.Model):
	def initialize(self, outchns, strides, num_units, num_fuses, is_last_stage=False, d=False):
		self.d = d 
		self.is_last_stage = is_last_stage
		self.num_fuses = num_fuses
		self.transition = Transition(outchns, strides)
		self.blocks = nn.ModuleList()
		self.fuses = nn.ModuleList()
		for j in range(num_fuses):
			block = nn.ModuleList()
			for i in range(len(outchns)):
				block.append(ResBasicBlock(outchns[i], num_units))
			self.blocks.append(block)
			if not (self.d and j==(self.num_fuses-1)):
				self.fuses.append(Fuse(outchns))
			
	def forward(self, x ):
		x = self.transition(x)
		for i in range(self.num_fuses):
			out = []
			for o,b in zip(x, self.blocks[i]):
				out.append(b(o))
			if not (self.d and i==(self.num_fuses-1)):
				x = self.fuses[i](out)
			else:
				x = out 
		return x 

class Body(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(3, 64, pad='SAME_LEFT', stride=2, activation=M.PARAM_RELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer(3, 64, pad='SAME_LEFT', stride=2, activation=M.PARAM_RELU, usebias=False, batch_norm=True)
		self.layer1 = ResBlock(256, 1, 4)
		self.stage1 = Stage([32, 64], [1, 2], 4, 1)
		self.stage2 = Stage([32, 64, 128], [None, None, 2], 4, 4)
		self.stage3 = Stage([32, 64, 128, 256], [None,None,None,2], 4, 3)
		# self.lastfuse = FuseLast([32,64,128,256])

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.layer1(x)
		x = self.stage1([x,x])
		x = self.stage2(x)
		x = self.stage3(x)
		# x = self.lastfuse(x)
		return x

class HRNET(M.Model):
	def initialize(self, num_pts):
		self.backbone = Body()
		self.lastconv = M.ConvLayer(1, num_pts)
	def forward(self, x):
		x = self.backbone(x)
		x = self.lastconv(x)
		return x

class AdaptConvBlk(M.Model):
	def initialize(self):
		self.c = M.AdaptConv3(15, batch_norm=True, usebias=False)
	
	def forward(self, x):
		branch = self.c(x)
		x = branch + x 
		x = torch.relu(x)
		return x 

class DEKR(M.Model):
	def initialize(self, num_pts):
		self.backbone = Body()

		self.transition_hmap = M.ConvLayer(1, 32, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.head_hmap = ResBasicBlock(32, 1)
		self.conv_hmap = M.ConvLayer(1, num_pts+1)

		self.transition_off = M.ConvLayer(1, 15*num_pts, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.reg_blks_off = nn.ModuleList()
		self.convs_off = nn.ModuleList()
		for _ in range(num_pts):
			self.reg_blks_off.append(AdaptConvBlk())
			self.reg_blks_off.append(AdaptConvBlk())
			self.convs_off.append(M.ConvLayer(1, 2))

	def forward(self, x):
		x = self.backbone(x)
		h, w = x[0].shape[2], x[0].shape[3]
		x = torch.cat([x[0], F.interpolate(x[1], (h,w), mode='bilinear'), \
			F.interpolate(x[2], (h,w), mode='bilinear'), F.interpolate(x[3], (h,w), mode='bilinear')], dim=1)
		# x0_h, x0_w = x[0].size(2), x[0].size(3)
		# x = torch.cat([x[0], \
		# 	F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear'), \
		# 	F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear'), \
		# 	F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')], 1)
		# return x 

		hmap = self.conv_hmap(self.head_hmap(self.transition_hmap(x)))

		offsets = []
		off = self.transition_off(x)
		for i in range(len(self.convs_off)):
			o = self.reg_blks_off[i*2](off[:,i*15:i*15+15])
			o = self.reg_blks_off[i*2+1](o)
			o = self.convs_off[i](o)
			offsets.append(o)
		offsets = torch.cat(offsets, dim=1)
		return hmap, offsets

if __name__=='__main__':
	net = HRNET(17)
	M.Saver(net).restore('./model_imagenet/')
	M.Saver(net.backbone).save('./model_body_imagenet/w48.pth')
