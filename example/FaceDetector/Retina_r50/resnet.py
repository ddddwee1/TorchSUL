import numpy as np 
from TorchSUL import Model as M 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Unit(M.Model):
	def initialize(self, chn, stride=1, shortcut=False):
		self.bn0 = M.BatchNorm()
		self.act = M.Activation(M.PARAM_RELU)
		self.c1 = M.ConvLayer(1, chn, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c2 = M.ConvLayer(3, chn, stride=stride, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c3 = M.ConvLayer(1, chn*4, usebias=False)
		self.shortcut = shortcut
		if shortcut:
			self.sc = M.ConvLayer(1, chn*4, stride=stride, usebias=False)

	def forward(self, inp):
		if self.shortcut:
			inp = self.bn0(inp)
			inp = self.act(inp)
			x2 = x = self.c1(inp)
			x = self.c2(x)
			x = self.c3(x)
			sc = self.sc(inp)
			x = sc + x 
		else:
			x = self.bn0(inp)
			x = self.act(x)
			x2 = x = self.c1(x)
			x = self.c2(x)
			x = self.c3(x)
			x = inp + x 
		return x, x2 

class Stage(M.Model):
	def initialize(self, outchn, num_units, stride):
		self.units = nn.ModuleList()
		for i in range(num_units):
			self.units.append(Unit(outchn, stride=stride if i==0 else 1, shortcut = i==0))

	def forward(self, x):
		for i,u in enumerate(self.units):
			if i==0:
				x, x2 = u(x)
			else:
				x, _ = u(x)
		return x, x2

class DETHead(M.Model):
	def initialize(self):
		self.c11 = M.ConvLayer(3, 256, batch_norm=True)
		self.c21 = M.ConvLayer(3, 128, batch_norm=True, activation=M.PARAM_RELU)
		self.c22 = M.ConvLayer(3, 128, batch_norm=True)
		self.c31 = M.ConvLayer(3, 128, batch_norm=True, activation=M.PARAM_RELU)
		self.c32 = M.ConvLayer(3, 128, batch_norm=True)
		self.act = M.Activation(M.PARAM_RELU)

	def forward(self, x):
		x1 = self.c11(x)
		x2 = self.c21(x)
		x3 = self.c31(x2)
		x3 = self.c32(x3)
		x2 = self.c22(x2)
		x = torch.cat([x1, x2, x3], dim=1)
		x = self.act(x)
		return x 

class RegressHead(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(1,4)
		self.c2 = M.ConvLayer(1,8)
		self.c3 = M.ConvLayer(1,20)

	def forward(self, x):
		prob = self.c1(x)
		bbox = self.c2(x)
		kpts = self.c3(x)
		prob = prob.view(prob.shape[0],2,prob.shape[2]*2,prob.shape[3])
		prob = F.softmax(prob, dim=1)
		prob = prob.view(prob.shape[0],4,-1, prob.shape[3])
		return prob, bbox, kpts

class Detector(M.Model):
	def initialize(self):
		self.bn0 = M.BatchNorm()
		self.c1 = M.ConvLayer(7, 64, stride=2, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.pool = M.MaxPool2D(3, 2)
		self.stage1 = Stage(64, num_units=3, stride=1)
		self.stage2 = Stage(128, num_units=4, stride=2)
		self.stage3 = Stage(256, num_units=6, stride=2)
		self.stage4 = Stage(512, num_units=3, stride=2)
		self.bn1 = M.BatchNorm()
		self.act = M.Activation(M.PARAM_RELU)

		self.ssh_c3_lateral = M.ConvLayer(1, 256, batch_norm=True, activation=M.PARAM_RELU)
		self.det3 = DETHead()
		self.head32 = RegressHead()

		self.ssh_c2_lateral = M.ConvLayer(1, 256, batch_norm=True, activation=M.PARAM_RELU)
		self.ssh_c3_upsampling = M.NNUpSample(2)
		self.ssh_c2_aggr = M.ConvLayer(3, 256, batch_norm=True, activation=M.PARAM_RELU)
		self.det2 = DETHead()
		self.head16 = RegressHead()

		self.ssh_m1_red_conv = M.ConvLayer(1, 256, batch_norm=True, activation=M.PARAM_RELU)
		self.ssh_c2_upsampling = M.NNUpSample(2)
		self.ssh_c1_aggr = M.ConvLayer(3, 256, batch_norm=True, activation=M.PARAM_RELU)
		self.det1 = DETHead()
		self.head8 = RegressHead()


	def forward(self, x):
		x = self.bn0(x)
		x = self.c1(x)
		x = self.pool(x)
		x, _ = self.stage1(x)
		x, _ = self.stage2(x)
		x, f1 = self.stage3(x)
		x, f2 = self.stage4(x)
		x = self.bn1(x)
		x = self.act(x)

		fc3 = x = self.ssh_c3_lateral(x)
		d3 = x = self.det3(x)
		scr32, box32, lmk32 = self.head32(d3)

		fc2 = self.ssh_c2_lateral(f2)
		x = self.ssh_c3_upsampling(fc3)
		x = x[:,:,:fc2.shape[2],:fc2.shape[3]]
		plus100 = x = fc2 + x
		fc2_aggr = x = self.ssh_c2_aggr(x)
		d2 = x = self.det2(x)
		scr16, box16, lmk16 = self.head16(d2)

		fc1 = self.ssh_m1_red_conv(f1)
		x = self.ssh_c2_upsampling(fc2_aggr)
		x = x[:,:,:fc1.shape[2],:fc1.shape[3]]
		x = fc1 + x 
		fc1_aggr = x = self.ssh_c1_aggr(x)
		d1 = x = self.det1(x)
		scr8, box8, lmk8 = self.head8(d1)

		results = [scr32, box32, lmk32, scr16, box16, lmk16, scr8, box8, lmk8]
		return results

