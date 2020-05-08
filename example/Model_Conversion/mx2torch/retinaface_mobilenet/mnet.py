import torch 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import numpy as np 

class DETHead(M.Model):
	def initialize(self):
		self.c11 = M.ConvLayer(3, 32, batch_norm=True)
		self.c21 = M.ConvLayer(3, 16, batch_norm=True, activation=M.PARAM_RELU)
		self.c22 = M.ConvLayer(3, 16, batch_norm=True)
		self.c31 = M.ConvLayer(3, 16, batch_norm=True, activation=M.PARAM_RELU)
		self.c32 = M.ConvLayer(3, 16, batch_norm=True)
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

class Backbone(M.Model):
	def initialize(self):
		self.c1 = M.ConvLayer(3, 8, stride=2, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c2 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c3 = M.ConvLayer(1, 16, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c4 = M.DWConvLayer(3, 1, stride=2, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c5 = M.ConvLayer(1, 32, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c6 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c7 = M.ConvLayer(1, 32, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c8 = M.DWConvLayer(3, 1, stride=2, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c9 = M.ConvLayer(1, 64, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c10 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c11 = M.ConvLayer(1, 64, usebias=False, batch_norm=True, activation=M.PARAM_RELU)

		self.c12 = M.DWConvLayer(3, 1, stride=2, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c13 = M.ConvLayer(1, 128, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c14 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c15 = M.ConvLayer(1, 128, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c16 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c17 = M.ConvLayer(1, 128, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c18 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c19 = M.ConvLayer(1, 128, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c20 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c21 = M.ConvLayer(1, 128, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c22 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c23 = M.ConvLayer(1, 128, usebias=False, batch_norm=True, activation=M.PARAM_RELU)

		self.c24 = M.DWConvLayer(3, 1, stride=2, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c25 = M.ConvLayer(1, 256, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c26 = M.DWConvLayer(3, 1, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.c27 = M.ConvLayer(1, 256, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
		self.bn_eps(1e-5)

		self.rf_c3_lateral = M.ConvLayer(1, 64, batch_norm=True, activation=M.PARAM_RELU)
		self.rf_c3_lateral.bn_eps(2e-5)
		self.det3 = DETHead()
		self.det3.bn_eps(2e-5)

		self.rf_c2_lateral = M.ConvLayer(1, 64, batch_norm=True, activation=M.PARAM_RELU)
		self.rf_c2_lateral.bn_eps(2e-5)
		self.rf_c3_upsampling = M.NNUpSample(2)
		self.rf_c2_aggr = M.ConvLayer(3, 64, batch_norm=True, activation=M.PARAM_RELU)
		self.rf_c2_aggr.bn_eps(2e-5)
		self.det2 = DETHead()
		self.det2.bn_eps(2e-5)

		self.rf_c1_red_conv = M.ConvLayer(1, 64, batch_norm=True, activation=M.PARAM_RELU)
		self.rf_c1_red_conv.bn_eps(2e-5)
		self.rf_c2_upsampling = M.NNUpSample(2)
		self.rf_c1_aggr = M.ConvLayer(3, 64, batch_norm=True, activation=M.PARAM_RELU)
		self.rf_c1_aggr.bn_eps(2e-5)
		self.det1 = DETHead()
		self.det1.bn_eps(2e-5)
		
		self.head32 = RegressHead()
		self.head16 = RegressHead()
		self.head8 = RegressHead()

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = self.c4(x)
		x = self.c5(x)
		x = self.c6(x)
		x = self.c7(x)
		x = self.c8(x)
		x = self.c9(x)
		x = self.c10(x)
		f1 = x = self.c11(x)

		x = self.c12(x)
		x = self.c13(x)
		x = self.c14(x)
		x = self.c15(x)
		x = self.c16(x)
		x = self.c17(x)
		x = self.c18(x)
		x = self.c19(x)
		x = self.c20(x)
		x = self.c21(x)
		x = self.c22(x)
		f2 = x = self.c23(x)

		x = self.c24(x)
		x = self.c25(x)
		x = self.c26(x)
		x = self.c27(x)
		fc3 = x = self.rf_c3_lateral(x)
		d3 = x = self.det3(x)

		fc2 = self.rf_c2_lateral(f2)
		x = self.rf_c3_upsampling(fc3)
		
		x = x[:,:,:fc2.shape[2],:fc2.shape[3]]
		x = fc2 + x
		fc2_aggr = x = self.rf_c2_aggr(x)
		d2 = x = self.det2(x)

		fc1 = self.rf_c1_red_conv(f1)
		x = self.rf_c2_upsampling(fc2_aggr)
		
		x = x[:,:,:fc1.shape[2],:fc1.shape[3]]
		x = fc1 + x 
		fc1_aggr = x = self.rf_c1_aggr(x)
		d1 = x = self.det1(x)

		scr32, box32, lmk32 = self.head32(d3)
		scr16, box16, lmk16 = self.head16(d2)
		scr8, box8, lmk8 = self.head8(d1)

		results = [scr32, box32, lmk32, scr16, box16, lmk16, scr8, box8, lmk8]

		return results

class Detector(M.Model):
	def initialize(self):
		self.backbone = Backbone()
	def forward(self, x):
		x = self.backbone(x)
		return x 
