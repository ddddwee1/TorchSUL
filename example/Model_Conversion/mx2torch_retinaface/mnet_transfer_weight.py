import torch 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import numpy as np 
import source 

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

if __name__=='__main__':
	net = Detector()
	net.eval()

	x = torch.from_numpy(np.ones([1,3,640,640]).astype(np.float32))
	_ = net(x)
	# net.bn_eps(1e-5)
	# net.backbone.det1.bn_eps(2e-5)

	res = {}
	ps = net.named_parameters()
	for p in ps:
		name, p = p 
		res[name] = p
	ps = net.named_buffers()
	for p in ps:
		name, p = p 
		res[name] = p
	# print(res.keys())

	def get_conv(l1, l2, bias=False):
		a = [l1 + '.weight']
		b = [l2 + '_weight']
		if bias:
			a.append(l1+'.bias')
			b.append(l2+'_bias')
		return a,b

	def get_bn(l1, l2):
		a = []
		b = []
		a.append(l1+'.weight')
		a.append(l1+'.bias')
		a.append(l1+'.running_mean')
		a.append(l1+'.running_var')
		b.append(l2+'_gamma')
		b.append(l2+'_beta')
		b.append(l2+'_running_mean')
		b.append(l2+'_running_var')
		return a, b

	def get_bn2(l1, l2):
		a = []
		b = []
		a.append(l1+'.weight')
		a.append(l1+'.bias')
		a.append(l1+'.running_mean')
		a.append(l1+'.running_var')
		b.append(l2+'_gamma')
		b.append(l2+'_beta')
		b.append(l2+'_moving_mean')
		b.append(l2+'_moving_var')
		return a, b

	def get_layer(l1, l2, bias=False):
		res = []
		res.append(get_conv(l1 + '.conv', l2%('conv')))
		res.append(get_bn(l1 + '.bn', l2%('batchnorm')))
		return res 

	def get_convbn(l1, l2, bias=False):
		res = []
		res.append(get_conv(l1 + '.conv', l2, bias=bias))
		res.append(get_bn2(l1 + '.bn', l2 + '_bn'))
		return res 

	def get_dethead(l1, l2):
		res = []
		res += get_convbn(l1+'.c11', l2+'_conv1', bias=True)
		res += get_convbn(l1+'.c21', l2+'_context_conv1', bias=True)
		res += get_convbn(l1+'.c22', l2+'_context_conv2', bias=True)
		res += get_convbn(l1+'.c31', l2+'_context_conv3_1', bias=True)
		res += get_convbn(l1+'.c32', l2+'_context_conv3_2', bias=True)
		return res 

	def get_regress(l1, l2):
		res = []
		res.append(get_conv(l1+'.c1.conv', l2%('cls_score'), bias=True))
		res.append(get_conv(l1+'.c2.conv', l2%('bbox_pred'), bias=True))
		res.append(get_conv(l1+'.c3.conv', l2%('landmark_pred'), bias=True))
		return res 

	def totonoi(l):
		a = []
		b = []
		for i in l:
			a += i[0]
			b += i[1]
		return a,b

	l = []
	l += get_layer('backbone.c1', 'mobilenet0_%s0')
	l += get_layer('backbone.c2', 'mobilenet0_%s1')
	l += get_layer('backbone.c3', 'mobilenet0_%s2')
	l += get_layer('backbone.c4', 'mobilenet0_%s3')
	l += get_layer('backbone.c5', 'mobilenet0_%s4')
	l += get_layer('backbone.c6', 'mobilenet0_%s5')
	l += get_layer('backbone.c7', 'mobilenet0_%s6')
	l += get_layer('backbone.c8', 'mobilenet0_%s7')
	l += get_layer('backbone.c9', 'mobilenet0_%s8')
	l += get_layer('backbone.c10', 'mobilenet0_%s9')
	l += get_layer('backbone.c11', 'mobilenet0_%s10')

	l += get_layer('backbone.c12', 'mobilenet0_%s11')
	l += get_layer('backbone.c13', 'mobilenet0_%s12')
	l += get_layer('backbone.c14', 'mobilenet0_%s13')
	l += get_layer('backbone.c15', 'mobilenet0_%s14')
	l += get_layer('backbone.c16', 'mobilenet0_%s15')
	l += get_layer('backbone.c17', 'mobilenet0_%s16')
	l += get_layer('backbone.c18', 'mobilenet0_%s17')
	l += get_layer('backbone.c19', 'mobilenet0_%s18')
	l += get_layer('backbone.c20', 'mobilenet0_%s19')
	l += get_layer('backbone.c21', 'mobilenet0_%s20')
	l += get_layer('backbone.c22', 'mobilenet0_%s21')
	l += get_layer('backbone.c23', 'mobilenet0_%s22')

	l += get_layer('backbone.c24', 'mobilenet0_%s23')
	l += get_layer('backbone.c25', 'mobilenet0_%s24')
	l += get_layer('backbone.c26', 'mobilenet0_%s25')
	l += get_layer('backbone.c27', 'mobilenet0_%s26')

	l += get_convbn('backbone.rf_c3_lateral', 'rf_c3_lateral', bias=True)
	l += get_convbn('backbone.rf_c2_lateral', 'rf_c2_lateral', bias=True)
	l += get_convbn('backbone.rf_c1_red_conv', 'rf_c1_red_conv', bias=True)
	l += get_convbn('backbone.rf_c2_aggr', 'rf_c2_aggr', bias=True)
	l += get_convbn('backbone.rf_c1_aggr', 'rf_c1_aggr', bias=True)

	l += get_dethead('backbone.det3', 'rf_c3_det')
	l += get_dethead('backbone.det2', 'rf_c2_det')
	l += get_dethead('backbone.det1', 'rf_c1_det')

	l += get_regress('backbone.head32', 'face_rpn_%s_stride32')
	l += get_regress('backbone.head16', 'face_rpn_%s_stride16')
	l += get_regress('backbone.head8' , 'face_rpn_%s_stride8')

	a,b = totonoi(l)

	for i,j in zip(a,b):
		print(i,j)
		value = source.res[j].asnumpy()
		print(value.shape)
		print(res[i].shape)
		res[i].data[:] = torch.from_numpy(value)[:]

	y = net(x)
	print(y)
	print(y.shape)

	M.Saver(net).save('./model/mbnet.pth')
