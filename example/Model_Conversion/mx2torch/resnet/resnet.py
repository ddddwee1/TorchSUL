import Model as M 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import pickle 

class ResBlock_v1(M.Model):
	def initialize(self, outchn, stride):
		self.stride = stride
		self.outchn = outchn
		self.bn0 = M.BatchNorm()
		self.c1 = M.ConvLayer(3, outchn, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)
		self.c2 = M.ConvLayer(3, outchn, stride=stride, usebias=False, batch_norm=True)

		# se module 
		#self.c3 = M.ConvLayer(1, outchn//16, activation=M.PARAM_PRELU)
		#self.c4 = M.ConvLayer(1, outchn, activation=M.PARAM_SIGMOID)

		# shortcut 
		self.sc = M.ConvLayer(1, outchn, stride=stride, usebias=False, batch_norm=True)

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
		self.c1 = M.ConvLayer(3, channel_list[0], 1, usebias=False, activation=M.PARAM_PRELU, batch_norm=True)
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
		x = self.fc1(x)
		return x 

def Res50():
	return ResNet([64,64,128,256,512],[3,4,14,3],512)
	
def Res100():
	return ResNet([64,64,128,256,512],[3,13,30,3],512)

if __name__=='__main__':
	net = Res100()
	net.eval()
	x = (np.float32(np.ones([1,3,112,112])) * 255 - 127.5) / 128 
	x = torch.from_numpy(x)
	y = net(x)

	res = {}

	ps = net.named_parameters()
	for p in ps:
		# print(p)
		name, p = p 
		res[name] = p
	ps = net.named_buffers()
	for p in ps:
		# print(p)
		name, p = p 
		res[name] = p

	print(res.keys())
	saved_params = pickle.load(open('params.pkl', 'rb'))

	# assign weights 
	def get_conv(l1, l2):
		a = l1 + '.weight'
		b = l2 + '_weight'
		return [a], [b]

	def get_bn(l1, l2):
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

	def get_act(l1, l2):
		a = l1 + '.weight'
		b = l2 + '_gamma'
		return [a], [b]

	def totonoi(l):
		a = []
		b = []
		for i in l:
			a += i[0]
			b += i[1]
		return a,b

	def get_unit(l1, l2, sc=False):
		res = []
		res.append(get_bn(l1+'.bn0', l2+'_bn1'))
		res.append(get_conv(l1+'.c1.conv', l2+'_conv1'))
		res.append(get_bn(l1+'.c1.bn', l2+'_bn2'))
		res.append(get_act(l1+'.c1.act', l2+'_relu1'))
		res.append(get_conv(l1+'.c2.conv', l2+'_conv2'))
		res.append(get_bn(l1+'.c2.bn', l2+'_bn3'))
		if sc:
			res.append(get_conv(l1+'.sc.conv', l2+'_conv1sc'))
			res.append(get_bn(l1+'.sc.bn', l2+'_sc'))
		return res 

	def get_stage(l1, l2, blocknum):
		res = []
		for i in range(blocknum):
			res += get_unit(l1+'.units.%d'%i, l2+'_unit%d'%(i+1), sc= i==0)
		return res 

	l = []
	l.append(get_conv('c1.conv','conv0'))
	l.append(get_bn('c1.bn', 'bn0'))
	l.append(get_act('c1.act', 'relu0'))
	l += get_stage('stage1', 'stage1', 3)
	l += get_stage('stage2', 'stage2', 13)
	l += get_stage('stage3', 'stage3', 30)
	l += get_stage('stage4', 'stage4', 3)
	l.append(get_bn('bn1', 'bn1'))
	l.append(get_conv('fc1.fc', 'pre_fc1'))
	l.append(get_bn('fc1.bn', 'fc1'))
	a,b = totonoi(l)

	for i,j in zip(a,b):
		res[i].data[:] = torch.from_numpy(saved_params[j])

	y = net(x)
	print(y)
	print(y[0,100])

	saver = M.Saver(net)
	saver.save('./saved_model/converted.pth')
