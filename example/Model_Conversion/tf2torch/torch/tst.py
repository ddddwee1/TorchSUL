import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import Model as M 
import hrnet 

class HRNET(M.Model):
	def initialize(self, num_pts):
		self.backbone = hrnet.ResNet()
		self.lastconv = M.ConvLayer(1, num_pts)
	def forward(self, x):
		x = self.backbone(x)
		x = self.lastconv(x)
		return x 

net = HRNET(17)
net.eval()
net.record()
print(net.lastconv.record)

dummy_inp = np.ones([1,3,256,256])
dummy_inp = np.float32(dummy_inp)
dummy_inp = torch.from_numpy(dummy_inp)

y = net(dummy_inp)
# print(y.shape)

import Layers as L 
vs = L.record_params
# print(len(L.record_params))
# print(vs[0].keys())

import pickle 
data = pickle.load(open('hrnet_variables.pkl' , 'rb'))
for vsrc, vtgt in zip(data, vs):
	print(vsrc.keys())
	print(vtgt.keys())
	print('------')
	for k in vsrc.keys():
		
		if 'kernel' in k:
			v = torch.from_numpy(np.transpose(vsrc[k], [3,2,0,1]))
			vtgt['conv.weight'].data[:] = v
		else:
			v = torch.from_numpy(vsrc[k])
		if 'bias' in k:
			vtgt['conv.bias'].data[:] = v 
		if 'gamma' in k:
			vtgt['bn.weight'].data[:] = v
		if 'beta' in k:
			vtgt['bn.bias'].data[:] = v 
		if 'moving_average' in k:
			vtgt['bn.running_mean'].data[:] = v 
		if 'variance' in k:
			vtgt['bn.running_var'].data[:] = v 

y = net(dummy_inp)
print(y)
print(y.shape)

M.Saver(net).save('./modeltorch/hrnet.pth')
