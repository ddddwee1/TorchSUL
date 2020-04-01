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

dummy_inp = np.ones([1,3,256,256], dtype=np.float32)
dummy_inp = torch.from_numpy(dummy_inp)
y = net(dummy_inp)

M.Saver(net).restore('./modeltorch/')

## Do what the fuck you want.
