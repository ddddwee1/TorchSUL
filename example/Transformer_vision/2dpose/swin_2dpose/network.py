from swin_transformer import SwinTransformer 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import TorchSUL.Model as M 
import numpy as np 
import config 

class FPN(M.Model):
	def initialize(self, chn, num_scales):
		self.convs = nn.ModuleList()
		for _ in range(num_scales):
			self.convs.append(M.ConvLayer(3, chn))
		self.final_conv = M.ConvLayer(3, chn)

	def forward(self, *fmaps):
		# xs is a list from resolution low to high
		assert len(fmaps)==len(self.convs)
		res = None
		for c,fmap in zip(self.convs, fmaps):
			if res is None:
				res = c(fmap)
			else:
				x = F.interpolate(res, fmap.shape[-2:], mode='nearest')
				res = x + c(fmap)
		res = self.final_conv(res)
		return res 

class DepthToSpace(M.Model):
	def initialize(self, block_size):
		self.block_size = block_size
	def forward(self, x):
		bsize, chn, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		assert chn%(self.block_size**2)==0, 'DepthToSpace: Channel must be divided by square(block_size)'
		x = x.view(bsize, -1, self.block_size, self.block_size, h, w)
		x = x.permute(0,1,4,2,5,3)
		x = x.reshape(bsize, -1, h*self.block_size, w*self.block_size)
		return x 

class UpSample(M.Model):
	def initialize(self, upsample_layers, upsample_chn):
		self.prevlayers = nn.ModuleList()
		#self.uplayer = M.DeConvLayer(3, upsample_chn, stride=2, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.uplayer = M.ConvLayer(3, upsample_chn*4, activation=M.PARAM_PRELU, usebias=False)
		self.d2s = DepthToSpace(2)
		self.postlayers = nn.ModuleList()
		for i in range(upsample_layers):
			self.prevlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
		for i in range(upsample_layers):
			self.postlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for p in self.prevlayers:
			x = p(x)
		x = self.uplayer(x)
		x = self.d2s(x)
		# print('UPUP', x.shape)
		for p in self.postlayers:
			x = p(x)
		return x 

class SWinNet(M.Model):
	def initialize(self):
		self.backbone = SwinTransformer()
		self.fpn = FPN(64, 4)
		self.upsample = UpSample(1, 32)
		self.final_conv = M.ConvLayer(1, config.num_pts)

	def forward(self, x):
		fmaps = self.backbone.forward_fmaps(x)
		fmap = self.fpn(*fmaps)
		fmap = self.upsample(fmap)
		out = self.final_conv(fmap)
		return out

if __name__=='__main__':
	net = SWinNet()
	
	x = torch.zeros(1, 3, 384, 384)
	y = net(x)
	print(y.shape)
