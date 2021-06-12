import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from TorchSUL import Model as M 
import config 
import hrnet 
from torch.nn.parallel import replicate, scatter, parallel_apply, gather

def normalize_adj_mtx(mtx):
	with torch.no_grad():
		# we dont need to plus I matrix, because the diag is already 1.
		# I = torch.eye(mtx.shape[0])
		# mtx = mtx + I 
		# if mtx.shape[0]>0:
		# 	print(mtx.min())
		S = torch.sum(mtx, dim=1)
		# print(S)
		S = torch.sqrt(S)
		S = 1. / S
		S = torch.diag(S)
		
		# A_ = (mtx + I) 
		A_ = mtx 
		A_ = torch.mm(S, A_)
		A_ = torch.mm(A_, S)
	return A_

class DensityBranch(M.Model):
	def initialize(self, num_layers, channel, density_level):
		self.layers = nn.ModuleList()
		for i in range(num_layers):
			self.layers.append(M.ConvLayer(3, channel, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
		self.layers.append(M.ConvLayer(1, density_level))
	def forward(self, x):
		for l in self.layers:
			x = l(x)
		return x 

class SamplingLayer(M.Model):
	def build(self, *inputs):
		density_shape = inputs[1].shape 
		density_levels = density_shape[1]
		maph = density_shape[2]
		mapw = density_shape[3]
		self.xys = nn.ParameterList()
		for level in range(1, density_levels):
			sample_rate = level * 2 - 1 
			x = torch.linspace(-1.0, 1.0, steps=maph*sample_rate, dtype=torch.float32)
			y = torch.linspace(-1.0, 1.0, steps=mapw*sample_rate, dtype=torch.float32)
			xx, yy = torch.meshgrid(x, y)
			xx = xx.view(maph, sample_rate, mapw, sample_rate)
			xx = torch.transpose(xx, 1,2)
			xx = xx.reshape(maph, mapw, sample_rate*sample_rate)
			yy = yy.view(maph, sample_rate, mapw, sample_rate)
			yy = torch.transpose(yy, 1,2)
			yy = yy.reshape(maph, mapw, sample_rate*sample_rate)
			xy = torch.stack([yy,xx], dim=-1)
			xy = Parameter(xy, requires_grad=False)
			self.xys.append(xy)

	def build_forward(self, x, density):
		x = torch.zeros(x.shape[1], 10)
		affinity = torch.eye(10)
		coords = torch.zeros(10, 2)
		return x, affinity, coords
		
	def forward(self, x, density):
		density_levels = density.shape[1]
		with torch.no_grad():
			density = torch.argmax(density, dim=1) # N, H, W
		coords = []
		for i in range(1, density_levels):
			idx = torch.where(density==i)
			yidx, xidx = idx[1], idx[2]
			if len(yidx)==0:
				coords.append(torch.zeros(0,2).to(x.device))
			else:
				selected = self.xys[i-1][yidx, xidx, :, :]
				coords.append(selected.reshape(-1, 2))
		# sample at least one pixel for each pixel 
		coords.append(self.xys[0].reshape([-1, 2]))
		coords = torch.cat(coords, dim=0)
		coords_temp = coords.unsqueeze(0).unsqueeze(0)
		x = F.grid_sample(x, coords_temp, align_corners=False)
		x = x.squeeze(0)
		x = x.squeeze(1)
		# compute affinity based on Normal(L2)?
		# XX = torch.sum(torch.pow(coords, 2), dim=1, keepdim=True)
		# XY = torch.mm(coords, torch.transpose(coords, 0, 1))
		# dist = XX - 2*XY + torch.transpose(XX, 0, 1)
		# affinity = torch.exp( - dist * 0.5 / config.affinity_sigma)
		# affinity = F.threshold(affinity, config.affinity_threshold, 0.0, inplace=False)
		# affinity = normalize_adj_mtx(affinity)
		affinity = None
		return x, affinity, coords
	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		pass
	def _save_to_state_dict(self, destination, prefix, keep_vars):
		pass

# class GCNBranch(M.Model):
# 	def initialize(self, num_layers, channel, ):
# 		self.layers = nn.ModuleList()
# 		for i in range(num_layers):
# 			self.layers.append(M.GraphConvLayer(channel, activation=M.PARAM_PRELU, usebias=False, batch_norm=True, norm=False)) # we do Laplasian norm in previous step
# 		self.layers.append(M.GraphConvLayer((2 + 1) * config.num_pts, norm=False))
# 	def forward(self, x, adj):
# 		x = torch.transpose(x, 0, 1)
# 		for l in self.layers:
# 			x = l(x, adj, affinity_grad=False)
# 		return x 

class GCNBranch(M.Model):
	def initialize(self, num_layers, channel, final_chn):
		self.layers = nn.ModuleList()
		for i in range(num_layers):
			self.layers.append(M.Dense(channel, activation=M.PARAM_PRELU, usebias=False, batch_norm=True)) # we do Laplasian norm in previous step
		self.layers.append(M.Dense(final_chn))
	def forward(self, x, adj):
		x = torch.transpose(x, 0, 1)
		for l in self.layers:
			x = l(x)
		return x 

class Head(M.Model):
	def initialize(self, head_layernum, head_chn):
		self.layers = nn.ModuleList()
		for i in range(head_layernum):
			self.layers.append(M.ConvLayer(3, head_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for l in self.layers:
			x = l(x)
		return x 

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

class InstSegBranch(M.Model):
	def initialize(self, num_layers, channel, dim):
		self.layers = nn.ModuleList()
		for i in range(num_layers):
			self.layers.append(M.ConvLayer(3, channel, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
		self.layers.append(M.ConvLayer(1, dim, usebias=False))
	def forward(self, x):
		for l in self.layers:
			x = l(x)
		return x 

class DensityNet(M.Model):
	def initialize(self, density_num_layers, density_channels, density_level, gcn_layers, gcn_channels, head_layernum, head_chn, upsample_layers, upsample_chn):
		self.backbone = hrnet.Body()
		self.upsample = UpSample(upsample_layers, upsample_chn)
		self.head = Head(head_layernum, head_chn)
		self.head2 = Head(head_layernum, head_chn)
		# self.head_density = Head(head_layernum, head_chn)
		# self.density_branch = DensityBranch(density_num_layers, density_channels, config.num_pts*2)
		# self.id_branch = DensityBranch(density_num_layers, density_channels, config.id_featdim * config.num_pts)
		# self.density_branch = M.ConvLayer(1, config.num_pts)
		self.c1 = M.ConvLayer(1, config.num_pts)
		self.c2 = M.ConvLayer(1, config.num_pts)
		# self.c2 = M.ConvLayer(1, config.num_pts*2)
		# add one block to estimate the block 
		# add one block to estimate higher resolution (512)
		# just train them, think about merge in post-process 

	# def build_forward(self, x, *args, **kwargs):
	# 	feat = self.backbone(x)
	# 	feat = self.upsample(feat)
	# 	# feat = self.head(feat)
	# 	result = self.c2(feat)
	# 	nn.init.normal_(self.c2.conv.weight, std=0.001)
	# 	print('normal init for last conv ')
	# 	outs, idout = result[:,:config.num_pts], result[:,config.num_pts:]
	# 	return outs, idout

	def build_forward(self, x, *args, **kwargs):
		feat = self.backbone(x)
		feat = self.upsample(feat)
		feat1 = self.head(feat)
		feat2 = self.head2(feat)
		outs = self.c1(feat1)
		idout = self.c2(feat2)
		nn.init.normal_(self.c1.conv.weight, std=0.001)
		nn.init.normal_(self.c2.conv.weight, std=0.001)
		print('normal init for last conv ')
		return outs, idout
		
	def forward(self, x, density_only=False):
		feat = self.backbone(x)
		feat = self.upsample(feat)
		h1 = self.head(feat)
		h2 = self.head2(feat)
		# results = self.density_branch(feat)
		# idout = self.id_branch(feat)
		outs = self.c1(h1)
		idout = self.c2(h2)
		# result = self.c2(feat)
		# outs, idout = result[:,:config.num_pts], result[:,config.num_pts:]
		# result = torch.cat([outs, idout], dim=1)
		# result = self.c2(feat)
		return outs, idout

class ParallelDensityNet(M.Model):
	def initialize(self, module, device_ids):
		self.network = module
		self.device_ids = device_ids

	def forward(self, *inputs, **kwargs):
		inputs = scatter(inputs, self.device_ids, dim=0)
		kwargs = scatter(kwargs, self.device_ids, dim=0)
		replicas = replicate(self.network, self.device_ids[:len(inputs)])
		outputs = parallel_apply(replicas, inputs, kwargs)
		outputs = list(zip(*outputs))

		res = []
		for i in range(len(outputs)):
			buf = []
			for j in range(len(outputs[i])):
				if isinstance(outputs[i][j], int):
					if outputs[i][j]<0:
						buf.append(outputs[i][j])
				else:
					buf.append(outputs[i][j].to(self.device_ids[0]))
			res.append(buf)
		return res


