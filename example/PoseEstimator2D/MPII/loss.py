import torch 
from TorchSUL import Model as M 
import torch.nn as nn 
import config 
import torch.nn.functional as F 

class HmapLoss(M.Model):
	def forward(self, hmap, gt):
		# hmap = torch.sigmoid(hmap)
		loss = torch.pow(hmap - gt, 2)
		loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
		return loss

class ModelWithLoss(M.Model):
	def initialize(self, model):
		self.HM = HmapLoss()
		self.model = model 

	def forward(self, img, hmap, hmap_match):
		hmap_match = F.interpolate(hmap_match, (img.shape[2], img.shape[3]))
		inp = torch.cat([img, hmap_match], dim=1)
		outs = self.model(inp)
		# print(outs.shape, hmap.shape)
		hm = self.HM(outs, hmap)
		return hm, outs

	def run(self, img, hmap, hmap_match):
		inp = torch.cat([img, hmap_match], dim=1)
		outs = self.model(inp)
		return outs
