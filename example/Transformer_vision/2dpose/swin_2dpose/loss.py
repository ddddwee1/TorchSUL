import torch 
from TorchSUL import Model as M 
import torch.nn as nn 
import config 
import torch.nn.functional as F 

class SkeletalLoss(M.Model):
	def inst_skeleton(self, idmap, pts):
		diff_scale = []
		diff_angle = []

		area = 0
		scale = pts[:,2]
		ang = pts[:,3]
		conf = pts[:,4]
		pts = pts[:,:2].long()
		for i in range(pts.shape[0]):
			if conf[i]<=0:
				continue 
			x, y = pts[i,0], pts[i,1]
			for xx in range(x-area, x+area+1):
				for yy in range(y-area, y+area+1):
					if xx<0 or xx>=config.out_size or yy<0 or yy>=config.out_size:
						continue 
					tag_s = idmap[i*2, yy,xx]
					tag_a = idmap[i*2+1, yy,xx]
					diff_scale.append(tag_s - scale[i])
					diff_angle.append(tag_a - ang[i])
		diff_scale = torch.stack(diff_scale)
		diff_angle = torch.stack(diff_angle)
		diff_scale = torch.pow(diff_scale, 2).mean()
		diff_angle = torch.pow(diff_angle, 2).mean()
		return diff_scale, diff_angle

	def forward(self, idmap, pts):
		losses_angle = []
		losses_scale = []
		bsize = idmap.shape[0]
		for i in range(bsize):
			l_s, l_a = self.inst_skeleton(idmap[i], pts[i])
			losses_scale.append(l_s)
			losses_angle.append(l_a)
		losses_scale = torch.stack(losses_scale)
		losses_angle = torch.stack(losses_angle)
		return losses_scale, losses_angle

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

	def forward(self, img, hmap):
		outs = self.model(img)
		hm = self.HM(outs, hmap)
		return hm, outs

	def run(self, img, hmap, hmap_match):
		inp = torch.cat([img, hmap_match], dim=1)
		outs = self.model(inp)
		return outs
