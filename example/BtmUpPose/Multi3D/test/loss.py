import torch 
from TorchSUL import Model as M 
import torch.nn as nn 
import config 

class AELoss(M.Model):
	def inst_ae(self, idmap, pts, debugmap=None):
		area = 0
		pull_loss = 0

		conf = pts[:,:,2]
		pts = pts.long()
		tags_means = []
		for i in range(config.max_inst):
			tags = []
			for j in range(config.num_pts):
				if conf[i,j]>0:
					x,y = pts[i,j,0], pts[i,j,1]
					for xx in range(x-area, x+area+1):
						for yy in range(y-area, y+area+1):
							if xx<0 or xx>=config.out_size or yy<0 or yy>=config.out_size:
								continue 
							tag = idmap[j,yy,xx]
							tags.append(tag)
							if debugmap is not None:
								debugmap[j,yy,xx] = 1
			if len(tags)==0:
				continue
			tags = torch.stack(tags)
			tags_mean = tags.mean()
			intra = torch.pow(tags - tags_mean, 2).mean()
			pull_loss = pull_loss + intra
			tags_means.append(tags_mean)

		num_tags = len(tags_means)

		if num_tags==0:
			return torch.zeros(1)[0].float().to(idmap.device), torch.zeros(1)[0].float().to(idmap.device)
		
		pull_loss = pull_loss / num_tags
		if num_tags==1:
			return torch.zeros(1)[0].float().to(idmap.device), pull_loss

		# tags_means = torch.stack(tags_means)
		# A = tags_means.unsqueeze(0)
		# B = A.permute(0,1)
		# diff = torch.triu(A-B, diagonal=1)
		# push = torch.sum(torch.exp(- torch.pow(diff, 2)))
		# push_loss = push / ((num_tags-1) * num_tags)

		# A = tags_means.expand(num_tags, num_tags)
		# B = A.permute(0, 1)
		# diff = torch.pow(A-B, 2)
		# push = torch.sum(torch.exp(-diff)) - num_tags
		# push_loss = push * 0.5 / ((num_tags-1) * num_tags)

		push_loss = 0
		for i in range(num_tags):
			for j in range(num_tags):
				if i!=j:
					diff = torch.pow(tags_means[i] - tags_means[j], 2)
					diff = torch.exp(-diff)
					push_loss = push_loss + diff
		push_loss = push_loss * 0.5 / ((num_tags-1) * num_tags)

		return push_loss, pull_loss

	def forward(self, idmap, pts, debugmaps=None):
		bsize = idmap.shape[0]
		push_loss = []
		pull_loss = []
		for i in range(bsize):
			push, pull = self.inst_ae(idmap[i], pts[i], debugmap=None if debugmaps is None else debugmaps[i])
			push_loss.append(push)
			pull_loss.append(pull)
		push_loss = torch.stack(push_loss)
		pull_loss = torch.stack(pull_loss)
		return push_loss, pull_loss

class DepthLoss(M.Model):
	def inst_dep(self, depout, depth):
		area = 1
		pts = depth[:,:2]
		d = depth[:,2]
		pts = pts.long()
		total_loss = 0
		counter = 0 
		for i in range(depth.shape[0]):
			x,y = pts[i,0], pts[i,1]
			gt = d[i]
			for xx in range(x-area, x+area+1):
				for yy in range(y-area, y+area+1):
					if xx<0 or xx>=config.out_size or yy<0 or yy>=config.out_size:
						continue 
					val = depout[0,yy,xx]
					ls = torch.mean(torch.pow( val - gt , 2))
					total_loss = total_loss + ls 
					counter += 1 
		if counter == 0:
			return torch.zeros(1)[0].float().to(depout.device)
		else:
			total_loss = total_loss / counter
			return total_loss

	def forward(self, depout, depth):
		bsize = depout.shape[0]
		losses = []
		for i in range(bsize):
			ls = self.inst_dep(depout[i], depth[i])
			losses.append(ls)
		losses = torch.stack(losses)
		return losses 

class DepthAllLoss(M.Model):
	def inst_dep(self, depout, depth):
		area = 0
		pts = depth[:,:,:2]
		d = depth[:,:,2]
		pts = pts.long()
		total_loss = 0
		counter = 0 
		for i in range(depth.shape[0]):
			for j in range(config.num_pts-1):
				x,y = pts[i,j,0], pts[i,j,1]
				gt = d[i,j]
				for xx in range(x-area, x+area+1):
					for yy in range(y-area, y+area+1):
						if xx<0 or xx>=config.out_size or yy<0 or yy>=config.out_size:
							continue 
						val = depout[j,yy,xx]
						ls = torch.mean(torch.pow( val - gt , 2))
						total_loss = total_loss + ls 
						counter += 1 
		if counter == 0:
			return torch.zeros(1)[0].float().to(depout.device)
		else:
			total_loss = total_loss / counter
			return total_loss

	def forward(self, depout, depth):
		bsize = depout.shape[0]
		losses = []
		for i in range(bsize):
			ls = self.inst_dep(depout[i], depth[i])
			losses.append(ls)
		losses = torch.stack(losses)
		return losses 

class HmapLoss(M.Model):
	def forward(self, hmap, gt):
		# hmap = torch.sigmoid(hmap)
		loss = torch.pow(hmap - gt, 2)
		# loss = loss * (gt.detach() * 10 + 1)
		loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
		return loss 

class ModelWithLoss(M.Model):
	def initialize(self, model):
		self.AE = AELoss()
		self.HM = HmapLoss()
		self.RDEP = DepthLoss()
		self.DEP = DepthAllLoss()
		self.model = model 
	def forward(self, img, hmap, pts, depth, depth_all, debugmaps=None):
		# depth = depth.float()
		outs, idout, depout, depallout = self.model(img)
		push, pull = self.AE(idout, pts, debugmaps=debugmaps)
		hm = self.HM(outs, hmap)
		rdep = self.RDEP(depout, depth)
		dep = self.DEP(depallout, depth_all)
		return hm, push, pull, rdep, dep, outs, idout, depout, depallout
	def run(self, img):
		outs, idout = self.model(img)
		return outs, idout
