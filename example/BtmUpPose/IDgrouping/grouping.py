import torch 
import numpy as np 
import torch.nn as nn 
from TorchSUL import Model as M 
import pickle 
import config 
import matplotlib.pyplot as plt 
import cv2 

class NMS(M.Model):
	def initialize(self, ksize):
		self.pool = nn.MaxPool2d(ksize, 1, (ksize-1)//2)

	def forward(self, x):
		maxmap = self.pool(x)
		mask = torch.eq(maxmap, x).float()
		return x * mask

class Parser():
	def __init__(self):
		self.nms = NMS(5)

	def _min_dist(self, feats_dict, query):
		# query: dim
		# mean_feats: [ids, dim]
		mean_feats = []
		keys = []
		for k in feats_dict:
			buff = []
			for pt in feats_dict[k]:
				buff.append(feats_dict[k][pt])
			buff = torch.stack(buff, 0)
			m = torch.mean(buff, dim=0)
			mean_feats.append(m)
			keys.append(k)
		mean_feats = torch.stack(mean_feats, 0)
		dist = torch.norm(mean_feats - query, dim=1)
		minval, argmin = torch.min(dist, 0)
		return minval, keys[argmin]

	def _parse_idx_dict(self, idx_dict):
		res = {}
		for k in idx_dict:
			inst = idx_dict[k]
			buff = {}
			for pt in inst:
				idx = inst[pt].cpu().numpy()
				x = idx % config.out_size
				y = idx // config.out_size
				buff[pt] = [x,y]
			res[k] = buff
		return res 

	def grouping(self, outmap, idmap):
		results = self.nms(outmap)
		results = results.view(config.num_pts, config.out_size*config.out_size)
		vals, idx = results.topk(config.max_inst, dim=1)

		# print(idmap.shape)
		idmap = idmap.reshape(config.num_pts, config.out_size*config.out_size, 1)
		# print(idmap.shape)
		id_selected = []
		for i in range(config.id_featdim):
			ids = torch.gather(idmap[:,:,i], dim=1, index=idx)
			id_selected.append(ids)
		id_selected = torch.stack(id_selected, -1)

		res = {}
		feats = {}
		scores = {}
		for i in range(config.num_pts):
			if len(feats)==0:
				for pt in range(config.max_inst):
					if vals[i, pt]>config.conf_thresh:
						res[len(res)] = {i:idx[i,pt]}
						feats[len(feats)] = {i:id_selected[i,pt]}
						scores[len(scores)] = {i:vals[i,pt]}
			else:
				for pt in range(config.max_inst):
					if vals[i, pt]>config.conf_thresh:

						minval, k = self._min_dist(feats, id_selected[i,pt])
						if i==13 or i==14:
							x = idx[i,pt] % config.out_size
							y = idx[i,pt] // config.out_size
							# print(i, vals[i,pt], minval, x, y)

						if minval<config.id_dist_thresh:
							if not i in res[k]:
								res[k][i] = idx[i,pt]
								feats[k][i] = id_selected[i,pt]
								scores[k][i] = vals[i,pt]
						else:
							res[len(res)] = {i:idx[i,pt]}
							feats[len(feats)] = {i:id_selected[i,pt]}
							scores[len(scores)] = {i:vals[i,pt]}
		res = self._parse_idx_dict(res)
		# res = self._convert_to_np(res, scores)
		return res, feats, scores

	def refine_map(self, outmap, idmap, feats_dict):
		result = {}
		result_feat = {}
		result_score = {}
		outmap = outmap[0]
		# print(idmap.shape, outmap.shape)
		idmap = idmap[0]
		# print(idmap.shape, outmap.shape)
		for k in feats_dict:
			result[k] = {}
			result_feat[k] = {}
			result_score[k] = {}

			mean_feat = []
			for i in feats_dict[k]:
				mean_feat.append(feats_dict[k][i])
			mean_feat = torch.stack(mean_feat, dim=0)
			mean_feat = torch.mean(mean_feat, dim=0)

			for pt in range(config.num_pts):
				o = outmap[pt]
				i = idmap[pt]
				dist = torch.norm(i - mean_feat, dim=-1)
				o_2 = o - torch.floor(dist) 
				# o_2 = o - dist * 16
				o_2 = o_2.view(config.out_size * config.out_size)

				val, idx = torch.max(o_2, dim=0)
				idx = idx.cpu().numpy()
				x = idx % config.out_size
				y = idx // config.out_size
				xx = x 
				yy = y 

				if o[y,max(0,x-1)]<o[y,min(config.out_size-1,x+1)]:
					xx = xx + 0.75
				else:
					xx = xx + 0.25 

				if o[max(0,y-1),x]<o[min(config.out_size-1,y+1),x]:
					yy = yy + 0.75 
				else:
					yy = yy + 0.25

				result[k][pt] = [xx,yy]
				result_score[k][pt] = val
				result_feat[k][pt] = i[y,x]
		return result, result_feat, result_score

	def refine_sample(self, outs, idout, coords, feats_dict):
		result = {}
		result_feat = {}
		result_score = {}
		for k in feats_dict:
			result[k] = {}
			result_feat[k] = {}
			result_score[k] = {}

			mean_feat = []
			for i in feats_dict[k]:
				mean_feat.append(feats_dict[k][i])
			mean_feat = torch.stack(mean_feat, dim=0)
			mean_feat = torch.mean(mean_feat, dim=0)

			for pt in range(config.num_pts):
				o = outs[:,pt]
				i = idout[:,pt]
				dist = torch.norm(i - mean_feat, dim=-1)
				o_2 = o - torch.floor(dist / 1.5) * 16 
				# o_2 = o - dist * 16
				# o_2 = o_2.view(config.out_size * config.out_size)

				# print(o_2.shape)
				val, idx = torch.max(o_2, dim=0)
				x, y = coords[idx,0], coords[idx,1]

				result[k][pt] = [x,y]
				result_score[k][pt] = val
				result_feat[k][pt] = i[idx]
		return result, result_feat, result_score

	def parse_dicts(self, idx_dict, scores_dict):
		results = []
		for k in idx_dict:
			inst = idx_dict[k]
			buff = np.zeros([config.num_pts, 3], dtype=np.float32)
			for pt in inst:
				buff[pt,0] = inst[pt][0]
				buff[pt,1] = inst[pt][1]
				buff[pt,2] = scores_dict[k][pt]
				# print(scores_dict[k][pt])
			results.append(buff)
		return results

def visualize(img, idx_dict, scores_dict):
	for inst_idx,k in enumerate(idx_dict):
		plt.figure()
		plt.imshow(img[:,:,::-1])
		inst = idx_dict[k]
		for pt in inst:
			x, y = inst[pt]
			scr = scores_dict[k][pt]
			plt.plot(x, y, 'o')
			plt.text(x, y, str(pt) + '_%.2f'%scr, color='red')
		plt.savefig('./visualize_one/pts_%d.png'%inst_idx)

if __name__=='__main__':
	outmap = pickle.load(open('outmap.pkl', 'rb'))
	idmap = pickle.load(open('idmap.pkl', 'rb'))

	outmap = np.transpose(outmap, (2,0,1))
	# plt.imshow(outmap[5], cmap='jet', vmin=0.0, vmax=16.0)
	# plt.show()
	outmap = torch.from_numpy(outmap)

	idmap = np.transpose(idmap, (2,0,1,3))
	idmap = torch.from_numpy(idmap)

	p = Parser()
	idx_dict, feats_dict, scores_dict = p.grouping(outmap, idmap)
	# idx_dict2, feats_dict, scores_dict = p.refine_map(outmap, idmap, feats_dict)
	# we can use the out,idout,coords to find the refined maximum 

	outs = pickle.load(open('outs.pkl' , 'rb'))
	idout = pickle.load(open('idout.pkl', 'rb'))
	coords = pickle.load(open('coords.pkl', 'rb'))
	print(outs.shape, idout.shape, coords.shape)
	outs = outs[:, :config.num_pts]
	outs = torch.from_numpy(outs)
	idout = torch.from_numpy(idout)
	coords = torch.from_numpy(coords)

	idx_dict2, feats_dict2, scores_dict2 = p.refine_sample(outs, idout, coords, feats_dict)

	pts = p.parse_dicts(idx_dict2, scores_dict2)
	# print(pts)
	print(len(pts))

	img = cv2.imread('img.png')
	img = cv2.resize(img, (256,256))
	# visualize(img, idx_dict, scores_dict)
	# plt.figure()
	visualize(img, idx_dict2, scores_dict2)
	# plt.show()
