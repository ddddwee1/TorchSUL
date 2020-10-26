import numpy as np 
import cv2 
import torch 
import torch.nn.functional as F 
import config 
import pickle 

from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from TorchSUL import Model as M 

from munkres import Munkres

def pre_process(img):
	img = np.float32(img)
	img = img / 255 
	# img = img[:,:,::-1]
	img = img - np.float32([0.485, 0.456, 0.406])
	img = img / np.float32([0.229, 0.224, 0.225])

	img = np.transpose(img, [2,0,1])
	# img = torch.from_numpy(img)
	return img 

def rescale_img2(img):
	h,w = img.shape[:2]
	if h<w:
		hh = htarget = int(config.inp_size) 
		wtarget = int((w * config.inp_size / h + 63) // 64 * 64)
		ww = int(w * hh / h)
	else:
		ww = wtarget = int(config.inp_size)
		htarget = int((h * config.inp_size / w + 63) // 64 * 64)
		hh = int(h * ww / w)
	scaleh = hh / h 
	scalew = ww / w 
	padh = (htarget - hh) // 2 
	padw = (wtarget - ww) // 2 
	canvas = np.zeros([htarget, wtarget, 3], dtype=np.uint8)
	img = cv2.resize(img, (ww, hh))
	canvas[padh:padh+hh, padw:padw+ww] = img 
	return canvas, [padw, padh, scaleh, scalew]

def compare(arr, pkl):
	data = pickle.load(open(pkl, 'rb'))
	diff = np.float32(arr) - np.float32(data) 
	diff = np.sum(np.abs(diff))
	print('DIFF', diff, arr.shape, data.shape, arr.dtype, data.dtype)

def process_img(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img, center, scale = resize_align_multi_scale(img, config.inp_size, 1, 1)
	# print(center)
	# print(scale)

	img = pre_process(img)
	img = img[None,...]
	# compare(img, 'image.pkl')
	img = torch.from_numpy(img).cuda()
	return img, center, scale

def get_heatmap(img, model):
	hmap = 0 
	tags = []
	roots = 0
	rels = 0
	outputs = model(img)
	for output in outputs:
		output = F.interpolate(output, size=(outputs[-1].shape[2], outputs[-1].shape[3]), mode='bilinear')
		hmap = hmap + output[:,:config.num_pts]
		tags.append(output[:,config.num_pts:config.num_pts*2])
		roots = roots + output[:,config.num_pts*2:config.num_pts*2+1]
		rels = rels + output[:, config.num_pts*2+1:]
	hmap = hmap / len(outputs)
	roots = roots / len(outputs)
	rels = rels / len(outputs)

	hmap_flip = 0
	roots_flip = 0 
	rels_flip = 0
	# flip_index = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
	flip_index = [0, 4,5,6, 1,2,3, 7, 11,12,13, 8,9,10]
	outputs_flip = model(torch.flip(img, [3]))
	for output in outputs_flip:
		outout = F.interpolate(output, size=(outputs_flip[-1].shape[2], outputs[-1].shape[3]), mode='bilinear')
		output = torch.flip(output, [3])
		hmap_flip = hmap_flip + output[:,:config.num_pts][:, flip_index]
		tags.append(output[:,config.num_pts:config.num_pts*2][:, flip_index])
		buff = output[:,config.num_pts*2:][:, flip_index]
		roots_flip = roots_flip + buff[:,0:1]
		rels_flip = rels_flip + buff[:,1:]
	hmap_flip = hmap_flip / len(outputs_flip)
	roots_flip = roots_flip / len(outputs_flip)
	rels_flip = rels_flip / len(outputs_flip)

	h, w = img.shape[2], img.shape[3]
	hmap = F.interpolate(hmap, size=(h,w), mode='bilinear')
	hmap_flip = F.interpolate(hmap_flip, size=(h,w), mode='bilinear')
	roots = F.interpolate(roots, size=(h,w), mode='bilinear')
	roots_flip = F.interpolate(roots_flip, size=(h,w), mode='bilinear')
	rels = F.interpolate(rels, size=(h,w), mode='bilinear')
	rels_flip = F.interpolate(rels_flip, size=(h,w), mode='bilinear')
	tags = [F.interpolate(t, size=(h,w), mode='bilinear') for t in tags]
	# print(hmap.shape)
	# hmap_final = (hmap + hmap_flip) / 2.0
	# roots_final = (roots + roots_flip) / 2.0
	# rels_final = (rels + rels_flip) / 2.0
	hmap_final = hmap
	roots_final = roots
	rels_final = rels
	# hmap_final = hmap_flip
	# roots_final = roots_flip
	# rels_final = rels_flip
	tags = [t.unsqueeze(4) for t in tags]
	tags = torch.cat(tags, dim=4)
	return hmap_final, tags, roots_final, rels_final

def top_k(hmap, tags):
	pool = M.MaxPool2D(config.nms_kernel)
	hmap_m = pool(hmap)
	hmap_m = torch.eq(hmap_m, hmap).float()
	hmap = hmap_m * hmap 

	bsize, num_pts, h, w = hmap.shape
	hmap = hmap.view(bsize, num_pts, h * w)
	tags = tags.view(bsize, num_pts, h * w, -1)
	val, ind = hmap.topk(config.max_inst, dim=2)
	ind_expanded = ind.unsqueeze(-1).expand(-1,-1,-1,tags.shape[-1])
	tagk = torch.gather(tags, 2, ind_expanded)

	x = ind%w 
	y = (ind/w).long()
	indk = torch.stack((x,y), dim=3)
	vals, indk, tagk = val[0], indk[0], tagk[0]
	return vals, indk, tagk

def hungarian(scores):
	if not isinstance(scores, np.ndarray):
		scores = scores.cpu().detach().numpy()
	m = Munkres()
	res = m.compute(scores)
	res = np.int32(res)
	return res 

def grouping(vals, indk, tagk):
	joint_order = [0,1,4,2,5,3,6,7,8,11,9,12,10,13]
	# res = torch.zeros(config.num_pts, 3+tagk.shape[2]).float()
	joint_dict = {}
	tag_dict = {}
	for i,idx in enumerate(joint_order):
		tags = tagk[idx]
		val = vals[idx]
		mask = torch.where(val>config.hmap_thresh)[0]
		if len(mask)==0:
			continue 
		tags = tags[mask] # [m, 2]
		val = val[mask] # [m]
		ind = indk[idx, mask] #[m, 2]
		if i==0 or len(joint_dict)==0:
			for t,v,ii in zip(tags, val, ind):
				key = t[0]
				value = torch.zeros(config.num_pts, 3+tagk.shape[2]).float().cuda()
				value[idx, :2] = ii 
				value[idx, 2] = v 
				value[idx, 3:] = t 
				joint_dict[key] = value
				tag_dict[key] = [t]
		else:
			grouped_keys = list(joint_dict.keys())
			grouped_tags = torch.stack([torch.mean(torch.stack(tag_dict[kk]),dim=0) for kk in grouped_keys], dim=0) # [g, 2]
			diff = tags.unsqueeze(1) - grouped_tags.unsqueeze(0)
			diff_norm = torch.norm(diff, dim=2) # [m,g]
			diff_normed2 = torch.round(diff_norm) * 100 - val.unsqueeze(1)
			num_added, num_grouped = diff.shape[0], diff.shape[1]
			if num_added > num_grouped:
				diff_normed2 = torch.cat([diff_normed2, torch.zeros(num_added, num_added-num_grouped).cuda()], dim=1)
			pairs = hungarian(diff_normed2)
			diffshape = diff_norm.shape 
			for row,col in pairs:
				is_matched = False
				if (row<num_added and col<num_grouped):
					if diff_norm[row,col]<config.tag_thresh:
						key = grouped_keys[col]
						value = joint_dict[key][idx]
						value[:2] = ind[row]
						value[2] = val[row]
						value[3:] = tags[row]
						tag_dict[key].append(tags[row])
						is_matched = True
				if not is_matched:
					key = tags[row][0]
					value = torch.zeros(config.num_pts, 3+tagk.shape[2]).float().cuda()
					value[idx,:2] = ind[row]
					value[idx,2] = val[row]
					value[idx,3:] = tags[row]
					joint_dict[key] = value 
					tag_dict[key] = [tags[row]]

	ans = [joint_dict[i] for i in joint_dict]
	if len(ans)==0:
		return [] 
	ans = torch.stack(ans, dim=0)
	return ans 

def adjust(pts_init, hmap):
	pts_init = pts_init.clone()
	for inst_id, inst in enumerate(pts_init):
		for p, pts in enumerate(inst):
			if pts[2]>0:
				x,y = pts[0:2]
				xx,yy = x.long(), y.long()
				h = hmap[0,p]
				if h[yy,min(xx+1,h.shape[1]-1)]>h[yy,max(xx-1,0)]:
					x += 0.25 
				else:
					x -= 0.25 
				if h[min(yy+1,h.shape[0]-1),xx]>h[max(yy-1,0),xx]:
					y += 0.25 
				else:
					y -= 0.25 
				pts[0] = x+0.5 
				pts[1] = y+0.5

	scores = pts_init[:,:,2].mean(dim=1) 
	return pts_init, scores 

def refine(pts_adjusted, hmap, tags):
	res = []
	for i in range(pts_adjusted.shape[0]):
		pts = pts_adjusted[i]
		init_tag = []
		for p in range(config.num_pts):
			if pts[p][2]>0:
				x,y = pts[p][0].long(), pts[p][1].long()
				init_tag.append(tags[0,p,y,x])
		init_tag = torch.mean(torch.stack(init_tag, dim=0), dim=0)

		buff = []
		for p in range(config.num_pts):
			dist = tags[0,p] - init_tag 
			dist = torch.sqrt(torch.sum(torch.pow(dist, 2), dim=2)) # [h,w]
			conf = hmap[0,p] - torch.round(dist / config.tag_distance)
			confmax = conf.max()
			idx = torch.argmax(conf)
			idx = idx.cpu().detach().numpy()
			idx = int(idx)
			x = idx % int(conf.shape[1])
			y = idx // int(conf.shape[1])
			xx = int(x)
			yy = int(y)
			v = hmap[0,p,yy,xx].cpu().detach().numpy()
			x += 0.5 
			y += 0.5 
			if hmap[0,p,yy,min(xx+1,conf.shape[1]-1)]>hmap[0,p,yy,max(xx-1,0)]:
				x += 0.25 
			else:
				x -= 0.25 
			if hmap[0,p,min(yy+1,conf.shape[0]-1),xx]>hmap[0,p,max(yy-1,0),xx]:
				y += 0.25
			else:
				y -= 0.25
			buff.append([x,y,v])
		res.append(buff)
	res = np.float32(res)
	original = pts_adjusted.cpu().detach().numpy()
	for i in range(original.shape[0]):
		# pts = original[i]
		for p in range(original.shape[1]):
			if res[i,p,2]>0 and original[i,p,2]==0:
				original[i,p,:3] = res[i,p,:3]
	return original 

def run_pipeline(img, model):
	img, center, scale = process_img(img)
	hmap, tags, roots, rels = get_heatmap(img, model)
	vals, indk, tagk = top_k(hmap, tags)
	pts_init = grouping(vals, indk, tagk)
	if len(pts_init)==0:
		return [], []
	pts_adjusted, scores = adjust(pts_init, hmap)
	pts_refined = refine(pts_adjusted, hmap, tags)
	pts_final = get_final_preds([pts_refined], center, scale, [hmap.size(3), hmap.size(2)])
	return pts_final, scores, roots, rels, hmap, img, tags

if __name__=='__main__':
	import custom.network 
	#%% load model 
	model = custom.network.DensityNet(config.density_num_layers, config.density_channels, config.density_level,\
							config.gcn_layers, config.gcn_channels, config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	x = np.float32(np.random.random(size=[1,3,512,512]))
	x = torch.from_numpy(x)
	with torch.no_grad():
		model(x)
	model.bn_eps(1e-5)
	saver = M.Saver(model)
	saver.restore('./model/')
	model.eval()
	model.cuda()

	#%% Main func
	pickle.dump(pts_refined, open('ref.pkl','wb'))

	print('abc')
