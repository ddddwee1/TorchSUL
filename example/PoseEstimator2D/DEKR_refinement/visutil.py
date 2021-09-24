import numpy as np 
import torch 
import config 
import cv2 

def deprocess(img):
	img = np.transpose(img, [1,2,0])
	img = img * np.float32([0.229, 0.224, 0.225])
	img = img + np.float32([0.485, 0.456, 0.406])
	# img = img[:,:,:,::-1]
	img = img * 255 
	img = np.uint8(img)
	return img 

def vis_kpt(img, hmap, minmax=False):
	img = img.copy()
	if minmax:
		hmap = hmap - hmap.min()
		hmap = hmap / hmap.max() 
	else:
		hmap = np.clip(hmap, 0.0, 1.0)
	hmap = np.uint8(hmap * 255)
	hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
	res = cv2.addWeighted(hmap, 0.7, img, 0.3, 0)
	return res 

def vis_one(img, hmap, minmax=False):
	img = deprocess(img)
	img = cv2.resize(img, (hmap.shape[-1], hmap.shape[-1]))
	if len(hmap.shape)==3:
		res = [vis_kpt(img, hmap[i], minmax=minmax) for i in range(hmap.shape[0])]
	else:
		res = [vis_kpt(img, hmap)]
	return res 

def vis_batch(imgs, hmaps, outdir=None, minmax=False):
	hmaps = hmaps.cpu().detach().numpy()
	imgs = imgs.cpu().detach().numpy()
	res = []
	for i in range(imgs.shape[0]):
		res.append(vis_one(imgs[i], hmaps[i], minmax=minmax))
	h = len(res)
	w = len(res[0])
	s = hmaps.shape[-1]
	canvas = np.zeros([h * s, w * s, 3], dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			canvas[i*s:i*s+s, j*s:j*s+s] = res[i][j]
	if outdir:
		cv2.imwrite(outdir, canvas)
	return canvas

def offset_one(img, offset, pt):
	s = offset.shape[-1]
	img = deprocess(img)
	img = cv2.resize(img, (s, s))
	res = []
	for j in range(config.num_pts):
		canvas = np.zeros([s, s], dtype=np.uint8)
		for i in range(pt.shape[0]):
			if pt[i,j,2]<=0:
				continue
			xx = int(pt[i,-1,0])
			yy = int(pt[i,-1,1])
			# print(xx, yy)
			if xx<0 or xx>=s or yy<0 or yy>=s:
				continue
			dx = offset[j*2, yy, xx]
			dy = offset[j*2+1, yy, xx]
			# print(dx, dy)
			# dx = dy = 0
			x = int(xx + dx)
			y = int(yy + dy)
			# print(x,y)
			for xx in range(x-2,x+3):
				for yy in range(y-2,y+3):
					if xx>=0 and xx<s and yy>=0 and yy<s:
						canvas[yy,xx] = 180
		canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
		canvas = cv2.addWeighted(canvas, 0.7, img, 0.3, 0)
		res.append(canvas)
	res = np.concatenate(res, axis=1)
	return res

def vis_offset(imgs, offsets, pts, outdir=None):
	imgs = imgs.cpu().detach().numpy()
	offsets = offsets.cpu().detach().numpy()
	pts = pts.cpu().detach().numpy()
	res = []
	for i in range(imgs.shape[0]):
		res.append(offset_one(imgs[i], offsets[i], pts[i]))
	res = np.concatenate(res, axis=0)
	if outdir:
		cv2.imwrite(outdir, res)
	return res 
