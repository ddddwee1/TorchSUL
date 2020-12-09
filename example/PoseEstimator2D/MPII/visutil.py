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
	img = cv2.resize(img, (config.out_size, config.out_size))
	if len(hmap.shape)==3:
		res = [vis_kpt(img, hmap[i], minmax=minmax) for i in range(hmap.shape[0])]
	else:
		res = [vis_kpt(img, hmap)]
	return res 

def vis_batch(imgs, hmaps, outdir=None, minmax=False):
	if not isinstance(hmaps, np.ndarray):
		hmaps = hmaps.cpu().detach().numpy()
		imgs = imgs.cpu().detach().numpy()
	res = []
	for i in range(imgs.shape[0]):
		res.append(vis_one(imgs[i], hmaps[i], minmax=minmax))
	h = len(res)
	w = len(res[0])
	s = config.out_size
	canvas = np.zeros([h * s, w * s, 3], dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			canvas[i*s:i*s+s, j*s:j*s+s] = res[i][j]
	if outdir:
		cv2.imwrite(outdir, canvas)
	return canvas
