from deepvit import ConvVit
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import numpy as np 
import cv2 
import glob 

def process(imgs):
	imgs = np.float32(imgs)
	imgs = imgs / 127.5 - 1.0
	imgs = np.transpose(imgs, (0,3,1,2))
	return imgs 

def deprocess(imgs):
	imgs = imgs + 1.
	imgs = imgs * 127.5
	imgs = np.transpose(imgs, (0, 2,3,1))
	imgs = np.uint8(imgs)
	return imgs 

def vis_single(img, att):
	hw = img.shape[1]
	att = att.reshape([7,7])
	att = cv2.resize(att, (hw, hw))
	att = att - att.min()
	att = att / att.max()
	att = np.uint8(att * 255)
	att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
	res = cv2.addWeighted(att, 0.7, img, 0.3, 0)
	return res 

def vis_img(img, att):
	# att: [h, 49]
	res = []
	for h in range(len(att)):
		res.append(vis_single(img, att[h]))
	res = np.concatenate(res, axis=0)
	return res 

def vis_batch(imgs, atts):
	imgs = deprocess(imgs)
	res_all = []
	for i in range(len(imgs)):
		res = vis_img(imgs[i], atts[i])
		res_all.append(res)
	res_all = np.concatenate(res_all, axis=1)
	return res_all

net = ConvVit()
dumb_x = torch.rand(2, 3, 112, 112)
dumb_f = net(dumb_x)

M.Saver(net).restore('./model/')
net.eval()
net.cuda()

imgs = [cv2.imread(i) for i in glob.glob('./imgs/*.jpg')]
imgs = process(imgs)
imgs = torch.from_numpy(imgs).cuda()

with torch.no_grad():
	outs = net(imgs)
	print(outs)
	outs, all_atts = net(imgs, return_attn=True)
	print(outs)
# print(outs.shape, atts.shape)

for i, atts in enumerate(all_atts):
	res = vis_batch(imgs.cpu().numpy(), atts.cpu().numpy())
	cv2.imwrite('./visatt/att_%d.png'%i, res)
