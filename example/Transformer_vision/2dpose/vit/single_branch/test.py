import torch 
import numpy as np 
import cv2 
from TorchSUL import Model as M 
import network 
import config 
import visutil 

def get_max_pos(out):
	c,h,w = out.shape
	out = out.reshape([-1, h*w])
	res = np.zeros([c, 2], dtype=np.int64)
	idx_max = np.argmax(out, axis=1)
	res[:,0] = idx_max // w 
	res[:,1] = idx_max % w 
	res = res // 2 
	return res 

def get_attn_map(attn, pos):
	# pos: [c, 2]
	# attn: [n_heads, 785, 785]
	n = attn.shape[0]
	c = pos.shape[0]
	attn = attn[:, 1:, 1:]
	idx = pos[:,0] * 28 + pos[:,1]  # [c]
	attn = attn[:, idx, :]  # [n_heads, c, 784]
	attn = np.transpose(attn, [1,0,2]).reshape([c, n, 28, 28])
	return attn

def vis_attn_map(img, attn, hmap, size=128):
	
	img = visutil.deprocess(img)
	img = cv2.resize(img, (size, size))
	pos = get_max_pos(hmap)

	attn = np.concatenate([attn, attn.sum(axis=0, keepdims=True)], axis=0)
	attn = get_attn_map(attn, pos)

	c, n = attn.shape[:2]
	canvas = np.zeros([size*n, size*c, 3], dtype=np.uint8)
	# print(canvas.shape)
	for i in range(c):
		for j in range(n):
			a = attn[i,j]
			a = cv2.resize(a, (size,size))
			a = visutil.vis_kpt(img, a, minmax=True)
			canvas[j*size:j*size+size, i*size:i*size+size] = a 
	return canvas

def _pre_process(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (config.inp_size, config.inp_size))
	img = np.float32(img)
	img = img / 255 
	img = img - np.float32([0.485, 0.456, 0.406])
	img = img / np.float32([0.229, 0.224, 0.225])

	img = np.transpose(img, [2,0,1])
	return img 

if __name__=='__main__':
	model_dnet = network.DinoNet()
	x = np.float32(np.random.random(size=[1,3, config.inp_size, config.inp_size]))
	x = torch.from_numpy(x)
	with torch.no_grad():
		outs = model_dnet(x)
	M.Saver(model_dnet).restore('./model/')
	model_dnet.eval()
	model_dnet.cuda()

	img = cv2.imread('0000.png')
	img = _pre_process(img)
	img = torch.from_numpy(img[None,...]).cuda()
	with torch.no_grad():
		out, attn = model_dnet(img, True)
	
	res = visutil.vis_one(img.cpu().numpy()[0], out.cpu().numpy()[0])
	res = np.concatenate(res, axis=1)
	# cv2.imwrite('out.png', res)

	attn_map = vis_attn_map(img.cpu().numpy()[0], attn.cpu().numpy()[0], out.cpu().numpy()[0], size=config.out_size)
	# cv2.imwrite('attn.png', attn_map)

	hmap_attn = np.concatenate([res, attn_map], axis=0)
	cv2.imwrite('hmap_attn.png', hmap_attn)