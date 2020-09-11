# basic imports 
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import numpy as np 
import torch 
import cv2 
import grouping 

# import visualization
import matplotlib.pyplot as plt 
import visutil
from tqdm import tqdm 

# following lines import the model 
from TorchSUL import Model as M 
import config 
import network
import loss 

def pre_process(img):
	img = np.float32(img)
	img = img / 255 
	img = img[:,:,::-1]
	img = img - np.float32([0.485, 0.456, 0.406])
	img = img / np.float32([0.229, 0.224, 0.225])

	img = np.transpose(img, [2,0,1])
	# img = torch.from_numpy(img)
	return img 

def rescale_img(img):
	h,w = img.shape[:2]
	imgsize = max(h, w)
	canvas = np.zeros([imgsize, imgsize, 3], dtype=np.uint8)
	padh = (imgsize - h) // 2
	padw = (imgsize - w) // 2 
	canvas[padh:padh+h, padw:padw+w] = img 
	canvas = cv2.resize(canvas, (config.inp_size, config.inp_size))
	return canvas, [padh, padw, imgsize/config.out_size]

def pts_to_origin(pts, meta):
	padh, padw, scale = meta 
	pts = pts.copy()
	pts[:, :2] *= scale
	pts[:, 0] -= padw
	pts[:, 1] -= padh
	return pts 

def run_model(img, net):
	img = pre_process(img)[None,...]
	img = torch.from_numpy(img).cuda()
	out_map, id_map = net(img)
	out_map = torch.sigmoid(out_map)
	return out_map, id_map

def parse_points(outmap, idmap):
	# outmap = outmap.permute(0,3,1,2)
	# idmap = idmap.permute(2,0,1,3)

	p = grouping.Parser()
	idx_dict, feats_dict, scores_dict = p.grouping(outmap, idmap)
	pts_grouped = p.parse_dicts(idx_dict, scores_dict)
	# print(pts_grouped)
	scores = [i[:,2].mean() for i in pts_grouped]

	idx_dict2, feats_dict2, scores_dict2 = p.refine_map(outmap, idmap, feats_dict)
	pts = p.parse_dicts(idx_dict2, scores_dict2)
	return pts, scores

def run_pipeline(img, net):
	img, meta = rescale_img(img)
	outmap, idmap = run_model(img, net)
	pts, scores = parse_points(outmap, idmap)
	pts = [pts_to_origin(i, meta) for i in pts]
	return pts, scores

def run_visualize(img, net):
	img, meta = rescale_img(img)
	outmap, idmap = run_model(img, net)
	outmap = outmap.detach().cpu().numpy()[0]
	outmap = np.amax(outmap, axis=0)
	res = visutil.vis_one(img, outmap, de=False)
	return res

def visualize(img, pts):
	for pidx in range(len(pts)):
		p = pts[pidx]
		plt.figure()
		for i in range(p.shape[0]):
			plt.imshow(img[:,:,::-1])
			x,y = p[i,0], p[i,1]
			scr = p[i,2]
			plt.plot(x,y,'o')
			plt.text(x,y,str(i)+'_%.2f'%scr, color='red')
		plt.savefig('./visualize_one/pts_%d.png'%pidx)

if __name__=='__main__':
	net = network.DensityNet(config.density_num_layers, config.density_channels, config.density_level,\
							config.gcn_layers, config.gcn_channels, config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	with torch.no_grad():
		x = np.float32(np.random.random(size=[1,3,512,512]))
		x = torch.from_numpy(x)
		outs, idout = net(x)
		net = loss.ModelWithLoss(net)
		saver = M.Saver(net)
		saver.restore('./model/')
		net.cuda()
		net.eval() # dont use eval model since inst-norm property 
		imgname = './data/000000000785.jpg'
		img = cv2.imread(imgname)
		print(img.shape)
		pts, scores = run_pipeline(img, net.model)
		# img, _ = rescale_img(img)
		print(pts)
		visualize(img, pts)

		
		