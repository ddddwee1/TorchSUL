from tqdm import tqdm 
import numpy as np 
import torch 
from TorchSUL import Model as M 
import cv2 
import pickle 
import testutil

import config 
import network 
import loss 
import visutil.util as vutil 
import matplotlib.pyplot as plt 
import visutil2 

def get_pts(pts, roots, rels):
	roots = roots.cpu().detach().numpy()
	rels = rels.cpu().detach().numpy()
	results = []
	for p in pts:
		buff = np.zeros([config.num_pts, 3], dtype=np.float32)
		x = int(p[0,0] / 4)
		y = int(p[0,1] / 4)
		dep = roots[0,0,y,x]
		buff[0,2] = dep
		for i in range(config.num_pts - 1):
			x = int(p[i+1, 0] / 4)
			y = int(p[i+1, 1] / 4)
			d = dep + rels[0,i,y,x]
			buff[i+1,2] = d 
		buff[:,:2] = p[:,:2]
		results.append(buff)
	return results

if __name__=='__main__':
	model_dnet = network.DensityNet(config.density_num_layers, config.density_channels, config.density_level,\
							config.gcn_layers, config.gcn_channels, config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)

	with torch.no_grad():
		# initialize model 
		x = np.float32(np.random.random(size=[1,3,512,512]))
		x = torch.from_numpy(x)
		model_dnet(x)
		model = loss.ModelWithLoss(model_dnet)
		M.Saver(model).restore('./model/')
		model.eval()
		model.cuda()

		imgname = '233.jpg'
		img = cv2.imread(imgname)
		pts, scores, roots, rels, hmap, img_processed, idout = testutil.run_pipeline(img, model.model)
		pts_results = get_pts(pts, roots, rels)

		print(len(pts_results))
		pts_final = []
		for p,s in zip(pts_results, scores):
			if s>0.2:
				pts_final.append(p)
		# print(pts_final[0][:,0].min(), pts_final[0][:,0].max(), pts_final[0][:,1].min(), pts_final[0][:,1].max(), pts_final[0][:,2].min(), pts_final[0][:,2].max())
		img = vutil.plot_skeleton3d(pts_final, alpha=[1.0]*len(pts_final))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imwrite('skeleton.png', img)
		# plt.show()
		# print(img_processed.shape, hmap.shape, idout[:,:,:,:,0].shape, roots.shape, rels.shape)
		# visutil2.vis_batch(img_processed, hmap, './outputs/out.jpg')
		# visutil2.vis_batch(img_processed, idout[:,:,:,:,0], './outputs/id.jpg', minmax=True)
		# visutil2.vis_batch(img_processed, roots, './outputs/dep.jpg', minmax=True)
		# visutil2.vis_batch(img_processed, rels, './outputs/rel.jpg', minmax=True)
