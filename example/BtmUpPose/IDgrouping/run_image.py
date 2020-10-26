import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from tqdm import tqdm 
import numpy as np 
import torch 
from TorchSUL import Model as M 
import cv2 
import pickle 
import testutil

from custom import config 
import custom.network 

if __name__=='__main__':
	model = custom.network.DensityNet(config.density_num_layers, config.density_channels, config.density_level,\
							config.gcn_layers, config.gcn_channels, config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	
	
	coco = COCO('person_keypoints_val2017.json')
	ids = list(coco.imgs.keys())

	with torch.no_grad():
		# initialize model 
		x = np.float32(np.random.random(size=[1,3,512,512]))
		x = torch.from_numpy(x)
		model(x)
		model.bn_eps(1e-5)
		saver = M.Saver(model)
		saver.restore('./model/')
		model.eval()
		model.cuda()
		# do testing 
		results = {}
		for i in tqdm(ids):
			fname = './val2017/%012d.jpg'%i
			img = cv2.imread(fname)
			pts, scores = testutil.run_pipeline(img, model)
			# print(len(pts), len(scores))
			results[i] = [pts, scores]
		pickle.dump(results, open('coco_results.pkl', 'wb'))
