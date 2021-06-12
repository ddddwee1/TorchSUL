import torch 
import network 
from TorchSUL import Model as M 
import config
import numpy as np 
import util 
import cv2 
import testutil 
# import visutil2 
# from pycocotools.cocoeval import COCOeval
# from pycocotools.coco import COCO

def vis_skeleton(img, pts):
	pts = pts[:,:3]
	# img = visutil2.deprocess(img)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	for i in range(len(pts)):
		x,y,conf = pts[i]
		if conf>0.1:
			cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
	return img 

def test_img(img, model):
	pts_all = []
	scores_all = []

	# original size 
	pts, scores,_,_,_ = testutil.run_pipeline(img, model)
	pts_all += pts
	scores_all += scores

	# crop ratio 2
	imgs2, metas2 = util.crop_images(img, 2)
	for i,m in zip(imgs2, metas2):
		pts, scores,_,_,_ = testutil.run_pipeline(i, model)
		pts = util.restore_pts(pts, m)
		pts_all += pts
		scores_all += scores

	# crop ratio 4
	imgs2, metas2 = util.crop_images(img, 4)
	for i,m in zip(imgs2, metas2):
		pts, scores,_,_,_ = testutil.run_pipeline(i, model)
		pts = util.restore_pts(pts, m)
		pts_all += pts
		scores_all += scores

	pts, scores = util.nms(pts_all, scores_all)
	# print(scores)

	# for i in range(len(pts)):
	# 	if scores[i]<0.3:
	# 		continue
	# 	imgcp = img.copy()
	# 	skltn = vis_skeleton(imgcp, pts[i])
	# 	cv2.imwrite('outputs/skt_%d.png'%i, skltn)

	return pts, scores

# initialize 
model = network.DensityNet(config.density_num_layers, config.density_channels, config.density_level,\
						config.gcn_layers, config.gcn_channels, config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)

# coco = COCO('person_keypoints_val2017.json')
# ids = list(coco.imgs.keys())

with torch.no_grad():
	x = np.float32(np.random.random(size=[1,3,config.inp_size,config.inp_size]))
	x = torch.from_numpy(x)
	model(x)
	M.Saver(model).restore('model_coco/')
	model.eval()
	model.cuda()

	# imgname = '000000410650.jpg'
	# img = cv2.imread(imgname)

	# pts, scores = test_img(img, model)

	results = {}
	for i in tqdm(ids):
		fname = './val2017/%012d.jpg'%i
		img = cv2.imread(fname)
		pts, scores = test_img(img, model)
		results[i] = [pts, scores]

	pickle.dump(results, open('coco_results.pkl', 'wb'))
