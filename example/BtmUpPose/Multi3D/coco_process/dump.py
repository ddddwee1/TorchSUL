import pickle 
import numpy as np 
import cv2 
import pycocotools
import pycocotools._mask
import os 
import json 
from tqdm import tqdm 

if __name__=='__main__':
	# annotation = pickle.load(open('kpts.pkl','rb'))['annotations']
	annotation = json.load(open('person_keypoints_train2017.json'))['annotations']

	data = {}
	for annot in annotation:
		imgid = annot['image_id']
		if not imgid in data:
			data[imgid] = []
		data[imgid].append(annot)


	data_kpts = {}
	for k in tqdm(data.keys()):
		if not k in data_kpts:
			data_kpts[k] = []
		for annot in data[k]:
			kpts = annot['keypoints']
			box = annot['bbox']
			kpts = np.float32(kpts).reshape([17,3])
			data_kpts[k].append([kpts, box])
	pickle.dump(data_kpts, open('imagewise_kpts.pkl', 'wb'))

	data_segs = {}
	for k in tqdm(data.keys()):
		if not k in data_segs:
			data_segs[k] = []
		for annot in data[k]:
			segs = annot['segmentation']
			# segs = np.float32(segs).reshape([17,3])
			data_segs[k].append(segs)
	pickle.dump(data_segs, open('imagewise_segments.pkl', 'wb'))

	if not os.path.exists('./masks/'):
		os.mkdir('./masks/')

	for k in tqdm(data.keys()):
		img = cv2.imread('./train2017/%012d.jpg'%k)
		h, w = img.shape[0], img.shape[1]
		mask = np.zeros([h ,w])
		for annot in data[k]:
			if annot['iscrowd']:
				seg = annot['segmentation']
				rle = pycocotools._mask.frPyObjects(seg, h, w)
				m = pycocotools._mask.decode([rle])
				m = np.amax(m, axis=-1)
				mask += m 
			elif annot['num_keypoints']==0:
				seg = annot['segmentation']
				# print(seg)
				rles = pycocotools._mask.frPyObjects(seg, h, w)
				m = pycocotools._mask.decode(rles)
				m = np.amax(m, axis=-1)
				mask += m 
				
		mask[mask>0] = 1 
		# masks[k] = mask

		pickle.dump(mask, open('masks/%012d.pkl'%k, 'wb'))
	

	# print(len(data))
	# print(data[77])

	# res = []
	# for annot in annotation:
	# 	imgid = annot['image_id']
	# 	if imgid == 77:
	# 		res.append(annot)
	# pickle.dump(res, open('77.pkl','wb'))

	# data = pickle.load(open('77.pkl', 'rb'))
	# print(data[3])
	# obj = data[3]
	# rle = pycocotools._mask.frPyObjects(obj['segmentation'], 375, 500)
	# print(rle)
	# mask = pycocotools._mask.decode(rle[0])
	# print(mask.shape)
	# print(mask.dtype)
	# import matplotlib.pyplot as plt 
	# plt.imshow(mask[:,:,0])
	# plt.show()
