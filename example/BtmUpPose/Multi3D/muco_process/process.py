import numpy as np 
import scipy.io as sio 
import cv2 
import matplotlib.pyplot as plt
import glob 
from tqdm import tqdm

import pickle 

pts_idx = np.int64([14, 11,12,13, 8,9,10, 16, 5,6,7, 2,3,4])
# 14 pelvis, 16 neck 

names = []
pts3ds = []
pts2ds = []

def parse_folder(foldername):
	# print(foldername)
	matf = glob.glob(foldername + '/*.mat')
	if len(matf)==0:
		return 
	f = sio.loadmat(matf[0])
	pts3d = np.float32(f['joint_loc3'])
	pts2d = np.float32(f['joint_loc2'])
	pts3d = pts3d[:,pts_idx]
	pts2d = pts2d[:,pts_idx]
	# print(pts3d.shape)
	imgnames = f['img_names']
	# # 3d jiushi camera 3d 
	for i in range(len(imgnames[0])):
		img = imgnames[0][i][0]
		p3d = pts3d[:,:,:,i]
		p3d = p3d.transpose([2,1,0])
		p2d = pts2d[:,:,:,i]
		p2d = p2d.transpose([2,1,0])
		# print(p3d.shape, p2d.shape, img)
		names.append(foldername + '/'+ img)
		pts3ds.append(p3d)
		pts2ds.append(p2d)

folders = glob.glob('./augmented_set/augmented_set/*')
for f in tqdm(folders):
	parse_folder(f)
print(names[0])

pickle.dump({'names':names, 'p2d':pts2ds, 'p3d':pts3ds}, open('augmented.pkl', 'wb'))

names = []
pts3ds = []
pts2ds = []
folders = glob.glob('./unaugmented_set/unaugmented_set/*')
for f in tqdm(folders):
	parse_folder(f)
print(names[0])

pickle.dump({'names':names, 'p2d':pts2ds, 'p3d':pts3ds}, open('unaugmented.pkl', 'wb'))