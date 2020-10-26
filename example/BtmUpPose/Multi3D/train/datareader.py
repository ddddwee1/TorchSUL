import cv2 
import numpy as np 
import config 
import pickle 
# from multiprocessing.pool import ThreadPool
from multiprocessing import Pool 
import random 
import pycocotools._mask
import torch 
import fastgaus.render_core_cython as rc
from torch.utils.data import Dataset

class RandomAffineTransform(object):
	# borrow from higher-hrnet
	def __init__(self,
				 input_size,
				 output_size,
				 max_rotation,
				 min_scale,
				 max_scale,
				 scale_type,
				 max_translate,
				 scale_aware_sigma=False):
		self.input_size = input_size
		self.output_size = output_size 

		self.max_rotation = max_rotation
		self.min_scale = min_scale
		self.max_scale = max_scale
		self.scale_type = scale_type
		self.max_translate = max_translate
		self.scale_aware_sigma = scale_aware_sigma

	def _get_affine_matrix(self, center, scale, res, rot=0):
		# Generate transformation matrix
		h = 200 * scale
		t = np.zeros((3, 3))
		t[0, 0] = float(res[1]) / h
		t[1, 1] = float(res[0]) / h
		t[0, 2] = res[1] * (-float(center[0]) / h + .5)
		t[1, 2] = res[0] * (-float(center[1]) / h + .5)
		t[2, 2] = 1
		if not rot == 0:
			rot = -rot  # To match direction of rotation from cropping
			rot_mat = np.zeros((3, 3))
			rot_rad = rot * np.pi / 180
			sn, cs = np.sin(rot_rad), np.cos(rot_rad)
			rot_mat[0, :2] = [cs, -sn]
			rot_mat[1, :2] = [sn, cs]
			rot_mat[2, 2] = 1
			# Need to rotate around center
			t_mat = np.eye(3)
			t_mat[0, 2] = -res[1]/2
			t_mat[1, 2] = -res[0]/2
			t_inv = t_mat.copy()
			t_inv[:2, 2] *= -1
			t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
		return t

	def _affine_joints(self, joints, mat):
		joints = np.array(joints)
		shape = joints.shape
		joints = joints.reshape(-1, 2)
		return np.dot(np.concatenate(
			(joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

	def __call__(self, image, mask, joints, boxes):
		# assert isinstance(mask, list)
		# assert isinstance(joints, list)
		# assert len(mask) == len(joints)
		# assert len(mask) == len(self.output_size)

		height, width = image.shape[:2]

		center = np.array((width/2, height/2))
		if self.scale_type == 'long':
			scale = max(height, width)/200
		elif self.scale_type == 'short':
			scale = min(height, width)/200
		else:
			raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
		aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
			+ self.min_scale
		scale *= aug_scale
		aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

		if self.max_translate > 0:
			dx = np.random.randint(
				-self.max_translate*scale, self.max_translate*scale)
			dy = np.random.randint(
				-self.max_translate*scale, self.max_translate*scale)
			center[0] += dx
			center[1] += dy

		mat_output = self._get_affine_matrix(center, scale, (self.output_size, self.output_size), aug_rot)[:2]
		# mask_out = np.zeros([self.output_size, self.output_size, mask.shape[-1]], dtype=np.uint8)
		# for i in range(mask.shape[-1]):
		   #  mask_out[:,:,i] = cv2.warpAffine((mask[:,:,i]*255).astype(np.uint8), mat_output, (self.output_size, self.output_size)) 
		# print('L95',mask.shape)
		if mask is None:
			mask_out = None
		else:
			mask_out = cv2.warpAffine((mask*255).astype(np.uint8), mat_output, (self.output_size, self.output_size)) 
			# print('L97', mask_out.shape)
			mask_out = np.float32(mask_out) / 255
			mask_out = (mask_out > 0.5).astype(np.float32)

		joints[:, :, 0:2] = self._affine_joints(joints[:, :, 0:2], mat_output)
		if boxes is None:
			whs = None
		else:
			whs = boxes[:,2:] * (float(self.output_size) / (200 * scale))

		if self.scale_aware_sigma:
			joints[:, :, 3] = joints[:, :, 3] / aug_scale

		mat_input = self._get_affine_matrix(center, scale, (self.input_size, self.input_size), aug_rot)[:2]
		image = cv2.warpAffine(image, mat_input, (self.input_size, self.input_size))

		return image, mask_out, joints, whs

class COCO_kpts(Dataset):
	def __init__(self):
		data = pickle.load(open('augmented.pkl' , 'rb'))
		self.names = data['names']
		self.pts_data = data['p2d']
		self.pts3d_data = data['p3d']
		print('Index pool MUCO:', len(self.names))
		self.aff = RandomAffineTransform(config.inp_size, config.out_size, config.rotation, config.min_scale, config.max_scale, 'short', config.max_translate, False)

		# coco 
		self.coco_pts_data = pickle.load(open('filtered_coco_kpts.pkl' , 'rb'))
		self.coco_img_indices = list(self.coco_pts_data.keys())
		print('Index pool COCO: ', len(self.coco_img_indices))
		self.coco_aff = RandomAffineTransform(config.inp_size, config.out_size, config.coco_rotation, config.coco_min_scale, config.coco_max_scale, 'short', config.coco_max_translate, False)

	def _blur_augmentation(self, img):
		if config.blur_prob==0.0:
			return img
		if random.random()<config.blur_prob:
			blur_type = random.choice(config.blur_type)
			blur_size = random.choice(config.blur_size)
			kernel = np.zeros([blur_size, blur_size])
			if blur_type=='vertical':
				kernel[:, (blur_size-1)//2] = 1.0
			elif blur_type=='horizontal':
				kernel[(blur_size-1)//2] = 1.0
			elif blur_type=='mean':
				kernel[:] = 1.0
			else:
				print('No such augmentation')
			kernel = kernel / kernel.sum()
			img = cv2.filter2D(img, -1, kernel)
		return img 

	def _pre_process(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = np.float32(img)
		img = img / 255 
		# img = img[:,:,::-1]
		img = img - np.float32([0.485, 0.456, 0.406])
		img = img / np.float32([0.229, 0.224, 0.225])

		img = np.transpose(img, [2,0,1])
		# img = torch.from_numpy(img)
		return img 

	def _get_coco_(self, idx):
		idx = self.coco_img_indices[idx]
		img = cv2.imread(config.coco_data_root + '%012d.jpg'%idx)
		mask = pickle.load(open(config.coco_mask_root + '%012d.pkl'%idx, 'rb'))
		mask = np.float32(mask)
		pts = self.coco_pts_data[idx]
		pts = [i[0] for i in pts]
		pts = np.float32(pts)
		img, mask, pts, _ = self.coco_aff(img, mask, pts, None)
		img = self._pre_process(img)
		hmaps = []
		for i in range(pts.shape[0]):
			hmap = rc.render_heatmap(pts[i,:,:3].copy(), config.out_size, config.base_sigma)
			hmaps.append(hmap)
		hmaps = np.float32(hmaps)
		hmap = np.amax(hmaps, axis=0)
		hmap_final = np.zeros([config.num_pts, config.out_size, config.out_size], dtype=np.float32)
		hmap_final[config.muco_coco_idx] = hmap[config.coco_idx]
		hmap = hmap_final
		pts_res = np.zeros([config.max_inst, config.num_pts, 3], dtype=np.float32)
		pts_res[:len(pts), config.muco_coco_idx] = pts[:,config.coco_idx,:3]
		depth = np.zeros([4,3], dtype=np.float32)  # create dumb array 
		all_depth = np.zeros([4, config.num_pts-1, 3], dtype=np.float32) # create dumb array 
		return img, hmap, mask, pts_res, depth, all_depth, 0

	def _get_muco_(self, idx):
		imgname = config.data_root + self.names[idx]
		img = cv2.imread(imgname)
		h, w = img.shape[0], img.shape[1]

		pts = self.pts_data[idx] # [N, 14, 2]
		pts = np.concatenate([pts, np.ones([pts.shape[0], pts.shape[1], 1], dtype=np.float32)], axis=-1) #[N, 14, 3]
		# print(pts.shape)
		
		img, _, pts, _ = self.aff(img, None, pts, None)
		img = self._pre_process(img)
		hmaps = []
		for i in range(pts.shape[0]):
			hmap = rc.render_heatmap(pts[i,:,:3].copy(), config.out_size, config.base_sigma)
			hmaps.append(hmap)
		hmaps = np.float32(hmaps)
		hmap = np.amax(hmaps, axis=0)
		pts_res = np.zeros([config.max_inst, config.num_pts, 3], dtype=np.float32)
		pts_res[:len(pts)] = pts[:,:,:3]

		p3d = self.pts3d_data[idx]
		depth = np.zeros([len(p3d), 3], dtype=np.float32)
		for i in range(len(p3d)):
			depth[i, :2] = pts_res[i, 0, :2]
			depth[i, 2] = p3d[i,0,2] / config.depth_scaling - config.depth_mean
		all_depth = np.zeros([len(p3d), config.num_pts-1, 3], dtype=np.float32)
		for i in range(len(p3d)):
			all_depth[i, :, :2] = pts_res[i, 1:, :2]
			all_depth[i, :, 2] = (p3d[i,1:,2] - p3d[i,0:1,2]) / config.rel_depth_scaling
		mask = np.zeros([config.out_size, config.out_size], dtype=np.float32)
		return img, hmap, mask, pts_res, depth, all_depth, 1

	def __getitem__(self, idx):
		# print(idx)
		if idx<len(self.names):
			img, hmap, mask, pts_res, depth, all_depth, is_muco = self._get_muco_(idx)
		else:
			idx = (idx - len(self.names)) % len(self.coco_img_indices)
			img, hmap, mask, pts_res, depth, all_depth, is_muco = self._get_coco_(idx)
		return img, hmap, mask, pts_res, depth, all_depth, is_muco

	def __len__(self):
		return len(self.names) + len(self.coco_img_indices) * 2

def get_train_dataloader(bsize, distributed=True):
	dataset = COCO_kpts()
	if config.distributed and distributed:
		sampler = torch.utils.data.distributed.DistributedSampler(dataset)
		shuffle = False
		print('Using distributed data sampler.')
	else:
		sampler = None
		shuffle = True 

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=8, pin_memory=True, sampler=sampler)
	return dataloader, sampler

if __name__=='__main__':
	# import time 
	# reader = DataReader(16)
	# t1 = time.time()
	# for i in range(1):
	# 	img, hmap, inst_maps, mask = reader.get_next()
	# 	# print('TYPE:', ptstype)
	# t2 = time.time()
	# print((t2-t1)/30)
	# import matplotlib.pyplot as plt 
	# plt.imshow(np.amax(hmap[0], axis=0), cmap='jet', vmin=0, vmax=1.0)
	# plt.savefig('a.png')

	# dataset = COCO_kpts()
	# for i in range(10):
	# 	img, hmap, mask, pts = dataset[i]
	# 	print(img.shape, hmap.shape, mask.shape, pts.shape)

	loader, sampler = get_train_dataloader(32, distributed=False)
	for i, (img, hmap, mask, pts_res, depth, all_depth, is_muco) in enumerate(loader):
		print(img.shape, hmap.shape, mask.shape, pts_res.shape, depth.shape, all_depth.shape, is_muco.shape)
		# idx = torch.where(is_muco==1)
		# print(idx)
		# print(depth[:,:,2].max(), depth[:,:,2].min())
		# print(all_depth[:,:,:,2].max(), all_depth[:,:,:,2].min())
