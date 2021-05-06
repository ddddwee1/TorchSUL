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

class MPII_kpts(Dataset):
	def __init__(self):
		self.data = pickle.load(open('mpii_3pts.pkl' , 'rb'))
		print('Index pool MPII:', len(self.data))
		self.aff = RandomAffineTransform(config.inp_size, config.out_size, config.rotation, config.min_scale, config.max_scale, 'short', config.max_translate, False)

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
		img = img - np.float32([0.485, 0.456, 0.406])
		img = img / np.float32([0.229, 0.224, 0.225])

		img = np.transpose(img, [2,0,1])
		return img 

	def _get_minmax(self, pts):
		xs = pts[:,0]
		ys = pts[:,1]
		conf = pts[:,2]
		idx = np.where(conf>0)[0]
		xs = xs[idx]
		ys = ys[idx]
		xmin, xmax = xs.min(), xs.max()
		ymin, ymax = ys.min(), ys.max()
		return xmin, xmax, ymin, ymax

	def _crop_norm(self, img, pts, augment=True):
		xmin, xmax, ymin, ymax = self._get_minmax(pts)
		
		wh = max(ymax - ymin, xmax - xmin)
		scale = config.inp_size / wh * (np.random.random() * 0.2 + 0.6)

		img = cv2.resize(img, None, fx=scale, fy=scale)
		pts[:,:2] = pts[:,:2] * scale

		xmin, xmax, ymin, ymax = self._get_minmax(pts)
		center = [0.5 * (xmax + xmin), 0.5 * (ymin + ymax)]

		xmin = center[0] - config.inp_size /2
		xmin = xmin - np.random.random() * 20 + 10
		ymin = center[1] - config.inp_size /2
		ymin = ymin - np.random.random() * 20 + 10

		H = np.float32([[1,0,-xmin], [0,1,-ymin]])
		img = cv2.warpAffine(img, H, (config.inp_size, config.inp_size))
		pts = pts - np.float32([xmin, ymin, 0])
		return img, pts 

	def _get_mpii_(self, idx):
		imgname = config.data_root + self.data[idx][0]
		pts = self.data[idx][1]

		img = cv2.imread(imgname)
		pts = pts[None, ...]
		pts = np.float32(pts)

		img, pts = self._crop_norm(img, pts[0])
		img, _, pts, _ = self.aff(img, None, pts[None, ...], None)
		pts = pts[0]
		img = self._pre_process(img)
		hmap = rc.render_heatmap(pts.copy(), config.out_size, config.base_sigma)

		return img, hmap

	def __getitem__(self, idx):
		# img, hmap, hmap_match = self._get_mpii_(idx)
		# return img, hmap, hmap_match 
		return self._get_mpii_(idx)

	def __len__(self):
		return len(self.data)

def get_train_dataloader(bsize, distributed=True):
	dataset = MPII_kpts()
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
	import visutil 
	dataset = MPII_kpts()
	img, hmap, kinematics = dataset[0]
	print(img.shape, hmap.shape, kinematics.shape)
	img = visutil.deprocess(img)


	# img = dataset[0]
	# img = visutil.deprocess(img)
	# cv2.imwrite('oo.png', img)
