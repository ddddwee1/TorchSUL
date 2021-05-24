import cv2 
import numpy as np 
import config 
import pickle 
# from multiprocessing.pool import ThreadPool
from multiprocessing import Pool 
import random 
import torch 
from torch.utils.data import Dataset
import os 

class RandomAffineTransform(object):
	# borrow from higher-hrnet
    def __init__(self,
                 input_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 scale_type,
                 max_translate,
                 scale_aware_sigma=False):
        self.input_size = input_size

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

    def __call__(self, image):
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

        mat_input = self._get_affine_matrix(center, scale, (self.input_size, self.input_size), aug_rot)[:2]
        image = cv2.warpAffine(image, mat_input, (self.input_size, self.input_size))

        return image

class ImageNet(Dataset):
	def __init__(self):
		self.img_paths = pickle.load(open('imglist.pkl' , 'rb'))

		print('Num images:', len(self.img_paths))
		self.aff = RandomAffineTransform(config.inp_size, config.rotation, config.min_scale, config.max_scale, 'short', config.max_translate, True)

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

	def __getitem__(self, index):
		path, lab = self.img_paths[index]
		path = os.path.join(config.data_root, path)
		# print(path)
		img = cv2.imread(path)
		img = self._blur_augmentation(img)

		img = self.aff(img)
		img = self._pre_process(img)

		return img, lab

	def __len__(self):
		return len(self.img_paths)

def get_train_dataloader(bsize, distributed=True):
	dataset = ImageNet()
	if config.distributed and distributed:
		sampler = torch.utils.data.distributed.DistributedSampler(dataset)
		shuffle = False
		print('Using distributed data sampler.')
	else:
		sampler = None
		shuffle = True 

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=4, pin_memory=True, sampler=sampler)
	return dataloader, 1000, sampler

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

	loader, _, _ = get_train_dataloader(256, False)
	for i, (img, labs) in enumerate(loader):
		print(img.shape, labs.shape)
