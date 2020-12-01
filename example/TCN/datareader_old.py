import numpy as np 
import pickle 
import random

def normalize_3d(pose):
	# pose: [bsize, 17, 3]
	xs = pose[:,:,0:1]
	ys = pose[:,:,1:2]
	ls = np.sqrt(xs[:,1:]**2 + ys[:,1:]**2)
	scale = np.mean(ls, axis=1, keepdims=True)
	pose = pose/scale
	pose = pose - pose[:,0:1,:]
	return pose

def RotationMatrix(phi, gamma, theta):
	Rx = np.float64([[1,0,0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
	Ry = np.float64([[np.cos(gamma), 0, -np.sin(gamma)], [0,1,0], [np.sin(gamma), 0, np.cos(gamma)]])
	Rz = np.float64([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0,0,1]])
	R = Rx.dot(Ry)
	R = R.dot(Rz)
	return R

def rotate(pose):
	y = random.random() * np.pi 
	x = random.random() * np.pi * 0.2 
	z = random.random() * np.pi * 0.2 
	R = RotationMatrix(x,y,z)
	pose = pose.reshape([-1, 3])
	pose = pose.dot(R)
	pose = pose.reshape([-1,17,3])
	return pose

def random_rot(p3d, cam, root):
	p3d[:,:,0] = 0
	x = np.random.random() * 0.1 - 0.05
	y = np.random.random() * 0.1 + 0.9
	z = np.random.random() * 0.1 - 0.05 
	ang = np.random.random() * np.pi * 2
	aug_vec = np.float32([x, y, z])
	aug_vec = aug_vec / np.linalg.norm(aug_vec) * np.sin(ang)
	aug_qt = np.float32([np.cos(ang), aug_vec[0], aug_vec[1], aug_vec[2]])
	aug_qt = aug_qt[None, ...]
	aug_qt = torch.from_numpy(aug_qt)
	p3d_aug = world_to_camera(p3d, R=aug_qt, t=0)
	p3d_aug = torch.from_numpy(p3d_aug.astype(np.float32))
	p2d_aug = project_to_2d(p3d_aug + root, cam).float()
	# p2d_aug = torch.from_numpy(p2d_aug)
	return p3d_aug, p2d_aug

SAMPLE_INTERVALS = [2,3,5]

class data_reader():
	def __init__(self, bsize, size):
		self.bsize=bsize
		self.size = size 
		data = pickle.load(open('points_flatten.pkl','rb'))
		self.data = []
		for i in data:
			for SAMPLE_INTERVAL in SAMPLE_INTERVALS:
				for s in range(SAMPLE_INTERVAL):
					self.data.append(i[s::SAMPLE_INTERVAL])

	def process_sample(self, data, size, missing_head=True):
		size = size*2+1
		if len(data)<size:
			if missing_head:
				append_head = np.stack([data[0]]*(size-len(data)),axis=0)
				res = np.concatenate([append_head, data], axis=0)
			else:
				append_tail = np.stack([data[-1]]*(size-len(data)),axis=0)
				res = np.concatenate([data, append_tail], axis=0)
		elif len(data)==size:
			res = data 
		else:
			raise ValueError('data size is larger than specified temporal elapse')
		return res 

	def get_next(self):
		res = []
		labels = []
		for i in range(self.bsize):
			sample = random.choice(self.data)
			point = random.randint(0,len(sample)-1)
			label = sample[point]
			sample = sample[ max(0,point-self.size) : min(len(sample), point+self.size+1) ]
			if point-self.size<0:
				missing_head = True
			else:
				missing_head = False
			sample = self.process_sample(sample, self.size, missing_head)
			sample = sample.reshape([-1, 17, 3])
			sample = rotate(sample)
			sample = normalize_3d(sample)
			sample = (random.random() * 0.9 + 0.2) * sample
			label = sample[self.size]
			# print(sample.shape)
			# print(sample[0])
			# input()
			res.append(sample)
			labels.append(label)

		res = np.float32(res)
		labels = np.float32(labels)
		# labels = np.expand_dims(labels, axis=1)

		res = res[:,:,:,:2]
		res = res.reshape([self.bsize, -1, 17*2])

		return res, labels

if __name__=='__main__':
	import time 
	reader = data_reader(64,32)
	t1 = time.time()
	for i in range(100):
		a = reader.get_next()
		print(a[0].shape)
	t2 = time.time()
	print(t2-t1)
