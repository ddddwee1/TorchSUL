import numpy as np 
import pickle 
import random 
from multiprocessing.pool import ThreadPool
import torch 

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

class PtsData():
	def __init__(self, seq_len):
		self.seq_len = seq_len
		self.half_seq_len = seq_len // 2 
		self.data = pickle.load(open('points_flatten2d.pkl' , 'rb'))
		self.idx_list = []
		for i in range(len(self.data)):
			for j in range(len(self.data[i][0])):
				self.idx_list.append([i,j])

	def pad_head_tail(self, seq, head, tail):
		if head>0:
			pad_head = np.float32([seq[0]] * head)
			seq = np.concatenate([pad_head, seq], axis=0)
		if tail>0:
			pad_tail = np.float32([seq[-1]] * tail)
			seq = np.concatenate([seq, pad_tail], axis=0)
		return seq 

	def __getitem__(self, index):
		i,j = self.idx_list[index]
		seq2d, seq3d = self.data[i]
		start = j - self.half_seq_len
		end = j + self.half_seq_len + 1
		head = - start
		tail = end - len(seq2d)
		cropped2d = seq2d[max(0,start) : min(len(seq2d), end)]
		res2d = self.pad_head_tail(cropped2d, head, tail)
		gt3d = seq3d[j:j+1]
		res2d = res2d / 1000
		gt3d = (gt3d - gt3d[:, :1]) / 1000
		return np.float32(res2d), np.float32(gt3d)

	def __len__(self):
		return len(self.idx_list)

	def reset(self):
		pass

class DataLoader():
	def __init__(self, dataset, bsize):
		self.dataset = dataset 
		self.dataset.reset()
		self.idx = list(range(len(dataset)))
		random.shuffle(self.idx)
		self.tpool = ThreadPool(processes=4)
		self.pos = 0
		self.bsize = bsize
		self.prefetch()

	def prefetch(self):
		self.ps = self.tpool.apply_async(self._fetch_func)
		self.pos += self.bsize

	def _fetch_func(self):
		if (self.pos+self.bsize) > len(self.idx):
			return None 
		else:
			batch = self.idx[self.pos: self.pos+self.bsize]
			res = [self.dataset[i] for i in batch]
			res = list(zip(*res))
			res = [np.array(i) for i in res]
			res = [torch.from_numpy(i) for i in res]
			return res 

	def __iter__(self):
		random.shuffle(self.idx)
		self.dataset.reset()
		self.pos = 0
		self.prefetch()
		return self 

	def __next__(self):
		res = self.ps.get()
		if res is None:
			raise StopIteration
		else:
			self.prefetch()
			return res 

	def __len__(self):
		return len(self.idx) // self.bsize

def get_loader(bsize, seq_len):
	dataset = PtsData(seq_len)
	loader = DataLoader(dataset, bsize)
	return loader

if __name__=='__main__':
	loader = get_loader(128, 243)
	for i,(p2d,p3d) in enumerate(loader):
		print(p2d.shape, p3d.shape)
		print(p3d[0])
		if i==2:
			break 
