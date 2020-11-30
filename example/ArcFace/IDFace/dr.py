import numpy as np 
import cv2 
import random 
from TorchSUL import sulio 
from torch.utils.data import Dataset
import torch 
import pickle 
from multiprocessing.pool import ThreadPool
import torch 

def adjust_img(img):
	img = cv2.resize(img, (128, 128))
	img = np.float32(img)
	img = img / 127.5 - 1.
	img = np.transpose(img, (2,0,1))
	return img 

class FaceDataset():
	def __init__(self):
		self.wild_nums = pickle.load(open('metas.pkl', 'rb'))
		self.max_label = len(self.wild_nums)
		print('Class num:',len(self.wild_nums))
		self.wild_idx_list = list(range(self.max_label))
		random.shuffle(self.wild_idx_list)

		self.start_class = 410063
		self.idpaths = pickle.load(open('idlist.pkl', 'rb'))
		self.captpaths = pickle.load(open('captlist.pkl', 'rb'))
		assert len(self.captpaths) == len(self.idpaths)
		assert len(self.captpaths)+self.start_class == self.max_label
		print('ID CAPT Class num:', len(self.captpaths))

	def _load_img(self, path):
		with open(path, 'rb') as f:
			img = f.read()
		img = np.frombuffer(img, dtype=np.uint8)
		img = cv2.imdecode(img, cv2.IMREAD_COLOR)
		img = adjust_img(img)
		return img 

	def reset(self):
		print('Reset dataset')
		random.shuffle(self.wild_idx_list)

	def __getitem__(self, index):
		capts = self.captpaths[index]
		ids = self.idpaths[index]

		path_idphoto = random.choice(ids)
		path_captphoto = random.choice(capts)
		
		img_id = self._load_img(path_idphoto)
		img_capt = self._load_img(path_captphoto)
		label = int(index + self.start_class)

		lab = index
		cnt = np.random.randint(0, self.wild_nums[lab]) 

		wild_path = '/data/face_data/500kfull_IDoldyoung_imgs/%d/%d.jpg'%(lab, cnt)
		img0 = self._load_img(wild_path)
		return img_id, img_capt, label, img0, lab
		# return img0, lab

	def __len__(self):
		return len(self.captpaths)
		# return self.max_label

class DataLoader():
	def __init__(self, dataset, bsize):
		self.dataset = dataset 
		self.dataset.reset()
		self.idx = list(range(len(dataset)))
		random.shuffle(self.idx)
		self.tpool = ThreadPool(processes=2)
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

def get_train_dataloader(bsize):
	print('Using DR...')
	dataset = FaceDataset()
	loader = DataLoader(dataset, bsize)
	max_label = dataset.max_label
	return loader, max_label, None

if __name__=='__main__':
	dataset = FaceDataset()
	loader = DataLoader(dataset, 128)
	for i,(img_id, img_capt, label, img0, lab) in enumerate(loader):
		print(img0.shape, lab.shape)
		print(lab)
		if i==3:
			break 
