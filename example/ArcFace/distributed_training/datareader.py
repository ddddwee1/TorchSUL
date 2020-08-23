import numpy as np 
import cv2 
import random 
from TorchSUL import sulio 
from torch.utils.data import Dataset
import torch 
import torch.multiprocessing as mp 

def adjust_img(img):
	# img = img[:16*4]
	if random.random()>0.5:
		img = np.flip(img, axis=1)
	img = np.float32(img)
	# img = img[:64]
	img = img / 127.5 - 1.
	img = np.transpose(img, (2,0,1))
	return img 

class FaceDataset(Dataset):
	def __init__(self):
		self.ioout = sulio.DataFrame('../500k_cleanv3/500kfull_v3_02/', debug=True)
		self.idx, self.max_label = self.get_data()
		# self.lock = mp.Lock()

	def get_data(self, img_thresh=0):
		result = []
		_,header0 = self.ioout.read(0)

		lab = 0
		for idd in range(header0[0], header0[1]):
			_, header = self.ioout.read(idd)

			imgrange = (header[0], header[1])
			if imgrange[1]-imgrange[0]<img_thresh:
				continue
			else:
				buff = list(range(imgrange[0], imgrange[1]))
				for item in buff:
					result.append([item, lab])
			lab += 1
		return result, lab

	def __getitem__(self, index):
		idx, lab = self.idx[index]
		# self.lock.acquire()
		img = self.ioout.read_data(idx)
		# self.lock.release()
		img = np.frombuffer(img, dtype=np.uint8)
		img = cv2.imdecode(img, cv2.IMREAD_COLOR)
		# print(img.shape)
		img = adjust_img(img)
		label = int(lab)
		return img, label

	def __len__(self):
		return len(self.idx)

def get_train_dataloader(bsize, distributed=False):
	# mp.set_start_method('spawn')
	dataset = FaceDataset()
	if distributed:
		sampler = torch.utils.data.distributed.DistributedSampler(dataset)
		shuffle = False
		print('Using distributed data sampler.')
	else:
		sampler = None
		shuffle = True 

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=shuffle, pin_memory=True, sampler=sampler, drop_last=True)
	return dataloader, dataset.max_label


if __name__=='__main__':
	import time 
	loader, max_label = get_train_dataloader(2048)
	t2 = time.time()
	for i,(img, label) in enumerate(loader):
		t1 = time.time()
		print(t1-t2)
		print(img.shape, label.shape)
		t2 = time.time()
		if i==2:
			break 
