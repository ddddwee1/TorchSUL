import numpy as np 
import cv2 
from TorchSUL import DataReader
import random 
from TorchSUL import sulio 

ioout = sulio.DataFrame('../data_emore_sul/', debug=True)
block_size = 32
stride = 16
n_grid = 6

def adjust_img(img):
	# img = img[:16*4]
	if random.random()>0.5:
		img = np.flip(img, axis=1)
	img = np.float32(img)
	img = img / 127.5 - 1.
	img = np.transpose(img, (2,0,1))
	res = []
	for i in range(n_grid):
		for j in range(n_grid):
			start_h = i*stride
			start_w = j*stride
			res.append(img[:,start_h:start_h+block_size, start_w:start_w+block_size])
	res = np.float32(res)
	return res 

def pre_process(inp):
	idx, labels = list(zip(*inp))
	frames = []
	for i in idx:
		# print(i)
		img = ioout.read_data(i)
		frames.append(img)
		# print(type(img))
	pack = list(zip(*[frames, labels]))
	return pack

def process(sample):
	# add more process here
	img, lab = sample
	img = np.frombuffer(img, dtype=np.uint8)
	img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	img = adjust_img(img)
	label = int(lab)
	return img, label

def post_process(inp):
	res = list(zip(*inp))
	imgs = np.float32(res[0])
	lb = np.int64(res[1])
	res = [imgs, lb]
	return res 

def get_data(img_thresh=0):
	result = []	
	_,header0 = ioout.read(0)

	lab = 0
	for idd in range(header0[0], header0[1]):
		_, header = ioout.read(idd)

		imgrange = (header[0], header[1])
		if imgrange[1]-imgrange[0]<img_thresh:
			continue
		else:
			buff = list(range(imgrange[0], imgrange[1]))
			for item in buff:
				result.append([item, lab])
		lab += 1
	return result, lab

def get_datareader(bsize, processes):
	reader = DataReader.DataReader(bsize, processes=processes, gpus=1, sample_policy='EPOCH')
	data, max_label = get_data()
	print('MAX LABEL:', max_label)
	print('DATA NUM:', len(data))
	reader.set_data(data)
	reader.set_param({'max_label':max_label})
	reader.set_pre_process_fn(pre_process)
	reader.set_process_fn(process)
	reader.set_post_process_fn(post_process)
	reader.prefetch()
	return reader

if __name__=='__main__':
	# s = imgrec.read_idx(1000)
	# hdd, img = recordio.unpack(s)
	# img = np.frombuffer(img, dtype=np.uint8)
	# img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	# cv2.imwrite('abc.jpg', img)
	# label = int(hdd.label)
	# print(label)

	# import time 
	# reader = get_datareader(16, 4)
	# t1 = time.time()
	# for i in range(100):
	# 	batch = reader.get_next()
	# 	print(i)
	# t2 = time.time()
	# print('TIME',t2-t1)
	# print(reader.iter_per_epoch, reader.max_label, len(reader.data))
	# print(batch[0].shape, batch[0].min(), batch[0].max())
	# print(batch[1].shape)
	# print(batch[1])

	from time import time 

	reader = get_datareader(1024, 4)

	for i in range(2):
		t1 = time()
		batch = reader.get_next()
		t2 = time()
		print(t2-t1)
	print(batch[0].shape)
	print(batch[1].shape)
