import numpy as np 
import cv2 
from TorchSUL import DataReader
import random 
import time 
from mxnet import recordio 
from mxnet import io 

recname = '../data_emore/emore_v2'
print('Reading rec:',recname)
imgrec = recordio.MXIndexedRecordIO(recname+'.idx', recname+'.rec', 'r')

def adjust_img(img):
	# img = img[:16*4]
	if random.random()>0.5:
		img = np.flip(img, axis=1)
	img = np.float32(img)
	img = img / 127.5 - 1.
	img = np.transpose(img, (2,0,1))
	return img 

def process(idx):
	# add more process here
	s = imgrec.read_idx(idx)
	hdd, img = recordio.unpack(s)
	img = np.frombuffer(img, dtype=np.uint8)
	img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	img = cv2.resize(img, (64,64))
	img = adjust_img(img)
	label = int(hdd.label)
	return img, label

def post_process(inp):
	res = list(zip(*inp))
	imgs = np.float32(res[0])
	# lb = np.zeros([len(imgs), res[2][0]], dtype=np.float32)
	# labels_row, labels_col = np.int32(list(range(len(imgs)))), np.int32(res[1])
	# lb[labels_row,labels_col] = 1
	lb = np.int64(res[1])
	res = [imgs, lb]
	return res 

def get_data(img_thresh=0, max_label=999999999):
	result = []
	s = imgrec.read_idx(0)
	header,_ = recordio.unpack(s)
	header0 = (int(header.label[0]), int(header.label[1]))
	max_label = min(header0[1] - header0[0], max_label)
	for idd in range(header0[0], min(header0[1], header0[0]+max_label)):
		s = imgrec.read_idx(idd)
		header, _ = recordio.unpack(s)
		imgrange = (int(header.label[0]), int(header.label[1]))
		if imgrange[1]-imgrange[0]<img_thresh:
			continue
		else:
			result += list(range(imgrange[0], imgrange[1]))
	return result, max_label

def get_datareader(bsize, processes):
	reader = DataReader.DataReader(bsize, processes=processes, gpus=1, sample_policy='EPOCH')
	data, max_label = get_data(max_label=10000)
	print('MAX LABEL:', max_label)
	print('DATA NUM:', len(data))
	reader.set_data(data)
	reader.set_param({'max_label':max_label})
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

	reader = get_datareader(16, 4)

	batch = reader.get_next()

	img = batch[0][0]
	cv2.imwrite('abc.jpg', img)
	cv2.imwrite('ddd.jpg', img[:16*4])