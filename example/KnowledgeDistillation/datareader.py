import numpy as np 
import cv2 
import SUL.DataReader
import random 
import time 

def adjust_img(img):
	if random.random()>0.5:
		img = np.flip(img, axis=1)
	img = np.float32(img)
	img = img / 127.5 - 1.
	img = np.transpose(img, (2,0,1))
	return img 

def process(sample):
	# add more process here
	img, label, max_label = sample
	img = img.replace('./','../dataset/')
	img = cv2.imread(img)
	img = adjust_img(img)
	return img, label, max_label

def post_process(inp):
	res = list(zip(*inp))
	imgs = np.float32(res[0])
	# lb = np.zeros([len(imgs), res[2][0]], dtype=np.float32)
	# labels_row, labels_col = np.int32(list(range(len(imgs)))), np.int32(res[1])
	# lb[labels_row,labels_col] = 1
	lb = np.int64(res[1])
	res = [imgs, lb]
	return res 

def get_data(listfile):
	print('Reading text file...')
	f = open(listfile, 'r')
	data = []
	max_label = 0
	for line in f:
		line = line.strip().split('\t')
		img = line[1]
		label = int(line[2])
		if label>max_label:
			max_label = label
		data.append([img, label])
	print('Text file loaded.')
	for i in range(len(data)):
		data[i].append(max_label+1)
	print('MAX LABEL', max_label)
	return data, max_label

def get_datareader(txtfile, bsize, processes):
	reader = SUL.DataReader.DataReader(bsize, processes=processes, gpus=1, sample_policy='EPOCH')
	data, max_label = get_data(txtfile)
	reader.set_data(data)
	reader.set_param({'max_label':max_label})
	reader.set_process_fn(process)
	reader.set_post_process_fn(post_process)
	reader.prefetch()
	return reader

if __name__=='__main__':
	import time 
	data_reader = get_datareader('imglist_iccv_small.txt', 1024)
	for i in range(10):
		t1 = time.time()
		batch = data_reader.get_next()
		batch = batch[0]
		print(len(batch))
		print(batch[0].shape)
		print(batch[1].shape)
		t2 = time.time()
		print(t2-t1)
