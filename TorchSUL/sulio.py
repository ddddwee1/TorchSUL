import os 
import os.path as osp 
import pickle 
import threading 
import logging 
import cv2 
from multiprocessing import Pool 

class DataFrame():
	def __init__(self, foldername, mode='r', debug=False):
		print('Initializing sulio dataframe...')
		assert mode in ['r', 'w'], 'only read or write mode'
		foldername = osp.normpath(foldername)
		basename = osp.basename(foldername)
		self.basename = basename 
		self.mode = mode
		self._closed = False
		self.logger = logging.getLogger('DataFrame')
		if debug:
			self.logger.setLevel(logging.DEBUG)
		else:
			self.logger.setLevel(logging.WARNING)

		if not osp.exists(foldername):
			os.makedirs(foldername)

		if mode=='r':
			self.idxfile = open(f'{foldername}/{basename}.idx', 'rb')
			self.datafile = open(f'{foldername}/{basename}.data', 'rb')
			self.metafile = open(f'{foldername}/{basename}.meta', 'rb')
			self.idxinfo = pickle.load(self.idxfile)
			self.metainfo = None 
		elif mode=='w':
			self.idxfile = open(f'{foldername}/{basename}.idx', 'wb')
			self.datafile = open(f'{foldername}/{basename}.data', 'wb')
			self.metafile = open(f'{foldername}/{basename}.meta', 'wb')
			self.idxinfo = {}
			self.metainfo = {}

	def write(self, bytedata, metadata=[]):
		assert self.mode=='w', 'Must in write mode'
		idxinfo_curr = [self.idxfile.tell(), len(bytedata)] # start position, datalength, 
		if (bytedata is None) or (len(bytedata)==0):
			self.logger.info('Write data length = 0')
		else:
			self.datafile.write(bytedata)
		self.idxinfo[len(self.idxinfo)] = idxinfo_curr
		self.metainfo[len(self.metainfo)] = metadata

	def write_idx(self, idx, bytedata, metadata=[]):
		assert self.mode=='w', 'Must in write mode'
		
		if idx in self.idxinfo:
			self.logger.warning('Duplicate index number: %d. This data will not be written to file.'%idx)
		
		if (bytedata is None) or (len(bytedata)==0):
			self.logger.info('Write data length = 0')
			idxinfo_curr = [self.datafile.tell(), 0]
		else:
			idxinfo_curr = [self.datafile.tell(), len(bytedata)] # start position, datalength, 
			self.datafile.write(bytedata)

		self.idxinfo[idx] = idxinfo_curr
		self.metainfo[idx] = metadata

	def read_meta(self, idx):
		assert self.mode=='r', 'Must in read mode'
		if self.metainfo is None:
			self.metainfo = pickle.load(self.metafile)
		return self.metainfo[idx]

	def read_data(self, idx):
		assert self.mode=='r', 'Must in read mode'
		start_pos, datalen = self.idxinfo[idx]
		if datalen==0:
			# return b''
			return None # None return should be more pythonic
		else:
			self.datafile.seek(start_pos)
			data = self.datafile.read(datalen)
			return data 

	def read(self, idx):
		data = self.read_data(idx)
		meta = self.read_meta(idx)
		return data, meta

	def close(self):
		if not self._closed:
			if self.mode=='w':
				print(self.mode)
				print('Dumping meta data...')
				pickle.dump(self.idxinfo, self.idxfile)
				pickle.dump(self.metainfo, self.metafile)
			self.idxfile.close()
			self.metafile.close()
			self.datafile.close()
			self._closed = True

	def __del__(self):
		self.close()

	def __len__(self):
		return len(self.idxinfo)

# TO-DO check whether it is faster
# class MultiWorkerDataFrame():
# 	def __init__(self, foldername, mode='r', worker=2, debug=False):
# 		self.mode = mode 
# 		self.lock = threading.Lock()
# 		self.df = DataFrame(foldername, mode, debug=debug)
# 		self.pool = Pool(processes=worker)

# 	def read(self, idx):
# 		return self.df.read(idx)

# 	def read_meta(self, idx):
# 		return self.df.read_meta(idx)

# 	def read_data(self, idx):
# 		return self.df.read_data(idx)

# 	def read_multi(self, idxlist):


class ThreadedDataFrame():
	def __init__(self, foldername, mode='r', debug=False):
		self.mode = mode 
		self.lock = threading.Lock()
		self.df = DataFrame(foldername, mode, debug=debug)

	def _write(self, bytedata, metadata):
		self.lock.acquire()
		self.df.write(bytedata, metadata)
		self.lock.release()

	def write(self, bytedata, metadata=[]):
		assert self.mode=='w', 'Must in write mode'
		# if there is a large speed gap between input and io write, this may cause memory problem 
		th = threading.Thread(target=self._write, args=(bytedata, metadata,))
		th.start()
		th.join()

	def _write_idx(self, idx, bytedata, metadata):
		self.lock.acquire()
		self.df.write_idx(idx, bytedata, metadata)
		self.lock.release()

	def write_idx(self, idx, bytedata, metadata=[]):
		assert self.mode=='w', 'Must in write mode'
		# if there is a large speed gap between input and io write, this may cause memory problem 
		th = threading.Thread(target=self._write_idx, args=(idx, bytedata, metadata,))
		th.start()
		th.join()

	def read_meta(self, idx):
		self.lock.acquire()
		meta = self.df.read_meta(idx)
		self.lock.release()
		return meta

	def read_data(self, idx):
		self.lock.acquire()
		data = self.df.read_data(idx)
		self.lock.release()
		return data 

	def read(self, idx):
		self.lock.acquire()
		result = self.df.read(idx)
		self.lock.release()
		return result

	def close(self):
		self.lock.acquire()
		self.df.close()
		self.lock.release()

	def __del__(self):
		self.close()

	def __len__(self):
		return len(self.df)

	def __getstate__(self):
		d = {}
		return d 

	def __setstate__(self):
		pass 

class SulIOExtractor():
	def __init__(self, inname):
		self.ioin = sulio.DataFrame(inname, debug=True)
		self.metas = []

	def write_imgs(self, outfolder, start_class=0):
		if not os.path.exists(outfolder):
			os.makedirs(outfolder)
		_,header0 = self.ioin.read(0)
		for idd in tqdm(range(header0[0], header0[1])):
			_, header = self.ioin.read(idd)
			idfolder = os.path.join(outfolder, '%d'%(len(self.metas)))
			if not os.path.exists(idfolder):
				os.makedirs(idfolder)
			cnt = 0
			for idx in range(header[0], header[1]):
				if len(self.metas)>=start_class:
					img = self.ioin.read_data(idx)
					imgpath = os.path.join(idfolder, '%d.jpg'%cnt)
					with open(imgpath, 'wb') as fout:
						fout.write(img)

				cnt += 1 
			self.metas.append(cnt)

	def finish(self):
		pickle.dump(self.metas, open('metas.pkl', 'wb'))

def decode_img(buf):
	img = np.frombuffer(buf, dtype=np.uint8)
	img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	return img 

def encode_img(img, quality=95, img_format='jpg'):
	jpg_formats = ['.JPG', '.JPEG']
	png_formats = ['.PNG']
	if img_fmt.upper() in jpg_formats:
		encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
	elif img_fmt.upper() in png_formats:
		encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
	ret, buf = cv2.imencode(img_format, img, encode_params)
	assert ret, 'failed to encode image'
	return buf 

def dump_pkl(data, name):
	pickle.dump(data, open(name, 'wb'))

def load_pkl(name):
	return pickle.load(open(name, 'rb'))

if __name__=='__main__':
	df = ThreadedDataFrame('./abc/', 'w')
	df.write(b'adbcbc')
	df.write(b'acbc')
	df.close()

	df = ThreadedDataFrame('./abc/', 'r')
	print(len(df))
	dt = df.read(0)
	print(dt)
	dt = df.read(1)
	print(dt)
