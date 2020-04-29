import os 
import os.path as osp 
import pickle 
import threading 
import logging 

class DataFrame():
	def __init__(self, foldername, mode='r', debug=False):
		assert mode in ['r', 'w'], 'only read or write mode'
		foldername = osp.normpath(foldername)
		basename = osp.basename(foldername)
		self.basename = basename 
		self.mode = mode
		self._closed = False
		self.logger = logging.getLogger('DataFrame')
		if debug:
			self.logger.setlevel(logging.DEBUG)
		else:
			self.logger.setlevel(logging.WARNING)

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
		self.idxinfo[len(self.idxinfo)] = idxinfo_curr
		self.metainfo[len(self.metainfo)] = metadata
		self.datafile.write(bytedata)

	def write_idx(self, idx, bytedata, metadata=[]):
		assert self.mode=='w', 'Must in write mode'
		idxinfo_curr = [self.idxfile.tell(), len(bytedata)] # start position, datalength, 
		if idx in self.idxinfo:
			self.logger.warning('Duplicate index number: %d. This data will not be written to file.'%idx)
		if len(bytedata)==0:
			self.logger.info('Write data length = 0')
		self.idxinfo[idx] = idxinfo_curr
		self.metainfo[idx] = metadata
		self.datafile.write(bytedata)

	def read_meta(self, idx):
		assert self.mode=='r', 'Must in read mode'
		if self.metainfo is None:
			self.metainfo = pickle.load(self.metafile)
		return self.metainfo[idx]

	def read_data(self, idx):
		assert self.mode=='r', 'Must in read mode'
		start_pos, datalen = self.idxinfo[idx]
		if datalen==0:
			return b''
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
		self.df.write(bytedata, metadata)
		self.lock.release()

	def write_idx(self, idx, bytedata, metadata=[]):
		assert self.mode=='w', 'Must in write mode'
		# if there is a large speed gap between input and io write, this may cause memory problem 
		th = threading.Thread(target=self._write, args=(idx, bytedata, metadata,))
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
