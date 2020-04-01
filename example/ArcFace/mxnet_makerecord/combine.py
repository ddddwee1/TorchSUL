from mxnet import recordio 
from mxnet import io 
import mxnet as mx 
from tqdm import tqdm 
import glob 
import cv2 

class Combiner():
	def __init__(self, recname):
		# recname = 'combine'
		self.recout = recordio.MXIndexedRecordIO( recname+'.idx', recname+'.rec', 'w')
		self.ID_idx = []
		self.idx = 1
		self.idnum = 0

	def push_record(self, recname):
		print('Pushing file: %s ...'%recname)
		imgrec = recordio.MXIndexedRecordIO(recname+'.idx', recname+'.rec', 'r')
		s = imgrec.read_idx(0)
		header,_ = recordio.unpack(s)
		header0 = (int(header.label[0]), int(header.label[1]))
		for idd in tqdm(range(header0[0], header0[1])):
			idxbuff = [self.idx]
			s = imgrec.read_idx(idd)
			header, _ = recordio.unpack(s)
			imgrange = range(int(header.label[0]), int(header.label[1]))
			for imgidx in imgrange:
				s = imgrec.read_idx(imgidx)
				hdd, img = recordio.unpack(s)
				hdd = mx.recordio.IRHeader(0, float(self.idnum), 0, 0)
				s = recordio.pack(hdd, img)
				self.recout.write_idx( self.idx, s)
				self.idx += 1
			idxbuff.append(self.idx)
			self.ID_idx.append(idxbuff)
			self.idnum += 1 

	def push_folder(self, foldername, process_fn=None):
		print('Pushing folder: %s ...'%foldername)
		foldernames = './%s/*'%foldername
		foldernames = glob.glob(foldernames)
		for folder in tqdm(foldernames):
			idxbuff = [self.idx]
			folder = folder + '/*.*'
			imgs = glob.glob(folder)
			for i in imgs:
				img = cv2.imread(i)
				# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				if process_fn:
					img = process_fn(img)
				hdd = mx.recordio.IRHeader(0, float(self.idnum), 0, 0)
				s = recordio.pack_img(hdd, img, quality=100)
				self.recout.write_idx( self.idx, s)
				self.idx += 1 
			idxbuff.append(self.idx)
			self.ID_idx.append(idxbuff)
			self.idnum += 1 

	def push_folder_raw(self, foldername):
		print('Pushing raw folder: %s ...'%foldername)
		foldernames = './%s/*'%foldername
		foldernames = glob.glob(foldernames)
		for folder in tqdm(foldernames):
			idxbuff = [self.idx]
			folder = folder + '/*.*'
			imgs = glob.glob(folder)
			for i in imgs:
				with open(i,'rb') as imraw:
					img = imraw.read()
				hdd = mx.recordio.IRHeader(0, float(self.idnum), 0, 0)
				s = recordio.pack(hdd, img)
				self.recout.write_idx( self.idx, s)
				self.idx += 1 
			idxbuff.append(self.idx)
			self.ID_idx.append(idxbuff)
			self.idnum += 1 

	def finish(self):
		startheader = mx.recordio.IRHeader(0, [float(self.idx), float(self.idx + len(self.ID_idx))], 0, 0)
		startheader = mx.recordio.pack(startheader, b'')
		self.recout.write_idx(0, startheader)

		print('Writing headers...')
		for i in tqdm(self.ID_idx):
			idheader = mx.recordio.IRHeader(0, [float(i[0]), float(i[1])], 0, 0)
			idheader = mx.recordio.pack(idheader, b'')
			self.recout.write_idx(self.idx, idheader)
			self.idx += 1

def crop_img(img):
	return img[33:-33, 33:-33]

comb = Combiner('emore_seg')
# comb.push_record('emore')
# comb.push_record('asian')
# comb.push_folder('part0',process_fn=crop_img)
# comb.push_folder('part1',process_fn=crop_img)
# comb.push_folder('part2',process_fn=crop_img)
# comb.push_folder('part3',process_fn=crop_img)
comb.push_folder_raw('images')
comb.finish()
