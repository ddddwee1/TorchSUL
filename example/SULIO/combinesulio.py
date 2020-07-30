from TorchSUL import sulio 
import numpy as np 
import cv2 
from tqdm import tqdm

class SulIOSaver():
	def __init__(self, outname, max_id):
		self.ioout = sulio.DataFrame(outname, 'w', debug=True)
		self.class_metas = []
		self.pos = 1

	def push_sulio(self, inname):
		print('Pushing SULIO:', inname)
		ioin = sulio.DataFrame(inname, debug=True)
		_,header0 = ioin.read(0)

		for idd in tqdm(range(header0[0], header0[1])):
			_, header = ioin.read(idd)
			meta = [self.pos]
			for idx in range(header[0], header[1]):
				img = ioin.read_data(idx)
				self.ioout.write_idx(self.pos, img)
				self.pos += 1 
			meta.append(self.pos)
			self.class_metas.append(meta)

	def push_mxrecord(self, recname):
		from mxnet import recordio 
		print('Pushing mxrec:', recname)
		imgrec = recordio.MXIndexedRecordIO(recname+'.idx', recname+'.rec', 'r')
		header,_ = recordio.unpack(imgrec.read_idx(0))
		header0 = (int(header.label[0]), int(header.label[1]))
		print('datalen', header0[1]-header0[0])
		bar = tqdm(range(header0[0], header0[1]))
		for idd in bar:
			s = imgrec.read_idx(idd)
			header, _ = recordio.unpack(s)
			imgrange = (int(header.label[0]), int(header.label[1]))
			meta = [self.pos]

			for idx in range(imgrange[0], imgrange[1]):
				s = imgrec.read_idx(idx)
				hdd, img = recordio.unpack(s)
				self.ioout.write_idx(self.pos, img)
				self.pos += 1 
			meta.append(self.pos)
			self.class_metas.append(meta)

	def push_folder(self, foldername):
		print('Pushing folder:', foldername)
		import glob 
		if foldername[-1]=='/':
			foldername = f'{foldername}*'
		else:
			foldername = f'{foldername}/*'
		foldernames = glob.glob(foldername)
		for folder in tqdm(foldernames):
			imgs = glob.glob(f'{folder}/*.*')

			meta = [self.pos]
			for i in imgs:
				img = open(i, 'rb').read()
				self.ioout.write_idx(self.pos, img)
				self.pos += 1 
			meta.append(self.pos)
			self.class_metas.append(meta)

	def finish(self):
		print('Dumping class meta...')
		meta0 = [self.pos]
		for meta in tqdm(self.class_metas):
			self.ioout.write_idx(self.pos, None, meta)
			self.pos += 1 
		meta0.append(self.pos)
		self.ioout.write_idx(0, None, meta0)


saver = SulIOSaver('/data/500k_cleanv3/data_emorechild/')
# saver.push_sulio('/data/500k_cleanv3/500kfull_v3_02/')
saver.push_mxrecord('/data/face_data/data_emore/emore_v2')
saver.push_folder('./children_cropped/')
saver.finish()
