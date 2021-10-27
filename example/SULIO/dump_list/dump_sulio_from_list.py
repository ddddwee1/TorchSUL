from TorchSUL import sulio 
import numpy as np 
import cv2 
from tqdm import tqdm
import pickle 
import os 
import shutil 

def check_suilio_ids(inname):
	print('ID num of SULIO:', inname)
	ioin = sulio.DataFrame(inname, debug=True)
	_,header0 = ioin.read(0)
	print(header0[1] - header0[0])

	imgnum_total = 0
	for idd in range(header0[0], header0[1]):
		_, header = ioin.read(idd)
		imgnum = header[1] - header[0]
		imgnum_total += imgnum
	print('Image num:', imgnum_total)

class SulIOListSaver():
	def __init__(self, outname, listfile):
		self.list = pickle.load(open(listfile, 'rb'))
		self.ioout = sulio.DataFrame(outname, 'w', debug=True)
		self.class_metas = []
		self.pos = 1
		self.info = []
		for item in self.list:
			t = item['type']
			if t=='sulio':
				self.push_sulio(item['data'])
			elif t=='mxrec':
				self.push_mxrecord(item['data'])
			elif t=='folder':
				self.push_folder(item['data'])
			self.info.append({'class_len':len(self.class_metas)})
		pickle.dump(self.info, open(os.path.join(outname, 'info.pkl'), 'wb'))
		shutil.copy(listfile, os.path.join(outname, 'pathinfo.pkl'))

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

	def push_folder(self, folderlist):
		for imgs in tqdm(folderlist):
			meta = [self.pos]
			for i in imgs:
				img = open(i, 'rb').read()
				self.ioout.write_idx(self.pos, img)
				self.pos += 1 
			meta.append(self.pos)
			if meta[1] == meta[0]:
				print('Warining: empty folder appears:', folder)
			self.class_metas.append(meta)

	def finish(self):
		print('Dumping class meta...')
		meta0 = [self.pos]
		for meta in tqdm(self.class_metas):
			self.ioout.write_idx(self.pos, None, meta)
			self.pos += 1 
		meta0.append(self.pos)
		self.ioout.write_idx(0, None, meta0)

saver = SulIOListSaver('./PesWebface_r1_id_old_yong/', 'PesWebface_r1_id_old_yong.pkl')
saver.finish()

# check_suilio_ids('./PesWebface_r2_id_old_yong/')
