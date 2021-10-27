import glob 
import numpy as np 
import pickle 
from tqdm import tqdm 
import os 

class FolderDumper():
	def __init__(self, outpath):
		self.outpath = outpath
		self.result = []

	def push_folder(self, folder_path):
		res = []
		folders = glob.glob(os.path.join(folder_path,'*'))
		folders = sorted(folders)
		for i in tqdm(folders):
			imgs = glob.glob(os.path.join(i, '*.*'))
			res.append(imgs)
		self.result.append({'type':'folder', 'data':res})

	def push_sulio(self, path):
		self.result.append({'type':'sulio', 'data':path})

	def push_mxrec(self, path):
		self.result.append({'type':'mxrec', 'data':path})

	def finish(self):
		pickle.dump(self.result, open(self.outpath + '.pkl', 'wb'))

dumper = FolderDumper('PesWebface_r1_id_old_yong')
dumper.push_sulio('WebfaceRound1')
# dumper.push_folder('pad_face_20210511/pad_face_20210511')
dumper.push_folder('ID_faces_cleaned/old')
dumper.push_folder('ID_faces_cleaned/yong')
dumper.finish()
