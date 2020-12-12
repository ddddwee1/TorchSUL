import resnet 
from TorchSUL import Model as M 
import pickle 
import os 
import torch 
import cv2 
import numpy as np 
import time 

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

class Reader():
	def __init__(self):
		self.captpaths = pickle.load(open('captlist_left.pkl', 'rb'))
		self.captpaths = self.captpaths
		self.pos = 0
		self.start_time = time.time()

	def adjust_img(self,img):
		img = cv2.resize(img, (128, 128))
		img = np.float32(img)
		img = img / 127.5 - 1.
		img = np.transpose(img, (2,0,1))
		return img 

	def get_next(self):
		t1 = time.time()
		remaining_time = (len(self.captpaths)-self.pos) * (t1-self.start_time) / (self.pos + 1) /60
		print('\r Class %d/%d  ETA(min):%.2f'%(self.pos, len(self.captpaths), remaining_time), end='')
		if self.pos>=len(self.captpaths):
			return None 
		else:
			imgpaths = self.captpaths[self.pos]
			if len(imgpaths)>256:
				imgpaths = imgpaths[:256]
			imgpaths = [i.replace('/data/face_data/', '/home/ddwe_cy/idface/data/') for i in imgpaths]
			imgs = [cv2.imread(i) for i in imgpaths]
			imgs = [self.adjust_img(i) for i in imgs]
			imgs = np.float32(imgs)
			# if imgs.shape[0]>512:
			# 	imgs = imgs[:512]
			self.pos += 1 
			return imgs 

if __name__=='__main__':
	DIM = 512
	devices = (0,1,2,3,4,5)
	with torch.no_grad():
		mtx_prev = torch.load('./classifier_ft/classifier_40_200.pth')
		print('Previous classifier shape:', mtx_prev.shape)
		feats = []

		net = resnet.Res50(dim=DIM)
		dumb_x = torch.from_numpy(np.float32(np.zeros([2,3,128,128])))
		zeros = torch.zeros(2, 1, 1, 1)
		
		net(dumb_x, zeros)
		saver = M.Saver(net)
		saver.restore('./model_ft/')

		net = torch.nn.DataParallel(net, device_ids=devices).cuda()
		net.eval()
		net.cuda()

		reader = Reader()
		zeros = zeros.cuda()

		while 1:
			imgs = reader.get_next()
			if imgs is None:
				break 
			imgs = torch.from_numpy(imgs)
			feat = net(imgs.cuda(), zeros)
			feat = feat.mean(dim=0)
			feat = feat / torch.norm(feat, p=2)
			feats.append(feat)
			if len(feats)==100:
				ff = torch.stack(feats)
				torch.save(ff.cpu(), 'abc.pth')


		feats = torch.stack(feats, dim=0)
		print('Appended classifier shape:',feats.shape)
		if not os.path.exists('./classifier_precom0/'):
			os.mkdir('./classifier_precom0/')
		torch.save(feats.cpu(), './classifier_precom0/pre_computed.pth')
		feats = torch.cat([mtx_prev, feats.cpu()], dim=0)
		print('Final classifier shape:',feats.shape)

		if not os.path.exists('./classifier_precom/'):
			os.mkdir('./classifier_precom/')
		torch.save(feats.cpu(), './classifier_precom/pre_computed.pth')

		ckpt = open('./classifier_precom/checkpoint','w')
		ckpt.write('pre_computed.pth')
		ckpt.close()
