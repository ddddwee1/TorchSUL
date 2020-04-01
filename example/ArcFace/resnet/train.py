import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import Model as M 
import losses 
import resnet 
import util 
import datareader 
import time 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class FaceRes100(M.Model):
	def initialize(self, num_classes):
		self.resnet = resnet.Res100()
		self.classifier = losses.MarginalCosineLayer(num_classes)

	def forward(self, x, label):
		feat = self.resnet(x)
		logits = self.classifier(feat, label, 1.0, 0.5, 0.0)
		logits = logits * 64
		return logits

def initialize(data, model):
	x,y = data 
	model(x,y)

if __name__=='__main__':
	BSIZE = 320 * 6
	reader = datareader.get_datareader('../datasetemoreasia/emore_asia_outs.txt', BSIZE, processes=16)
	gpus = (0,1,2,3,4,5)
	model = FaceRes100(reader.max_label + 1)
	# init model 
	batch = reader.get_next()
	imgs = torch.from_numpy(batch[0])[:2]
	labels = torch.from_numpy(batch[1])[:2]
	initialize((imgs, labels), model)
	# parallel training 
	model = nn.DataParallel(model, device_ids=gpus).cuda()
	saver = M.Saver(model)
	saver.restore('./model/')
	optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
	model.train()

	lossfunc = losses.NLLLoss()
	MAXEPOCH = 20
	t0 = time.time()
	for e in range(MAXEPOCH):
		for i in range(reader.iter_per_epoch):
			# t00 = time.time()
			batch = reader.get_next()
			imgs = torch.from_numpy(batch[0])
			labels = torch.from_numpy(batch[1])
			labels = labels.cuda()

			# go training 
			# t11 = time.time()
			optim.zero_grad()
			logits = model(imgs, labels)
			with torch.no_grad():
				acc = util.accuracy(logits, labels)
			loss = lossfunc(logits, labels)
			# t22 = time.time()
			loss2 = loss / len(gpus)
			loss2.backward()
			optim.step()
			# t33 = time.time()
			# print('DATA', t11-t00, 'Forward',t22-t11, 'Back',t33-t22)

			lr = optim.param_groups[0]['lr']
			if i%10==0:
				t1 =time.time()
				speed = BSIZE * 10 / (t1-t0)
				t0 = t1
				print('Epoch:%03d\tIter:%06d/%06d\tLoss:%.4f\tAcc:%.4f\tLR:%.1e\tSpeed:%.2f'%(e,i,reader.iter_per_epoch,loss.cpu().detach().numpy(), acc, lr, speed))
			if i%2000==0 and i>0:
				saver.save('./model/%04d_%06d.pth'%(e,i))
		if e%3==2:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr
		saver.save('./model/%04d_%06d.pth'%(e,i))
