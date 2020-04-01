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
		feat, fmaps = self.resnet(x)
		logits, logits_pure = self.classifier(feat, label, 1.0, 0.5, 0.0)
		logits = logits * 64
		return logits, logits_pure*64, feat, fmaps

class FaceRes34(M.Model):
	def initialize(self, num_classes):
		self.resnet = resnet.Res34()
		self.classifier = losses.MarginalCosineLayer(num_classes)

	def forward(self, x, label):
		feat, fmaps = self.resnet(x)
		logits, logits_pure = self.classifier(feat, label, 1.0, 0.5, 0.0)
		logits = logits * 64
		return logits, logits_pure*64, feat, fmaps

def initialize(data, model):
	x,y = data 
	model(x,y)

def loss_distil(x,y,student,teacher, lossfunc):
	tau = 1
	with torch.no_grad():
		logits_t, logits_t_pure, feat_t, fmaps_t = teacher(x,y)
		# acc_t_pure = util.accuracy(logits_t_pure, y)
		# acc_t_raw = util.accuracy(logits_t, y)
		# print(acc_t_pure, acc_t_raw)
		# for aa in fmaps_t:
		# 	print(aa.shape)

	logits_s, logits_s_pure, feat_s, fmaps_s = student(x,y)
	# for bb in fmaps_s:
	# 	print(bb.shape)
	acc = util.accuracy(logits_s, y)
	acc_pure = util.accuracy(logits_s_pure, y)
	loss_nll = lossfunc(logits_s, y)
	loss_logits = F.softmax(logits_t_pure/tau, dim=-1) * (F.log_softmax(logits_t_pure/tau, dim=-1) - F.log_softmax(logits_s_pure, dim=-1))
	loss_logits = torch.mean(loss_logits, dim=0) # average among batch 
	loss_logits = torch.sum(loss_logits) # sum among classes
	loss_fmap = sum([torch.mean(torch.abs(s - t)) for s,t in zip(fmaps_s, fmaps_t)]) / len(fmaps_t)
	loss_feat = torch.mean(torch.abs(feat_s - feat_t))
	loss_total = loss_nll + loss_feat + loss_logits + loss_fmap
	return acc, acc_pure, loss_total, loss_nll, loss_fmap, loss_feat, loss_logits

if __name__=='__main__':
	BSIZE = 320 * 6
	reader = datareader.get_datareader('../dataset/emore_asia_outs.txt', BSIZE, processes=16)
	gpus = (0,1,2,3,4,5)
	model_teacher = FaceRes100(reader.max_label + 1)
	model_student = FaceRes34(reader.max_label + 1)
	# init model 
	batch = reader.get_next()
	imgs = torch.from_numpy(batch[0])[:2]
	labels = torch.from_numpy(batch[1])[:2]
	initialize((imgs, labels), model_teacher)
	initialize((imgs, labels), model_student)
	# parallel training 
	model_teacher = nn.DataParallel(model_teacher, device_ids=gpus).cuda()
	M.Saver(model_teacher).restore('./model_teacher/')

	model_student = nn.DataParallel(model_student, device_ids=gpus).cuda()
	saver = M.Saver(model_student)
	saver.restore('./model_student/')

	optim = torch.optim.SGD(model_student.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
	model_teacher.train()
	model_student.train()

	lossfunc = losses.NLLLoss()
	MAXEPOCH = 20
	t0 = time.time()
	for e in range(MAXEPOCH):
		for i in range(reader.iter_per_epoch):
			batch = reader.get_next()
			imgs = torch.from_numpy(batch[0])
			labels = torch.from_numpy(batch[1])
			labels = labels.cuda()

			# go training 
			optim.zero_grad()
			acc, acc_pure, loss_total, loss_nll, loss_fmap, loss_feat, loss_logits = loss_distil(imgs, labels, model_student, model_teacher, lossfunc)
			loss_total.backward()
			optim.step()

			lr = optim.param_groups[0]['lr']
			if i%10==0:
				t1 =time.time()
				speed = BSIZE * 10 / (t1-t0)
				t0 = t1
				lossstr = 'Epoch:%03d\tIter:%06d/%06d\t'%(e,i,reader.iter_per_epoch)
				losses = ['Lossttl','Lossnll','Lossfmap','Lossfeat','Losslogits','Acc','AccPure']
				lossstr += util.format_losses(losses, [loss_total, loss_nll, loss_fmap, loss_feat, loss_logits,acc, acc_pure])
				lossstr += '\tLR:%.1e\tSpeed:%.2f'%(lr, speed)
				print(lossstr)
			if i%2000==0 and i>0:
				saver.save('./model/%04d_%06d.pth'%(e,i))
		if e%4==3:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr
		saver.save('./model/%04d_%06d.pth'%(e,i))
