import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import numpy as np 
from tqdm import tqdm 
import resnet 
import losses 
import datareader 
import time 

if __name__=='__main__':
	devices = (0,1,2,3,)

	BSIZE = 256 * 4
	reader = datareader.get_datareader(BSIZE, processes=16)
	# reader = datareader

	BackboneRes100 = resnet.Res100()
	classifier = losses.DistributedClassifier(reader.max_label, devices)

	# init 
	dumb_x = torch.from_numpy(np.float32(np.zeros([2,3,112,112])))
	dumb_y = torch.from_numpy(np.int64(np.zeros(2)))
	_ = BackboneRes100(dumb_x)
	_ = classifier(_, dumb_y)

	# restore 
	if devices is not None: 
		BackboneRes100 = nn.DataParallel(BackboneRes100, device_ids=devices).cuda()
		classifier = classifier.cuda()
	saver = M.Saver(BackboneRes100)
	saver_classifier = M.Saver(classifier)
	saver.restore('./model_r100/')
	saver_classifier.restore('./classifier/')

	# define optim 
	optim = torch.optim.SGD([{'params':BackboneRes100.parameters()}, {'params':classifier.parameters()}], lr=0.1, momentum=0.9, weight_decay=0.0005)
	classifier.train()
	BackboneRes100.train()

	MAXEPOCH = 10

	for ep in range(MAXEPOCH):
		bar = tqdm(range(reader.iter_per_epoch))
		for it in bar:
			imgs, labels = reader.get_next()
			# imgs, labels = reader.post_process(123)
			imgs = torch.from_numpy(imgs)
			labels = torch.from_numpy(labels)

			# training loop 
			optim.zero_grad()
			feats = BackboneRes100(imgs)
			loss, acc = classifier(feats, labels)
			loss.backward()
			optim.step()

			# output string 
			lr = optim.param_groups[0]['lr']
			loss = loss.cpu().detach().numpy()
			# acc = acc.cpu().detach().numpy()
			outstr = 'Ep:%d Ls:%.3f Ac:%.3f Lr:%.1e'%(ep, loss, acc, lr)
			bar.set_description(outstr)

			# save model 
			if it%8000==0 and it>0:
				saver.save('./model_r100/res100_%d_%d.pth'%(ep, it))
				saver_classifier.save('./classifier/classifier_%d_%d.pth'%(ep,it))
		saver.save('./model_r100/res50_%d_%d.pth'%(ep, it))
		saver_classifier.save('./classifier/classifier_%d_%d.pth'%(ep,it))

		if ep%3==2:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr
