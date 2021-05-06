import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import numpy as np 
from tqdm import tqdm 
import losses 
import datareader 
from vit import ViT 

if __name__=='__main__':
	devices = (0,1,2,3,4,5,6,7)

	BSIZE = 1024 * 8
	reader = datareader.get_datareader(BSIZE, processes=16)
	# reader = datareader

	BackboneRes100 = ViT(image_size=112, patch_size=16, dim=512, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
	classifier = losses.DistributedClassifier(reader.max_label, devices)

	# init 
	dumb_x = torch.rand(2, 3, 112, 112)
	dumb_y = torch.from_numpy(np.int64(np.zeros(2)))
	dumb_f = BackboneRes100(dumb_x)
	_ = classifier(dumb_f, dumb_y)
	print(dumb_f.shape)

	# restore 
	if devices is not None: 
		BackboneRes100 = nn.DataParallel(BackboneRes100, device_ids=devices).cuda()
		classifier = classifier.cuda()
	saver = M.Saver(BackboneRes100)
	saver_classifier = M.Saver(classifier)
	saver.restore('./model/')
	saver_classifier.restore('./classifier/')

	# define optim 
	# optim = torch.optim.SGD([{'params':BackboneRes100.parameters()}, {'params':classifier.parameters()}], lr=0.001, momentum=0.9, weight_decay=0.0005)
	optim = torch.optim.AdamW([{'params':BackboneRes100.parameters()}, {'params':classifier.parameters()}], lr=0.001, weight_decay=0.0005)

	classifier.train()
	BackboneRes100.train()

	MAXEPOCH = 45

	for ep in range(MAXEPOCH):
		if ep<5:
			m2 = 0.0
		else:
			m2 = 0.5
		print('m2=',m2)
		bar = tqdm(range(reader.iter_per_epoch))
		for it in bar:

			imgs, labels = reader.get_next()
			# imgs, labels = reader.post_process(123)
			imgs = torch.from_numpy(imgs)
			labels = torch.from_numpy(labels)

			# training loop 
			optim.zero_grad()
			feats = BackboneRes100(imgs)
			loss, acc = classifier(feats, labels, m2=m2)
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
				saver.save('./model/vit_%d_%d.pth'%(ep, it))
				saver_classifier.save('./classifier/classifier_%d_%d.pth'%(ep,it))

		saver.save('./model/vit_%d_%d.pth'%(ep, it))
		saver_classifier.save('./classifier/classifier_%d_%d.pth'%(ep,it))

		if ep%25==14:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr
