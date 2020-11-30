import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import numpy as np 
from tqdm import tqdm 
import resnet 
import losses 
import dr as datareader 
import time 
from time import gmtime, strftime

if __name__=='__main__':
	devices = (0,1,2,3,4,5,6,7)
	print('Devices:', devices)

	BSIZE = 128 * 8
	loader, max_label, _ = datareader.get_train_dataloader(BSIZE)

	BackboneRes100 = resnet.Res50(dim=512)
	print('Classifier class:', max_label)
	classifier = losses.DistributedClassifier(max_label, devices)

	# init 
	dumb_x = torch.from_numpy(np.float32(np.zeros([2,3,128,128])))
	dumb_y = torch.from_numpy(np.int64(np.zeros(2)))
	_ = BackboneRes100(dumb_x)
	_ = classifier(_, dumb_y)

	# restore 
	if devices is not None: 
		BackboneRes100 = nn.DataParallel(BackboneRes100, device_ids=devices).cuda()
		classifier = classifier.cuda()
	saver = M.Saver(BackboneRes100)
	saver_classifier = M.Saver(classifier)
	saver.restore('./model/')
	saver.restore('./model_r50_finetune/')
	classifier.load('./classifier/', start_class=0)
	classifier.load('./classifier_finetune/', start_class=0)

	# define optim 
	optim = torch.optim.SGD([{'params':BackboneRes100.parameters()}, {'params':classifier.parameters()}], lr=0.001, momentum=0.9, weight_decay=0.0005)
	# optim = torch.optim.SGD(BackboneRes100.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
	classifier.train()
	BackboneRes100.train()

	MAXEPOCH = 320

	for ep in range(MAXEPOCH):
		m2 = 0.0
		if ep >= 30 :
			m2 = 0.3
		if ep >= 45:
			m2 = 0.5
		print('Epoch:', ep, 'm2', m2)
		bar = tqdm(loader)
		for it, (imgs_id, imgs_capt, labels, imgs_wild, label2) in enumerate(bar):
		# for it, (imgs_wild, label2) in enumerate(bar):
			# training loop 
			optim.zero_grad()

			# labels = labels.long()
			# label2 = label2.long()
			feats_wild = BackboneRes100(imgs_wild)
			loss_wild, acc_wild = classifier(feats_wild, label2, m2=m2)

			if ep>=0:
				feats_capt = BackboneRes100(imgs_capt)
				loss_capt, acc_capt = classifier(feats_capt, labels, m2=m2)

				feats_id = BackboneRes100(imgs_id)
				loss_id, acc_id = classifier(feats_id, labels, m2=m2)

				feats_capt_norm = feats_capt / feats_capt.norm(p=2,dim=1,keepdim=True) 
				feats_id_norm = feats_id / feats_id.norm(p=2,dim=1,keepdim=True)
				distance = torch.sum(feats_capt_norm * feats_id_norm, dim=1) 
				feat_loss = torch.exp(2 - 2*distance) - 1.0
				feat_loss = feat_loss.mean()

				coef = loss_capt / loss_id
				coef = torch.pow(coef, 1.5)

			else:
				# loss_wild = 0
				# acc_wild = 0
				loss_id = 0
				acc_id = 0
				loss_capt = 0
				acc_capt = 0
				feat_loss = 0
				coef = 0

			loss = 1 * loss_id + 1 * loss_capt + loss_wild * 2
			# loss = loss_wild
			loss.backward()
			optim.step()

			# output string 
			lr = optim.param_groups[0]['lr']
			loss = loss.cpu().detach().numpy()
			# acc = acc.cpu().detach().numpy()
			outstr = 'Ep:%d LsID:%.3f LsCapt:%.3f LsWild:%.3f LsFeat:%.3f AcID:%.3f AcCapt:%.3f AcWild:%.3f Coef:%.3f Lr:%.1e'\
						%(ep, loss_id, loss_capt, loss_wild, feat_loss, acc_id, acc_capt, acc_wild, coef, lr)
			# print(outstr)
			bar.set_description(outstr)

			# save model 
			if it%8000==0 and it>0:
				saver.save('./model_r50_finetune/res50_%d_%d.pth'%(ep+1, it))
				classifier.save('./classifier_finetune/classifier_%d_%d.pth'%(ep+1,it))
		if ep%20==19 and ep>0:
			saver.save('./model_r50_finetune/res50_%d_%d.pth'%(ep+1, it))
			classifier.save('./classifier_finetune/classifier_%d_%d.pth'%(ep+1,it))

		if (ep-0)%90==0 and ep>20:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr
