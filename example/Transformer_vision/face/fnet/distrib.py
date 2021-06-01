import torch 
import torch.distributed as dist 
import numpy as np 
import torch.multiprocessing as mp 
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
import vit 
from TorchSUL import Model as M 
import logging 
import datareader 
import losses 
import random 

def main():
	world_size = torch.cuda.device_count()
	mp.spawn(main_worker, nprocs=world_size, args=(world_size,))

def main_worker(gpu, world_size):
	BSIZE = 384
	FORMAT = '%(asctime)-15s  Replica:%(name)s  %(message)s'
	logging.basicConfig(format=FORMAT)
	logger = logging.getLogger('%d'%gpu)
	logger.setLevel(10)
	logger.info('Initialize process.')
	torch.cuda.set_device(gpu)
	dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=world_size, rank=gpu)
	
	net = vit.FaceVit()
	dumb_x = torch.from_numpy(np.float32(np.zeros([2,3,112,112])))
	with torch.no_grad():
		net(dumb_x)
	saver = M.Saver(net)
	saver.restore('./model/')
	net.cuda(gpu)
	net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
	net.train()
	logger.info('Model initialization finished.')

	loader, max_label, sampler = datareader.get_train_dataloader(BSIZE, distributed=True)
	classifier = losses.SplitClassifier(max_label, world_size, gpu, logger)
	# classifier = losses.TotalClassifier(max_label, gpu, logger)
	dumb_x = torch.from_numpy(np.float32(np.zeros([2,512])))
	dumb_y = torch.from_numpy(np.int64(np.zeros(2)))
	with torch.no_grad():
		classifier(dumb_x, dumb_y)
	losses.load_classifier(max_label, world_size, classifier, gpu, './classifier/', logger)
	classifier.cuda(gpu)
	# classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[gpu])
	# optim = torch.optim.SGD([{'params':net.parameters(), 'weight_decay':0.0005}, {'params':classifier.parameters(), 'weight_decay':0.0}], lr=0.0001, momentum=0.9)
	optim = torch.optim.AdamW([{'params':net.parameters(), 'weight_decay':0.0005}, {'params':classifier.parameters(), 'weight_decay':0.0}], lr=0.0001)

	for e in range(3,25):
		if e<2:
			m2 = 0.0
		else:
			m2 = 0.5 
		sampler.set_epoch(e)
		logger.info('Epoch:%d'%e)
		if e%10==9 and e>0:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr

		for i, (img, label) in enumerate(loader):
			label = label.cuda(gpu)
			labels = [torch.zeros_like(label) for _ in range(world_size)]
			dist.all_gather(labels, label)
			labels = torch.cat(labels, dim=0)
			# labels = label
			optim.zero_grad()
			feat = net(img)
			feat_list = [torch.zeros_like(feat) for _ in range(world_size)]
			dist.all_gather(feat_list, feat)
			feat_cat = torch.cat(feat_list, dim=0)
			feat_cat = feat_cat.requires_grad_()
			# logger.info('%s  %s'%(feat_cat.shape, labels.shape))
			loss, correct = classifier(feat_cat, labels, m2=m2)
			# loss, correct = classifier(feat, labels, m2=0.0)
			loss = loss.sum()
			# logger.info(f'{loss}')
			loss.backward()
			
			# logger.info('%s'%feat_cat.grad)
			dist.all_reduce(feat_cat.grad, dist.ReduceOp.SUM)
			grad_feat = feat_cat.grad[BSIZE*gpu : BSIZE*gpu+BSIZE] 
			# logger.info('%s  %s'%(feat.shape, grad_feat.shape))
			feat.backward(gradient=grad_feat)
			# logger.info('%s'%net.module.c1.conv.weight.max())
			optim.step()

			dist.all_reduce(loss)
			dist.all_reduce(correct)
			# logger.info(f'{loss}  {feat_cat.shape}')
			# loss = loss / feat_cat.shape[0]
			# acc = correct / feat_cat.shape[0]
			loss = loss / BSIZE / world_size
			acc = correct / BSIZE / world_size
			lr = optim.param_groups[0]['lr']
			if gpu==0 and i%20==0:
				logger.info('Epoch:%d  Iter:%d/%d  Loss:%.4f Acc:%.4f LR:%.1e'%(e, i,len(loader),loss, acc, lr))

		stamp = random.randint(0, 1000000)
		if gpu==0:
			saver.save('./model/R50_%d_%d.pth'%(e, stamp))
		losses.save_classifier(max_label, world_size, classifier, './classifier/%d_%d.pth'%(e,stamp), logger, do_save=gpu==0)

if __name__=='__main__':
	FORMAT = '%(asctime)-15s  Replica:%(name)s  %(message)s'
	logging.basicConfig(format=FORMAT)

	# d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
	# logger.warning('Protocol problem: %s', 'connection reset', extra=d)
	main()

