import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch.distributed as dist
import torch
import torch.nn as nn 
import torch.multiprocessing as mp 
import config 
import datareader 
import numpy as np 
from TorchSUL import Model as M
import loss 
from time import gmtime, strftime
import random 
import visutil 
from config import cfg
from config import update_config
import hrnet

torch.backends.cudnn.benchmark = True

def main():
	ngpus_per_node = torch.cuda.device_count()
	worldsize = ngpus_per_node
	mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, ))

def main_worker(gpu, ngpus_per_node):
	print('Use GPU:', gpu)
	dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=ngpus_per_node, rank=gpu)
	print('Group initialized.')

	# initialize the network 
	model_dnet = hrnet.DEKRRefine(config.num_pts, 128)
	x = torch.zeros(1, 3, 128, 128)
	with torch.no_grad():
		model_dnet(x)
	M.Saver(model_dnet.dekr).restore('./model_dekr/')
	model = loss.ModelWithLoss(model_dnet)
	saver = M.Saver(model)
	saver.restore('./model/')

	torch.cuda.set_device(gpu)
	model.cuda(gpu)
	model.train()
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	print('Model initialized.')

	# get loader 
	loader, sampler = datareader.get_train_dataloader()
	optim = torch.optim.Adam(model.parameters(), lr=config.init_lr)

	for e in range(config.max_epoch):
		print('Replica:%d Epoch:%d'%(gpu, e))
		sampler.set_epoch(e)
		for i, (img, heatmap, mask, offset, offset_weight) in enumerate(loader):
			optim.zero_grad()
			hm, off, refined, hm_loss, off_loss, hm_refined_loss = model(img, heatmap, mask, offset, offset_weight)
			loss_total = hm_loss + off_loss * 0.03 + hm_refined_loss
			loss_total.backward()
			optim.step()
			lr = optim.param_groups[0]['lr']

			if i%100==0 and gpu==0:
				visutil.vis_batch(img, hm, './outputs/%d_out.jpg'%i)
				visutil.vis_batch(img, refined, './outputs/%d_ref.jpg'%i)
				visutil.vis_batch(img, heatmap, './outputs/%d_gt.jpg'%i)

			if i%20==0 and gpu==0:
				curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
				print('%s  Replica:%d  Progress:%d/%d  LsHM:%.3e  LsRef:%.3e  LsOff:%.3e  LR:%.1e'%(curr_time, gpu, i, len(loader), hm_loss, hm_refined_loss, off_loss, lr))

		if e in config.lr_epoch:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr

		if e%config.save_interval==0 and gpu==0:
			stamp = random.randint(0, 1000000)
			saver.save('./model/%d_%d.pth'%(e, stamp))

if __name__=='__main__':
	main()
