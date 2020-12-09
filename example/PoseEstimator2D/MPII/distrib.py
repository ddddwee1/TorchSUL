import torch.distributed as dist
import torch
import torch.nn as nn 
import torch.multiprocessing as mp 
import config 
import network 
import datareader 
import numpy as np 
from TorchSUL import Model as M
import loss 
from time import gmtime, strftime
import random 
import visutil 
import os 

def main():
	ngpus_per_node = torch.cuda.device_count()
	worldsize = ngpus_per_node
	mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, ))

def main_worker(gpu, ngpus_per_node):
	print('Use GPU:', gpu)
	dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=ngpus_per_node, rank=gpu)
	print('Group initialized.')

	model_dnet = network.DensityNet(config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	x = np.float32(np.random.random(size=[1,3+config.num_match_pts, config.inp_size, config.inp_size]))
	x = torch.from_numpy(x)
	with torch.no_grad():
		outs = model_dnet(x)
	# input()
	saver = M.Saver(model_dnet.backbone)
	saver.restore('./model_imagenet_w32/', strict = False, exclude=['c1.conv.weight'])
	saver.restore('./model/')
	# model_dnet.bn_eps(1e-5)
	model = loss.ModelWithLoss(model_dnet)

	torch.cuda.set_device(gpu)
	model.cuda(gpu)
	model.train()
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	print('Model get.')

	loader, sampler = datareader.get_train_dataloader(22)
	optim = torch.optim.Adam(model.parameters(), lr=config.init_lr)

	for e in range(config.max_epoch):
		print('Replica:%d Epoch:%d'%(gpu, e))
		sampler.set_epoch(e)
		for i, (img, hmap, hmap_match) in enumerate(loader):
			# print(img.shape, hmap.shape, hmap_match.shape)
			optim.zero_grad()
			hmap_loss, outs = model(img, hmap, hmap_match)

			hmap_loss = hmap_loss.mean()

			hmap_loss.backward()
			optim.step()
			lr = optim.param_groups[0]['lr']

			if i%100==0 and gpu==0:
				if not os.path.exists('./outputs/'):
					os.mkdir('./outputs/')

				visutil.vis_batch(img, outs, './outputs/%d_out.jpg'%i)
				visutil.vis_batch(img, hmap, './outputs/%d_gt.jpg'%i)
				visutil.vis_batch(img, hmap_match, './outputs/%d_hmm.jpg'%i, minmax=True)

			if i%20==0:
				curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
				print('%s  Replica:%d  Progress:%d/%d  LsC:%.3e LR:%.1e'%(curr_time, gpu, i, len(loader), hmap_loss, lr))

		if e in config.lr_epoch:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr

		if e%config.save_interval==0 and gpu==0:
			stamp = random.randint(0, 1000000)
			saver.save('./model/%d_%d.pth'%(e, stamp))

if __name__=='__main__':
	main()

