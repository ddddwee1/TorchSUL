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

	model_dnet = network.DensityNet(config.density_num_layers, config.density_channels, config.density_level,\
							config.gcn_layers, config.gcn_channels, config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	x = np.float32(np.random.random(size=[1,3,512,512]))
	x = torch.from_numpy(x)
	with torch.no_grad():
		outs, idout, depout = model_dnet(x)
	# input()
	M.Saver(model_dnet.backbone).restore('./model_imagenet_w32/')
	# model_dnet.bn_eps(1e-5)
	model = loss.ModelWithLoss(model_dnet)
	M.Saver(model.model.backbone).restore('./backbone/')
	M.Saver(model.model.upsample).restore('./upsample/')
	saver = M.Saver(model)
	saver.restore('./model/')

	torch.cuda.set_device(gpu)
	model.cuda(gpu)
	model.train()
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	print('Model get.')

	loader, sampler = datareader.get_train_dataloader(28)
	optim = torch.optim.Adam(model.parameters(), lr=config.init_lr)

	for e in range(config.max_epoch):
		print('Replica:%d Epoch:%d'%(gpu, e))
		sampler.set_epoch(e)
		for i, (img, hmap, mask, pts, depth, depth_all, is_muco) in enumerate(loader):
			optim.zero_grad()
			hmap_loss, push_loss, pull_loss, rdepth_loss, depth_loss, outs, idout, depout, depallout = model(img, hmap, mask, pts, depth, depth_all, is_muco)
			# print(hmap_loss.shape)

			hmap_loss = hmap_loss.mean()
			push_loss = push_loss.mean()
			pull_loss = pull_loss.mean()
			rdepth_loss = rdepth_loss.mean()
			depth_loss = depth_loss.mean()
			if e<0:
				loss_total = hmap_loss + 0.05*push_loss + 0.001*pull_loss + 0.01 * rdepth_loss + 0.01 * depth_loss
			else:
				loss_total = hmap_loss + 0.003*push_loss + 0.001*pull_loss + 0.003 * rdepth_loss + 0.003 * depth_loss
			loss_total.backward()
			optim.step()
			lr = optim.param_groups[0]['lr']

			if i%100==0 and gpu==0:
				if not os.path.exists('./outputs/'):
					os.mkdir('./outputs/')
				# outs = torch.sigmoid(outs)
				# outs, idout = model_dnet(img.cuda())
				visutil.vis_batch(img, outs, './outputs/%d_out.jpg'%i)
				visutil.vis_batch(img, hmap, './outputs/%d_gt.jpg'%i)
				#visutil.vis_batch(img, mask, './outputs/%d_mask.jpg'%i)
				visutil.vis_batch(img, idout, './outputs/%d_id.jpg'%i, minmax=True)
				visutil.vis_batch(img, depout, './outputs/%d_dep.jpg'%i, minmax=True)
				visutil.vis_batch(img, depallout, './outputs/%d_rel.jpg'%i, minmax=True)
				print(outs.max(), outs.min(), depout.max(), depout.min(), depth[:,:,2].max(), depth[:,:,2].min())

			if i%20==0:
				curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
				print('%s  Replica:%d  Progress:%d/%d  Ls:%.3e  LsC:%.3e  IDs:%.3e  IDd:%.3e rD:%.3e D:%.3e LR:%.1e'%(curr_time, gpu, i, len(loader), loss_total, hmap_loss, pull_loss, push_loss, rdepth_loss, depth_loss, lr))

		if e in config.lr_epoch:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr

		if e%config.save_interval==0 and gpu==0:
			stamp = random.randint(0, 1000000)
			saver.save('./model/%d_%d.pth'%(e, stamp))

if __name__=='__main__':
	main()
