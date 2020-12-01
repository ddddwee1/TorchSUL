import network 
import torch 
import numpy as np 
from TorchSUL import Model as M 
import datareader 
from tqdm import tqdm

if __name__=='__main__':
	seq_len = 243
	bsize = 256
	max_epoch = 100

	net = network.Refine2dNet(17, 243)
	a = np.zeros([2, 243, 17*2], dtype=np.float32)
	a = torch.from_numpy(a)
	b = net(a)
	# print(b.shape)

	saver = M.Saver(net)
	saver.restore('./model/')

	net.cuda()
	net.train()
	loader = datareader.get_loader(bsize, seq_len)
	optim = torch.optim.Adam(net.parameters(), lr=0.001)

	for ep in range(max_epoch):
		print('Epoch:', ep)
		lr = optim.param_groups[0]['lr']
		if max_epoch in [60, 85]:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr
			lr = newlr

		# print(len(loader))
		bar = tqdm(loader)
		for it,(p2d, gt3d) in enumerate(bar):
			p2d = p2d.cuda()
			gt3d = gt3d.cuda()

			optim.zero_grad()

			pred = net(p2d)
			loss = torch.sqrt(torch.pow(pred - gt3d, 2).sum(dim=-1)).mean()

			loss.backward()
			optim.step()

			bar.set_description('Ls:%.3f  LR:%.1e'%(loss, lr))

		postfix = np.random.randint(0, 100000)
		saver.save('./model/%d_%d.pth'%(ep, postfix))

