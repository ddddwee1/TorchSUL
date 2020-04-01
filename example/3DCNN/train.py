import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import Model as M 
import vgg3d 
import datareader 
from tqdm import tqdm 
import util 

if __name__=='__main__':
	net = vgg3d.VGG3D()

	x = np.float32(np.ones([1,3,128,128,16]))
	x = torch.from_numpy(x)
	y = net(x)
	# print(y.shape)

	saver = M.Saver(net)
	saver.restore('./model/')

	reader = datareader.get_datareader('imgslist.txt', 32, 4)

	net.cuda()
	net.train()

	optim = torch.optim.Adam(net.parameters(), 0.0001 )
	lossfunc = nn.BCEWithLogitsLoss()

	MAXEPOCH = 1000
	for ep in range(MAXEPOCH):
		maxiter = reader.iter_per_epoch
		bar = tqdm(range(maxiter))
		for it in bar:
			batch = reader.get_next()
			imgs = torch.from_numpy(batch[0]).cuda()
			labels = torch.from_numpy(batch[1]).cuda()

			optim.zero_grad()
			out = net(imgs)
			loss = lossfunc(out, labels)
			with torch.no_grad():
				acc = util.accuracy(out, labels)
			loss.backward()
			optim.step()

			outstr = 'Ls:%.4f Acc:%.3f'%(loss, acc)
			bar.set_description(outstr)
