import Model as M 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import hrnet 
from tqdm import tqdm
import numpy as np 
import datareader 
import cv2 

net = hrnet.HRNET(17)
x = torch.from_numpy(np.zeros([1,3+12,256,256], dtype=np.float32))
net(x)
net = nn.DataParallel(net, device_ids=(0,1,2,3)).cuda()
saver = M.Saver(net)
saver.restore('./model/')

optim = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.999))

reader = datareader.get_datareader(32, processes=4)

MAXITER = 60001
bar = tqdm(range(MAXITER))

def vis(x):
	x = x.cpu().detach().numpy()
	x = x[0]
	x = np.clip(x, 0., 255.)
	x = np.amax(x, axis=0)
	x = np.uint8(x)
	return x 

for i in bar:
	img, hmap_match, hmap = reader.get_next()
	img = torch.from_numpy(img)
	hmap_match = torch.from_numpy(hmap_match)
	hmap_match = F.interpolate(hmap_match, size=(256,256), mode='bilinear')
	x = torch.cat([img, hmap_match], dim=1)
	hmap = torch.from_numpy(hmap).cuda()

	optim.zero_grad()
	# print(x.dtype)
	out = net(x)
	loss = torch.mean(torch.pow( out - hmap , 2))
	loss.backward()
	optim.step()

	lr = optim.param_groups[0]['lr']
	outstr = 'loss: %.4f LR: %.1e'%(loss, lr)
	bar.set_description(outstr)

	if i%100==0:
		outsample = vis(out)
		labsample = vis(hmap)
		cv2.imwrite('./vis/%08d_gt.jpg'%i, labsample)
		cv2.imwrite('./vis/%08d_out.jpg'%i, outsample)

	if i%3000==0 and i>0:
		saver.save('./model/%06d.pth'%i)

	if i%20000==0 and i>0:
		newlr = lr * 0.1 
		for param_group in optim.param_groups:
			param_group['lr'] = newlr
