# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import Model as M 
import resnet 
import time 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

net = resnet.Res100()
net.eval()
x = torch.from_numpy(np.zeros([1,3,112,112], dtype=np.float32))
net(x)

saver = M.Saver(net)
saver.restore('./saved_model/')
net.cuda()
net.eval()

print('Verifying network...')
net(x.cuda())
print('Verified.')

def get_features(imgs):
	if imgs.shape[-1]==3:
		imgs = np.transpose(imgs, [0,3,1,2])
	imgs = (imgs - 127.5) / 128
	with torch.no_grad():
		imgs = torch.from_numpy(imgs).cuda()
		feats = net(imgs)
		feats = feats.cpu().detach().numpy()
	feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
	return feats 

if __name__=='__main__':
	a = np.float32(np.ones([1,3,112,112]))* 255
	feat = get_features(a)
	print(feat)
