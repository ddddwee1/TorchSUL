import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from TorchSUL import Model as M 
import resnet 
import resnet2
from tqdm import tqdm 

x = torch.from_numpy(np.zeros([1,3,16*4,112], dtype=np.float32))
net = resnet.Res34()
net.eval()
net(x)

print('Test Net1 CPU...')
for i in tqdm(range(1000)):
	net(x)

print('Test Net1 GPU...')
net.cuda()
xg = x.cuda()
net(xg)
for i in tqdm(range(10000)):
	net(xg)

net2 = resnet2.Res34()
net2.eval()
net2(x)
print('Test Net1 CPU...')
for i in tqdm(range(1000)):
	net2(x)

print('Test Net1 GPU...')
net2.cuda()
net2(xg)
for i in tqdm(range(10000)):
	net2(xg)
