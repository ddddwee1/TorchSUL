import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from TorchSUL import Model as M 
import effnetocc
import time 

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

net = effnetocc.effnet()

net.eval()
x = torch.from_numpy(np.zeros([2,3,112,112], dtype=np.float32))
net(x)

saver = M.Saver(net)
saver.restore('./model/', strict=True)
net.eval()

def get_features(imgs):
	imgs = np.float32(imgs)
	if imgs.shape[-1]==3:
		imgs = np.transpose(imgs, [0,3,1,2])
	imgs[:,:,16*4:] = 0
	imgs = imgs / 127.5 - 1.
	# imgs = imgs[:,:,:16*4]
	# imgs = imgs[:,::-1]
	# imgs = (imgs - 127.5 ) / 128.
	with torch.no_grad():
		imgs = torch.from_numpy(imgs).cuda()
		feats = net(imgs)[0]
		feats = feats.cpu().detach().numpy()
	feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
	return feats 


res = {}
ps = net.named_parameters()
for p in ps:
    # print(p)
    name, p = p 
    res[name] = p
    print(name)
ps = net.named_buffers()
for p in ps:
    # print(p)
    name, p = p 
    res[name] = p

print('PARAMS from Torch:', len(res))
input()

def get_conv(l1, l2, bias=False):
    a = [l1 + '.weight']
    b = [l2 + '_weight']
    if bias:
        a.append(l1 + '.bias')
        b.append(l2 + '_bias')
    return a,b

def get_bn(l1, l2):
    a = []
    b = []
    a.append(l1+'.weight')
    a.append(l1+'.bias')
    a.append(l1+'.running_mean')
    a.append(l1+'.running_var')
    b.append(l2+'_gamma')
    b.append(l2+'_beta')
    b.append(l2+'_moving_mean')
    b.append(l2+'_moving_var')
    return a, b

def get_act(l1, l2):
    a = l1 + '.weight'
    b = l2 + '_gamma'
    return [a], [b]

def totonoi(l):
    a = []
    b = []
    for i in l:
        a += i[0]
        b += i[1]
    return a,b

def get_unit(l1, l2, sc=False):
    res = []
    res.append(get_bn(l1+'.bn0', l2+'_prebn'))

    res.append(get_conv(l1+'.c0.conv', l2+'_conv0'))
    res.append(get_bn(l1+'.c0.bn', l2+'_bn0'))
    res.append(get_act(l1+'.c0.act', l2+'_relu0'))

    res.append(get_conv(l1+'.c1.conv', l2+'_conv1'))
    res.append(get_bn(l1+'.c1.bn', l2+'_bn1'))
    res.append(get_act(l1+'.c1.act', l2+'_relu1'))

    res.append(get_conv(l1+'.se1.conv', l2+'_seconv2', bias=True))
    res.append(get_act(l1+'.se1.act', l2+'_serelu2'))
    res.append(get_conv(l1+'.se2.conv', l2+'_seconv3', bias=True))

    res.append(get_conv(l1+'.c2.conv', l2+'_conv4'))
    res.append(get_bn(l1+'.c2.bn', l2+'_bn4'))
    if sc:
        res.append(get_conv(l1+'.sc.conv', l2+'_convsc'))
        res.append(get_bn(l1+'.sc.bn', l2+'_bnsc'))
    return res 

l = []
l.append(get_conv('phi.u1','upsampling1'))
l.append(get_conv('phi.u2','upsampling2'))
l.append(get_conv('c0.conv','c1'))
l.append(get_bn('c0.bn', 'bn1'))
l.append(get_act('c0.act', 'relu1'))

repeats = [1,2,2,4,3,4,3]
for block, r in enumerate(repeats):
    for j in range(r):
        sc = j==0
        l += get_unit('body.%d.%d'%(block,j), 'Stage%d_Unit%d'%(block,j), sc=sc)

l.append(get_conv('phi.c1.conv', 'phi_conv0', bias=True))
a,b = totonoi(l)

import transfer
import os 
for i,j in zip(a,b):
    # res[i].data[:] = torch.from_numpy(saved_params[j])
    value = res[i].detach().numpy()
    transfer.assign_value(j, value)

if not os.path.exists('./transferred/'):
    os.mkdir('./transferred/')
transfer.save_model('./transferred/efflargeE', 0)
print('Model saved.')
