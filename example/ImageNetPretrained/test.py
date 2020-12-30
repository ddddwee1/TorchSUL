import resnet 
import torch 
import numpy as np 
from TorchSUL import Model as M 

x = torch.zeros(2,3,224,224)
r50_2 = resnet.Res50()
r50_2(x)
r50_2.eval()
r50_2.bn_eps(0.00001)

a = torch.from_numpy(np.random.random([1,3,224,224])).float()
y = r50_2(a)
