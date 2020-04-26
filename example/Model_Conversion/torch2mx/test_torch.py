import torch 
import numpy as np 
from TorchSUL import Model as M 
import effnetocc

net = effnetocc.effnet()

net.eval()
x = torch.from_numpy(np.zeros([1,3,112,112], dtype=np.float32))
net(x)

saver = M.Saver(net)
saver.restore('./model/', strict=True)
net.eval()

y2 = net(x)
y2 = y2.cpu().detach().numpy()
# y2 = y2 / np.linalg.norm(y2, axis=-1, keepdims=True)
print(y2)
print(y2.shape)
