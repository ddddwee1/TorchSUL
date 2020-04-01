import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from TorchSUL import Model as M 
import resnet 
import resnet2

x = torch.from_numpy(np.zeros([1,3,16*4,112], dtype=np.float32))
net = resnet.Res34()
net.eval()
# initialize
net(x)
saver = M.Saver(net)
saver.restore('./newmodel/')
# end: initialize
# record params 
net.record()
net.merge_bn()
net(x)
c1param = M.get_record()
# end record params 

# record simplified network 
net2 = resnet2.Res34()
net2.eval()
M.reset_record()
net2.record()
net2(x)
c2param = M.get_record()
# end: record

# assign the params 
for i,j in zip(c1param, c2param):
    for k in j:
        try:
            j[k].data[:] = i[k].data[:]
        except:
            continue

# test if outputs are same 
y1 = net.debug(x)
y2 = net2.debug(x)
print(y1)
print(y2)

# save simplified network 
M.Saver(net2).save('./simplified/r34.pth')
