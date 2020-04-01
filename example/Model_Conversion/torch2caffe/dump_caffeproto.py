import Layers as L 
import torch
import numpy as np 
import Model as M 
import pickle 
import resnet2 as resnet 

# c = L.dwconv2D(3, 1)
# c = L.Flatten()
# class ABC(M.Model):
# 	def initialize(self):
# 		self.f = L.Flatten()
# 		self.fc = L.fclayer(5)
# 	def forward(self, x):
# 		x = self.f(x)
# 		x = self.fc(x)
# 		return x 

# c = ABC()

c = resnet.Res100()
c.eval()
x = torch.from_numpy(np.float32(np.ones([1,3,112,112])))
x = [x, 'inputdata']
L.init_caffe_input(x)

y = c(x)

# M.Saver(c).save('./model/a.pth')
M.Saver(c).restore('./new/')

fout = open('proto.prototxt', 'w')
fout.write(L.caffe_string)
fout.close()

L.compile_params_dict()
pickle.dump(L.params_dict, open('params.pkl', 'wb'))

print(c.debug(x))
