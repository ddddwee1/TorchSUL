import torch
import torch.nn as nn 
import torch.nn.functional as F 
import Model as M 
import numpy as np 

class MarginalCosineLayer(M.Model):
	def initialize(self, num_classes):
		self.classifier = M.Dense(num_classes, usebias=False, norm=True)
	def forward(self, x, label, m1=1.0, m2=0.0, m3=0.0):
		x = self.classifier(x)
		x_pure = x.clone()
		if not (m1==1.0 and m2==0.0 and m3==0.0):
			# print('MarginalCosineLayer used.')
			idx = torch.arange(0,label.size(0), dtype=torch.long)
			t = x[idx, label]
			t = torch.acos(t)
			if m1!=1.0:
				t = t * m1 
			if m2!=0.0:
				t = t + m2 
			t = torch.cos(t)
			x[idx, label] = t - m3 
		return x, x_pure

class NLLLoss(M.Model):
	def initialize(self):
		self.lgsm = nn.LogSoftmax(dim=1)
		# self.lsfn = nn.NLLLoss()
	def forward(self, x, label):
		x = self.lgsm(x)
		label = label.unsqueeze(-1)
		loss = torch.gather(x, 1, label)
		loss = -loss.mean()
		return loss

# a = np.float32([[0,1],[2,3]])
# b = np.int64([0,1])
# a = torch.from_numpy(a)
# b = torch.from_numpy(b)
# idx = torch.arange(0,a.size(0), dtype=torch.long)
# c = a[idx, b] 
# print(a)
# print(c)
# a[idx, b] = 0
# print(a)
