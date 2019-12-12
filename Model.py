import Layers as L 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import os 

Model = L.Model
activation = L.activation
flatten = L.flatten
GlobalAvgPool = L.GlobalAvgPool2D
BatchNorm = L.BatchNorm

# activation const
PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6
PARAM_SWISH = 7
PARAM_PRELU = 8

class Saver():
	def __init__(self, module):
		self.model = module

	def _get_checkpoint(self, path):
		path = path.replace('\\','/')
		ckpt = path + 'checkpoint'
		if os.path.exists(ckpt):
			fname = open(ckpt).readline().strip()
			return path + fname
		else:
			return False

	def restore(self, path):
		print('Trying to load from:',path)
		# print(path[-4:])
		if path[-4:] == '.pth':
			if not os.path.exists(path):
				print('Path:',path, 'does not exsist.')
			elif isinstance(self.model, nn.DataParallel):
				self.model.module.load_state_dict(torch.load(path))
				print('Model loaded from:', path)
			else:
				self.model.load_state_dict(torch.load(path))
				print('Model loaded from:', path)
		else:
			path = self._get_checkpoint(path)
			if path:
				if isinstance(self.model, nn.DataParallel):
					self.model.module.load_state_dict(torch.load(path))
				else:
					self.model.load_state_dict(torch.load(path))
				print('Model loaded from:', path)
			else:
				print('No checkpoint found. No restoration is performed.')

	def save(self, path):
		directory = os.path.dirname(path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		if isinstance(self.model, nn.DataParallel):
			torch.save(self.model.module.state_dict(), path)
		else:
			torch.save(self.model.state_dict(), path)
		print('Model saved to:',path)
		ckpt = open(directory + '/checkpoint', 'w')
		ckpt.write(os.path.basename(path))
		ckpt.close()

class ConvLayer(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, activation=-1, batch_norm=False, affine=True, usebias=True, groups=1):
		self.conv = L.conv2D(size, outchn, stride, pad, dilation_rate, usebias, groups)
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
		self.batch_norm = batch_norm
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU()
	def forward(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
		return x 

class Dense(Model):
	def initialize(self, outsize, batch_norm=False, affine=True, activation=-1 , usebias=True, norm=False):
		self.fc = L.fclayer(outsize, usebias, norm)
		self.batch_norm = batch_norm
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU()
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
	def forward(self, x):
		x = self.fc(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
		return x 
