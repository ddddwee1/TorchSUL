import torch
import torch.nn as nn 
import torch.nn.init as init 
import torch.nn.functional as F 
from TorchSUL import Model as M 
from torch.nn.parameter import Parameter
from torch.nn.parallel import replicate, scatter, parallel_apply, gather
import numpy as np 
import torch.distributed as dist 
import os 

def accuracy(pred, label):
	_, predicted = torch.max(pred.data, 1)
	total = label.size(0)
	correct = (predicted == label).sum().item()
	acc = correct / total 
	return acc 

def classify(feat, weight, label, m1=1.0, m2=0.5, m3=0.0, s=64):
	feat = feat / feat.norm(p=2, dim=1, keepdim=True)
	weight = weight / weight.norm(p=2, dim=1, keepdim=True)
	x = torch.mm(feat, weight.t())
	bsize = feat.shape[0]
	if not (m1==1.0 and m2==0.0 and m3==0.0):
		idlen = weight.shape[0]
		idx = torch.where(label>=0)[0]
		if idx.shape[0]==0:
			# print('Not in this patch.')
			x = x * s
			# print('xmax', x.max())
			xexp = torch.exp(x)
			xsum = xexp.sum(dim=1, keepdim=True)
			xmax, xargmax = torch.max(xexp, dim=1)
			# print('XMAX', xmax)
			return x, xexp, xsum, xargmax, xmax
		label = label[label>=0]

		t = x[idx, label]
		t = torch.acos(t)
		if m1!=1.0:
			t = t * m1 
		if m2!=0.0:
			# print('M2 applied')
			# t = t + m2 
			t = t + 0.5
		# with torch.no_grad():
		# 	delta = t - (np.pi * 2 - 1e-6) 
		# 	delta.clamp_(min=0)
		# t = t - delta
		t = torch.cos(t)
		# print('TMAX',t.min())
		x[idx, label] = t - m3 
		# print('XMAX', x.min())

	x = x * s 
	# print('xmax2', x.max())
	xexp = torch.exp(x)
	xsum = xexp.sum(dim=1, keepdim=True)
	xmax, xargmax = torch.max(xexp, dim=1)
	
	return x, xexp, xsum, xargmax, xmax

# change backward in autograd
class NLLDistributed(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, xexp, label, sums):
		# xexp = torch.exp(x)
		results = xexp / sums 
		idx = torch.where(label>=0)[0]
		grad = results.clone().detach()
		if idx.shape[0]!=0:
			# print(len(idx))
			label = label[idx]
			grad[idx,label] -= 1. 
			grad = grad / xexp.shape[0]
			results = results[idx, label]
		else:
			results = (results[0,0] + 1) / (results[0,0] + 1) # avoid nan
		ctx.save_for_backward(grad)
		results = - torch.log(results)
		return results

	@staticmethod
	def backward(ctx, grad_out):
		grad = ctx.saved_tensors
		return grad[0], None, None, None

nllDistributed = NLLDistributed.apply

class SplitClassifier(M.Model):
	def initialize(self, num_classes, world_size, rank, logger):
		split_size = num_classes // world_size + int(num_classes%world_size>0)
		self.start = split_size * rank
		self.outsize = min(num_classes-self.start, split_size)
		self.end = min(num_classes, self.start + split_size)
		self.world_size = world_size
		self.rank = rank 
		self.logger = logger
		self.logger.info('Create classifier for class %d~%d'%(self.start , self.end))

	def parse_args(self, input_shape):
		self.weight_shape = [self.outsize, input_shape[1]]

	def build(self, *inputs):
		self.parse_args(inputs[0].shape)
		self.weight = Parameter(torch.ones(*self.weight_shape))
		init.normal_(self.weight, std=0.01)

	def forward(self, feats, label, **kwargs):
		label = label - self.start
		label[label>=self.outsize] = -1
		x, xexp, xsum, xargmax, xmax = classify(feats, self.weight, label, **kwargs)
		# print('adf',xsum.device, xsum.shape)
		dist.all_reduce(xsum) # [bsize, 1]
		# self.logger.info('xsum %s'%(xsum.shape))
		# print('XSUM', xsum.shape)
		loss = nllDistributed(x, xexp, label, xsum)
		# dist.all_reduce(loss)
		bsize = label.shape[0]
		start_bidx = bsize * self.rank
		correct = (xargmax==label.cuda(self.rank)).float().sum()
		return loss , correct

	def build_forward(self, feats, label, **kwargs):
		return None 

class TotalClassifier(M.Model):
	def initialize(self, num_classes, rank, logger):
		self.logger = logger
		self.num_classes = num_classes
		self.rank = rank 

	def parse_args(self, input_shape):
		self.weight_shape = [self.num_classes, input_shape[1]]

	def build(self, *inputs):
		self.parse_args(inputs[0].shape)
		self.weight = Parameter(torch.ones(*self.weight_shape))
		init.normal_(self.weight, std=0.01)

	def forward(self, feats, label, **kwargs):
		x, xexp, xsum, xargmax, xmax = classify(feats, self.weight, label, **kwargs)
		# print(xexp.device, xsum.device, xargmax.device, xmax.device)
		loss = nllDistributed(x, xexp, label, xsum)
		correct = (xargmax==label.cuda(self.rank)).float().sum()
		# x = xexp / xsum
		# x = torch.log(x)
		# label = label.unsqueeze(-1)
		# loss = torch.gather(x, 1, label)
		# loss = - loss.mean()
		return loss, correct
	def build_forward(self, feats, label, **kwargs):
		return None 

def save_classifier(num_classes, world_size, classifier, path, logger):
	tensor = classifier.weight
	split_size = num_classes // world_size + int(num_classes%world_size>0)
	outsizes = [min(split_size, num_classes - split_size*rank) for rank in range(world_size)]
	results_list = [torch.zeros(outsizes[i],tensor.shape[1]) for i in range(world_size)]
	dist.gather(tensor, gather_list=results_list, dst=0)
	result = torch.cat(results_list, dim=0)
	result = result.detach().cpu()
	directory = os.path.dirname(path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	torch.save(result, path)
	logger.info('Classifier saved to: %s'%path)

def load_classifier(num_classes, world_size, classifier, rank, path, logger):
	if os.path.exists(path):
		tensor = classifier.weight
		split_size = num_classes // world_size + int(num_classes%world_size>0)
		start_idx = split_size * rank
		end_idx = min(num_classes, start_idx + split_size)
		mtx = torch.load(path)
		classifier.weight.data[:] = mtx[start_idx:end_idx]
		logger.info('Classifier loaded from: %s'%path)
	else:
		logger.warning('No savings at %s'%path)
	