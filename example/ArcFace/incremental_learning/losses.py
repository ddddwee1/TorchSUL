import torch
import torch.nn as nn 
import torch.nn.init as init 
import torch.nn.functional as F 
from TorchSUL import Model as M 
from torch.nn.parameter import Parameter
from torch.nn.parallel import replicate, scatter, parallel_apply, gather
import numpy as np 

def accuracy(pred, label):
	_, predicted = torch.max(pred.data, 1)
	total = label.size(0)
	correct = (predicted == label).sum().item()
	acc = correct / total 
	return acc 

def classify(feat, weight, label, m1=1.0, m2=0.5, m3=0.0, s=64, simple_output=False):
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
			xexp = torch.exp(x)
			xsum = xexp.sum(dim=1, keepdim=True)
			xmax, xargmax = torch.max(xexp, dim=1)
			return x, xexp, xsum, xargmax, xmax
		label = label[(label>=0) & (label<idlen)]

		t = x[idx, label]
		t = torch.acos(t)
		if m1!=1.0:
			t = t * m1 
		if m2!=0.0:
			t = t + m2 
		# with torch.no_grad():
		# 	delta = t - (np.pi * 2 - 1e-6) 
		# 	delta.clamp_(min=0)
		# t = t - delta
		t = torch.cos(t)
		x[idx, label] = t - m3 

	x = x * s 
	if simple_output:
		return x 
	else:
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

class DistributedClassifier(M.Model):
	def initialize(self, num_classes, gpus):
		# if self.gpus is None, then this layer runs in cpu mode 
		# otherwise, distribute the classifier into gpus independently
		self.num_classes = num_classes
		self.gpus = gpus
		assert (self.gpus is None) or isinstance(self.gpus, list) or isinstance(self.gpus, tuple), 'gpus must be either None or python list'

	def parse_args(self, input_shape):
		if self.gpus is None:
			self.weight_shape = [self.num_classes, input_shape[1]]
		else:
			id_per_split = self.num_classes // len(self.gpus) + int(self.num_classes % len(self.gpus) > 0)
			self.weight_shape = []
			self.weight_idx = [0]
			for i in range(len(self.gpus)):
				self.weight_idx.append(self.weight_idx[-1] + id_per_split)
			self.weight_idx[-1] = self.num_classes
			for i in range(len(self.gpus)):
				outnum = self.weight_idx[i+1] - self.weight_idx[i]
				shape = [outnum, input_shape[1]]
				self.weight_shape.append(shape)

	def build(self, *inputs):
		input_shape = inputs[0].shape 
		self.parse_args(input_shape)

		if self.gpus is None:
			self.weight = Parameter(torch.ones(*self.weight_shape))
			init.normal_(self.weight, std=0.01)
		else:
			self.weights = []
			for idx, i in enumerate(self.gpus):
				# weight = Parameter(torch.ones(*self.weight_shape[idx], device=torch.device('cuda:%d'%i)))
				weight = Parameter(torch.ones(*self.weight_shape[idx]))
				self.register_parameter('w%d'%idx,weight)
				init.normal_(weight, std=0.01)
				self.weights.append(weight)

	def zero_grad(self, idx):
		if idx<=0:
			return 
		curr_idx = 0
		for w in self.weights:
			end_idx = min(idx - curr_idx, w.shape[0])
			w.grad[:end_idx] = 0
			curr_idx += end_idx
			if curr_idx == idx:
				break 

	def forward(self, x, label, **kwargs):
		if self.gpus is None:
			# cpu mode, normal fc layer
			x = classify(x, self.weight, label, simple_output=True, **kwargs)
			with torch.no_grad():
				acc = accuracy(x, label)
			x = F.log_softmax(x, dim=1)
			label = label.unsqueeze(-1)
			loss = torch.gather(x, 1, label)
			loss = -loss.mean()
			return loss, acc
		else:
			weight_scattered = (w.to(i) for w,i in zip(self.weights, self.gpus) )
			feat_copies = [x.to(i) for i in self.gpus]
			labels_scattered = []
			for i in range(len(self.weights)):
				labels_new = label.clone()
				labels_new[(labels_new>=self.weight_idx[i+1]) | (labels_new<self.weight_idx[i])] = -1
				labels_new = labels_new - self.weight_idx[i]
				labels_scattered.append(labels_new)
			kwargs_scattered = scatter(kwargs, self.gpus)
			input_scattered = list(zip(feat_copies, weight_scattered, labels_scattered))
			modules = [classify] * len(self.weights)
			results_scattered = parallel_apply(modules, input_scattered, kwargs_scattered, self.gpus)

			logits = [i[0] for i in results_scattered]
			xexps = [i[1] for i in results_scattered]
			sums = [i[2] for i in results_scattered]
			argmaxs = [i[3] for i in results_scattered]
			maxs = [i[4] for i in results_scattered]

			sums = gather(sums, 0, dim=1)
			sums = sums.sum(dim=1, keepdim=True)
			sums_scattered = [sums.to(i) for i in self.gpus]
			loss_input_scattered = list(zip(logits, xexps, labels_scattered, sums_scattered))
			loss_results_scattered = parallel_apply([nllDistributed] * len(self.gpus), loss_input_scattered, None, self.gpus)
			loss_results_scattered = [i.sum() for i in loss_results_scattered]
			
			loss_results_scattered = [i.to(0) for i in loss_results_scattered]
			loss = sum(loss_results_scattered)
			loss = loss / x.shape[0]

			for i in range(len(argmaxs)):
				argmaxs[i] = argmaxs[i] + self.weight_idx[i]
			maxs = [i.to(0) for i in maxs]
			maxs = torch.stack(maxs, dim=1)
			
			_, max_split = torch.max(maxs, dim=1)
			idx = torch.arange(0,maxs.size(0), dtype=torch.long)
			argmaxs = [i.to(0) for i in argmaxs]
			argmaxs = torch.stack(argmaxs, dim=1)
			predicted = argmaxs[idx, max_split]

			total = label.size(0)
			predicted = predicted.cpu()
			label = label.cpu()
			correct = (predicted == label).sum().item()
			acc = correct / total 

			return loss, acc

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		for hook in self._load_state_dict_pre_hooks.values():
			hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
		w = state_dict[prefix + 'weight']
		if self.gpus is None:
			self.weight.data = w 
		else:
			weight_idx = self.weight_idx+[self.num_classes]

			class_old = w.shape[0]
			dim_old = w.shape[1]
			dim_new = self.weights[0].shape[1]
			if dim_old!=dim_new:
				print('Feature dim does not match. Got old: %d and new: %d'%(dim_old, dim_new))
			if class_old!=self.num_classes:
				print('Class number does not match. Got old: %d and new %d'%(class_old, self.num_classes))
			curr = 0

			for i in range(len(self.gpus)):
				end_idx = min(class_old - curr, self.weights[i].shape[0])
				self.weights[i].data[:end_idx, :dim_old] = w[curr: curr+end_idx, :dim_old]
				self.weights[i].data[:end_idx,dim_old:] = 0.0
				curr += end_idx
				if curr == class_old:
					break 

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self.gpus is None:
			buf = self.weight.data
		else:
			buf = []
			for w in self.weights:
				buf.append(w.data)
			buf = torch.cat(buf, dim=0)
		destination[prefix + 'weight'] = buf 
