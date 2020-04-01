import torch
import torch.nn as nn 
import torch.nn.functional as F 

def accuracy(pred, label):
	_, predicted = torch.max(pred.data, 1)
	total = label.size(0)
	correct = (predicted == label).sum().item()
	acc = correct / total 
	return acc 

def format_losses(keys, losses):
	strout = []
	for k,l in zip(keys, losses):
		buff = '%s:%.4f'%(k,l)
		strout.append(buff)
	strout = '\t'.join(strout)
	return strout