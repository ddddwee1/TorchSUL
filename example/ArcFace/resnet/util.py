import torch
import torch.nn as nn 
import torch.nn.functional as F 

def accuracy(pred, label):
	_, predicted = torch.max(pred.data, 1)
	total = label.size(0)
	correct = (predicted == label).sum().item()
	acc = correct / total 
	return acc 
