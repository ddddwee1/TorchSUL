import torch 

def accuracy(output, label):
	pred = torch.round(torch.sigmoid(output))
	total = label.size(0)
	correct = (pred==label).sum().item()
	acc = correct / total
	return acc
