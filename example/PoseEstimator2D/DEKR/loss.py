import torch 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import torch.nn as nn 
import config 

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss


class OffsetsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        return loss

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred, gt) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss

class ModelWithLoss(M.Model):
	def initialize(self, model):
		self.Off = OffsetsLoss()
		self.HM = HeatmapLoss()
		self.model = model 

	def forward(self, img, heatmap, mask, offset, offset_weight):
		all_losses_fused = []
		all_offls_fused = []
		all_maps_fused = []

		hm, off = self.model(img)

		hm_loss = self.HM(hm, heatmap, mask)
		off_loss = self.Off(off, offset, offset_weight)

		return hm, off, hm_loss, off_loss

