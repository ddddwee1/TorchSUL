import torch.nn.functional as F 

from torch import Tensor


def sigmoid_focal_loss(x: Tensor, label: Tensor, alpha: float=0.25, gamma: float=2):
    prob = x.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(x, label, reduction='none')
    p_t = prob * label + (1 - prob) * (1 -label)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >=0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    return loss

