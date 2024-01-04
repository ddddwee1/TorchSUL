import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Tuple 

from ..Base import Model


class NNUpSample(Model):
    inchannel: int 
    scale: int 
    size: Tuple[int,int,int,int]
    
    def __init__(self, scale:int):
        super().__init__(scale=scale)

    def initialize(self, scale: int):
        self.scale = scale

    def _parse_args(self, input_shape):
        self.inchannel = input_shape[1]
        self.size = (self.inchannel, 1, self.scale, self.scale)

    def build(self, *inputs):
        inp = inputs[0]
        self._parse_args(inp.shape)
        self.weight = Parameter(torch.Tensor(*self.size), requires_grad=False)
        self.reset_params()

    def reset_params(self):
        init.ones_(self.weight)

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, None, self.scale, 0, 0, self.inchannel, 1)
    
    def __call__(self, x:Tensor) -> Tensor:
        return super().__call__(x)
    
    def to_torch(self) -> nn.ConvTranspose2d:
        conv = nn.ConvTranspose2d(self.inchannel, self.inchannel, self.scale, self.scale, 0, 0, self.inchannel, bias=False)
        conv.weight.data[:] = self.weight.data[:]
        return conv 
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        pass


class PadAround2D(nn.Module):
    def __init__(self, n_pix: int):
        super().__init__()
        self.n_pix = n_pix

    def forward(self, x: Tensor) -> Tensor: 
        return F.pad(x, (self.n_pix, self.n_pix, self.n_pix, self.n_pix))
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)


class BilinearUpSample(Model):
    factor: int 
    pad0: int 
    inchannel: int 
    weight: Parameter

    def __init__(self, factor: int):
        super().__init__(factor)

    def initialize(self, factor: int):
        self.factor = factor
        self.pad0 = factor//2 * 3 + factor%2

    def build(self, *inputs):
        inp = inputs[0]
        self.inchannel = inp.shape[1]
        filter_size = 2*self.factor - self.factor%2
        k = self.upsample_kernel(filter_size)
        k = k[None,...]
        k = np.repeat(k, self.inchannel, axis=0)
        self.weight = Parameter(torch.from_numpy(k), requires_grad=False)

    def upsample_kernel(self,size):
        factor = (size +1)//2
        if size%2==1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        k = (1 - abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)
        return np.array(k, dtype=np.float32)

    def forward(self, x):
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = F.conv_transpose2d(x, self.weight, None, self.factor, self.pad0, 0, self.inchannel, 1)
        return x 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
    
    def to_torch(self) -> nn.Sequential:
        conv = nn.ConvTranspose2d(self.inchannel, self.inchannel, self.factor, self.factor, 0, 0, self.inchannel, bias=False)
        conv.weight.data[:] = self.weight.data[:]
        mod = nn.Sequential(PadAround2D(1), conv)
        return mod 

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        pass
