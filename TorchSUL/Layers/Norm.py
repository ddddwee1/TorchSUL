from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from ..Base import Model


class Transpose_last(nn.Module):
    dim:int 

    def __init__(self, dim:int):
        self.dim = dim 

    def forward(self, x:Tensor) -> Tensor:
        if self.dim==-1:
            return x 
        return x.transpose(-1, self.dim)


class BatchNorm(Model):
    eps: Union[float, None]
    momentum: float 
    affine: bool 
    track_running_stats: bool 
    num_batches_tracked: Tensor
    running_mean: Tensor 
    running_var: Tensor
    n_dims: int

    def __init__(self, eps:Union[float, None]=None, momentum:float=0.01, affine:bool=True, track_running_stats:bool=True):
        super().__init__(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def initialize(self, eps:Union[float, None]=None, momentum:float=0.01, affine:bool=True, track_running_stats:bool=True):
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
    def build(self, *inputs):
        num_features = inputs[0].shape[1]
        self.n_dims = len(inputs[0].shape)
        self.num_features = num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()
        eps = self.get_flag('bn_eps')
        if eps is not None:
            self.eps = eps 
        if self.eps is None:
            self.eps = 1e-5

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_() 
            self.running_var.fill_(1) 
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.eps is not None:
            result =  F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            raise Exception('eps is not set for batch norm layer.')
        return result
    
    def __call__(self, x:Tensor) -> Tensor:
        return super().__call__(x)

    def to_torch(self) -> nn.modules.batchnorm._BatchNorm:
        if self.eps is not None:
            if self.n_dims==3 or self.n_dims==2:
                bn = nn.BatchNorm1d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)
            elif self.n_dims==4:
                bn = nn.BatchNorm2d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)
            elif self.n_dims==5:
                bn = nn.BatchNorm3d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)
            else:
                raise Exception('Number of dims in batch norm should be either 3,4,5')
            if self.affine:
                bn.weight.data[:] = self.weight.data[:]
                bn.bias.data[:] = self.bias.data[:]
            if (bn.running_mean is not None) and (bn.running_var is not None):
                bn.running_mean.data[:] = self.running_mean.data[:]
                bn.running_var.data[:] = self.running_var.data[:]
            return bn 
        else:
            raise Exception('eps is not set for batch norm layer')


class LayerNorm(Model):
    dim: int 
    affine: bool 
    eps: Union[float, None]
    weight: Parameter
    bias: Parameter
    chn: int

    def initialize(self, dim=-1, affine=True, eps=None):
        self.dim = dim
        self.affine = affine
        self.eps = eps 

    def build(self, *inputs):
        shape = inputs[0].shape
        self.chn = shape[self.dim]
        if self.affine:
            self.weight = Parameter(torch.Tensor(shape[self.dim]))
            self.bias = Parameter(torch.Tensor(shape[self.dim]))
        self.reset_params()
        eps = self.get_flag('ln_eps')
        if eps is not None:
            self.eps = eps 
        if self.eps is None:
            self.eps = 1e-5

    def reset_params(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.dim!=-1:
            x = x.transpose(self.dim, -1)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, correction=0, keepdim=True)
        if self.affine:
            out = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            out = (x - mean) / torch.sqrt(var + self.eps)
        if self.dim!=-1:
            out = out.transpose(self.dim, -1)
        return out 
        
    def __call__(self, x:Tensor) -> Tensor:
        return super().__call__(x)

    def to_torch(self) -> nn.Sequential:
        mod = nn.Sequential(Transpose_last(self.dim), nn.LayerNorm(normalized_shape=self.chn), Transpose_last(self.dim))
        return mod 

