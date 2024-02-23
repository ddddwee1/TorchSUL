import math
from typing import Union, List, Tuple

import torch
import torch.nn.init as init
import torchvision.ops as ops
from torch import Tensor
from torch.nn.parameter import Parameter

from ..Base import Model
from ..Consts.Types import *


## Resnet style initialization 
def _resnet_normal(tensor):
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_out))
    return init._no_grad_normal_(tensor, 0., std)


class DeformConv2D(Model):
    TypeKSize = Union[int, List[int], Tuple[int,int]]
    size: TypeKSize
    kernel_size = Tuple[int,int,int,int]
    outchn: int 
    stride: TypeKSize 
    pad_mode: PadModes
    dilation_rate: int 
    usebias: bool
    pad: Union[int, Tuple[int,int]]
    groups: int
    inchn: int
    weight: Parameter
    bias: Union[Parameter, None]
    
    def __init__(self, size:TypeKSize, outchn:int, stride:TypeKSize=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias)

    def initialize(self, size:TypeKSize, outchn:int, stride:TypeKSize=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True):
        self.size = size
        self.outchn = outchn
        self.stride = stride
        self.usebias = usebias
        self.dilation_rate = dilation_rate
        assert (pad in ['VALID','SAME_LEFT'])
        self.pad_mode = pad 

    def _parse_args(self, input_shape: List[int]):
        inchannel = input_shape[1]
        self.inchn = inchannel
        # parse args
        if isinstance(self.size, list) or isinstance(self.size, tuple):
            if self.pad_mode == 'VALID':
                self.pad = 0
            else:
                self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size[0]-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * (self.size[1]-1))//2)
            self.kernel_size = (self.outchn, inchannel // self.groups, self.size[0], self.size[1])
        elif isinstance(self.size, int):
            if self.pad_mode == 'VALID':
                self.pad = 0
            else:
                self.pad = (self.size + (self.dilation_rate-1) * (self.size-1))//2
            self.kernel_size = (self.outchn, inchannel // self.groups, self.size, self.size)

    def build(self, *inputs):
        inp = inputs[0]
        self._parse_args(inp.shape)
        self.weight = Parameter(torch.Tensor(*self.size))
        if self.usebias:
            self.bias = Parameter(torch.Tensor(self.outchn))
        else:
            self.register_parameter('bias', None)
        self.reset_params()

    def reset_params(self):
        if self.get_flag('conv_init_mode')=='normal':
            init.normal_(self.weight, std=0.001)
        elif self.get_flag('conv_init_mode')=='resnet':
            _resnet_normal(self.weight)
        else:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        if self.bias is not None:
            if self.get_flag('conv_init_mode')=='normal':
                init.zeros_(self.bias)
            elif self.get_flag('conv_init_mode')=='resnet':
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
            else:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor, offset: Tensor, mask: Union[Tensor,None]=None) -> Tensor:
        return ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.pad, self.dilation_rate, mask)       # type: ignore

    def __call__(self, x: Tensor, offset: Tensor, mask: Union[Tensor,None]=None) -> Tensor:
        return super().__call__(x, offset, mask)

