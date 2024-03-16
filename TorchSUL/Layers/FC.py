import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from ..Base import Model
from ..Consts.Types import *
from ..Quant import QQuantizers, QuantizerBase


class fcLayer(Model):
    outsize: int 
    usebias: bool
    norm: bool
    insize: int
    weight: Parameter
    bias: Union[Parameter, None]
    input_quantizer: QuantizerBase
    w_quantizer: QuantizerBase

    def __init__(self, outsize: int, usebias:bool=True, norm:bool=False):
        super().__init__(outsize=outsize, usebias=usebias, norm=norm)

    def initialize(self, outsize: int, usebias:bool=True, norm:bool=False):
        self.outsize = outsize
        self.usebias = usebias
        self.norm = norm
        # self.bias = None 

    def build(self, *inputs):
        self.insize = inputs[0].shape[-1]
        self.weight = Parameter(torch.Tensor(self.outsize, self.insize))
        if self.usebias:
            self.bias = Parameter(torch.Tensor(self.outsize))
        else:
            self.register_parameter('bias', None)
        self.reset_params()

        if self._quant:
            bit_type: QBitTypes = self.get_flag('QActBit')
            if bit_type is None:
                bit_type = 'int8'
            obs_type: QObserverTypes = self.get_flag('QActObserver')
            if obs_type is None:
                obs_type = 'minmax'
            self.input_quantizer = QQuantizers['uniform'](zero_offset=False, bit_type=bit_type, observer=obs_type)
            self.w_quantizer = QQuantizers['uniform'](zero_offset=True, mode='channel_wise', is_weight=True)

    def set_input_quantizer(self, quantizer: QuantizerBase):
        if quantizer is not None:
            assert isinstance(quantizer, QuantizerBase), 'Must assign quantizer'
            self.input_quantizer = quantizer

    def reset_params(self):
        if self.get_flag('fc_init_mode')=='normal':
            init.normal_(self.weight, std=0.001)
            if self.bias is not None:
                init.zeros_(self.bias)
        else:
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm:
            with torch.no_grad():
                norm = x.norm(p=2, dim=1, keepdim=True)
                wnorm = self.weight.norm(p=2,dim=1, keepdim=True)
            x = x / norm
            weight = self.weight / wnorm
        else:
            weight = self.weight

        if self._quant:
            x = self.input_quantizer(x).contiguous()
            weight = self.w_quantizer(weight).contiguous()
        return F.linear(x, weight, self.bias)
    
    def __call__(self, x:Tensor) -> Tensor:
        return super().__call__(x)

    def to_torch(self) -> nn.Linear:
        fc = nn.Linear(self.insize, self.outsize, self.usebias)
        fc.weight.data[:] = self.weight.data[:]
        if self.bias is not None:
            fc.bias.data[:] = self.bias.data[:]
        return fc 
