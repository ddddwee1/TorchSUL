import math
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from ..Base import Model
from ..Consts.Types import *
from ..Quant import QQuantizers, QuantizerBase


## Resnet style initialization 
def _resnet_normal(tensor):
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(2.0 / float(fan_out))
    return init._no_grad_normal_(tensor, 0., std)


######  Layers 
class ConvBase(Model, ABC):
    size: TypeKSize
    stride: TypeKSize
    kernel_size = Tuple[int,int,int,int]
    outchn: int 
    pad_mode: PadModes
    dilation_rate: int 
    usebias: bool
    groups: int 
    pad: Union[int, Tuple[int,int]]
    groups: int
    inchn: int
    weight: Parameter
    bias: Union[Parameter, None]
    input_quantizer: QuantizerBase
    w_quantizer: QuantizerBase

    def __init__(self, size:TypeKSize, outchn:int, stride:TypeKSize=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)

    def initialize(self, size:TypeKSize, outchn:int, stride:TypeKSize=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1) -> None:
        self.size = size
        self.outchn = outchn
        self.stride = stride
        self.usebias = usebias
        self.groups = groups
        self.dilation_rate = dilation_rate
        assert (pad in ['VALID','SAME_LEFT'])
        self.pad_mode = pad 
        self.bias = None

    def build(self, *inputs):
        inp = inputs[0]
        self._parse_args(inp.shape)
        self.weight = Parameter(torch.Tensor(*self.kernel_size))
        if self.usebias:
            self.bias = Parameter(torch.Tensor(self.outchn))
        self.reset_params()

        if self._quant:
            bit_type = self.get_flag('QActBit')
            if bit_type is None:
                bit_type = 'int8'
            obs_type = self.get_flag('QActObserver')
            if obs_type is None:
                obs_type = 'minmax'
            self.input_quantizer = QQuantizers['uniform'](zero_offset=False, bit_type=bit_type, observer=obs_type)
            self.w_quantizer = QQuantizers['uniform'](zero_offset=True, mode='channel_wise', is_weight=True)

    @abstractmethod
    def _parse_args(self, shape: List[int]):
        ...

    def set_input_quantizer(self, quantizer: QuantizerBase):
        if quantizer is not None:
            assert isinstance(quantizer, QuantizerBase), 'Must assign quantizer'
            self.input_quantizer = quantizer

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
    
    def __call__(self, x:Tensor) -> Tensor:
        return super().__call__(x)


class conv2D(ConvBase):
    size: TypeKSize2D
    stride: TypeKSize2D
    def __init__(self, size:TypeKSize2D, outchn:int, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)

    def initialize(self, size:TypeKSize2D, outchn:int, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1) -> None:
        super().initialize(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)
    
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

    def forward(self, x) -> Tensor:
        weight = self.weight
        if self._quant:
            x = self.input_quantizer(x)
            weight = self.w_quantizer(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.pad, self.dilation_rate, self.groups)

    def to_torch(self) -> nn.Conv2d:
        conv = nn.Conv2d(in_channels = self.inchn, out_channels = self.outchn, kernel_size = self.kernel_size[2:], stride = self.stride,      # type: ignore
                        padding = self.pad, padding_mode = 'zeros', dilation = self.dilation_rate, groups = self.groups, bias = self.usebias)
        conv.weight.data[:] = self.weight.data[:]
        if (self.bias is not None) and (conv.bias is not None):
            conv.bias.data[:] = self.bias.data[:]
        return conv 


class deconv2D(ConvBase):
    size: int
    out_pad: int
    stride: int

    def __init__(self, size:int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)

    def initialize(self, size:int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1) -> None:
        super().initialize(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)

    def _parse_args(self, input_shape):
        inchannel = input_shape[1]
        # parse args
        if isinstance(self.size,int):
            if self.pad_mode == 'VALID':
                self.pad = 0
                self.out_pad = 0
            else:
                self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2 - (1 - self.size%2)
                self.out_pad = self.stride - 1         
            self.kernel_size = (inchannel, self.outchn // self.groups, self.size, self.size)
        else:
            raise NotImplementedError("Deconv kernel only supports int")

    def forward(self, x: Tensor) -> Tensor:
        inh, inw = x.shape[2], x.shape[3]
        weight = self.weight
        if self._quant:
            x = self.input_quantizer(x)
            weight = self.w_quantizer(weight)
        x = F.conv_transpose2d(x, weight, self.bias, self.stride, self.pad, self.out_pad, self.groups, self.dilation_rate)
        outh, outw = x.shape[2], x.shape[3]
        if self.padmethod=='SAME_LEFT':
            if outh!=inh*self.stride or outw!=inw*self.stride:
                x = x[:,:,:inh*self.stride,:inw*self.stride]
        return x 
    
    def to_torch(self) -> nn.ConvTranspose2d:
        conv = nn.ConvTranspose2d(in_channels = self.inchn, out_channels = self.outchn, kernel_size = self.kernel_size[2:], stride = self.stride,       # type: ignore
                        padding = self.pad, output_padding = self.out_pad, padding_mode = 'zeros', dilation = self.dilation_rate, groups = self.groups, bias = self.usebias)
        conv.weight.data[:] = self.weight.data[:]
        if (self.bias is not None) and (conv.bias is not None):
            conv.bias.data[:] = self.bias.data[:]
        return conv 


class conv1D(ConvBase):
    size: int
    kernel_size: Tuple[int,int,int]
    pad: int
    stride: int

    def __init__(self, size:int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)

    def initialize(self, size:int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1) -> None:
        super().initialize(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)

    def _parse_args(self, input_shape):
        inchannel = input_shape[1]
        # parse args
        if self.pad_mode == 'VALID':
            self.pad = 0
        else:
            self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
        self.kernel_size = (self.outchn, inchannel // self.gropus, self.size)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.groups)
    
    def to_torch(self) -> nn.Conv1d:
        conv = nn.Conv1d(in_channels = self.inchn, out_channels = self.outchn, kernel_size = self.kernel_size[2:], stride = self.stride,        # type: ignore
                        padding = self.pad, padding_mode = 'zeros', dilation = self.dilation_rate, groups = self.groups, bias = self.usebias)
        conv.weight.data[:] = self.weight.data[:]
        if (self.bias is not None) and (conv.bias is not None):
            conv.bias.data[:] = self.bias.data[:]
        return conv 


class conv3D(ConvBase):
    size: TypeKSize3D
    stride: TypeKSize3D
    kernel_size: Tuple[int,int,int,int,int]
    pad: Union[int, Tuple[int,int,int]]

    def __init__(self, size:TypeKSize3D, outchn:int, stride:TypeKSize3D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)

    def initialize(self, size:TypeKSize3D, outchn:int, stride:TypeKSize3D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, usebias:bool=True, groups:int=1) -> None:
        super().initialize(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, usebias=usebias, groups=groups)
    
    def _parse_args(self, input_shape):
        inchannel = input_shape[1]
        # parse args
        if isinstance(self.size,list) or isinstance(self.size, tuple):
            if self.pad == 'VALID':
                self.pad = 0
            else:
                self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size[0]-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size[1]-1 ))//2, (self.size[2]+ (self.dilation_rate-1) * ( self.size[2]-1 ))//2)
            self.size = [self.outchn, inchannel // self.gropus, self.size[0], self.size[1], self.size[2]]
        else:
            if self.pad == 'VALID':
                self.pad = 0
            else:
                self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
            self.kernel_size = (self.outchn, inchannel // self.gropus, self.size, self.size, self.size)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv3d(x, self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.groups)
    
    def to_torch(self) -> nn.Conv3d:
        conv = nn.Conv3d(in_channels = self.inchn, out_channels = self.outchn, kernel_size = self.kernel_size[2:], stride = self.stride,       # type: ignore
                        padding = self.pad, padding_mode = 'zeros', dilation = self.dilation_rate, groups = self.groups, bias = self.usebias)
        conv.weight.data[:] = self.weight.data[:]
        if (self.bias is not None) and (conv.bias is not None):
            conv.bias.data[:] = self.bias.data[:]
        return conv 
