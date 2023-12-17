from typing import Literal, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..Base import Model
from ..Consts.Types import *


## Pool layers 
class MaxPool2d(Model):
    size: TypeKSize2D
    stride: TypeKSize2D
    pad_mode: PadModes
    dilation_rate: int
    pad: Union[int, tuple[int,int]]

    def __init__(self, size:TypeKSize2D, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1):
        super().__init__(size, stride, pad, dilation_rate)

    def initialize(self, size:TypeKSize2D, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1):
        self.size = size
        self.stride = stride
        self.pad_mode = pad
        self.dilation_rate = dilation_rate
        
    def build(self, x:Tensor):
        self._parse_args()

    def _parse_args(self):
        # parse args
        if isinstance(self.size,list) or isinstance(self.size, tuple):
            if self.pad_mode == 'VALID':
                self.pad = 0
            else:
                self.pad = (self.size[0]//2, self.size[1]//2)
        else:
            if self.pad_mode == 'VALID':
                self.pad = 0
            else:
                self.pad = self.size//2

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.size, self.stride, self.pad, self.dilation_rate, False, False)
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
    
    def to_torch(self) -> nn.MaxPool2d:
        pool = nn.MaxPool2d(self.size, self.stride, padding=self.pad, dilation=self.dilation_rate) #type: ignore
        return pool


class AvgPool2d(Model):
    size: TypeKSize2D
    stride: TypeKSize2D
    pad_mode: PadModes
    dilation_rate: int
    pad: Union[int, tuple[int,int]]

    def __init__(self, size:TypeKSize2D, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1):
        super().__init__(size, stride, pad, dilation_rate)

    def initialize(self, size:TypeKSize2D, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1):
        self.size = size
        self.stride = stride
        self.pad_mode = pad
        self.dilation_rate = dilation_rate
        
    def build(self, x:Tensor):
        self._parse_args()

    def _parse_args(self):
        # parse args
        if isinstance(self.size,list) or isinstance(self.size, tuple):
            if self.pad_mode == 'VALID':
                self.pad = 0
            else:
                self.pad = (self.size[0]//2, self.size[1]//2)
        else:
            if self.pad_mode == 'VALID':
                self.pad = 0
            else:
                self.pad = self.size//2

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, self.size, self.stride, self.pad, self.dilation_rate, False, False)
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
    
    def to_torch(self) -> nn.AvgPool2d:
        pool = nn.AvgPool2d(self.size, self.stride, padding=self.pad, dilation=self.dilation_rate) #type: ignore
        return pool
