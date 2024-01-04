from abc import ABC, abstractmethod
from typing import Type, Union, Tuple, Dict

import torch
from loguru import logger
from torch import Tensor
from torch.nn.parameter import Parameter

from ..Base import Model
from ..Consts.Types import *
from .QTypes import QTypeBase


##### START: Observer classes 
class ObserverBase(Model, ABC):
    min_val: Union[Tensor, None]
    max_val: Union[Tensor, None]
    mode: QuantModes
    bit_type: Type[QTypeBase]
    zero_offset: bool
    is_weight: bool
    scale: Union[Tensor, None]
    zero_point: Union[Tensor, None]
    dim: int

    def __init__(self, bit_type: Type[QTypeBase], zero_offset: bool, mode: QuantModes, is_weight: bool):
        super().__init__(bit_type, zero_offset, mode, is_weight)
        assert mode in ['layer_wise', 'channel_wise']
        self.min_val = None 
        self.max_val = None 
        self.mode = mode 
        self.bit_type = bit_type
        self.zero_offset = zero_offset
        self.is_weight = is_weight
        self.scale = None 
        self.zero_point = None 

    def build(self, x: Tensor):
        if self.is_weight:
            self.dim = 0
        else:
            if len(x.shape)==4:
                self.dim = 1 
            else:
                self.dim = -1

    @abstractmethod
    def observe(self, x: Tensor):
        ...

    @torch.no_grad()
    def get_quant_params(self, max_val: Tensor, min_val: Tensor) -> Tuple[Tensor, Tensor]:
        if self.zero_offset:
            max_val = torch.max(max_val, -min_val)
            scale = max_val / min(self.bit_type.max_val, -self.bit_type.min_val)
            zero_point = torch.zeros_like(max_val)
        else:
            scale = (max_val - min_val) / float(self.bit_type.max_val - self.bit_type.min_val)
            zero_point = self.bit_type.min_val - torch.round(self.min_val / scale)
        scale.clamp(torch.finfo(torch.float32).eps)
        return scale, zero_point

    def _finish_calibrate(self):
        if self.scale is None:
            if (self.min_val is None) or (self.max_val is None):
                logger.warning('This quant layer is not fed with any data and it will be omitted. Use M.inspect_quant_params to get specific layer name')
            elif self.scale is None:
                s,z = self.get_quant_params(self.max_val, self.min_val)
                self.scale = Parameter(s)
                self.zero_point = Parameter(z)

    def forward(self, x: Tensor) -> Tensor:
        if self._quant_calibrating:
            self.observe(x)
        return x 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix+'scale' in state_dict:
            self.scale = Parameter(state_dict[prefix + 'scale'])
            self.zero_point = Parameter(state_dict[prefix + 'zero_point'])
        else:
            logger.debug(f'No scale in checkpoint for layer [{prefix}]')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self._quant_calibrated:
            destination[prefix + 'scale'] = self.scale 
            destination[prefix + 'zero_point'] = self.zero_point


class PercentileObserver(ObserverBase):
    sigma: float
    percentile: float

    def initialize(self, *args, **kwargs):
        self.sigma = 0.01 
        self.percentile = 0.99999

    def observe(self, x):
        if self.mode == 'channel_wise':
            x = x.transpose(0, self.dim)
            x = x.flatten(1)
            minv = torch.quantile(x.cuda(), 1-self.percentile, dim=1).cpu()
            maxv = torch.quantile(x.cuda(), self.percentile, dim=1).cpu()
        else:
            minv = torch.quantile(x.cuda(), 1-self.percentile).cpu()
            maxv = torch.quantile(x.cuda(), self.percentile).cpu()
        if (self.min_val is None) or (self.max_val is None):
            self.min_val = minv
            self.max_val = maxv
        else:
            self.min_val = self.sigma * minv + (1 - self.sigma) * self.min_val
            self.max_val = self.sigma * maxv + (1 - self.sigma) * self.max_val


class MinMaxObserver(ObserverBase):
    def observe(self, x):
        if self.mode == 'channel_wise':
            x = x.transpose(0, self.dim)
            x = x.flatten(1)
            minv = x.min(dim=1)[0]
            maxv = x.max(dim=1)[0]
        else:
            minv = x.min()
            maxv = x.max()
        if (self.min_val is None) or (self.max_val is None):
            self.min_val = minv
            self.max_val = maxv
        else:
            self.min_val = torch.minimum(self.min_val, minv)
            self.max_val = torch.maximum(self.max_val, maxv)


class OmseObserver(ObserverBase):
    x_buffer: Tensor

    def observe(self, x):
        if self.mode == 'channel_wise':
            x = x.transpose(-1, self.dim)
            x = x.flatten(0,-2)
            minv = x.min(dim=0)[0]
            maxv = x.max(dim=0)[0]
        else:
            minv = x.min()
            maxv = x.max()
        if (self.min_val is None) or (self.max_val is None):
            self.min_val = minv
            self.max_val = maxv
        else:
            self.min_val = torch.minimum(self.min_val, minv)
            self.max_val = torch.maximum(self.max_val, maxv)

        self.x_buffer = x

    @torch.no_grad()
    def _least_square(self, max_val: Tensor, min_val: Tensor, scale: Tensor, factor: float):
        if self.zero_offset:
            zero_point = torch.zeros_like(max_val)
        else:
            zero_point = self.bit_type.min_val - torch.round(min_val * factor / scale)

        scale = scale.cuda()
        zero_point = zero_point.cuda()
        x_buffer = self.x_buffer.cuda()

        scale = scale * factor
        x_buffer_q = x_buffer / scale + zero_point
        x_buffer_q = x_buffer_q.round().clamp(self.bit_type.min_val, self.bit_type.max_val)
        x_buffer_q = (x_buffer_q - zero_point) * scale 
        return torch.pow(x_buffer_q - x_buffer, 2).mean(), scale, zero_point

    @torch.no_grad()
    def get_quant_params(self, max_val: Tensor, min_val: Tensor):
        if self.zero_offset:
            max_val = torch.max(max_val, -min_val)
            scale = max_val / min(self.bit_type.max_val, -self.bit_type.min_val)
        else:
            scale = (max_val - min_val) / float(self.bit_type.max_val - self.bit_type.min_val)
        scale.clamp(torch.finfo(torch.float32).eps)
        min_score, scale_best, zero_point_best = self._least_square(max_val, min_val, scale, 1.0)
        for i in range(50):
            factor = 1 - 0.01*i
            score, scale_i, zero_point_i = self._least_square(max_val, min_val, scale, factor)
            if min_score is None:
                min_score = score
                scale_best = scale_i 
                zero_point_best = zero_point_i
            if score<min_score:
                min_score = score
                scale_best = scale_i
                zero_point_best = zero_point_i
        return scale_best, zero_point_best


QObservers: Dict[Union[QObserverTypes, str], Type[ObserverBase]] = {"minmax": MinMaxObserver, 'percentile': PercentileObserver, 'omse': OmseObserver}
##### END: Observer classes 

