from typing import Union

from torch import Tensor

from ..Base import Model
from ..Consts.Types import *
from .Quantizers import QQuantizers, QuantizerBase


class QAct(Model):
    bit_type: Union[QBitTypes, None]
    mode: QuantModes
    observer_str: Union[QObserverTypes, None]
    is_weight: bool
    zero_offset: bool

    def __init__(self, zero_offset: bool=False, mode: QuantModes='layer_wise', \
                    observer: Union[QObserverTypes,None]=None, bit_type: Union[QBitTypes, None]=None, is_weight: bool=False):
        super().__init__(self, zero_offset=zero_offset, mode=mode, observer=observer, bit_type=bit_type, is_weight=is_weight)           # make typechecker happy
        
    def initialize(self, zero_offset: bool=False, mode: QuantModes='layer_wise', \
                    observer: Union[QObserverTypes,None]=None, bit_type: Union[QBitTypes, None]=None, is_weight: bool=False):
        self.mode = mode 
        self.zero_offset = zero_offset
        self.observer_str = observer
        self.bit_type = bit_type
        self.is_weight = is_weight

    def build(self, x):
        if self._quant:
            bit_type: Union[QBitTypes, None] = self.get_flag('QActBit')
            if bit_type is None:
                bit_type = 'int8'
            if self.bit_type is not None:
                bit_type = self.bit_type

            obs_type = self.get_flag('QActObserver')
            if obs_type is None:
                obs_type = 'minmax'
            if self.observer_str is not None:
                obs_type = self.observer_str
            self.quantizer = QQuantizers['uniform'](bit_type=bit_type, zero_offset=self.zero_offset, mode=self.mode, observer=obs_type, is_weight=self.is_weight)

    def forward(self, x: Tensor) -> Tensor:
        if self._quant:
            x = self.quantizer(x)
        return x 

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
    
    def get_quantizer(self) -> QuantizerBase:
        return self.quantizer
    
    def set_quantizer(self, quantizer: QuantizerBase):
        if quantizer is not None:
            assert isinstance(quantizer, QuantizerBase), 'Must assign quantizer'
            self.quantizer = quantizer
    
