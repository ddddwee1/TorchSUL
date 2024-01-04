from abc import ABC, abstractmethod

from torch import Tensor
from typing import Dict, Type

from ..Base import Model
from ..Consts.Types import *
from .Observers import QObservers
from .QTypes import QTYPES
from .QATop import QATFunc


##### START: Quantizer 
class QuantizerBase(Model, ABC):
    def __init__(self, bit_type: QBitTypes='int8', observer: QObserverTypes='minmax', zero_offset=False, mode: QuantModes='layer_wise', is_weight=False):
        super().__init__(bit_type=bit_type, observer=observer, zero_offset=zero_offset, mode=mode, is_weight=is_weight)              # make typechecker happy

    def initialize(self, bit_type: QBitTypes='int8', observer: QObserverTypes='minmax', zero_offset=False, mode: QuantModes='layer_wise', is_weight=False):
        self.bit_type = QTYPES[bit_type]
        self.observer = QObservers[observer](self.bit_type, zero_offset, mode, is_weight)
        self.mode = mode 
        self.is_weight = is_weight
        self.zero_offset = zero_offset

    def build(self, x):
        if self.is_weight:
            self.dim = 0
        else:
            if len(x.shape)==4:
                self.dim = 1 
            else:
                self.dim = -1

    @abstractmethod
    def forward(self):
        ...

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)


class UniformQuantizer(QuantizerBase):
    def forward(self, x: Tensor) -> Tensor:
        x = x.contiguous()
        if self._quant_calibrating:
            x = self.observer(x)
        if self._quant and self._quant_calibrated and (self.observer.scale is not None):
            if self.observer.scale.device!=x.device:
                self.observer.to(x.device)
            if self.get_flag('dump_onnx'):
                x = QATFunc.apply(x, self.observer.scale.data.reshape(-1), self.observer.zero_point.data.reshape(-1),     # type: ignore
                      self.bit_type.min_val, self.bit_type.max_val, self.zero_offset, self.mode, self.dim)     # type: ignore
            else:
                x = QATFunc.apply(x, self.observer.scale.contiguous().reshape(-1), self.observer.zero_point.contiguous().reshape(-1),     # type: ignore
                      self.bit_type.min_val, self.bit_type.max_val, self.zero_offset, self.mode, self.dim)     # type: ignore
        return x.contiguous() 


QQuantizers: Dict[Union[QuantizerTypes, str], Type[QuantizerBase]] = {"uniform": UniformQuantizer}
##### END: Quantizer 

