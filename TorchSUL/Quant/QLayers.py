from .QCalibrator import LayerCalibrator
from .Quantizers import QQuantizers
from ..Base import Model
from torch import Tensor 


class QMatmul(Model):
    def __init__(self, swap_ab: bool=False):
        super().__init__(swap_ab)

    def initialize(self, swap_ab: bool=False):
        self.swap_ab = swap_ab

    def build(self, *args, **kwargs):
        if self._quant:
            bit_type = self.get_flag('QActBit')
            if bit_type is None:
                bit_type = 'int8'

            if self.get_flag('LayerwiseQuant'):
                self.a_quantizer = QQuantizers['uniform'](zero_offset=True, observer='placeholder')
                self.b_quantizer = QQuantizers['uniform'](zero_offset=True, observer='placeholder')
                self.calibrator = LayerCalibrator([self.a_quantizer, self.b_quantizer], self.forward)
                self.register_forward_hook(self.calibrator.layer_hook)
            else:
                obs_type = self.get_flag('QActObserver')
                if obs_type is None:
                    obs_type = 'minmax'
                self.a_quantizer = QQuantizers['uniform'](zero_offset=True, observer=obs_type)
                self.b_quantizer = QQuantizers['uniform'](zero_offset=True, observer=obs_type)

    def forward(self, a, b):
        if self._quant:
            if self.swap_ab:
                a = self.b_quantizer(a)
                b = self.a_quantizer(b)
            else:
                a = self.a_quantizer(a)
                b = self.b_quantizer(b)
        return a @ b 
    
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return super().__call__(a, b)


class QAdd(Model):
    def __init__(self, swap_ab: bool=False, quant_left: bool=True, quant_right: bool=True):
        super().__init__(swap_ab, quant_left, quant_right)

    def initialize(self, swap_ab: bool=False, quant_left: bool=True, quant_right: bool=True) -> None:
        self.swap_ab = swap_ab
        self.quant_left = quant_left
        self.quant_right = quant_right

    def build(self, *args, **kwargs):
        if self._quant:
            bit_type = self.get_flag('QActBit')
            if bit_type is None:
                bit_type = 'int8'

            if self.get_flag('LayerwiseQuant'):
                quantizers = []
                if (self.quant_left and not self.swap_ab) or (self.quant_right and self.swap_ab):
                    self.a_quantizer = QQuantizers['uniform'](zero_offset=False, observer='placeholder')
                    quantizers.append(self.a_quantizer)
                if (self.quant_right and not self.swap_ab) or (self.quant_left and self.swap_ab):
                    self.b_quantizer = QQuantizers['uniform'](zero_offset=False, observer='placeholder')
                    quantizers.append(self.b_quantizer)
                self.calibrator = LayerCalibrator(quantizers, self.forward)
                self.register_forward_hook(self.calibrator.layer_hook)
            else:
                obs_type = self.get_flag('QActObserver')
                if obs_type is None:
                    obs_type = 'minmax'
                if (self.quant_left and not self.swap_ab) or (self.quant_right and self.swap_ab):
                    self.a_quantizer = QQuantizers['uniform'](zero_offset=False, observer=obs_type)
                if (self.quant_right and not self.swap_ab) or (self.quant_left and self.swap_ab):
                    self.b_quantizer = QQuantizers['uniform'](zero_offset=False, observer=obs_type)

    def forward(self, a, b):
        if self._quant:
            if self.swap_ab:
                if self.quant_left:
                    a = self.b_quantizer(a)
                if self.quant_right:
                    b = self.a_quantizer(b)
            else:
                if self.quant_left:
                    a = self.a_quantizer(a)
                if self.quant_right:
                    b = self.b_quantizer(b)
        return a + b 

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return super().__call__(a, b)
    
