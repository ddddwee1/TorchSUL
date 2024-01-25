from .QCalibrator import LayerCalibrator
from .Quantizers import QQuantizers
from ..Base import Model


class QMatmul(Model):
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
            a = self.a_quantizer(a)
            b = self.b_quantizer(b)
        return a @ b 


class QAdd(Model):
    def build(self, *args, **kwargs):
        if self._quant:
            bit_type = self.get_flag('QActBit')
            if bit_type is None:
                bit_type = 'int8'

            if self.get_flag('LayerwiseQuant'):
                self.a_quantizer = QQuantizers['uniform'](zero_offset=False, observer='placeholder')
                self.b_quantizer = QQuantizers['uniform'](zero_offset=False, observer='placeholder')
                self.calibrator = LayerCalibrator([self.a_quantizer, self.b_quantizer], self.forward)
                self.register_forward_hook(self.calibrator.layer_hook)
            else:
                obs_type = self.get_flag('QActObserver')
                if obs_type is None:
                    obs_type = 'minmax'
                self.a_quantizer = QQuantizers['uniform'](zero_offset=False, observer=obs_type)
                self.b_quantizer = QQuantizers['uniform'](zero_offset=False, observer=obs_type)

    def forward(self, a, b):
        if self._quant:
            a = self.a_quantizer(a)
            b = self.b_quantizer(b)
        return a + b 


