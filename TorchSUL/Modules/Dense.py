import torch.nn as nn
from torch import Tensor

from .. import Layers as L
from ..Base import Model
from ..Consts.Activate import *
from ..Consts.Types import *
from ..Quant import QuantizerBase


class Dense(Model):
    fc: L.fcLayer
    bn: L.BatchNorm
    batch_norm: bool 
    act: L.Activation

    def __init__(self, outsize: int, batch_norm:bool=False, affine:bool=True, activation:int=-1, usebias:bool=True, norm:bool=False):
        super().__init__(outsize=outsize, batch_norm=batch_norm, affine=affine, activation=activation, usebias=usebias, norm=norm)

    def initialize(self, outsize, batch_norm=False, affine=True, activation=-1 , usebias=True, norm=False):
        self.fc = L.fcLayer(outsize, usebias, norm)
        self.batch_norm = batch_norm
        self.act = L.Activation(activation)
        if batch_norm:
            self.bn = L.BatchNorm(affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.batch_norm:
            assert len(x.shape)==2, f'Batch norm in dense layers only accepts num_dims=2, but got [{len(x.shape)}]'
            x = self.bn(x)
        x = self.act(x)
        return x 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def get_input_quantizer(self) -> QuantizerBase:
        return self.fc.input_quantizer

    def get_weight_quantizer(self) -> QuantizerBase:
        return self.fc.w_quantizer

    def set_input_quantizer(self, quantizer: QuantizerBase):
        self.fc.set_input_quantizer(quantizer)

    def to_torch_module(self):
        fc = self.fc.to_torch()
        setattr(self, 'fc', fc)
        if self.batch_norm:
            bn = self.bn.to_torch()
            setattr(self, 'bn', bn)
        if self.activation==PARAM_RELU:
            relu = nn.ReLU()
            setattr(self, 'act', relu)

    def _load_from_state_dict2(self, state_dict, prefix):
        if prefix+'fc.weight' in state_dict:
            return 
        if self.get_flag('from_torch'):
            w = state_dict.pop(prefix + 'weight')
            if self.fc.usebias:
                try:
                    b = state_dict.pop(prefix + 'bias')
                except:
                    raise KeyError(f'Bias is set for layer [{prefix}] but not found in checkpoint.')
            else:
                b = None

            state_dict[prefix+'fc.weight'] = w
            if b is not None: 
                state_dict[prefix+'fc.bias'] = b

