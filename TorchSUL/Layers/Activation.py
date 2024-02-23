import torch
import torch.nn.functional as F
from torch import Tensor

from ..Base import Model


def activation(x: Tensor, act, **kwargs) -> Tensor:
    if act==-1:
        return x
    elif act==0:
        return F.relu(x)
    elif act==1:
        return F.leaky_relu(x, negative_slope=0.1)
    elif act==2:
        return F.elu(x)
    elif act==3:
        return F.tanh(x)
    elif act==6:
        return torch.sigmoid(x)
    elif act==10:
        return F.gelu(x)
    elif act==11:
        return F.silu(x)
    else:
        raise NotImplementedError(f'Activation [{act}] is not supported')


class Activation(Model):
    act_num: int

    def __init__(self, act: int):
        super().__init__(act)

    def initialize(self, act: int):
        self.act_num = act 
        if act==9:
            self.act = torch.nn.PReLU(num_parameters=1)
    
    def build(self, x:Tensor):
        if self.act_num==8:
            self.act = torch.nn.PReLU(num_parameters=x.shape[-1]) 

    def forward(self, x: Tensor) -> Tensor:
        if self.act_num==8 or self.act_num==9:
            return self.act(x)
        else:
            return activation(x, self.act_num)

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

