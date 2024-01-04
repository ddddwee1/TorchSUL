import torch
from torch import Tensor
from typing import Tuple, List 

from .. import Layers as L

from.Conv import ConvLayer
from ..Base import Model
from ..Consts.Types import *
from .Dense import Dense


class LSTMCell(Model):
    def __init__(self, outdim: int, usebias: bool=False):
        super().__init__(outdim=outdim, usebias=usebias)

    def initialize(self, outdim: int, usebias: bool=False):
        self.F = Dense(outdim, usebias=usebias, norm=False)
        self.O = Dense(outdim, usebias=usebias, norm=False)
        self.I = Dense(outdim, usebias=usebias, norm=False)
        self.C = Dense(outdim, usebias=usebias, norm=False)

        self.hF = Dense(outdim, usebias=usebias, norm=False)
        self.hO = Dense(outdim, usebias=usebias, norm=False)
        self.hI = Dense(outdim, usebias=usebias, norm=False)
        self.hC = Dense(outdim, usebias=usebias, norm=False)

    def forward(self, x: Tensor, h: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        f = self.F(x) + self.hF(h)
        o = self.O(x) + self.hO(h)
        i = self.I(x) + self.hI(h)
        c = self.C(x) + self.hC(h)

        f_ = torch.sigmoid(f)
        c_ = torch.tanh(c) * torch.sigmoid(i)
        o_ = torch.sigmoid(o)

        next_c = c_prev * f_ + c_ 
        next_h = o_ * torch.tanh(next_c)
        return next_h, next_c
    
    def __call__(self, x: Tensor, h: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(x, h, c_prev)


class ConvLSTM(Model):
    def __init__(self, chn: int, usebias: bool=False):
        super().__init__(chn=chn, usebias=usebias)

    def initialize(self, chn: int, usebias: bool=False):
        self.gx = ConvLayer(3, chn, usebias=usebias)
        self.gh = ConvLayer(3, chn, usebias=usebias)
        self.fx = ConvLayer(3, chn, usebias=usebias)
        self.fh = ConvLayer(3, chn, usebias=usebias)
        self.ox = ConvLayer(3, chn, usebias=usebias)
        self.oh = ConvLayer(3, chn, usebias=usebias)
        self.ix = ConvLayer(3, chn, usebias=usebias)
        self.ih = ConvLayer(3, chn, usebias=usebias)

    def forward(self, x: Tensor, c: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        gx = self.gx(x)
        gh = self.gh(h)

        ox = self.ox(x)
        oh = self.oh(h)

        fx = self.fx(x)
        fh = self.fh(h)

        ix = self.ix(x)
        ih = self.ih(h)

        g = torch.tanh(gx + gh)
        o = torch.sigmoid(ox + oh)
        i = torch.sigmoid(ix + ih)
        f = torch.sigmoid(fx + fh)

        cell = f*c + i*g 
        h = o * torch.tanh(cell)
        return cell, h 
    
    def __call__(self, x: Tensor, c: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        return super().__call__(x, c, h)


class AdaptConv3(Model):
    # Originally proposed in DEKR
    regular_matrix: Tensor
    transform_conv: ConvLayer
    translation_conv: ConvLayer
    deform_conv: L.DeformConv2D
    batch_norm: bool
    bn: L.BatchNorm
    act: L.Activation

    def __init__(self, outchn: int, stride: int=1, pad: PadModes='SAME_LEFT', dilation_rate: int=1, batch_norm: bool=False, activation: int=-1, usebias: bool=True):
        super().__init__(outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, batch_norm=batch_norm, activation=activation, usebias=usebias)

    def initialize(self, outchn: int, stride: int=1, pad: PadModes='SAME_LEFT', dilation_rate: int=1, batch_norm: bool=False, activation: int=-1, usebias: bool=True):
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],\
            [-1, 0, 1, -1 ,0 ,1 ,-1, 0, 1]])
        # register to buffer that it can be mangaed by cuda or cpu
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.transform_conv = ConvLayer(3, 4)
        self.translation_conv = ConvLayer(3, 2)
        self.deform_conv = L.DeformConv2D(3, outchn, stride, pad, dilation_rate, usebias)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = L.BatchNorm()
        self.act = L.Activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        trans_mtx = self.transform_conv(x)
        trans_mtx = trans_mtx.permute(0,2,3,1).reshape((N*H*W,2,2))
        offset = torch.matmul(trans_mtx, self.regular_matrix)
        offset = offset-self.regular_matrix
        offset = offset.transpose(1,2).reshape((N,H,W,18)).permute(0,3,1,2)

        translation = self.translation_conv(x)
        offset[:,0::2,:,:] += translation[:,0:1,:,:]
        offset[:,1::2,:,:] += translation[:,1:2,:,:]

        out = self.deform_conv(x, offset)
        
        if self.batch_norm:
            out = self.bn(out)
        
        if self.activation!=-1:
            out = self.act(out)
        return out 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
