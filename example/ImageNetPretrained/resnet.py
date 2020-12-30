import torch.nn as nn 
from TorchSUL import Model as M

class Unit(M.Model):
    def initialize(self, outchn, stride):
        self.stride = stride 
        self.outchn = outchn 
        self.c1 = M.ConvLayer(1, outchn, batch_norm=True, usebias=False, activation=M.PARAM_RELU)
        self.c2 = M.ConvLayer(3, outchn, batch_norm=True, stride=stride, usebias=False, activation=M.PARAM_RELU)
        self.c3 = M.ConvLayer(1, outchn*4, batch_norm=True, usebias=False)
        self.sc = M.ConvLayer(1, outchn*4, batch_norm=True, stride=stride, usebias=False)

    def build(self, *inputs):
        self.inchn = inputs[0].shape[1]

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)

        if self.inchn==self.outchn*4 and self.stride==1:
            sc = x 
        else:
            sc = self.sc(x)
        out = out + sc 
        out = M.activation(out, M.PARAM_RELU)
        return out 

class Stage(M.Model):
    def initialize(self,outchn, blocknum, stride):
        self.units = nn.ModuleList()
        for i in range(blocknum):
            self.units.append(Unit(outchn, stride = stride if i==0 else 1))
    
    def forward(self, x):
        for u in self.units:
            x = u(x)
        return x 

class ResNet(M.Model):
    def initialize(self, channel_list, blocknum_list):
        self.c1 = M.ConvLayer(7, channel_list[0], stride=2, usebias=False, batch_norm=True, activation=M.PARAM_RELU)
        self.maxpool = M.MaxPool2D(3, 2)
        self.stage1 = Stage(channel_list[1], blocknum_list[0], stride=1)
        self.stage2 = Stage(channel_list[2], blocknum_list[1], stride=2)
        self.stage3 = Stage(channel_list[3], blocknum_list[2], stride=2)
        self.stage4 = Stage(channel_list[4], blocknum_list[3], stride=2)
        self.fc1 = M.Dense(1000)

    def forward(self, x):
        x = self.c1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = M.GlobalAvgPool(x)
        x = x.squeeze()
        x = self.fc1(x)
        return x 

    def debug(self, x):
        x = self.c1(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        return x 

def Res50():
    return ResNet([64, 64, 128, 256, 512], [3,4,6,3])
