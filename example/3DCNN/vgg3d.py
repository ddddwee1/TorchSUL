import Model as M 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class VGG3D(M.Model):
	def initialize(self):
		self.c11 = M.ConvLayer3D((5,5,1), 32, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c12 = M.ConvLayer3D((5,5,1), 32, stride=(2,2,1), activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		self.c21 = M.ConvLayer3D((3,3,3), 64, stride=(1,1,2), activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c22 = M.ConvLayer3D((3,3,1), 64, stride=(2,2,1), activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		self.c31 = M.ConvLayer3D((3,3,3), 64, stride=(1,1,2), activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c32 = M.ConvLayer3D((3,3,1), 64, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c33 = M.ConvLayer3D((3,3,1), 64, stride=(2,2,1), activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		self.c41 = M.ConvLayer3D((3,3,3), 128, stride=(1,1,2), activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c42 = M.ConvLayer3D((3,3,1), 128, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c43 = M.ConvLayer3D((3,3,1), 128, stride=(2,2,1), activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		self.c51 = M.ConvLayer3D((3,3,3), 256, stride=(1,1,2), activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c52 = M.ConvLayer3D((3,3,1), 256, activation=M.PARAM_RELU, batch_norm=True, usebias=False)
		self.c53 = M.ConvLayer3D((3,3,1), 256, activation=M.PARAM_RELU, batch_norm=True, usebias=False)

		self.fc = M.Dense(512, usebias=False)
		self.classifier = M.Dense(1, usebias=False)

	def forward(self, x):
		x = self.c11(x)
		x = self.c12(x)

		x = self.c21(x)
		x = self.c22(x)

		x = self.c31(x)
		x = self.c32(x)
		x = self.c33(x)

		x = self.c41(x)
		x = self.c42(x)
		x = self.c43(x)

		x = self.c51(x)
		x = self.c52(x)
		x = self.c53(x)

		x = M.flatten(x)
		x = self.fc(x)
		x = self.classifier(x)
		return x 

# class Classifier(M.Model):
# 	def initialize(self):
# 		self.fc = M.Dense()