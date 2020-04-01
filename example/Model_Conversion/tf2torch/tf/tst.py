import model3 as M 
import hrnet 
import numpy as np 
import tensorflow as tf 
import cv2 
import glob 
import pickle
import json  
import os 
import config 
from tqdm import tqdm 
import network 
import util
import matplotlib.pyplot as plt 

# for 2D prediction 
class HRNET(M.Model):
	def initialize(self, num_pts):
		self.backbone = hrnet.ResNet()
		self.lastconv = M.ConvLayer(1, num_pts)
	def forward(self, x):
		feat = self.backbone(x)
		hmap = self.lastconv(feat)
		return hmap
tf.keras.backend.set_learning_phase(False)

net = HRNET(17)
M.Saver(net).restore('./modelhr/')

dumb_inp = np.ones([1,256,256,3])

out = net(dumb_inp)

out = tf.transpose(out, [0,3,1,2])
print(out.shape)
print(out)
# import buff 
# print(len(buff.variables))

# res = []
# vs = buff.variables
# for layer in vs:
# 	buff = {}
# 	for v in layer:
# 		buff[v.name] = v.numpy()
# 	res.append(buff)

# import pickle 
# pickle.dump(res, open('hrnet_variables.pkl', 'wb'))
# print('variables dumped.')

# print(len(res))

# # dump variables 
# vs = net.variables 

# print(len(vs))

# variables = {}

# for v in vs:
# 	print(v.name)
# 	variables[v.name] = v.numpy()

# import pickle 
# pickle.dump(variables, open('hrnet_variables.pkl', 'wb'))
# print('variables dumped.')

