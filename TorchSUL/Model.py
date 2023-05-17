from . import Layers as L 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import os 
import copy 
from distutils.version import LooseVersion
from torch.nn.parameter import Parameter
import torch.nn.init as init 
import torchvision.ops as ops

Model = L.Model
activation = L.activation
Activation = L.Activation
flatten = L.flatten
Flatten = L.Flatten
GlobalAvgPool = L.GlobalAvgPool2D
GlobalAvgPoolLayer = L.GlobalAvgPool2DLayer
BatchNorm = L.BatchNorm
LayerNorm = L.LayerNorm
MaxPool2D = L.MaxPool2d
AvgPool2D = L.AvgPool2d
NNUpSample = L.NNUpSample
BilinearUpSample = L.BilinearUpSample

# activation const
PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_MFM_FC = 5
PARAM_SIGMOID = 6
PARAM_SWISH = 7
PARAM_PRELU = 8
PARAM_PRELU1 = 9
PARAM_GELU = 10

# recorded paramters 
def get_record():
	return L.record_params
def reset_record():
	L.record_params = []

def init_caffe_input(x):
	global caffe_string, layer_counter
	if not 'caffe_string' in globals():
		caffe_string = ''
	if not 'layer_counter' in globals():
		layer_counter = 0
	caffe_string += 'layer{\n'
	caffe_string += ' name: "%s"\n'%x[1]()
	caffe_string += ' type: "Input"\n'
	caffe_string += ' top: "%s"\n'%x[1]()
	caffe_string += ' input_param{\n  shape{\n   dim:%d\n   dim:%d\n   dim:%d\n   dim:%d\n  }\n }\n}\n'%(x[0].shape[0], x[0].shape[3], x[0].shape[1], x[0].shape[2])
	layer_counter += 1 

def init_model(model, *args, **kwargs):
	# run one forward loop to initialize model 
	with torch.no_grad():
		return model(*args, **kwargs)

def to_standard_torch(model, inplace=True):
	if not inplace:
		model = copy.deepcopy(model)
	convertible_layers = (ConvLayer, Dense, DWConvLayer)
	replacable_layers = (BatchNorm,)
	for n,c in model.named_children():
		if isinstance(c, convertible_layers):
			c.to_torch()
		elif isinstance(c, replacable_layers):
			rep = c.to_torch()
			setattr(model, n.split('.')[-1], rep)
		else:
			to_standard_torch(c)
	return model 

class Saver():
	def __init__(self, module):
		self.model = module

	def _get_checkpoint(self, path):
		path = path.replace('\\','/')
		ckpt = path + 'checkpoint'
		if os.path.exists(ckpt):
			fname = open(ckpt).readline().strip()
			return path + fname
		else:
			return False

	def _exclude(self, d, exclude):
		if exclude is not None:
			for e in exclude:
				if e in d:
					d.pop(e)
		return d 

	def restore(self, path, strict=True, exclude=None):
		print('Trying to load from:',path)
		# print(path[-4:])
		device = torch.device('cpu')
		if path[-4:] == '.pth':
			if not os.path.exists(path):
				print('Path:',path, 'does not exsist.')
			elif isinstance(self.model, nn.DataParallel):
				state_dict = torch.load(path, map_location=device)
				state_dict = self._exclude(state_dict, exclude)
				self.model.module.load_state_dict(state_dict, strict=strict)
				print('Model loaded from:', path)
			else:
				state_dict = torch.load(path, map_location=device)
				state_dict = self._exclude(state_dict, exclude)
				self.model.load_state_dict(state_dict, strict=strict)
				print('Model loaded from:', path)
		else:
			path = self._get_checkpoint(path)
			if path:
				if isinstance(self.model, nn.DataParallel):
					state_dict = torch.load(path, map_location=device)
					state_dict = self._exclude(state_dict, exclude)
					self.model.module.load_state_dict(state_dict, strict=strict)
				else:
					state_dict = torch.load(path, map_location=device)
					state_dict = self._exclude(state_dict, exclude)
					self.model.load_state_dict(state_dict, strict=strict)
				print('Model loaded from:', path)
			else:
				print('No checkpoint found. No restoration will be performed.')

	def save(self, path):
		directory = os.path.dirname(path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		if isinstance(self.model, nn.DataParallel):
			if LooseVersion(torch.__version__)>=LooseVersion('1.6.0'):
				torch.save(self.model.module.state_dict(), path, _use_new_zipfile_serialization=False)
			else:
				torch.save(self.model.module.state_dict(), path)
		else:
			if LooseVersion(torch.__version__)>=LooseVersion('1.6.0'):
				torch.save(self.model.state_dict(), path, _use_new_zipfile_serialization=False)
			else:
				torch.save(self.model.state_dict(), path)
		print('Model saved to:',path)
		ckpt = open(directory + '/checkpoint', 'w')
		ckpt.write(os.path.basename(path))
		ckpt.close()

class ConvLayer(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, activation=-1, batch_norm=False, affine=True, usebias=True, groups=1):
		self.conv = L.conv2D(size, outchn, stride, pad, dilation_rate, usebias, groups)
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
		self.batch_norm = batch_norm
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=outchn)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)
		else:
			self.act = L.Activation(activation)
	def forward(self, x):
		if self._record:
			record_flag = True
			self.un_record()
		else:
			record_flag = False
		# print('Record:',record_flag)
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)

		if record_flag:
			# print(self._merge_bn)
			# do record
			if self._merge_bn:
				miu = self.bn.running_mean
				var = self.bn.running_var
				gamma = self.bn.weight
				beta = self.bn.bias 
				eps = self.bn.eps 
				if gamma is None:
					gamma = 1 
				if beta is None:
					beta = 0
				weight = self.conv.weight
				bias = self.conv.bias

				if bias is not None:
					b = gamma * (bias - miu) / torch.sqrt(eps + var) + beta 
				else:
					b = beta - (gamma * miu / torch.sqrt(eps + var))
				bn_w = gamma / torch.sqrt(eps + var)
				bn_w = bn_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
				w = bn_w * weight

				res = {}
				base_name = list(self.named_parameters())[0][0]
				base_name = '.'.join(base_name.split('.')[:-1])
				res[base_name+'.weight'] = w 
				res[base_name+'.bias'] = b

				if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
					actw = self.act.weight
					res['act.weight'] = actw
				# print(res.keys())
			else:
				res = {}
				for p in self.named_parameters():
					res[p[0]] = p[1]
				for p in self.named_buffers():
					res[p[0]] = p[1]
			L.record_params.append(res)
		return x 

	def to_torch(self):
		conv = self.conv.to_torch()
		setattr(self, 'conv', conv)
		if self.batch_norm:
			bn = self.bn.to_torch()
			setattr(self, 'bn', bn)
		if self.activation==PARAM_RELU:
			relu = nn.ReLU()
			setattr(self, 'act', relu)

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		keys = list(state_dict.keys())
		if (prefix+'conv.weight') in keys:
			super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
		elif (self.get_flag('fc2conv')) and ((prefix + 'fc.weight') in keys):
			assert self.conv.size==1, 'fc2conv must set kernel size to 1'
			w = state_dict[prefix+'fc.weight'].unsqueeze(-1).unsqueeze(-1)
			self.conv.weight.data[:] = w.data[:]
			if self.conv.usebias:
				if (prefix + 'fc.bias') in keys:
					b = state_dict[prefix+'fc.bias']
					self.conv.bias.data[:] = b.data[:]
				else:
					raise Exception('Bias not exists in checkpoint for layer', prefix)
			else:
				if strict and ((prefix + 'fc.bias') in keys):
					raise Exception('Bias is not used but exists in checkpoint for layer', prefix)


class DeConvLayer(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, activation=-1, batch_norm=False, affine=True, usebias=True, groups=1):
		self.conv = L.deconv2D(size, outchn, stride, pad, dilation_rate, usebias, groups)
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
		self.batch_norm = batch_norm
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=outchn)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)
	def forward(self, x):
		if self._record:
			record_flag = True
			self.un_record()
		else:
			record_flag = False
		# print('Record:',record_flag)
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)

		if record_flag:
			# print(self._merge_bn)
			# do record
			if self._merge_bn:
				miu = self.bn.running_mean
				var = self.bn.running_var
				gamma = self.bn.weight
				beta = self.bn.bias 
				eps = self.bn.eps 
				if gamma is None:
					gamma = 1 
				if beta is None:
					beta = 0
				weight = self.conv.weight
				bias = self.conv.bias

				if bias is not None:
					b = gamma * (bias - miu) / torch.sqrt(eps + var) + beta 
				else:
					b = beta - (gamma * miu / torch.sqrt(eps + var))
				bn_w = gamma / torch.sqrt(eps + var)
				bn_w = bn_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
				w = bn_w * weight

				res = {}
				base_name = list(self.named_parameters())[0][0]
				base_name = '.'.join(base_name.split('.')[:-1])
				res[base_name+'.weight'] = w 
				res[base_name+'.bias'] = b

				if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
					actw = self.act.weight
					res['act.weight'] = actw
				# print(res.keys())
			else:
				res = {}
				for p in self.named_parameters():
					res[p[0]] = p[1]
				for p in self.named_buffers():
					res[p[0]] = p[1]
			L.record_params.append(res)
		return x 


class DWConvLayer(Model):
	def initialize(self, size, multiplier, stride=1, pad='SAME_LEFT', dilation_rate=1, activation=-1, batch_norm=False, affine=True, usebias=True):
		self.conv = L.dwconv2D(size, multiplier, stride, pad, dilation_rate, usebias)
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
		self.batch_norm = batch_norm
		self.activation = activation
		self.multiplier = multiplier
	def build(self, *inputs):
		inp = inputs[0]
		inchannel = inp.shape[1]
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=inchannel*self.multiplier)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)
		else:
			self.act = L.Activation(self.activation)
	def forward(self, x):
		if self._record:
			record_flag = True
			self.un_record()
		else:
			record_flag = False
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)


		if record_flag:
			# print(self._merge_bn)
			# do record
			if self._merge_bn:
				miu = self.bn.running_mean
				var = self.bn.running_var
				gamma = self.bn.weight
				beta = self.bn.bias 
				eps = self.bn.eps 
				if gamma is None:
					gamma = 1 
				if beta is None:
					beta = 0
				weight = self.conv.weight
				bias = self.conv.bias

				if bias is not None:
					b = gamma * (bias - miu) / torch.sqrt(eps + var) + beta 
				else:
					b = beta - (gamma * miu / torch.sqrt(eps + var))
				bn_w = gamma / torch.sqrt(eps + var)
				bn_w = bn_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
				w = bn_w * weight

				res = {}
				base_name = list(self.named_parameters())[0][0]
				base_name = '.'.join(base_name.split('.')[:-1])
				res[base_name+'.weight'] = w 
				res[base_name+'.bias'] = b

				if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
					actw = self.act.weight
					res['act.weight'] = actw
				# print(res.keys())
			else:
				res = {}
				for p in self.named_parameters():
					res[p[0]] = p[1]
				for p in self.named_buffers():
					res[p[0]] = p[1]
			L.record_params.append(res)
		return x 
	def to_torch(self):
		conv = self.conv.to_torch()
		setattr(self, 'conv', conv)
		if self.batch_norm:
			bn = self.bn.to_torch()
			setattr(self, 'bn', bn)
		if self.activation==PARAM_RELU:
			relu = nn.ReLU()
			setattr(self, 'act', relu)

class ConvLayer1D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, activation=-1, batch_norm=False, affine=True, usebias=True, groups=1):
		self.conv = L.conv1D(size, outchn, stride, pad, dilation_rate, usebias, groups)
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
		self.batch_norm = batch_norm
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=outchn)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)
	def forward(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
		return x 

class ConvLayer3D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, activation=-1, batch_norm=False, affine=True, usebias=True, groups=1):
		self.conv = L.conv3D(size, outchn, stride, pad, dilation_rate, usebias, groups)
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
		self.batch_norm = batch_norm
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=outchn)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)
	def forward(self, x):
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
		return x 
		
class Dense(Model):
	def initialize(self, outsize, batch_norm=False, affine=True, activation=-1 , usebias=True, norm=False):
		self.fc = L.fclayer(outsize, usebias, norm)
		self.batch_norm = batch_norm
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=outsize)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)
		if batch_norm:
			self.bn = L.BatchNorm(affine=affine)
	def forward(self, x):
		if self._record:
			record_flag = True
			self.un_record()
		else:
			record_flag = False
		x = self.fc(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
		
		if record_flag:
			# print(self._merge_bn)
			# do record
			if hasattr(self, 'bn') and self._merge_bn:
				miu = self.bn.running_mean
				var = self.bn.running_var
				gamma = self.bn.weight
				beta = self.bn.bias 
				eps = self.bn.eps 
				if gamma is None:
					gamma = 1 
				if beta is None:
					beta = 0
				weight = self.fc.weight
				bias = self.fc.bias

				if bias is not None:
					b = gamma * (bias - miu) / torch.sqrt(eps + var) + beta 
				else:
					b = beta - (gamma * miu / torch.sqrt(eps + var))
				bn_w = gamma / torch.sqrt(eps + var)
				bn_w = bn_w.unsqueeze(-1)
				w = bn_w * weight

				res = {}
				base_name = list(self.named_parameters())[0][0]
				base_name = '.'.join(base_name.split('.')[:-1])
				res[base_name+'.weight'] = w 
				res[base_name+'.bias'] = b
				# print(res.keys())

			else:
				res = {}
				for p in self.named_parameters():
					res[p[0]] = p[1]
				for p in self.named_buffers():
					res[p[0]] = p[1]
			L.record_params.append(res)

		return x 
	def to_torch(self):
		fc = self.fc.to_torch()
		setattr(self, 'fc', fc)
		if self.batch_norm:
			bn = self.bn.to_torch()
			setattr(self, 'bn', bn)
		if self.activation==PARAM_RELU:
			relu = nn.ReLU()
			setattr(self, 'act', relu)

class LSTMCell(Model):
	def initialize(self, outdim):
		self.F = L.fcLayer(outdim, usebias=False, norm=False)
		self.O = L.fcLayer(outdim, usebias=False, norm=False)
		self.I = L.fcLayer(outdim, usebias=False, norm=False)
		self.C = L.fcLayer(outdim, usebias=False, norm=False)

		self.hF = L.fcLayer(outdim, usebias=False, norm=False)
		self.hO = L.fcLayer(outdim, usebias=False, norm=False)
		self.hI = L.fcLayer(outdim, usebias=False, norm=False)
		self.hC = L.fcLayer(outdim, usebias=False, norm=False)

	def forward(self, x, h, c_prev):
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

class ConvLSTM(Model):
	def initialize(self, chn):
		self.gx = L.conv2D(3, chn)
		self.gh = L.conv2D(3, chn)
		self.fx = L.conv2D(3, chn)
		self.fh = L.conv2D(3, chn)
		self.ox = L.conv2D(3, chn)
		self.oh = L.conv2D(3, chn)
		self.gx = L.conv2D(3, chn)
		self.gh = L.conv2D(3, chn)

	def forward(self, x, c, h):
		gx = self.gx(x)
		gh = self.gh(h)

		ox = self.ox(x)
		oh = self.oh(h)

		fx = self.fx(x)
		fh = self.fh(h)

		gx = self.gx(x)
		gh = self.gh(h)

		g = torch.tanh(gx + gh)
		o = torch.sigmoid(ox + oh)
		i = torch.sigmoid(ix + ih)
		f = torch.sigmoid(fx + fh)

		cell = f*c + i*g 
		h = o * torch.tanh(cell)
		return cell, h 

class GraphConvLayer(Model):
	def initialize(self, outsize, usebias=True, norm=True, activation=-1, batch_norm=False):
		self.GCL = L.graphConvLayer(outsize, usebias=usebias, norm=norm)
		self.batch_norm = batch_norm
		self.activation = activation
		if batch_norm:
			self.bn = L.BatchNorm()
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=outsize)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)

	def forward(self, x, adj, affinity_grad=True):
		x = self.GCL(x, adj, affinity_grad)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
		return x 

class AdaptConv3(Model):
	def initialize(self, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, batch_norm=False, activation=-1, usebias=True):
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
		self.activation = activation
		if self.activation == PARAM_PRELU:
			self.act = torch.nn.PReLU(num_parameters=outchn)
		elif self.activation==PARAM_PRELU1:
			self.act = torch.nn.PReLU(num_parameters=1)
		else:
			self.act = L.Activation(activation)
		
		self.usebias = usebias
		self.outchn = outchn

	def forward(self, x):
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
