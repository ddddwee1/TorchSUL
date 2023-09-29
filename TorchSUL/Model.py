from . import Layers as L 
from . import Quant as Qnt 
from . import Base 

import torch 
import torch.nn as nn 

import os 
import copy
import inspect
from distutils.version import LooseVersion
from loguru import logger

Model = Base.Model
activation = L.activation
Activation = L.Activation
BatchNorm = L.BatchNorm
LayerNorm = L.LayerNorm
MaxPool2D = L.MaxPool2d
AvgPool2D = L.AvgPool2d
NNUpSample = L.NNUpSample
BilinearUpSample = L.BilinearUpSample
DeformConv2D = L.DeformConv2D
QAct = Qnt.QAct
QQuantizers = Qnt.QQuantizers
quant = Qnt   # alias for quantization module 

# activation const
# some values are no longer supported 
PARAM_RELU = 0
PARAM_LRELU = 1
PARAM_ELU = 2
PARAM_TANH = 3
PARAM_MFM = 4
PARAM_SIGMOID = 6
PARAM_PRELU = 8
PARAM_PRELU1 = 9
PARAM_GELU = 10


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

def inspect_quant_params(module, result_dict=dict(), prefix=''):
	if isinstance(module, QAct):
		try:
			zero_point = module.quantizer.observer.zero_point
			scale = module.quantizer.observer.scale 
			result_dict[prefix] = [scale, zero_point.round()]
		except:
			logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
		return result_dict
	if isinstance(module, ConvLayer):
		try:
			if hasattr(module.conv, 'input_quantizer'):
				scale = module.conv.input_quantizer.observer.scale
				zero_point = module.conv.input_quantizer.observer.zero_point
				result_dict[prefix+'/conv/Conv__input'] = [scale, zero_point.round()]
				scale = module.conv.w_quantizer.observer.scale
				zero_point = module.conv.w_quantizer.observer.zero_point
				result_dict[prefix+'/conv/Conv__weight'] = [scale, zero_point.round()]
		except:
			logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
		return result_dict
	if isinstance(module, DeConvLayer):
		try:
			if hasattr(module.conv, 'input_quantizer'):
				scale = module.conv.input_quantizer.observer.scale
				zero_point = module.conv.input_quantizer.observer.zero_point
				result_dict[prefix+'/conv/DeConv__input'] = [scale, zero_point.round()]
				scale = module.conv.w_quantizer.observer.scale
				zero_point = module.conv.w_quantizer.observer.zero_point
				result_dict[prefix+'/conv/DeConv__weight'] = [scale, zero_point.round()]
		except:
			logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
		return result_dict
	if isinstance(module, Dense):
		try:
			if hasattr(module.fc, 'input_quantizer'):
				scale = module.fc.input_quantizer.observer.scale
				zero_point = module.fc.input_quantizer.observer.zero_point
				result_dict[prefix+'/fc/Dense__input'] = [scale, zero_point.round()]
				scale = module.fc.w_quantizer.observer.scale
				zero_point = module.fc.w_quantizer.observer.zero_point
				result_dict[prefix+'/fc/Dense__weight'] = [scale, zero_point.round()]
		except:
			logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
		return result_dict
	if isinstance(module, (nn.ModuleList, nn.Sequential)):
		for i in range(len(module)):
			inspect_quant_params(module[i], result_dict=result_dict, prefix=prefix+'.%d'%i)
		return result_dict
	results = inspect.getmembers(module)
	for name, child_module in results:
		if isinstance(child_module, nn.Module):
			inspect_quant_params(child_module, result_dict=result_dict, prefix=prefix+'/'+name)
	return result_dict


class Saver():
	def __init__(self, module):
		self.model = module

	def _get_checkpoint(self, path):
		path = path.replace('\\','/')  # for windows 
		# ckpt = path + 'checkpoint'
		ckpt = os.path.join(path, 'checkpoint')
		if os.path.exists(ckpt):
			fname = open(ckpt).readline().strip()
			# return path + fname
			return os.path.join(path, fname)
		else:
			return False

	def _exclude(self, d, exclude):
		if exclude is not None:
			for e in exclude:
				if e in d:
					d.pop(e)
		return d 

	def restore(self, path, strict=True, exclude=None):
		logger.info('Trying to load from: %s'%path)
		device = torch.device('cpu')
		if path[-4:] == '.pth':
			if not os.path.exists(path):
				logger.warning('Path: %s does not exsist. No restoration will be performed.'%path)
			elif isinstance(self.model, nn.DataParallel):
				state_dict = torch.load(path, map_location=device)
				state_dict = self._exclude(state_dict, exclude)
				self.model.module.load_state_dict(state_dict, strict=strict)
				logger.info('Model loaded from: %s'%path)
			else:
				state_dict = torch.load(path, map_location=device)
				state_dict = self._exclude(state_dict, exclude)
				self.model.load_state_dict(state_dict, strict=strict)
				logger.info('Model loaded from: %s'%path)
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
				logger.info('Model loaded from: %s'%path)
			else:
				logger.warning('No checkpoint found. No restoration will be performed.')

	def save(self, path):
		# To make it compatible with older pytorch 
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
		logger.info('Model saved to: %s'%path)
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
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
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

	def _load_from_state_dict2(self, state_dict, prefix):
		def _load_weight(k):
			if not k in state_dict:
				raise Exception('Attenpt to find', k, 'but only exist', state_dict.keys(), 'Cannot find weight in checkpoint for layer:', prefix)
			return state_dict.pop(k)
		def _load_bias(k):
			if self.conv.usebias:
				try:
					b = state_dict.pop(k)
				except:
					raise Exception('Attenpt to find', k, 'but only exist', state_dict.keys(), 'Bias is set for layer', prefix, 'but not found in checkpoint. Try to set usebias=False to fix this problem')
			else:
				b = None
			return b 

		# get names for params
		if prefix+'conv.weight' in state_dict:
			# normal load 
			w = prefix + 'conv.weight'
			b = prefix + 'conv.bias'
		elif self.get_flag('fc2conv') and ((prefix+'fc.weight') in state_dict):
			w = prefix + 'fc.weight'
			b = prefix + 'fc.bias'
		elif self.get_flag('from_torch'):
			w = prefix + 'weight'
			b = prefix + 'bias'
		else:
			raise Exception('Cannot find weight in checkpoint for layer:', prefix)

		# laod weight and bias 
		w = _load_weight(w)
		b = _load_bias(b)
		if self.get_flag('fc2conv') and len(w.shape)==2:
			w = w.unsqueeze(-1).unsqueeze(-1)

		# write processed params to state dict 
		state_dict[prefix+'conv.weight'] = w 
		if b is not None:
			state_dict[prefix+'conv.bias'] = b


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
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
		return x 

	def _load_from_state_dict2(self, state_dict, prefix):
		def _load_weight(k):
			if not k in state_dict:
				raise Exception('Attenpt to find', k, 'but only exist', state_dict.keys(), 'Cannot find weight in checkpoint for layer:', prefix)
			return state_dict.pop(k)
		def _load_bias(k):
			if self.conv.usebias:
				try:
					b = state_dict.pop(k)
				except:
					raise Exception('Attenpt to find', k, 'but only exist', state_dict.keys(), 'Bias is set for layer', prefix, 'but not found in checkpoint. Try to set usebias=False to fix this problem')
			else:
				b = None
			return b 

		# get names for params
		if prefix+'conv.weight' in state_dict:
			# normal load 
			w = prefix + 'conv.weight'
			b = prefix + 'conv.bias'
		elif self.get_flag('from_torch'):
			w = prefix + 'weight'
			b = prefix + 'bias'
		else:
			raise Exception('Cannot find weight in checkpoint for layer:', prefix)

		# laod weight and bias 
		w = _load_weight(w)
		b = _load_bias(b)

		# write processed params to state dict 
		state_dict[prefix+'conv.weight'] = w 
		if b is not None:
			state_dict[prefix+'conv.bias'] = b


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
		x = self.conv(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation!=-1:
			x = self.act(x)
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
		x = self.fc(x)
		if self.batch_norm:
			x = self.bn(x)
		if self.activation==PARAM_PRELU or self.activation==PARAM_PRELU1:
			x = self.act(x)
		else:
			x = L.activation(x, self.activation)
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

	def _load_from_state_dict2(self, state_dict, prefix):
		if prefix+'fc.weight' in state_dict:
			return 
		if self.get_flag('from_torch'):
			w = state_dict.pop(prefix + 'weight')
			if self.fc.usebias:
				try:
					b = state_dict.pop(prefix + 'bias')
				except:
					raise Exception('Bias is set for layer', prefix, 'but not found in checkpoint. Try to set usebias=False to fix this problem')
			else:
				b = None

			state_dict[prefix+'fc.weight'] = w
			if b is not None: 
				state_dict[prefix+'fc.bias'] = b


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
		self.ix = L.conv2D(3, chn)
		self.ih = L.conv2D(3, chn)

	def forward(self, x, c, h):
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
