import functools
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from torch.nn.parameter import Parameter
import math 
import numpy as np 
import torchvision.ops as ops

record_params = []

def _resnet_normal(tensor):
	fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
	std = math.sqrt(2.0 / float(fan_out))
	return init._no_grad_normal_(tensor, 0., std)

class Model(nn.Module):
	def __init__(self, *args, **kwargs):
		super(Model, self).__init__()
		self._record = False
		self._merge_bn = False
		self._quant_calibrating = False
		self._quant_calibrated = False
		self._quant = False
		self.is_built = False
		self._model_flags = {}
		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		pass 

	def build(self, *inputs, **kwargs):
		if self._quant:
			self.start_quant()
		for k in self._model_flags:
			self.set_flag(k, self._model_flags[k])

	def build_forward(self, *inputs, **kwargs):
		if self._quant:
			self.start_quant()
		for k in self._model_flags:
			self.set_flag(k, self._model_flags[k])
		return self.forward(*inputs, **kwargs)

	def __call__(self, *input, **kwargs):
		if not self.is_built:
			self.build(*input)
		for hook in self._forward_pre_hooks.values():
			result = hook(self, input)
			if result is not None:
				if not isinstance(result, tuple):
					result = (result,)
				input = result
		if torch._C._get_tracing_state():
			result = self._slow_forward(*input, **kwargs)
		else:
			if not self.is_built:
				result = self.build_forward(*input, **kwargs)
			else:
				result = self.forward(*input, **kwargs)
		for hook in self._forward_hooks.values():
			hook_result = hook(self, input, result)
			if hook_result is not None:
				result = hook_result
		if len(self._backward_hooks) > 0:
			var = result
			while not isinstance(var, torch.Tensor):
				if isinstance(var, dict):
					var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
				else:
					var = var[0]
			grad_fn = var.grad_fn
			if grad_fn is not None:
				for hook in self._backward_hooks.values():
					wrapper = functools.partial(hook, self)
					functools.update_wrapper(wrapper, hook)
					grad_fn.register_hook(wrapper)
		self.is_built = True
		return result

	def record(self):
		def set_record_flag(obj):
			obj._record = True
		self.apply(set_record_flag)

	def un_record(self):
		def unset_record_flag(obj):
			obj._record = False
		self.apply(unset_record_flag)

	def merge_bn(self):
		def set_merge_bn(obj):
			obj._merge_bn = True 
		self.apply(set_merge_bn)

	def set_flag(self, k, v):
		def set_model_flag(obj):
			if hasattr(obj, '_model_flags'):
				obj._model_flags[k] = v 
		self.apply(set_model_flag)

	def get_flag(self, k):
		return self._model_flags.get(k, None)

	def bn_eps(self, value):
		def set_eps(obj):
			obj.eps = value
		self.apply(set_eps)

	def start_calibrate(self):
		def set_calibarte(obj):
			if obj._quant:
				obj._quant_calibrating = True
		self.apply(set_calibarte)

	def end_calibrate(self):
		def unset_calibrate(obj):
			if obj._quant:
				obj._quant_calibrating = False
				if hasattr(obj, '_finish_calibrate'):
					obj._finish_calibrate()
				obj._quant_calibrated = True
		self.apply(unset_calibrate)

	def start_quant(self):
		def set_quant(obj):
			obj._quant = True
		self.apply(set_quant)

	def end_quant(self):
		def unset_quant(obj):
			obj._quant = False
		self.apply(unset_quant)

##### In this version, quant objects are stored here. This is not a good programming behavior. 
##### The code will be re-organized in later version 

class QUint8():
	max_val = 2 ** 8 -1 
	min_val = 0 
	signed = False

class QInt8():
	max_val = 2 ** 7 - 1 
	min_val = - 2 ** 7 
	signed = True 

QTYPES = {"uint8": QUint8, "int8": QInt8}

class PercentileObserver(Model):
	def initialize(self, bit_type, zero_offset, mode='layer_wise', is_weight=False):
		assert mode in ['layer_wise', 'channel_wise']
		self.min_val = None 
		self.max_val = None 
		self.mode = mode 
		self.bit_type = bit_type
		self.zero_offset = zero_offset
		self.is_weight = is_weight
		self.scale = None 
		self.zero_point = None 
		self.sigma = 0.01 
		self.percentile = 0.999

	def build(self, x):
		if len(x.shape)==4:
			# b,c,h,w
			self.dim = -3
			if self.is_weight:
				self.dim = -4
		else:
			self.dim = -1

	def observe(self, x):
		if self.mode == 'channel_wise':
			x = x.transpose(0, self.dim)
			x = x.flatten(1)
			minv = torch.quantile(x.cuda(), 1-self.percentile, dim=1).cpu()
			maxv = torch.quantile(x.cuda(), self.percentile, dim=1).cpu()
		else:
			minv = torch.quantile(x.cuda(), 1-self.percentile).cpu()
			maxv = torch.quantile(x.cuda(), self.percentile).cpu()
		if self.min_val is None:
			self.min_val = minv
			self.max_val = maxv
		else:
			self.min_val = self.sigma * minv + (1 - self.sigma) * self.min_val
			self.max_val = self.sigma * maxv + (1 - self.sigma) * self.max_val

	def get_quant_params(self):
		if self.zero_offset:
			max_val = torch.max(self.max_val, -self.min_val)
			scale = max_val / min(self.bit_type.max_val, -self.bit_type.min_val)
			zero_point = torch.zeros_like(max_val)
		else:
			scale = (self.max_val - self.min_val) / float(self.bit_type.max_val - self.bit_type.min_val)
			zero_point = self.bit_type.min_val - torch.round(self.min_val / scale)
		scale.clamp(torch.finfo(torch.float32).eps)
		return scale, zero_point

	def _finish_calibrate(self):
		# print('Recording calibrated values...')
		if self.scale is None:
			self.scale, self.zero_point = self.get_quant_params()

	def forward(self, x):
		if self._quant_calibrating:
			self.observe(x)
		return x 

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		if prefix+'scale' in state_dict:
			print('Quant params loaded.')
			self.scale = state_dict[prefix + 'scale']
			self.zero_point = state_dict[prefix + 'zero_point']

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self._quant_calibrated:
			destination[prefix + 'scale'] = self.scale 
			destination[prefix + 'zero_point'] = self.zero_point
			print('Quant param saved.')


class MinMaxObserver(Model):
	def initialize(self, bit_type, zero_offset, mode='layer_wise', is_weight=False):
		assert mode in ['layer_wise', 'channel_wise']
		self.min_val = None 
		self.max_val = None 
		self.mode = mode 
		self.bit_type = bit_type
		self.zero_offset = zero_offset
		self.is_weight = is_weight
		self.scale = None 
		self.zero_point = None 

	def build(self, x):
		if len(x.shape)==4:
			# b,c,h,w
			self.dim = -3
			if self.is_weight:
				self.dim = -4
		else:
			self.dim = -1

	def observe(self, x):
		if self.mode == 'channel_wise':
			x = x.transpose(0, self.dim)
			x = x.flatten(1)
			minv = x.min(dim=1)[0]
			maxv = x.max(dim=1)[0]
		else:
			minv = x.min()
			maxv = x.max()
		if self.min_val is None:
			self.min_val = minv
			self.max_val = maxv
		else:
			self.min_val = torch.minimum(self.min_val, minv)
			self.max_val = torch.maximum(self.max_val, maxv)

	def get_quant_params(self):
		if self.zero_offset:
			max_val = torch.max(self.max_val, -self.min_val)
			scale = max_val / min(self.bit_type.max_val, -self.bit_type.min_val)
			zero_point = torch.zeros_like(max_val)
		else:
			scale = (self.max_val - self.min_val) / float(self.bit_type.max_val - self.bit_type.min_val)
			zero_point = self.bit_type.min_val - torch.round(self.min_val / scale)
		scale.clamp(torch.finfo(torch.float32).eps)
		return scale, zero_point

	def _finish_calibrate(self):
		# print('Recording calibrated values...')
		if self.scale is None:
			self.scale, self.zero_point = self.get_quant_params()

	def forward(self, x):
		if self._quant_calibrating:
			self.observe(x)
		return x 

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		if prefix+'scale' in state_dict:
			print('Quant params loaded.')
			self.scale = state_dict[prefix + 'scale']
			self.zero_point = state_dict[prefix + 'zero_point']

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self._quant_calibrated:
			destination[prefix + 'scale'] = self.scale 
			destination[prefix + 'zero_point'] = self.zero_point
			print('Quant param saved.')


QObservers = {"minmax": MinMaxObserver, "percentile": PercentileObserver}

class UniformQuantizer(Model):
	def initialize(self, bit_type='int8', observer='minmax', zero_offset=False, mode='layer_wise', is_weight=False):
		self.bit_type = QTYPES[bit_type]
		self.observer = QObservers[observer](self.bit_type, zero_offset, mode, is_weight)

	def build(self, x):
		if len(x.shape)==4:
			self.dim = -3
		else:
			self.dim = -1 

	def quant(self, x):
		if self.dim!=-1 and mode=='channel_wise':
			# make everything run at last dim, no need manually reshape 
			x = x.transpose(-1, self.dim)
		x = x / self.observer.scale + self.observer.zero_point
		x = x.round().clamp(self.bit_type.min_val, self.bit_type.max_val)
		if self.dim!=-1 and mode=='channel_wise':
			x = x.transpose(-1, self.dim)
		return x 

	def dequant(self, x):
		if self.dim!=-1 and mode=='channel_wise':
			x = x.transpose(-1, self.dim)
		x = (x - self.observer.zero_point) * self.observer.scale 
		if self.dim!=-1 and mode=='channel_wise':
			x = x.transpose(-1, self.dim)
		return x 

	def forward(self, x):
		if self._quant_calibrating:
			x = self.observer(x)
		if self._quant and self._quant_calibrated:
			x = self.quant(x)
			x = self.dequant(x)
		return x 


QQuantizers = {"uniform": UniformQuantizer}

class QAct(Model):
	def initialize(self, zero_offset=False, mode='layer_wise', observer='minmax'):
		self.mode = mode 
		self.zero_offset = zero_offset
		self.observer_str = observer
	def build(self, x):
		if self._quant:
			self.quantizer = QQuantizers['uniform'](zero_offset=self.zero_offset, mode=self.mode, observer=self.observer_str)
	def forward(self, x):
		if self._quant:
			x = self.quantizer(x)
		return x 

######  Layers 

class conv2D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, usebias=True, gropus=1):
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.gropus = gropus
		self.dilation_rate = dilation_rate
		assert (pad in ['VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		self.inchannel = inchannel
		# print('INC', inchannel)
		# parse args
		if isinstance(self.size,list):
			# self.size = [self.size[0],self.size[1],inchannel,self.outchn]
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size-1 ))//2)
			self.size = [self.outchn, inchannel // self.gropus, self.size[0], self.size[1]]
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
			self.size = [self.outchn, inchannel // self.gropus, self.size, self.size]

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		# self.inchannel = inp
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

		if self._quant:
			self.input_quantizer = QQuantizers['uniform'](zero_offset=False)
			self.w_quantizer = QQuantizers['uniform'](zero_offset=True, mode='channel_wise', is_weight=True)

	def reset_params(self):
		_resnet_normal(self.weight)
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		weight = self.weight
		if self._quant:
			x = self.input_quantizer(x)
			weight = self.w_quantizer(weight)
		return F.conv2d(x, weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)

	def to_torch(self):
		# print('inchannel',self.inchannel)
		conv = nn.Conv2d(in_channels = self.inchannel, out_channels = self.outchn, kernel_size = tuple(self.size[2:]), stride = self.stride,\
						padding = self.pad, padding_mode = 'zeros', dilation = self.dilation_rate, groups = self.gropus, bias = self.usebias)
		conv.weight.data[:] = self.weight.data[:]
		if self.usebias:
			conv.bias.data[:] = self.bias.data[:]
		return conv 

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		if prefix+'weight' in state_dict:
			super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
		else:
			# print(self.get_flag('fc2conv'), strict)
			if self.get_flag('fc2conv') or (not strict):
				# print('Warning: No weight found for layer:', prefix)
				pass
			else:
				raise Exception('No weight found for layer:', prefix)


class deconv2D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, usebias=True, gropus=1):
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.gropus = gropus
		self.dilation_rate = dilation_rate
		assert (pad in ['VALID','SAME_LEFT'])
		self.pad = pad 
		self.padmethod = pad

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		# parse args
		if isinstance(self.size,int):
			if self.pad == 'VALID':
				self.pad = 0
				self.out_pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2 - (1 - self.size%2)
				# self.pad = self.dilation_rate * (self.size - 1 ) 
				self.out_pad = self.stride - 1
			self.size = [inchannel, self.outchn // self.gropus, self.size, self.size]
		else:
			raise Exception("Deconv kernel only supports int")

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		_resnet_normal(self.weight)
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		inh, inw = x.shape[2], x.shape[3]
		x = F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.pad, self.out_pad, self.gropus, self.dilation_rate)
		outh, outw = x.shape[2], x.shape[3]
		if self.padmethod=='SAME_LEFT':
			if outh!=inh*self.stride or outw!=inw*self.stride:
				x = x[:,:,:inh*self.stride,:inw*self.stride]
		return x 

class dwconv2D(Model):
	# depth-wise conv2d
	def initialize(self, size, multiplier, stride=1, pad='SAME_LEFT', dilation_rate=1, usebias=True):
		self.size = size
		self.multiplier = multiplier
		self.stride = stride
		self.usebias = usebias
		self.dilation_rate = dilation_rate
		assert (pad in ['VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		self.gropus = inchannel
		self.inchannel = inchannel
		self.outchn = self.multiplier * inchannel
		# parse args
		if isinstance(self.size,list):
			# self.size = [self.size[0],self.size[1],inchannel,self.outchn]
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size-1 ))//2)
			self.size = [self.multiplier * inchannel, 1, self.size[0], self.size[1]]
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
			self.size = [self.multiplier * inchannel, 1, self.size, self.size]

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.size[0]))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		_resnet_normal(self.weight)
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		return F.conv2d(x, self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)

	def to_torch(self):
		conv = nn.Conv2d(in_channels = self.inchannel, out_channels = self.outchn, kernel_size = tuple(self.size[2:]), stride = self.stride,\
						padding = self.pad, padding_mode = 'zeros', dilation = self.dilation_rate, groups = self.gropus, bias = self.usebias)
		conv.weight.data[:] = self.weight.data[:]
		if self.usebias:
			conv.bias.data[:] = self.bias.data[:]
		return conv 

class conv1D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, usebias=True, gropus=1):
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.gropus = gropus
		self.dilation_rate = dilation_rate
		assert (pad in ['VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		# parse args
		if self.pad == 'VALID':
			self.pad = 0
		else:
			self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
		self.size = [self.outchn, inchannel // self.gropus, self.size]

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		return F.conv1d(x, self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)

class conv3D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, usebias=True, gropus=1):
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.gropus = gropus
		self.dilation_rate = dilation_rate
		assert (pad in ['VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		# parse args
		if isinstance(self.size,list) or isinstance(self.size, tuple):
			# self.size = [self.size[0],self.size[1],inchannel,self.outchn]
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size-1 ))//2, (self.size[2]+ (self.dilation_rate-1) * ( self.size-1 ))//2)
			self.size = [self.outchn, inchannel // self.gropus, self.size[0], self.size[1], self.size[2]]
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
			self.size = [self.outchn, inchannel // self.gropus, self.size, self.size, self.size]

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		return F.conv3d(x, self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)

class fclayer(Model):
	def initialize(self, outsize, usebias=True, norm=False):
		self.outsize = outsize
		self.usebias = usebias
		self.norm = norm

	def build(self, *inputs):
		# print('building...')
		self.insize = inputs[0].shape[-1]
		self.weight = Parameter(torch.Tensor(self.outsize, self.insize))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outsize))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		# init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		init.normal_(self.weight, std=0.001)
		# _resnet_normal(self.weight)
		# print('Reset fc params...')
		if self.bias is not None:
		# 	fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
		# 	bound = 1 / math.sqrt(fan_in)
		# 	init.uniform_(self.bias, -bound, bound)
			init.zeros_(self.bias)

	def forward(self, x):
		if self.norm:
			with torch.no_grad():
				norm = x.norm(p=2, dim=1, keepdim=True)
				wnorm = self.weight.norm(p=2,dim=1, keepdim=True)
			x = x / norm
			weight = self.weight / wnorm
		else:
			weight = self.weight
		return F.linear(x, weight, self.bias)

	def to_torch(self):
		fc = nn.Linear(self.insize, self.outsize, self.usebias)
		fc.weight.data[:] = self.weight.data[:]
		if self.usebias:
			fc.bias.data[:] = self.bias.data[:]
		return fc 

def flatten(x):
	x = x.reshape(x.size(0), -1)
	return x 

class Flatten(Model):
	def forward(self, x):
		return flatten(x)

class MaxPool2d(Model):
	def initialize(self, size, stride=1, pad='SAME_LEFT', dilation_rate=1):
		self.size = size
		self.stride = stride
		self.pad = pad
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		# parse args
		if isinstance(self.size,list) or isinstance(self.size, tuple):
			# self.size = [self.size[0],self.size[1],inchannel,self.outchn]
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size[0]//2, self.size[1]//2, self.size[2]//2)
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = self.size//2

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)

	def forward(self, x):
		return F.max_pool2d(x, self.size, self.stride, self.pad, self.dilation_rate, False, False)

class AvgPool2d(Model):
	def initialize(self, size, stride=1, pad='SAME_LEFT'):
		self.size = size
		self.stride = stride
		self.pad = pad

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		# parse args
		if isinstance(self.size,list) or isinstance(self.size, tuple):
			# self.size = [self.size[0],self.size[1],inchannel,self.outchn]
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size[0]//2, self.size[1]//2, self.size[2]//2)
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = self.size//2

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)

	def forward(self, x):
		return F.avg_pool2d(x, self.size, self.stride, self.pad, False, True)

class BatchNorm(Model):
	# _version = 2
	# __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
	# 				 'running_mean', 'running_var', 'num_batches_tracked',
	# 				 'num_features', 'affine', 'weight', 'bias']

	def initialize(self, eps=2e-5, momentum=0.01, affine=True,
				 track_running_stats=True):
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		
	def build(self, *inputs):
		# print('building...')
		num_features = inputs[0].shape[1]
		self.num_features = num_features
		if self.affine:
			self.weight = Parameter(torch.Tensor(num_features))
			self.bias = Parameter(torch.Tensor(num_features))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		if self.track_running_stats:
			self.register_buffer('running_mean', torch.zeros(num_features))
			self.register_buffer('running_var', torch.ones(num_features))
			self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
		else:
			self.register_parameter('running_mean', None)
			self.register_parameter('running_var', None)
			self.register_parameter('num_batches_tracked', None)
		self.reset_parameters()

	def reset_running_stats(self):
		if self.track_running_stats:
			self.running_mean.zero_()
			self.running_var.fill_(1)
			self.num_batches_tracked.zero_()

	def reset_parameters(self):
		self.reset_running_stats()
		if self.affine:
			init.ones_(self.weight)
			init.zeros_(self.bias)

	def forward(self, input):
		global record_params
		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum

		if self.training and self.track_running_stats:
			# TODO: if statement only here to tell the jit to skip emitting this when it is None
			if self.num_batches_tracked is not None:
				self.num_batches_tracked += 1
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		result =  F.batch_norm(
			input, self.running_mean, self.running_var, self.weight, self.bias,
			self.training or not self.track_running_stats,
			exponential_average_factor, self.eps)
		if hasattr(self, '_record'):
			if self._record:
				res = {}
				for p in self.named_parameters():
					res[p[0]] = p[1]
				for p in self.named_buffers():
					res[p[0]] = p[1]
				record_params.append(res)
			self.un_record()
		return result

	def to_torch(self):
		bn = nn.BatchNorm2d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)
		if self.affine:
			bn.weight.data[:] = self.weight.data[:]
			bn.bias.data[:] = self.bias.data[:]
		if self.track_running_stats:
			bn.running_mean.data[:] = self.running_mean.data[:]
			bn.running_var.data[:] = self.running_var.data[:]
		return bn 

	# def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
	# 						  missing_keys, unexpected_keys, error_msgs):
	# 	version = local_metadata.get('version', None)

	# 	if (version is None or version < 2) and self.track_running_stats:
	# 		# at version 2: added num_batches_tracked buffer
	# 		#			   this should have a default value of 0
	# 		num_batches_tracked_key = prefix + 'num_batches_tracked'
	# 		if num_batches_tracked_key not in state_dict:
	# 			state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

	# 	super(_BatchNorm, self)._load_from_state_dict(
	# 		state_dict, prefix, local_metadata, strict,
	# 		missing_keys, unexpected_keys, error_msgs)

class LayerNorm(Model):
	def initialize(self, n_dims_to_keep, affine=True, eps=1e-5):
		self.n_dims_to_keep = n_dims_to_keep
		self.affine = affine
		self.eps = eps 

	def build(self, *inputs):
		shape = inputs[0].shape
		n_dims = len(shape)
		self.dim_to_reduce = tuple(range(n_dims - self.n_dims_to_keep, n_dims))
		if self.affine:
			self.weight = Parameter(torch.Tensor(*shape[-self.n_dims_to_keep:]))
			self.bias = Parameter(torch.Tensor(*shape[-self.n_dims_to_keep:]))
		self.reset_params()

	def reset_params(self):
		if self.affine:
			init.ones_(self.weight)
			init.zeros_(self.bias)

	def forward(self, x):
		mean = torch.mean(x, dim=self.dim_to_reduce, keepdim=True)
		std = torch.std(x, dim=self.dim_to_reduce, keepdim=True)
		if self.affine:
			return self.weight * (x - mean) / (std + self.eps) + self.bias
		else:
			return (x - mean) / (std + self.eps)

def GlobalAvgPool2D(x):
	x = x.mean(dim=(2,3), keepdim=True)
	return x 

class GlobalAvgPool2DLayer(Model):
	def forward(self, x):
		return GlobalAvgPool2D(x)

class NNUpSample(Model):
	def initialize(self, scale):
		self.scale = scale

	def _parse_args(self, input_shape):
		self.inchannel = input_shape[1]
		self.size = [self.inchannel, 1, self.scale, self.scale]

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size), requires_grad=False)
		self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		init.ones_(self.weight)

	def forward(self, x):
		# w = self.weight.detach()
		w = self.weight
		return F.conv_transpose2d(x, w, self.bias, self.scale, 0, 0, self.inchannel, 1)

def activation(x, act, **kwargs):
	if act==-1:
		return x
	elif act==0:
		return F.relu(x)
	elif act==1:
		return F.leaky_relu(x, negative_slope=0.2)
	elif act==2:
		return F.elu(x)
	elif act==3:
		return F.tanh(x)
	elif act==6:
		return torch.sigmoid(x)
	elif act==10:
		return F.gelu(x)

class Activation(Model):
	def initialize(self, act):
		self.act = act 
		if act==8:
			self.act = torch.nn.PReLU(num_parameters=outchn)  # this line is buggy
		elif act==9:
			self.act = torch.nn.PReLU(num_parameters=1)
	def forward(self, x):
		if self.act==8 or self.act==9:
			return self.act(x)
		else:
			return activation(x, self.act)

class graphConvLayer(Model):
	def initialize(self, outsize, usebias=True, norm=True):
		self.outsize = outsize
		self.usebias = usebias
		self.norm = norm 

	def _parse_args(self, input_shape):
		# set size
		insize = input_shape[-1]
		self.size = [self.outsize, insize]

	def build(self, *inputs):
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outsize))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def _normalize_adj_mtx(self, mtx):
		with torch.no_grad():
			S = torch.sum(mtx, dim=1)
			S = torch.sqrt(S)
			S = 1. / S
			S = torch.diag(S)
			I = torch.eye(S.shape[0])
			A_ = (mtx + I) 
			A_ = torch.mm(S, A_)
			A_ = torch.mm(A_, S)
		return A_

	def forward(self, x, adj, affinity_grad=True):
		if self.norm:
			adj = self._normalize_adj_mtx(adj)
		if not affinity_grad:
			adj = adj.detach()
		res = torch.mm(adj, x)
		res = F.linear(res, self.weight, self.bias)
		return res 
	
class BilinearUpSample(Model):
	def initialize(self, factor):
		self.factor = factor
		self.pad0 = factor//2 * 3 + factor%2
	def build(self, *inputs):
		inp = inputs[0]
		self.inchn = inp.shape[1]
		filter_size = 2*self.factor - self.factor%2
		k = self.upsample_kernel(filter_size)
		k = k[None, ...]
		k = k[None, ...]
		k = np.repeat(k, self.inchn, axis=0)
		self.weight = Parameter(torch.from_numpy(k), requires_grad=False)
	def upsample_kernel(self,size):
		factor = (size +1)//2
		if size%2==1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size, :size]
		k = (1 - abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)
		return np.float32(k)
	def forward(self, x):
		x = F.pad(x, (1,1,1,1), 'replicate')
		x = F.conv_transpose2d(x, self.weight, None, self.factor, self.pad0, 0, self.inchn, 1)
		return x 
	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		pass
	def _save_to_state_dict(self, destination, prefix, keep_vars):
		pass

class DeformConv2D(Model):
	def initialize(self, size, outchn, stride=1, pad='SAME_LEFT', dilation_rate=1, usebias=True):
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.dilation_rate = dilation_rate
		assert (pad in ['VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[1]
		self.inchannel = inchannel
		# print('INC', inchannel)
		# parse args
		if isinstance(self.size,list):
			# self.size = [self.size[0],self.size[1],inchannel,self.outchn]
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size-1 ))//2)
			self.size = [self.outchn, inchannel, self.size[0], self.size[1]]
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
			self.size = [self.outchn, inchannel, self.size, self.size]

	def build(self, *inputs):
		# print('building...')
		inp = inputs[0]
		# self.inchannel = inp
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		_resnet_normal(self.weight)
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x, offset):
		return ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.pad, self.dilation_rate)

