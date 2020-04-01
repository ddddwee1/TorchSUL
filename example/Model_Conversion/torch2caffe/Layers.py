import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from torch.nn.parameter import Parameter
import math 
import numpy as np 

record_params = []
params_dict = {}

def init_caffe_input(x):
	global caffe_string, layer_counter
	if not 'caffe_string' in globals():
		caffe_string = ''
	if not 'layer_counter' in globals():
		layer_counter = 0
	caffe_string += 'layer{\n'
	caffe_string += ' name: "%s"\n'%x[1]
	caffe_string += ' type: "Input"\n'
	caffe_string += ' top: "%s"\n'%x[1]
	caffe_string += ' input_param{\n  shape{\n   dim:%d\n   dim:%d\n   dim:%d\n   dim:%d\n  }\n }\n}\n'%(x[0].shape[0], x[0].shape[1], x[0].shape[2], x[0].shape[3])
	layer_counter += 1 

def compile_params_dict():
	global params_dict
	for l in params_dict.keys():
		layer = params_dict[l]
		for k in layer.keys():
			# print(l, k, layer[k])
			layer[k] = layer[k].cpu().detach().numpy()

def _resnet_normal(tensor):
	fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
	std = math.sqrt(2.0 / float(fan_out))
	return init._no_grad_normal_(tensor, 0., std)

class Model(nn.Module):
	def __init__(self, *args, **kwargs):
		super(Model, self).__init__()
		self.is_built = False
		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		pass 

	def build(self, *inputs):
		pass 

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
			obj.record = True
		self.apply(set_record_flag)

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
		inp = inputs[0][0]
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

	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'conv%d'%layer_counter

		stride = self.stride
		pad = self.pad 
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Convolution"\n'
		caffe_string += ' bottom: "%s"\n'%name
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' convolution_param{\n'
		caffe_string += '  num_output: %d\n'%self.outchn
		caffe_string += '  bias_term: %s\n'%('true' if self.usebias else 'false')
		caffe_string += '  group: 1\n'
		caffe_string += '  stride: %d\n'%stride
		caffe_string += '  pad_h: %d\n'%pad
		caffe_string += '  pad_w: %d\n'%pad
		caffe_string += '  kernel_h: %d\n'%(self.size[2])
		caffe_string += '  kernel_w: %d\n'%(self.size[3])
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['kernel'] = self.weight
		if self.usebias:
			params_dict[layer_name]['bias'] = self.bias

		layer_counter += 1 
		return layer_name

	def forward(self, x):
		res =  F.conv2d(x[0], self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)
		lname = self._write_caffe(x[1])
		return res, lname

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
		inp = inputs[0][0]
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

	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'conv%d'%layer_counter

		stride = self.stride
		pad = self.pad
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Convolution"\n'
		caffe_string += ' bottom: "%s"\n'%name
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' convolution_param{\n'
		caffe_string += '  num_output: %d\n'%(self.multiplier * self.gropus)
		caffe_string += '  bias_term: %s\n'%('true' if self.usebias else 'false')
		caffe_string += '  group: %d\n'%self.gropus
		caffe_string += '  stride: %d\n'%stride
		caffe_string += '  pad_h: %d\n'%pad
		caffe_string += '  pad_w: %d\n'%pad
		caffe_string += '  kernel_h: %d\n'%(self.size[2])
		caffe_string += '  kernel_w: %d\n'%(self.size[3])
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['dwkernel'] = self.weight
		if self.usebias:
			params_dict[layer_name]['bias'] = self.bias

		layer_counter += 1 
		return layer_name

	def forward(self, x):
		res = F.conv2d(x[0], self.weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)
		lname = self._write_caffe(x[1])
		return res, lname

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
		self.insize = inputs[0][0].shape[1]
		self.weight = Parameter(torch.Tensor(self.outsize, self.insize))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outsize))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

	def reset_params(self):
		# init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		# init.normal_(self.weight, std=0.01)
		_resnet_normal(self.weight)
		print('Reset fc params...')
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'fc%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "InnerProduct"\n'
		caffe_string += ' bottom: "%s"\n'%name
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' inner_product_param{\n'
		caffe_string += '  num_output: %d\n'%self.outsize
		caffe_string += '  bias_term: %s\n'%('true' if self.usebias else 'false')
		caffe_string += ' }\n}\n'

		params_dict[layer_name] = {}
		params_dict[layer_name]['fckernel'] = self.weight

		if self.usebias:
			params_dict[layer_name]['bias'] = self.bias

		layer_counter += 1 
		return layer_name

	def forward(self, x):
		lname = self._write_caffe(x[1])
		x = x[0]
		if self.norm:
			with torch.no_grad():
				norm = x.norm(p=2, dim=1, keepdim=True)
				wnorm = self.weight.norm(p=2,dim=1, keepdim=True)
			x = x / norm
			weight = self.weight / wnorm
		else:
			weight = self.weight
		return F.linear(x, weight, self.bias), lname

def flatten(x):
	x = x.view(x.size(0), -1)
	return x 

class Flatten(Model):
	def _write_caffe(self, name):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'flatten%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Flatten"\n'
		caffe_string += ' bottom: "%s"\n'%name
		caffe_string += ' top: "%s"\n'%layer_name
		# caffe_string += ' crop_param{\n  offset:%d\n  offset:%d\n  }\n}\n'%(1,1)
		caffe_string += '}\n'

		layer_counter += 1 
		return layer_name

	def forward(self, x):
		# print(x)
		lname = self._write_caffe(x[1])
		return flatten(x[0]), lname

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
		num_features = inputs[0][0].shape[1]
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

	def _write_caffe(self, btm):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0

		layer_name_bn = 'bn%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name_bn
		caffe_string += ' type: "BatchNorm"\n'
		caffe_string += ' bottom: "%s"\n'%btm
		caffe_string += ' top: "%s"\n'%layer_name_bn
		caffe_string += ' batch_norm_param{\n  use_global_stats:true\n  eps:2e-5\n }\n'
		caffe_string += '}\n'

		params_dict[layer_name_bn] = {}
		params_dict[layer_name_bn]['mean'] = self.running_mean
		params_dict[layer_name_bn]['var'] = self.running_var
		params_dict[layer_name_bn]['scale'] = torch.from_numpy(np.float32([1.]))

		layer_name_scale = 'scale%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name_scale
		caffe_string += ' type: "Scale"\n'
		caffe_string += ' bottom: "%s"\n'%layer_name_bn
		caffe_string += ' top: "%s"\n'%layer_name_bn
		caffe_string += ' scale_param{\n  bias_term:true\n }\n'
		caffe_string += '}\n'
		params_dict[layer_name_scale] = {}
		params_dict[layer_name_scale]['scale'] = self.weight
		# print(layer_name, self.weight)
		params_dict[layer_name_scale]['bias'] = self.bias
		return layer_name_bn

	def forward(self, input):
		lname = self._write_caffe(input[1])
		input = input[0]
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
		return result, lname

def GlobalAvgPool2D(x):
	x = x.mean(dim=(2,3), keepdim=True)
	return x 

class GlobalAvgPool2DLayer(Model):
	def _write_caffe(self, name, ksize):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'gavgpool%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Pooling"\n'
		caffe_string += ' bottom:"%s"\n'%name
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' pooling_param{\n  pool:AVE\n  kernel_size:%d\n }\n'%ksize
		caffe_string += '}\n'
		return layer_name

	def forward(self, x):
		lname = self._write_caffe(x[1], x.shape[2])
		return GlobalAvgPool2D(x[0]), lname

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

class Activation(Model):
	def initialize(self, act):
		self.param = act 
		if act==9:
			self.act = torch.nn.PReLU(num_parameters=1)

	def build(self, *inputs):
		outchn = inputs[0][0].shape[1]
		if self.param==8:
			self.act = torch.nn.PReLU(num_parameters=outchn)

	def _write_caffe(self, btm):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'actv%d'%(layer_counter)

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		if self.param == 0:
			caffe_string += ' type: "ReLU"\n'
		elif self.param in [1,8,9]:
			caffe_string += ' type: "PReLU"\n'
			params_dict[layer_name] = {}
			if self.param==1:
				params_dict[layer_name]['gamma'] = torch.from_numpy(np.float32([0.2]))
			else:
				params_dict[layer_name]['gamma'] = list(self.parameters())[0]
		elif self.param == 6:
			caffe_string += ' type: "Sigmoid"\n'
		caffe_string += ' bottom: "%s"\n'%btm
		caffe_string += ' top: "%s"\n'%btm
		caffe_string += '}\n'

		layer_counter += 1 
		return btm 

	def forward(self, x):
		if self.param == -1:
			lname = x[1]
		else:
			lname = self._write_caffe(x[1])
		x = x[0]
		# print(x.shape)
		if self.param==8 or self.param==9:
			res = self.act(x)
		else:
			res = activation(x, self.param)
		return res, lname

class BroadcastMUL(Model):
	def _write_caffe(self, names, tiles):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0

		# manual tiling layers to match the size 
		layer_name = 'tile_0_%d'%layer_counter
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Tile"\n'
		caffe_string += ' bottom:"%s"\n'%names[0]
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' tile_param{\n  axis:2\n  tiles:%d\n }\n'%tiles[0]
		caffe_string += '}\n'

		layer_name = 'tile_1_%d'%layer_counter
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Tile"\n'
		caffe_string += ' bottom:"tile_0_%d"\n'%layer_counter
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' tile_param{\n  axis:3\n  tiles:%d\n }\n'%tiles[1]
		caffe_string += '}\n'

		# do multiplication
		layer_name = 'mul%d'%layer_counter
		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Eltwise"\n'
		caffe_string += ' bottom:"tile_1_%d"\n'%layer_counter
		caffe_string += ' bottom:"%s"\n'%names[1]
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' eltwise_param{\n  operation:PROD\n }\n'
		caffe_string += '}\n'
		layer_counter += 1
		return layer_name

	def forward(self, x):
		names = [i[1] for i in x]
		xs = [i[0] for i in x]
		out = xs[0]*xs[1]
		lname = self._write_caffe(names, [xs[1].shape[2], xs[1].shape[3]])
		return out, lname

class SUM(Model):
	def _write_caffe(self, names):
		global caffe_string, layer_counter
		if not 'caffe_string' in globals():
			caffe_string = ''
		if not 'layer_counter' in globals():
			layer_counter = 0
		layer_name = 'add%d'%layer_counter

		caffe_string += 'layer{\n'
		caffe_string += ' name: "%s"\n'%layer_name
		caffe_string += ' type: "Eltwise"\n'
		for n in names:
			caffe_string += ' bottom:"%s"\n'%n
		caffe_string += ' top: "%s"\n'%layer_name
		caffe_string += ' eltwise_param{\n  operation:SUM\n }\n'
		caffe_string += '}\n'
		layer_counter += 1
		return layer_name

	def forward(self, *x):
		names = [i[1] for i in x]
		xs = [i[0] for i in x]
		lname = self._write_caffe(names)
		return sum(xs), lname
