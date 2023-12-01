import functools
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from torch.nn.parameter import Parameter
import math 
import numpy as np 
import torchvision.ops as ops
from .Base import Model 
from .Quant import QQuantizers, QObservers, QTYPES


def _resnet_normal(tensor):
	fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
	std = math.sqrt(2.0 / float(fan_out))
	return init._no_grad_normal_(tensor, 0., std)

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
		# parse args
		if isinstance(self.size,list):
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size[0]-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size[1]-1 ))//2)
			self.size = [self.outchn, inchannel // self.gropus, self.size[0], self.size[1]]
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
			self.size = [self.outchn, inchannel // self.gropus, self.size, self.size]

	def build(self, *inputs):
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

		if self._quant:
			bit_type = self.get_flag('QActBit')
			if bit_type is None:
				bit_type = 'int8'
			obs_type = self.get_flag('QActObserver')
			if obs_type is None:
				obs_type = 'minmax'
			self.input_quantizer = QQuantizers['uniform'](zero_offset=False, bit_type=bit_type, observer=obs_type)
			self.w_quantizer = QQuantizers['uniform'](zero_offset=True, mode='channel_wise', is_weight=True)

	def reset_params(self):
		if self.get_flag('conv_init_mode')=='normal':
			init.normal_(self.weight, std=0.001)
		elif self.get_flag('conv_init_mode')=='resnet':
			_resnet_normal(self.weight)
		else:
			init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			if self.get_flag('conv_init_mode')=='normal':
				init.zeros_(self.bias)
			elif self.get_flag('conv_init_mode')=='resnet':
				fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
				bound = 1 / math.sqrt(fan_in)
				init.uniform_(self.bias, -bound, bound)
			else:
				fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
				if fan_in != 0:
					bound = 1 / math.sqrt(fan_in)
					init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		weight = self.weight
		if self._quant:
			x = self.input_quantizer(x)
			weight = self.w_quantizer(weight)
		return F.conv2d(x, weight, self.bias, self.stride, self.pad, self.dilation_rate, self.gropus)

	def to_torch(self):
		conv = nn.Conv2d(in_channels = self.inchannel, out_channels = self.outchn, kernel_size = tuple(self.size[2:]), stride = self.stride,\
						padding = self.pad, padding_mode = 'zeros', dilation = self.dilation_rate, groups = self.gropus, bias = self.usebias)
		conv.weight.data[:] = self.weight.data[:]
		if self.usebias:
			conv.bias.data[:] = self.bias.data[:]
		return conv 


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
				self.out_pad = self.stride - 1
			self.size = [inchannel, self.outchn // self.gropus, self.size, self.size]
		else:
			raise Exception("Deconv kernel only supports int")

	def build(self, *inputs):
		inp = inputs[0]
		self._parse_args(inp.shape)
		self.weight = Parameter(torch.Tensor(*self.size))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outchn))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

		if self._quant:
			bit_type = self.get_flag('QActBit')
			if bit_type is None:
				bit_type = 'int8'
			obs_type = self.get_flag('QActObserver')
			if obs_type is None:
				obs_type = 'minmax'
			self.input_quantizer = QQuantizers['uniform'](zero_offset=False, bit_type=bit_type, observer=obs_type)
			self.w_quantizer = QQuantizers['uniform'](zero_offset=True, mode='channel_wise', is_weight=True)

	def reset_params(self):
		_resnet_normal(self.weight)
		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		inh, inw = x.shape[2], x.shape[3]
		weight = self.weight
		if self._quant:
			x = self.input_quantizer(x)
			weight = self.w_quantizer(weight)
		x = F.conv_transpose2d(x, weight, self.bias, self.stride, self.pad, self.out_pad, self.gropus, self.dilation_rate)
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
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size[0]-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size[1]-1 ))//2)
			self.size = [self.multiplier * inchannel, 1, self.size[0], self.size[1]]
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
			self.size = [self.multiplier * inchannel, 1, self.size, self.size]

	def build(self, *inputs):
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
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = ((self.size[0]+ (self.dilation_rate-1) * ( self.size[0]-1 ))//2, (self.size[1]+ (self.dilation_rate-1) * ( self.size[1]-1 ))//2, (self.size[2]+ (self.dilation_rate-1) * ( self.size[2]-1 ))//2)
			self.size = [self.outchn, inchannel // self.gropus, self.size[0], self.size[1], self.size[2]]
		else:
			if self.pad == 'VALID':
				self.pad = 0
			else:
				self.pad = (self.size + (self.dilation_rate-1) * ( self.size-1 ))//2
			self.size = [self.outchn, inchannel // self.gropus, self.size, self.size, self.size]

	def build(self, *inputs):
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
		self.insize = inputs[0].shape[-1]
		self.weight = Parameter(torch.Tensor(self.outsize, self.insize))
		if self.usebias:
			self.bias = Parameter(torch.Tensor(self.outsize))
		else:
			self.register_parameter('bias', None)
		self.reset_params()

		if self._quant:
			bit_type = self.get_flag('QActBit')
			if bit_type is None:
				bit_type = 'int8'
			obs_type = self.get_flag('QActObserver')
			if obs_type is None:
				obs_type = 'minmax'
			self.input_quantizer = QQuantizers['uniform'](zero_offset=False, bit_type=bit_type, observer=obs_type)
			self.w_quantizer = QQuantizers['uniform'](zero_offset=True, mode='channel_wise', is_weight=True)


	def reset_params(self):
		if self.get_flag('fc_init_mode')=='normal':
			init.normal_(self.weight, std=0.001)
			if self.bias is not None:
				init.zeros_(self.bias)
		else:
			init.kaiming_uniform_(self.weight, a=math.sqrt(5))
			if self.bias is not None:
				fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
				bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
				init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		if self.norm:
			with torch.no_grad():
				norm = x.norm(p=2, dim=1, keepdim=True)
				wnorm = self.weight.norm(p=2,dim=1, keepdim=True)
			x = x / norm
			weight = self.weight / wnorm
		else:
			weight = self.weight

		if self._quant:
			x = self.input_quantizer(x).contiguous()
			weight = self.w_quantizer(weight).contiguous()
		return F.linear(x, weight, self.bias)

	def to_torch(self):
		fc = nn.Linear(self.insize, self.outsize, self.usebias)
		fc.weight.data[:] = self.weight.data[:]
		if self.usebias:
			fc.bias.data[:] = self.bias.data[:]
		return fc 


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
		inp = inputs[0]
		self._parse_args(inp.shape)

	def forward(self, x):
		return F.avg_pool2d(x, self.size, self.stride, self.pad, False, True)


class BatchNorm(Model):
	# _version = 2
	# __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
	# 				 'running_mean', 'running_var', 'num_batches_tracked',
	# 				 'num_features', 'affine', 'weight', 'bias']

	def initialize(self, eps=None, momentum=0.01, affine=True,
				 track_running_stats=True):
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		
	def build(self, *inputs):
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
		eps = self.get_flag('bn_eps')
		self.eps = 2e-5 if eps is None else eps 

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


class NNUpSample(Model):
	def initialize(self, scale):
		self.scale = scale

	def _parse_args(self, input_shape):
		self.inchannel = input_shape[1]
		self.size = [self.inchannel, 1, self.scale, self.scale]

	def build(self, *inputs):
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
		return F.leaky_relu(x, negative_slope=0.1)
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
		# parse args
		if isinstance(self.size,list):
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

	def forward(self, x, offset):
		return ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.pad, self.dilation_rate)

