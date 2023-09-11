from .Base import Model 
import torch 
import torch.nn as nn 
from torch.nn.parameter import Parameter
from torch.autograd import Function 
from torch.autograd.function import once_differentiable
from loguru import logger 

class QATFunc(Function):
	# quant-aware training function 
	@staticmethod 
	def forward(ctx, x, scale, zero_point, Qn, Qp, zero_offset=False, mode='layer_wise', dim=-1):
		ctx.Qn = Qn 
		ctx.Qp = Qp 
		ctx.mode = mode 
		ctx.dim = dim 
		ctx.zero_offset = zero_offset

		if dim!=-1 and mode=='channel_wise':
			# make everything run at last dim, no need manually reshape 
			x = x.transpose(-1, dim)

		zero_point = zero_point.round()
		x_back = x 
		x = x / scale + zero_point
		x = x.round().clamp(Qn, Qp)
		ctx.save_for_backward(x_back, x, scale, zero_point)     # save here to save computation for backward

		x = (x - zero_point) * scale 
		if dim!=-1 and mode=='channel_wise':
			x = x.transpose(-1, dim)
		return x

	@staticmethod
	@once_differentiable
	def backward(ctx, out_grad):
		out_grad = out_grad.contiguous()
		x, x1, scale, zero_point = ctx.saved_tensors

		# compute ds 
		ds = x1.clone()
		ds -= zero_point
		idx = (x1>ctx.Qn) & (x1<ctx.Qp)
		if ctx.mode=='channel_wise':
			coef = torch.zeros_like(x)
			coef[idx] += x[idx]
			ds -= coef / scale
			ds *= out_grad.transpose(ctx.dim, -1)
		else:
			ds[idx] -= x[idx]/scale
			ds *= out_grad

		# compute db 
		db = torch.zeros_like(x1)
		if not ctx.zero_offset:
			idx = (x1<=ctx.Qn) | (x1>=ctx.Qp)
			if ctx.mode=='channel_wise':
				db[idx] += 1
				db *= -scale
				db *= out_grad.transpose(ctx.dim, -1)
			else:
				db[idx] -= scale
				db *= out_grad

		# compute dx 
		dx = out_grad.clone()
		if ctx.mode=='channel_wise':
			x1 = x1.transpose(ctx.dim, -1)
		dx[(x1<=ctx.Qn) | (x1>=ctx.Qp)] = 0

		if ctx.mode=='channel_wise':
			# use mean here 
			ds = ds.flatten(0,-2).sum(dim=0)
			db = db.flatten(0,-2).sum(dim=0)
		else:
			ds = ds.sum()
			db = db.sum()

		return dx.contiguous(), ds.contiguous(), db.contiguous(), None, None, None, None, None

	@staticmethod
	def symbolic(g, x, scale, zero_point, Qn, Qp, zero_offset=False, mode='layer_wise', dim=-1):
		return g.op('custom_ops::quant', x, scale, zero_point, Qn_i=int(Qn), Qp_i=int(Qp), zero_offset_i=int(zero_offset), mode_s=mode, dim_i=int(dim)).setType(x.type().with_sizes(x.type().sizes()))


##### Quant classes 

class QUint8():
	max_val = 2 ** 8 -1 
	min_val = 0 
	signed = False

class QInt8():
	max_val = 2 ** 7 - 1 
	min_val = - 2 ** 7 
	signed = True 

class QInt16():
	max_val = 2 ** 15 - 1 
	min_val = - 2 ** 15 
	signed = True 

QTYPES = {"uint8": QUint8, "int8": QInt8, 'int16':QInt16}


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
		self.percentile = 0.99999

	def build(self, x):
		if self.is_weight:
			self.dim = 0
		else:
			if len(x.shape)==4:
				self.dim = 1 
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

	@torch.no_grad()
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
		if (self.min_val is None) or (self.max_val is None):
			logger.warning('This quant layer is not fed with any data and it will be omitted. Use M.inspect_quant_params to get specific layer name')
		elif self.scale is None:
			s,z = self.get_quant_params()
			self.scale = Parameter(s)
			self.zero_point = Parameter(z)

	def forward(self, x):
		if self._quant_calibrating:
			self.observe(x)
		return x 

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		if prefix+'scale' in state_dict:
			self.scale = Parameter(state_dict[prefix + 'scale'])
			self.zero_point = Parameter(state_dict[prefix + 'zero_point'])
		else:
			logger.debug('Loading quant layer... no scale for layer ', prefix)

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self._quant_calibrated:
			destination[prefix + 'scale'] = self.scale 
			destination[prefix + 'zero_point'] = self.zero_point


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
		if self.is_weight:
			self.dim = 0
		else:
			if len(x.shape)==4:
				self.dim = 1 
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

	@torch.no_grad()
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
		if (self.min_val is None) or (self.max_val is None):
			logger.warning('This quant layer is not fed with any data and it will be omitted. Use M.inspect_quant_params to get specific layer name')
		elif self.scale is None:
			s,z = self.get_quant_params()
			self.scale = Parameter(s)
			self.zero_point = Parameter(z)

	def forward(self, x):
		if self._quant_calibrating:
			self.observe(x)
		return x 

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		if prefix+'scale' in state_dict:
			self.scale = Parameter(state_dict[prefix + 'scale'])
			self.zero_point = Parameter(state_dict[prefix + 'zero_point'])
		else:
			logger.debug('Loading quant layer... no scale for layer %s'%prefix)

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self._quant_calibrated:
			destination[prefix + 'scale'] = self.scale 
			destination[prefix + 'zero_point'] = self.zero_point


class OmseObserver(Model):
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
		# TODO: mannually fit dim?
		if self.is_weight:
			self.dim = 0
		else:
			if len(x.shape)==4:
				self.dim = 1 
			else:
				self.dim = -1

	def observe(self, x):
		if self.mode == 'channel_wise':
			x = x.transpose(-1, self.dim)
			x = x.flatten(0,-2)
			minv = x.min(dim=0)[0]
			maxv = x.max(dim=0)[0]
		else:
			minv = x.min()
			maxv = x.max()
		if self.min_val is None:
			self.min_val = minv
			self.max_val = maxv
		else:
			self.min_val = torch.minimum(self.min_val, minv)
			self.max_val = torch.maximum(self.max_val, maxv)

		self.x_buffer = x

	@torch.no_grad()
	def _least_square(self, scale, factor):
		if self.zero_offset:
			zero_point = torch.zeros_like(self.max_val)
		else:
			zero_point = self.bit_type.min_val - torch.round(self.min_val * factor / scale)

		scale = scale.cuda()
		zero_point = zero_point.cuda()
		x_buffer = self.x_buffer.cuda()

		scale = scale * factor
		x_buffer_q = x_buffer / scale + zero_point
		x_buffer_q = x_buffer_q.round().clamp(self.bit_type.min_val, self.bit_type.max_val)
		x_buffer_q = (x_buffer_q - zero_point) * scale 
		return torch.pow(x_buffer_q - x_buffer, 2).mean(), scale, zero_point

	@torch.no_grad()
	def get_quant_params(self):
		if self.zero_offset:
			max_val = torch.max(self.max_val, -self.min_val)
			scale = max_val / min(self.bit_type.max_val, -self.bit_type.min_val)
		else:
			scale = (self.max_val - self.min_val) / float(self.bit_type.max_val - self.bit_type.min_val)
		scale.clamp(torch.finfo(torch.float32).eps)
		min_score = None 
		scale_best = None 
		zero_point_best = None
		for i in range(50):
			factor = 1 - 0.01*i
			score, scale_i, zero_point_i = self._least_square(scale, factor)
			if min_score is None:
				min_score = score
				scale_best = scale_i 
				zero_point_best = zero_point_i
			if score<min_score:
				min_score = score
				scale_best = scale_i
				zero_point_best = zero_point_i
		return scale_best, zero_point_best

	def _finish_calibrate(self):
		if (self.min_val is None) or (self.max_val is None):
			logger.warning('This quant layer is not fed with any data and it will be omitted. Use M.inspect_quant_params to get specific layer name')
		elif self.scale is None:
			s,z = self.get_quant_params()
			self.scale = Parameter(s)
			self.zero_point = Parameter(z)

	def forward(self, x):
		if self._quant_calibrating:
			self.observe(x)
		return x 

	def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
		if prefix+'scale' in state_dict:
			self.scale = Parameter(state_dict[prefix + 'scale'])
			self.zero_point = Parameter(state_dict[prefix + 'zero_point'])
		else:
			logger.debug('Loading quant layer... no scale for layer ', prefix)

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self._quant_calibrated:
			destination[prefix + 'scale'] = self.scale 
			destination[prefix + 'zero_point'] = self.zero_point


QObservers = {"minmax": MinMaxObserver, 'percentile': PercentileObserver, 'omse': OmseObserver}

class UniformQuantizer(Model):
	def initialize(self, bit_type='int8', observer='minmax', zero_offset=False, mode='layer_wise', is_weight=False):
		self.bit_type = QTYPES[bit_type]
		self.observer = QObservers[observer](self.bit_type, zero_offset, mode, is_weight)
		self.mode = mode 
		self.is_weight = is_weight
		self.zero_offset = zero_offset

	def build(self, x):
		if self.is_weight:
			self.dim = 0
		else:
			if len(x.shape)==4:
				self.dim = 1 
			else:
				self.dim = -1

	def quant_dequant(self, x):
		if self.dim!=-1 and self.mode=='channel_wise':
			# make everything run at last dim, no need manually reshape 
			x = x.transpose(-1, self.dim)
		x = x / self.observer.scale + self.observer.zero_point
		x = x.round().clamp(self.bit_type.min_val, self.bit_type.max_val)
		x = (x - self.observer.zero_point) * self.observer.scale 
		if self.dim!=-1 and self.mode=='channel_wise':
			x = x.transpose(-1, self.dim)
		return x 

	def forward(self, x):
		x = x.contiguous()
		if self._quant_calibrating:
			x = self.observer(x)
		if self._quant and self._quant_calibrated and (self.observer.scale is not None):
			if self.observer.scale.device!=x.device:
				self.observer.to(x.device)
			# x = self.quant_dequant(x)
			if self.get_flag('dump_onnx'):
				x = QATFunc.apply(x, self.observer.scale.data, self.observer.zero_point.data, self.bit_type.min_val, self.bit_type.max_val, self.zero_offset, self.mode, self.dim)
			else:
				x = QATFunc.apply(x, self.observer.scale.contiguous(), self.observer.zero_point.contiguous(), self.bit_type.min_val, self.bit_type.max_val, self.zero_offset, self.mode, self.dim)
		return x.contiguous() 


QQuantizers = {"uniform": UniformQuantizer}


class QAct(Model):
	def initialize(self, zero_offset=False, mode='layer_wise', observer=None, bit_type=None):
		self.mode = mode 
		self.zero_offset = zero_offset
		self.observer_str = observer
		self.bit_type = bit_type
	def build(self, x):
		if self._quant:
			bit_type = self.get_flag('QActBit')
			if bit_type is None:
				bit_type = 'int8'
			if self.bit_type is not None:
				bit_type = self.bit_type

			obs_type = self.get_flag('QActObserver')
			if obs_type is None:
				obs_type = 'minmax'
			if self.observer_str is not None:
				obs_type = self.observer_str
			self.quantizer = QQuantizers['uniform'](bit_type=bit_type, zero_offset=self.zero_offset, mode=self.mode, observer=obs_type)
	def forward(self, x):
		if self._quant:
			x = self.quantizer(x)
		return x 
