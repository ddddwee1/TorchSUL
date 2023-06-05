from .Base import Model 
import torch 
import torch.nn as nn 

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
			# print('Quant params loaded.')
			self.scale = state_dict[prefix + 'scale']
			self.zero_point = state_dict[prefix + 'zero_point']
		else:
			print('no scale for ', prefix)

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self._quant_calibrated:
			destination[prefix + 'scale'] = self.scale 
			destination[prefix + 'zero_point'] = self.zero_point
			# print('Quant param saved.')


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
			# print('Quant params loaded.')
			self.scale = state_dict[prefix + 'scale']
			self.zero_point = state_dict[prefix + 'zero_point']
		else:
			print('no scale for ', prefix)

	def _save_to_state_dict(self, destination, prefix, keep_vars):
		if self._quant_calibrated:
			destination[prefix + 'scale'] = self.scale 
			destination[prefix + 'zero_point'] = self.zero_point
			# print('Quant param saved.')


QObservers = {"minmax": MinMaxObserver, 'percentile': PercentileObserver}

class UniformQuantizer(Model):
	def initialize(self, bit_type='int8', observer='minmax', zero_offset=False, mode='layer_wise', is_weight=False):
		self.bit_type = QTYPES[bit_type]
		self.observer = QObservers[observer](self.bit_type, zero_offset, mode, is_weight)
		self.mode = mode 
		self.is_weight = is_weight

	def build(self, x):
		if len(x.shape)==4:
			if self.is_weight:
				self.dim = -4
			else:
				self.dim = -3
		else:
			self.dim = -1 

	def quant(self, x):
		if self.dim!=-1 and self.mode=='channel_wise':
			# make everything run at last dim, no need manually reshape 
			x = x.transpose(-1, self.dim)
		x = x / self.observer.scale + self.observer.zero_point
		x = x.round().clamp(self.bit_type.min_val, self.bit_type.max_val)
		if self.dim!=-1 and self.mode=='channel_wise':
			x = x.transpose(-1, self.dim)
		return x 

	def dequant(self, x):
		if self.dim!=-1 and self.mode=='channel_wise':
			x = x.transpose(-1, self.dim)
		x = (x - self.observer.zero_point) * self.observer.scale 
		if self.dim!=-1 and self.mode=='channel_wise':
			x = x.transpose(-1, self.dim)
		return x 

	def forward(self, x, debug=False):
		if self._quant_calibrating:
			x = self.observer(x)
		if self._quant and self._quant_calibrated:
			if self.observer.scale.device!=x.device:
				self.observer.scale = self.observer.scale.to(x.device)
				self.observer.zero_point = self.observer.zero_point.to(x.device)
			x = self.quant(x)
			if debug:
				print(x, self.observer.scale, self.observer.zero_point)
			x = self.dequant(x)
		return x 


QQuantizers = {"uniform": UniformQuantizer}

class QAct(Model):
	def initialize(self, zero_offset=False, mode='layer_wise', observer='minmax', bit_type=None):
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
			# else:
			# 	print('QAct using bit_type:', bit_type)
			self.quantizer = QQuantizers['uniform'](bit_type=bit_type, zero_offset=self.zero_offset, mode=self.mode, observer=self.observer_str)
	def forward(self, x, debug=False):
		if self._quant:
			x = self.quantizer(x, debug=debug)
		return x 
