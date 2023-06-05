import torch 
import torch.nn as nn 
import functools

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
			# if hasattr(obj, '_quant'):
			# 	print(type(obj))
			if hasattr(obj, '_quant') and obj._quant:
				obj._quant_calibrating = True
		self.apply(set_calibarte)

	def end_calibrate(self):
		def unset_calibrate(obj):
			if hasattr(obj, '_quant') and obj._quant:
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
