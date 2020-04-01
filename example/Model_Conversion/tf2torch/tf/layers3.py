import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf 
from tensorflow.keras.layers import Layer as KLayer 
import numpy as np 
import time

# dumb layer declaration
class Layer(KLayer):
	"""
	Layer template. Implement some basic functions by override initialize, build and forward.
	"""
	def __init__(self, *args, **kwargs):
		"""
		Default initialization. Not recommended to touch.
		"""
		super(Layer, self).__init__()
		self.initialize(*args, **kwargs)

	def initialize(self, *args, **kwargs):
		"""
		Write initialization logic here.

		This is a method to assign pre-defined parameters to the class.
		"""
		pass 

	def build(self, input_shape):
		pass 

	def call(self, x, *args, **kwargs):
		return self.forward(x, *args, **kwargs)

	def forward(self, x, *args, **kwargs):
		"""
		Alternative for *call*.

		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		pass 

class conv2D(KLayer):
	"""
	Basic convolution 2D layer
	"""
	def __init__(self, size, outchn, stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type outchn: int
		:param outchn: Number of output channels

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME', 'VALID', 'SAME_LEFT'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding and 'SAME_LEFT' uses pytorch-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(conv2D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate
		assert (pad in ['SAME','VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[-1]
		# parse args
		if isinstance(self.size,list):
			self.size = [self.size[0],self.size[1],inchannel,self.outchn]
		else:
			self.size = [self.size, self.size, inchannel, self.outchn]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1],1]
		else:
			self.stride = [1,self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,1]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
			# self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))
		if self.pad == 'SAME_LEFT':
			self.pad_value = [self.size[0]//2, self.size[1]//2]

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.pad=='SAME_LEFT':
			x = tf.pad(x, [[0,0], [self.pad_value[0], self.pad_value[0]], [self.pad_value[1], self.pad_value[1]], [0,0]])
			pad = 'VALID'
		else:
			pad = self.pad
		out = tf.nn.conv2d(x, self.kernel, self.stride, pad, dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out, self.bias)
		return out 

class conv3D(KLayer):
	"""
	Basic convolution 3D layer
	"""
	def __init__(self,size,outchn,stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type outchn: int
		:param outchn: Number of output channels

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME' or 'VALID'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(conv3D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0], self.size[1], self.size[2],inchannel,self.outchn]
		else:
			self.size = [self.size, self.size, self.size, inchannel, self.outchn]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1], self.stride[2],1]
		else:
			self.stride = [1,self.stride, self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],self.dilation_rate[2],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,self.dilation_rate,1]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))
		
	def call(self,x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		out = tf.nn.conv3d(x,self.kernel,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		return out 

class conv1D(KLayer):
	"""
	Basic convolution 1D layer
	"""
	def __init__(self,size,outchn,stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type outchn: int
		:param outchn: Number of output channels

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME' or 'VALID'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(conv1D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		self.size = [self.size, inchannel, self.outchn]
		# set stride
		self.stride = [1, self.stride, 1]
		# set dilation
		self.dilation_rate = [1,self.dilation_rate,1]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv1d(x,self.kernel,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class dwconv2D(KLayer):
	"""
	Basic depth-wise convolution layer.
	"""
	def __init__(self, size, multiplier, stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type multiplier: int
		:param multiplier: Multiplier of number of output channel. (outchannel = multiplier * inchannel)

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME', 'VALID', 'SAME_LEFT'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding and 'SAME_LEFT' uses pytorch-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(dwconv2D, self).__init__()
		self.size = size
		self.multiplier = multiplier
		self.stride = stride
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate
		assert (pad in ['SAME','VALID','SAME_LEFT'])
		self.pad = pad 

	def _parse_args(self, input_shape):
		inchannel = input_shape[-1]
		self.outchn = inchannel * self.multiplier
		# parse args
		if isinstance(self.size,list):
			self.size = [self.size[0],self.size[1],inchannel,self.multiplier]
		else:
			self.size = [self.size, self.size, inchannel, self.multiplier]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1],1]
		else:
			self.stride = [1,self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [self.dilation_rate[0],self.dilation_rate[1]]
		else:
			self.dilation_rate = [self.dilation_rate,self.dilation_rate]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))
		if self.pad == 'SAME_LEFT':
			self.pad_value = [self.size[0]//2, self.size[1]//2]

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.pad=='SAME_LEFT':
			x = tf.pad(x, [[0,0], [self.pad_value[0], self.pad_value[0]], [self.pad_value[1], self.pad_value[1]], [0,0]])
			pad = 'VALID'
		else:
			pad = self.pad

		out = tf.nn.depthwise_conv2d(x, self.kernel, self.stride, pad, dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out, self.bias)
		return out 

class deconv1D(KLayer):
	"""
	Basic transposed convolution 1D layer
	"""
	def __init__(self,size,outchn,stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type outchn: int
		:param outchn: Number of output channels

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME' or 'VALID'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(deconv1D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		self.size = [self.size, self.outchn, inchannel]
		# set stride
		self.stride = [1, self.stride, 1]
		# set dilation
		self.dilation_rate = [1,self.dilation_rate,1]
		# compute output shape
		if self.pad=='SAME':
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1], self.outchn]
		else:
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1]+self.size[1]-self.stride[1], self.outchn]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv1d_transpose(x,self.kernel,self.outshape,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class deconv2D(KLayer):
	"""
	Basic transposed convolution 2D layer
	"""
	def __init__(self,size,outchn,stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type outchn: int
		:param outchn: Number of output channels

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME' or 'VALID'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(deconv2D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0],self.size[1],self.outchn,inchannel]
		else:
			self.size = [self.size, self.size, self.outchn, inchannel]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1],1]
		else:
			self.stride = [1,self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,1]

		# compute output shape
		if self.pad=='SAME':
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1], input_shape[2]*self.stride[2], self.outchn]
		else:
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1]+self.size[1]-self.stride[1], inp_shape[2]*self.stride[2]+self.size[2]-self.stride[2], self.outchn]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal'))
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		# self.outshape[0] = x.get_shape().as_list()[0]
		input_shape = x.shape
		# compute output shape
		if self.pad=='SAME':
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1], input_shape[2]*self.stride[2], self.outchn]
		else:
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1]+self.size[1]-self.stride[1], inp_shape[2]*self.stride[2]+self.size[2]-self.stride[2], self.outchn]
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv2d_transpose(x,self.kernel,self.outshape,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class deconv3D(KLayer):
	"""
	Basic transposed convolution 3D layer
	"""
	def __init__(self,size,outchn,stride=1,pad='SAME',dilation_rate=1,usebias=True,values=None):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of convolution kernel.

		:type outchn: int
		:param outchn: Number of output channels

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME' or 'VALID'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding.

		:type dilation_rate: int or list[int]
		:param dilation_rate: Dilation rate. Can be either integer or list of integers. When dilation_rate is larger than 1, stride should be 1.

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(deconv3D, self).__init__()
		self.size = size
		self.outchn = outchn
		self.stride = stride
		self.pad = pad 
		self.usebias = usebias
		self.values = values
		self.dilation_rate = dilation_rate

	def _parse_args(self, input_shape):
		# set size
		inchannel = input_shape[-1]
		if isinstance(self.size,list):
			self.size = [self.size[0], self.size[1], self.size[2],self.outchn,inchannel]
		else:
			self.size = [self.size, self.size, self.size, self.outchn, inchannel]
		# set stride
		if isinstance(self.stride,list):
			self.stride = [1,self.stride[0],self.stride[1], self.stride[2],1]
		else:
			self.stride = [1,self.stride, self.stride, self.stride, 1]
		# set dilation
		if isinstance(self.dilation_rate,list):
			self.dilation_rate = [1,self.dilation_rate[0],self.dilation_rate[1],self.dilation_rate[2],1]
		else:
			self.dilation_rate = [1,self.dilation_rate,self.dilation_rate,self.dilation_rate,1]
		# compute output shape
		if self.pad=='SAME':
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1], input_shape[2]*self.stride[2], input_shape[3]*self.stride[3], self.outchn]
		else:
			self.outshape = [input_shape[0], input_shape[1]*self.stride[1]+self.size[1]-self.stride[1], input_shape[2]*self.stride[2]+self.size[2]-self.stride[2], input_shape[3]*self.stride[3]+self.size[3]-self.stride[3], self.outchn]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outchn], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))
		
	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		# x = tf.expand_dims(x, axis=1)
		out = tf.nn.conv3d_transpose(x,self.kernel,self.outshape,self.stride,self.pad,dilations=self.dilation_rate)
		if self.usebias:
			out = tf.nn.bias_add(out,self.bias)
		# out = tf.squeeze(out, axis=1)
		return out 

class maxpoolLayer(KLayer):
	"""
	Basic max pooling layer
	"""
	def __init__(self, size, stride, pad='SAME'):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of pooling kernel.

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME', 'VALID', 'SAME_LEFT'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding and 'SAME_LEFT' uses pytorch-style auto-padding.
		"""
		super(maxpoolLayer, self).__init__()
		self.size = size
		self.stride = stride
		self.pad = pad 

	def build(self, input_shape):
		self.dim = len(input_shape)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.pad=='SAME_LEFT':
			if isinstance(self.size, int):
				padpix = self.size//2
				padpix = [[0,0]] + [[padpix, padpix]]* (self.dim-2) + [[0,0]]
				x = tf.pad(x, padpix)
				pad = 'VALID'
			if isinstance(self.size, list):
				raise NotImplementedError('Not implmeneted for non-square kernel in pooling')
		else:
			pad = self.pad
		out = tf.nn.max_pool(x, self.size, self.stride, pad)
		return out 

class avgpoolLayer(KLayer):
	"""
	Basic average pooling layer
	"""
	def __init__(self, size, stride, pad='SAME'):
		"""
		:type size: int or list[int]
		:param size: Indicate the size of pooling kernel.

		:type stride: int or list[int]
		:param stride: Stride number. Can be either integer or list of integers

		:type pad: String
		:param pad: Padding method, must be one of 'SAME', 'VALID', 'SAME_LEFT'. 'VALID' does not use auto-padding scheme. 'SAME' uses tensorflow-style auto-padding and 'SAME_LEFT' uses pytorch-style auto-padding.
		"""
		super(avgpoolLayer, self).__init__()
		self.size = size 
		self.stride = stride
		self.pad = pad 

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.pad=='SAME_LEFT':
			if isinstance(self.size, int):
				padpix = self.size//2
				padpix = [[0,0]] + [[padpix, padpix]]* (self.dim-2) + [[0,0]]
				x = tf.pad(x, padpix)
				pad = 'VALID'
			if isinstance(self.size, list):
				raise NotImplementedError('Not implmeneted for non-square kernel in pooling')
		else:
			pad = self.pad
		out = tf.nn.avg_pool(x, self.size, self.stride, pad)
		return out 

class globalAvgpoolLayer(KLayer):
	"""
	Basic global average pooling layer
	"""
	def __init__(self):
		super(globalAvgpoolLayer, self).__init__()

	def build(self, input_shape):
		self.num_dim = len(input_shape)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.num_dim==3:
			res = tf.reduce_mean(x, axis=1, keepdims=True)
		elif self.num_dim==4:
			res = tf.reduce_mean(x, axis=[1,2], keepdims=True)
		elif self.num_dim==5:
			res = tf.reduce_mean(x, axis=[1,2,3], keepdims=True)
		return res 

class activation(KLayer):
	"""
	Basic activation layer
	"""
	def __init__(self, param, **kwargs):
		"""
		Possible values:
			- model3.PARAM_RELU
			- model3.PARAM_LRELU
			- model3.PARAM_ELU
			- model3.PARAM_TANH
			- model3.PARAM_MFM
			- model3.PARAM_MFM_FC
			- model3.PARAM_SIGMOID
			- model3.PARAM_SWISH
		"""
		super(activation, self).__init__()

		self.param = param
		self.kwargs = kwargs

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.param == 0:
			res =  tf.nn.relu(x)
		elif self.param == 1:
			if 'leaky' in self.kwargs:
				leaky = self.kwargs['leaky']
			else:
				leaky = 0.2
			res =  tf.maximum(x,x*leaky)
		elif self.param == 2:
			res =  tf.nn.elu(x)
		elif self.param == 3:
			res =  tf.tanh(x)
		elif self.param == 4:
			shape = x.get_shape().as_list()
			res = tf.reshape(x,[-1,shape[1],shape[2],2,shape[-1]//2]) # potential bug in conv_net
			res = tf.reduce_max(res,axis=[3])
		elif self.param == 5:
			shape = x.get_shape().as_list()
			res = tf.reduce_max(tf.reshape(x,[-1,2,shape[-1]//2]),axis=[1])
		elif self.param == 6:
			res =  tf.sigmoid(x)
		elif self.param == 7:
			# res = tf.nn.swish(x)
			# res = tf.sigmoid(x) * x 
			res = swish(x)
		else:
			res =  x
		return res

class fcLayer(KLayer):
	"""
	Basic fully connected layer
	"""
	def __init__(self, outsize, usebias=True, values=None, norm=False):
		"""
		:type outsize: int
		:param outsize: Number of output channels

		:type usebias: bool
		:param usebias: Whether to add bias term in this layer.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.

		:type norm: bool (default=False)
		:param norm: Whether to normalize the kernel (along axis 0) before matrix multiplication
		"""
		super(fcLayer, self).__init__()
		self.outsize = outsize
		self.usebias = usebias
		self.values = values
		self.norm = norm 

	def _parse_args(self, input_shape):
		# set size
		insize = input_shape[-1]
		self.size = [insize, self.outsize]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outsize], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outsize], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		shape, dim = x.shape, len(x.shape)
		if dim==3:
			x = tf.reshape(x, [-1, shape[2]])

		if self.norm:
			k = tf.nn.l2_normalize(self.kernel, axis=0)
			x = tf.nn.l2_normalize(x, axis=1)
		else:
			k = self.kernel
		res = tf.matmul(x, k)
		if self.usebias:
			res = tf.nn.bias_add(res, self.bias)

		if dim==3:
			res = tf.reshape(res, [shape[0], shape[1], -1])
		return res 

class batch_norm(KLayer):
	"""
	Basic batch normalization layer
	"""
	def __init__(self, decay=1e-2, epsilon=1e-5, is_training=None, values=None):
		"""
		:type decay: float
		:param decay: Decay rate.

		:type epsilon: float
		:param epsilon: Epsilon value to avoid 0 division.

		:type is_training: bool
		:param is_training: Define whether this layer is in training mode

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(batch_norm, self).__init__()

		self.decay = decay
		self.epsilon = epsilon
		self.is_training = is_training
		self.values = values

	def build(self, input_shape):
		values = self.values
		shape = input_shape[-1]
		if self.values is None:
			self.moving_average = self.add_weight('moving_average',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(0.0),trainable=False)
			self.variance = self.add_weight('variance',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(1.0),trainable=False)

			self.gamma = self.add_weight('gamma',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(1.0),trainable=True)
			self.beta = self.add_weight('beta',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(0.0),trainable=True)
		else:
			self.moving_average = self.add_weight('moving_average',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(self.values[0]),trainable=False)
			self.variance = self.add_weight('variance',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(values[1]),trainable=False)

			self.gamma = self.add_weight('gamma',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(values[2]),trainable=True)
			self.beta = self.add_weight('beta',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(values[3]),trainable=True)

	def update(self,variable,value):
		delta = (variable - value) * self.decay
		variable.assign_sub(delta)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.is_training is None:
			is_training = bool(tf.keras.backend.learning_phase())
		else:
			is_training = self.is_training
		# is_training = True
		# print(is_training, time.time())
		inp_shape = x.get_shape().as_list()
		inp_dim_num = len(inp_shape)
		if inp_dim_num==3:
			x = tf.expand_dims(x, axis=1)
		elif inp_dim_num==2:
			x = tf.expand_dims(x, axis=1)
			x = tf.expand_dims(x, axis=1)
		elif inp_dim_num==5:
			x = tf.reshape(x, [inp_shape[0], inp_shape[1], inp_shape[2]*inp_shape[3], inp_shape[4]])
		if is_training:
			# res, mean, var = tf.compat.v1.nn.fused_batch_norm(x, self.gamma, self.beta, None, None, self.epsilon, is_training=is_training)
			mean, var = tf.nn.moments(x, axes=[0,1,2])
			res = (x - mean) / tf.sqrt(self.epsilon + var) * self.gamma + self.beta
			self.update(self.moving_average, mean)
			self.update(self.variance, var)
		else:
			res = (x - self.moving_average) / tf.sqrt(self.epsilon + self.variance) * self.gamma + self.beta
			# res, mean, var = tf.compat.v1.nn.fused_batch_norm(x, self.gamma, self.beta, self.moving_average, self.variance, self.epsilon, is_training=is_training)
		if inp_dim_num==3:
			res = tf.squeeze(res , axis=1)
		elif inp_dim_num==2:
			res = tf.squeeze(res, axis=[1,2])
		elif inp_dim_num==5:
			res = tf.reshape(res, inp_shape)
		return res 

class inst_norm(KLayer):
	"""
	Basic instance normalization layer
	"""
	def __init__(self, decay=1e-2, epsilon=1e-5, is_training=None, values=None):
		"""
		:type decay: float
		:param decay: Decay rate.

		:type epsilon: float
		:param epsilon: Epsilon value to avoid 0 division.

		:type is_training: bool
		:param is_training: Define whether this layer is in training mode

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.
		"""
		super(batch_norm, self).__init__()

		self.decay = decay
		self.epsilon = epsilon
		self.is_training = is_training
		self.values = values

	def build(self, input_shape):
		values = self.values
		shape = input_shape[-1]
		if self.values is None:
			self.moving_average = self.add_weight('moving_average',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(0.0),trainable=False)
			self.variance = self.add_weight('variance',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(1.0),trainable=False)

			self.gamma = self.add_weight('gamma',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(1.0),trainable=True)
			self.beta = self.add_weight('beta',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(0.0),trainable=True)
		else:
			self.moving_average = self.add_weight('moving_average',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(self.values[0]),trainable=False)
			self.variance = self.add_weight('variance',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(values[1]),trainable=False)

			self.gamma = self.add_weight('gamma',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(values[2]),trainable=True)
			self.beta = self.add_weight('beta',[shape], dtype=tf.keras.backend.floatx(),initializer=tf.initializers.constant(values[3]),trainable=True)

	def update(self,variable,value):
		delta = (variable - value) * self.decay
		variable.assign_sub(delta)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.is_training is None:
			is_training = bool(tf.keras.backend.learning_phase())
		else:
			is_training = self.is_training
		# is_training = True
		# print(is_training, time.time())
		inp_shape = x.get_shape().as_list()
		inp_dim_num = len(inp_shape)
		if inp_dim_num==3:
			x = tf.expand_dims(x, axis=1)
		elif inp_dim_num==2:
			x = tf.expand_dims(x, axis=1)
			x = tf.expand_dims(x, axis=1)
		elif inp_dim_num==5:
			x = tf.reshape(x, [inp_shape[0], inp_shape[1], inp_shape[2]*inp_shape[3], inp_shape[4]])

		if is_training:
			mean = tf.reduce_mean(x, axis=[1,2], keepdims=True)
			std = tf.reduce_std(x, axis=[1,2], keepdims=True) 
			res = (x - mean) / (std + self.epsilon)
			mean = tf.reduce_mean(mean, axis=[0,1,2])
			std = tf.reduce_mean(std, axis=[0,1,2])
			self.update(self.moving_average, mean)
			self.update(self.variance, std)
		else:
			res = (x - self.moving_average) / (self.variance + self.epsilon)
			
		if inp_dim_num==3:
			res = tf.squeeze(res , axis=1)
		elif inp_dim_num==2:
			res = tf.squeeze(res, axis=[1,2])
		elif inp_dim_num==5:
			res = tf.reshape(res, inp_shape)
		return res 

class flatten(KLayer):
	"""
	Basic flatten layer
	"""
	def __init__(self):
		super(flatten, self).__init__()

	def build(self, input_shape):
		self.shape = input_shape

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		self.shape = x.get_shape().as_list()
		num = 1
		for k in self.shape[1:]:
			num *= k 
		res = tf.reshape(x, [-1, num])
		return res 

class graphConvLayer(KLayer):
	"""
	Basic graph convolution layer
	"""
	def __init__(self, outsize, adj_mtx=None, adj_fn=None, values=None, usebias=True):
		"""
		:type outsize: int
		:param outsize: Output unit number

		:type adj_mtx: np.array
		:param adj_mtx: A matrix that indicates the affinity.

		:type adj_fn: callable
		:param adj_fn: A function to infer the affinity matrix.

		:type values: list[np.array]
		:param values: If the param 'values' is set, the layer will be initialized with the list of numpy array.

		:type usebias: bool
		:param usebias: Whether to use bias in this layer.
		"""
		super(graphConvLayer, self).__init__()
		assert (adj_mtx is None) ^ (adj_fn is None), 'Assign either adj_mtx or adj_fn' 
		self.outsize = outsize
		self.adj_mtx = adj_mtx
		self.adj_fn = adj_fn
		self.values = values
		self.usebias = usebias
		self.normalized = False

	def _parse_args(self, input_shape):
		# set size
		insize = input_shape[-1]
		self.size = [insize, self.outsize]

	def build(self, input_shape):
		values = self.values
		self._parse_args(input_shape)
		if self.values is not None:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[0]))
		else:
			self.kernel = self.add_weight('kernel', shape=self.size, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		if self.usebias:
			if self.values is not None:
				self.bias = self.add_weight('bias', shape=[self.outsize], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(values[1]))
			else:
				self.bias = self.add_weight('bias', shape=[self.outsize], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(0.0))

	def _normalize_adj_mtx(self, mtx):
		S = tf.reduce_sum(mtx, axis=1)
		S = tf.sqrt(S)
		S = 1. / S
		S = tf.diag(S)
		I = tf.eye(tf.cast(S.shape[0], tf.int64))
		A_ = (mtx + I) 
		A_ = tf.matmul(S, A_)
		A_ = tf.matmul(A_, S)
		return tf.stop_gradient(A_)

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		if self.adj_mtx is not None:
			A = self.adj_mtx
			if not self.normalized:
				A = self._normalize_adj_mtx(A)
				self.adj_mtx = A 
				self.normalized = True
		else:
			A = self.adj_fn(x)
			A = self._normalize_adj_mtx(A)

		res = tf.matmul(A, x)
		res = tf.matmul(res, self.W)
		if self.usebias:
			res = tf.nn.bias_add(res, self.b)
		return res 

class bilinearUpSample(KLayer):
	"""
	Basic bilinear upsampling layer. The biliear upsampling in tensorflow 1.x version has some mismatch with cv2&PIL&pytorch.
	"""
	def __init__(self, factor):
		"""
		:type factor: int
		:param factor: The upsample factor. The output will become *factor* times larger as the input tensor.
		"""
		super(bilinearUpSample, self).__init__()
		self.factor = factor

	def upsample_kernel(self, size):
		factor = (size +1)//2
		if size%2==1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size, :size]
		return (1 - abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)

	def upsample_kernel_1d(self, size):
		factor = (size + 1)//2
		if size%2==1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size]
		og = og[None, :]
		kernel = 1 - abs(og - center)/factor
		return kernel

	def upsample_kernel_3d(self, size):
		factor = (size + 1)//2
		if size%2==1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size, :size, :size]
		kernel = (1 - abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor) * (1-abs(og[2]-center)/factor)
		return kernel

	def get_kernel(self, dim, chn, factor):
		filter_size = 2*factor - factor%2
		shape = [filter_size] * (dim-2) + [chn, chn]
		weights = np.zeros(shape, dtype=tf.keras.backend.floatx())
		if dim == 5:
			k = self.upsample_kernel_3d(filter_size)
			for i in range(chn):
				weights[:,:,:,i,i] = k
		elif dim==4:
			k = self.upsample_kernel(filter_size)
			for i in range(chn):
				weights[:,:,i,i] = k
		elif dim==3:
			k = self.upsample_kernel_1d(filter_size)
			for i in range(chn):
				weights[:,i,i] = k
		return weights

	def build(self, input_shape):
		self.dim = len(input_shape)
		self.num_chn = input_shape[-1]
		if self.dim==3:
			self.outshape = [None, (input_shape[1]+2)*self.factor, input_shape[2]]
		elif self.dim==4:
			self.outshape = [None, (input_shape[1]+2)*self.factor, (input_shape[2]+2)*self.factor, input_shape[3]]
		elif self.dim==5:
			self.outshape = [None, (input_shape[1]+2)*self.factor, (input_shape[2]+2)*self.factor, (input_shape[3]+2)*self.factor, input_shape[4]]
		self.stride = [1] + [self.factor]*(self.dim-2) + [1]
		kernel = self.get_kernel(self.dim, self.num_chn, self.factor)
		self.kernel = self.add_weight('kernel_upsample', shape=kernel.shape, dtype=tf.keras.backend.floatx(), initializer=tf.initializers.constant(kernel))

	def call(self, x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		pad = [[0,0]] + [[1,1]]*(self.dim-2) + [[0,0]]
		x = tf.pad(x, pad, mode='symmetric')
		self.outshape[0] = x.get_shape().as_list()[0]
		# print(self.outshape)
		if self.dim==3:
			res = tf.nn.conv1d_transpose(x, self.kernel, self.outshape, self.stride)
			res = res[:, self.factor:-self.factor, :]
		elif self.dim==4:
			res = tf.nn.conv2d_transpose(x, self.kernel, self.outshape, self.stride)
			res = res[:, self.factor:-self.factor, self.factor:-self.factor, :]
		elif self.dim==5:
			res = tf.nn.conv3d_transpose(x, self.kernel, self.outshape, self.stride)
			res = res[:, self.factor:-self.factor, self.factor:-self.factor, self.factor:-self.factor, :]
		return res 

class NALU(KLayer):
	"""
	Arxiv: 1808.00508
	"""
	def __init__(self, outdim):
		"""
		:type outdim: int
		:param outdim: Output unit number of this layer.
		"""
		super(NALU, self).__init__()
		self.outdim = outdim
	def build(self, input_shape):
		indim = input_shape[-1]
		self.W = self.add_weight('W', shape=[indim, self.outdim], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		self.M = self.add_weight('M', shape=[indim, self.outdim], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
		self.G = self.add_weight('G', shape=[indim, self.outdim], dtype=tf.keras.backend.floatx(), initializer=tf.initializers.GlorotUniform())
	def call(self,x):
		"""
		:param x: Input tensor or numpy array. The object will be automatically converted to tensor if the input is np.array. Note that other arrays in args or kwargs will not be auto-converted.
		"""
		W = tf.tanh(self.W) * tf.sigmoid(self.M)
		g = tf.sigmoid(tf.matmul(x, self.G))
		m = tf.exp(tf.matmul(tf.log(tf.abs(x) + 1e-8), W))
		a = tf.matmul(x, W)
		out = g*a + (1.-g)*m
		return out 

@tf.custom_gradient
def gradient_reverse(x):
	def grad(dy):
		return -dy 
	return x, grad

@tf.custom_gradient
def swish(x):
	sig = tf.sigmoid(x)
	o = sig * x
	def grad(dy):
		g = sig * (1. + x * (1. - sig))
		return dy * g 
	return o, grad 
