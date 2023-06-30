import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import json 
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axis3d as axis3d

class Plotter3D():
	def __init__(self, usebuffer=False, elev=None, azim=None, axis='on', axis_tick='on', no_margin=False):
		fig = plt.figure(figsize=(6,5))
		self.ax = fig.add_subplot(111, projection='3d')
		# self.ax = custom_Axes3D(fig)
		self.axis_tick = axis_tick
		self.axis = axis 

		self.ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		self.ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		self.ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		self.ax.set_zticks([])
		self.ax.zaxis.line.set_lw(0.)

		if no_margin:
			# axis = 'off'
			plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
			plt.margins(0,0,0)
			# self.ax.xaxis.set_major_locator(plt.NullLocator())
			# self.ax.yaxis.set_major_locator(plt.NullLocator())
		self.ax.view_init(elev=elev, azim=azim)
		self.ax.axis(axis)
		if axis_tick=='off':
			self.ax.set_xticklabels([])
			self.ax.set_yticklabels([])
			self.ax.set_zticklabels([])
		self.lines = []
		self.lines_buff = []
		self.line_pos = 0
		self.usebuffer = usebuffer
		self.fig = fig 

	def show(self, ion=True):
		if ion:
			plt.ion()
		plt.show()

	def clear(self):
		self.ax.clear()
		self.ax.set_zticks([])
		self.ax.zaxis.line.set_lw(0.)
		if self.axis_tick=='off':
			self.ax.set_xticklabels([])
			self.ax.set_yticklabels([])
			self.ax.set_zticklabels([])
		if self.axis=='off':
			self.ax.axis('off')
		# self.ax.zaxis.line.set_lw(0.)
		
	def plot(self, xs,ys,zs, lims=None, **kwargs):
		if lims is not None:
			self.ax.set_xlim(lims[0])
			self.ax.set_ylim(lims[1])
			self.ax.set_zlim(lims[2])

		# self.ax.plot(xs, ys, zs, **kwargs)

		if (len(self.lines)==0) or (not self.usebuffer):
			a = self.ax.plot(xs, ys, zs, **kwargs)
			self.lines_buff.append(a)
		else:
			line = self.lines[self.line_pos][0]
			line.set_data(xs,ys)
			line.set_3d_properties(zs)
			self.line_pos += 1
		
	def update(self, require_img=False):
		# slowest 
		# plt.pause(0.0001)
		# faster
		# self.fig.canvas.draw()
		# much faster
		try:
			self.ax.set_proj_type('persp')
			self.ax.draw_artist(self.ax.patch)
			for line in self.lines:
				self.ax.draw_artist(line)
			self.fig.canvas.update()
		except:
			self.ax.set_proj_type('persp')
			self.fig.canvas.draw()

		if require_img:
			# image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8')
			s, (width, height) = self.fig.canvas.print_to_buffer()
			image = np.fromstring(s, np.uint8).reshape((height, width, 4))

		self.fig.canvas.flush_events()
		if len(self.lines)==0:
			self.lines = self.lines_buff
			self.lines_buff = []
		self.line_pos = 0

		if require_img:
			return image

class Surface3D():
	def __init__(self, elev=None, azim=None, axis='on', axis_tick='on'):
		fig = plt.figure()
		self.ax = fig.add_subplot(111, projection='3d')
		self.ax.view_init(elev=elev, azim=azim)
		self.ax.axis(axis)
		if axis=='off':
			self.ax.set_xticklabels([])
			self.ax.set_yticklabels([])
			self.ax.set_zticklabels([])

	def show(self, ion=True):
		if ion:
			plt.ion()
		plt.show()

	def plot(self, X,Y,Z, lims=None, **kwargs):
		X = np.float32(X)
		Y = np.float32(Y)
		Z = np.float32(Z)
		if lims is not None:
			self.ax.set_xlim(lims[0])
			self.ax.set_ylim(lims[1])
			self.ax.set_zlim(lims[2])
		self.ax.clear()
		self.surf = self.ax.plot_surface(X,Y,Z, **kwargs)
		plt.pause(0.0001)

	def add_locator(self, num):
		self.ax.zaxis.set_major_locator(LinearLocator(num))

	def add_colorbar(self, **kwargs):
		self.ax.colorbar(self.surf, **kwargs)
	
	def set_label(self, labels):
		self.ax.set_xlabel(labels[0])
		self.ax.set_ylabel(labels[1])
		self.ax.set_zlabel(labels[2])

	def interpolate(self, X,Y, X_target, Y_target, values):
		if len(X.shape)==1:
			X,Y = np.meshgrid(X,Y)

		coord = np.stack([X,Y], axis=-1).reshape([-1,2])
		values = values.reshape([-1])
		res = griddata(coord, values, (X_target, Y_target), method='cubic')
		return res 

class Plotter2D():
	def __init__(self, usebuffer=False, elev=None, azim=None, axis='on', axis_tick='on', no_margin=False):
		fig = plt.figure()
		plt.tight_layout()
		self.ax = fig.add_subplot(111)
		if no_margin:
			axis = 'off'
			plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
			plt.margins(0,0)
			self.ax.xaxis.set_major_locator(plt.NullLocator())
			self.ax.yaxis.set_major_locator(plt.NullLocator())
		self.ax.axis(axis)
		if axis=='off':
			self.ax.set_xticklabels([])
			self.ax.set_yticklabels([])
		self.lines = []
		self.lines_buff = []
		self.line_pos = 0
		self.usebuffer = usebuffer
		self.fig = fig 

	def show(self, ion=True):
		if ion:
			plt.ion()
		plt.show()

	def clear(self):
		self.ax.clear()

	def plot(self, xs,ys, lims=None, **kwargs):
		if lims is not None:
			self.ax.set_xlim(lims[0])
			self.ax.set_ylim(lims[1])

		if (len(self.lines)==0) or (not self.usebuffer):
			a = self.ax.plot(xs, ys, **kwargs)
			self.lines_buff.append(a)
		else:
			line = self.lines[self.line_pos][0]
			line.set_data(xs,ys)
			self.line_pos += 1
		
	def update(self, require_img=False):
		# slowest 
		# plt.pause(0.0001)
		# faster
		# self.fig.canvas.draw()
		# much faster
		try:
			self.ax.draw_artist(self.ax.patch)
			for line in self.lines:
				self.ax.draw_artist(line)
			self.fig.canvas.update()
		except:
			self.fig.canvas.draw()

		if require_img:
			# image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8')
			s, (width, height) = self.fig.canvas.print_to_buffer()
			image = np.fromstring(s, np.uint8).reshape((height, width, 4))

		self.fig.canvas.flush_events()
		if len(self.lines)==0:
			self.lines = self.lines_buff
			self.lines_buff = []
		self.line_pos = 0

		if require_img:
			return image

	def imshow(self, img, **kwargs):
		self.ax.imshow(img, **kwargs)

class LossPlotter():
	def __init__(self, loss_file, skip_first=False, splitter='\t'):
		fig = plt.figure()
		self.ax = fig.add_subplot(111)
		self.losses = []
		f = open(loss_file)
		if skip_first:
			f.readline()
		for i in f:
			i = i.strip()
			buff = i.split(splitter)
			buff = [float(_) for _ in buff]
			self.losses.append(buff)
		self.losses = list(zip(*self.losses))
		self.losses = [np.array(_) for _ in self.losses]

	def apply_ema(self, ignore_index=0):
		def ema(arr, alpha=0.05):
			for i in range(len(arr)-1):
				arr[i+1] = alpha * arr[i+1] + (1 - alpha) * arr[i]

		for i in range(ignore_index, len(self.losses)):
			ema(self.losses[i])

	def plot(self, ignore_index=0, labels=None, lims=None, iteration_interval=1):
		if lims is not None:
			self.ax.set_xlim(lims[0])
			self.ax.set_ylim(lims[1])
		x = np.float32(list(range(len(self.losses[0])))) * iteration_interval
		data = self.losses[ignore_index:]
		if labels is None:
			for d in data:
				self.ax.plot(x, d)
		else:
			for d,lb in zip(data,labels):
				self.ax.plot(x, d, label=lb)

	def set_title(self, title):
		self.ax.title(title)
	def set_xylabel(self, label):
		self.ax.set_xlabel(label[0])
		self.ax.set_ylabel(label[1])
	def set_legend(self, location):
		self.ax.legend(loc=location)
	def show(self, ion=True):
		if ion:
			plt.ion()
		plt.show()

class LossPlotterJson():
	def __init__(self, loss_file, keys, alpha=0.1, scales=None):
		fig = plt.figure()
		self.ax = fig.add_subplot(111)
		dt = json.load(open(loss_file))
		self.losses = [dt[k] for k in keys]
		self.losses = [[_[1] for _ in i] for i in self.losses]
		self.keys = keys
		self.alpha = alpha
		if scales is None:
			self.scales = [1 for _ in range(len(losses))]
		else:
			self.scales = scales

	def apply_ema(self, ignore_index=0):
		def ema(arr):
			alpha = self.alpha
			for i in range(len(arr)-1):
				arr[i+1] = alpha * arr[i+1] + (1 - alpha) * arr[i]

		for i in range(ignore_index, len(self.losses)):
			ema(self.losses[i])

	def plot(self):
		x = np.float32(list(range(len(self.losses[0]))))
		for d,lb,s in zip(self.losses, self.keys, self.scales):
			d = np.float32(d)
			d = d * s 
			self.ax.plot(x, d, label=lb)

	def set_title(self, title):
		self.ax.title(title)

	def set_xylabel(self, label):
		self.ax.set_xlabel(label[0])
		self.ax.set_ylabel(label[1])
	def set_legend(self, location):
		self.ax.legend(loc=location)
	def show(self, ion=False):
		if ion:
			plt.ion()
		plt.show()

class FilterPlotter():
	def __init__(self):
		fig = plt.figure()
		self.ax = fig.add_subplot(111)

	def plot(self, kernel):
		f = np.fft.fft2(kernel)
		f = np.fft.fftshift(f)
		self.mag_spectrum = np.log(np.abs(f)+1)
		self.ax.imshow(self.mag_spectrum, cmap='gray', interpolation='bilinear')

	def show(self, ion=True):
		if ion:
			plt.ion()
		plt.show()
