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
		
	def plot(self, xs,ys,zs, lims=None, **kwargs):
		if lims is not None:
			self.ax.set_xlim(lims[0])
			self.ax.set_ylim(lims[1])
			self.ax.set_zlim(lims[2])

		if (len(self.lines)==0) or (not self.usebuffer):
			a = self.ax.plot(xs, ys, zs, **kwargs)
			self.lines_buff.append(a)
		else:
			line = self.lines[self.line_pos][0]
			line.set_data(xs,ys)
			line.set_3d_properties(zs)
			self.line_pos += 1
		
	def update(self, require_img=False):
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
