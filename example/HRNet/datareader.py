import numpy as np 
import pickle 
import cv2 
import SUL.DataReader

def plot_gaussian(pos, size=64):
	x, y = pos[0], pos[1]
	xx = np.linspace(0,size-1,size)
	yy = np.linspace(0,size-1,size)
	xx, yy = np.meshgrid(xx, yy)
	exp = np.exp( -((xx - x)**2 + (yy - y)**2 ) / 4 )
	hmap = exp  / exp.max()
	return hmap

def get_hmap(pts, size=64, scale=4):
	hmap = np.zeros([size, size, 17])
	if pts is None:
		return hmap
	pts = pts.copy()
	pts[:,:2] /= scale
	for i in range(len(pts)):
		if pts[i,2] > 0:
			hmap[:,:,i] = plot_gaussian(pts[i])
	return hmap

def get_minmax(pts):
	xs = pts[:,0]
	ys = pts[:,1]
	conf = pts[:,2]
	idx = np.where(conf>0)[0]
	xs = xs[idx]
	ys = ys[idx]
	xmin, xmax = xs.min(), xs.max()
	ymin, ymax = ys.min(), ys.max()
	return xmin, xmax, ymin, ymax

def crop_norm(img, pts, augment=True):
	# TODO: add random scale and random shift while transforming 
	xmin, xmax, ymin, ymax = get_minmax(pts)
	
	wh = max(ymax - ymin, xmax - xmin)
	scale = 256 / wh * (np.random.random() * 0.3 + 0.7)

	img = cv2.resize(img, None, fx=scale, fy=scale)
	pts[:,:2] = pts[:,:2] * scale

	xmin, xmax, ymin, ymax = get_minmax(pts)
	center = [0.5 * (xmax + xmin), 0.5 * (ymin + ymax)]
	xmin = center[0] - 128 
	if augment: xmin = xmin - np.random.random() * 80 + 40
	ymin = center[1] - 128
	if augment: ymin = ymin - np.random.random() * 80 + 40

	H = np.float32([[1,0,-xmin], [0,1,-ymin]])
	img = cv2.warpAffine(img, H, (256,256))
	pts = pts - np.float32([xmin, ymin, 0])
	return img, pts 

def hmap_to_match(hmap):
	# TODO: choose certain index
	idx = [1,2,3,4,5,6,11,12,13,14,15,16]
	idx = np.int32(idx)
	result = hmap[:,:,idx]
	return result 

def random_noise(hmap):
	noise = np.random.random(hmap.shape) * 30 - 15
	hmap += noise
	return hmap

def random_mask(hmap, thresh=0.25):
	domask = np.random.random()
	if domask<0.5:
		mask = np.random.random([12])
		mask[mask<thresh] = 0
		mask[mask>0] = 1 
		hmap *= mask

def augment_pts(pts):
	pts = pts.copy()
	random_shift = np.random.random(pts.shape) * 20 - 10
	pts[:,:2] += random_shift[:,:2]
	return pts 

def process(sample):
	# add more process here
	img, pts = sample
	is_centered = np.random.random()
	img = cv2.imread('./images/' + img)
	if is_centered<0.75:
		img, pts = crop_norm(img, pts)
		hmap_match = hmap_to_match(get_hmap(augment_pts(pts))) * 256
		random_mask(random_noise(hmap_match)) 
		hmap = get_hmap(pts) * 256
	else:
		img, pts = crop_norm(img, pts, augment=False)
		hmap_match = hmap_to_match(get_hmap(None)) * 256
		random_noise(hmap_match)
		hmap = get_hmap(pts) * 256
	return img, hmap_match, hmap

def post_process(inp):
	res = list(zip(*inp))
	imgs = np.float32(res[0])
	hmap_match = np.float32(res[1])
	hmap = np.float32(res[2])
	res = [imgs, hmap_match, hmap]
	res = [np.transpose(i, (0,3,1,2)) for i in res]
	return res 

def get_data(listfile):
	print('Reading pickle file...')
	data = pickle.load(open(listfile,'rb'))
	return data

def get_datareader(bsize, processes):
	reader = SUL.DataReader.DataReader(bsize, processes=processes, gpus=1, sample_policy='RANDOM')
	reader.set_data(get_data('mpii_3pts.pkl'))
	reader.set_process_fn(process)
	reader.set_post_process_fn(post_process)
	reader.prefetch()
	return reader

if __name__=='__main__':
	# img = cv2.imread('./images/005808361.jpg')
	# data = get_data('mpii_3pts.pkl')
	# for i in data:
	# 	if i[0]=='005808361.jpg':
	# 		pts = i[1]
	# 		break 

	# img, pts = crop_norm(img, pts)
	# hmap = get_hmap(pts)

	# import matplotlib.pyplot as plt 
	
	# hmap = np.amax(hmap, axis=-1)
	# plt.figure()
	# plt.imshow(hmap, cmap='jet')
	# plt.figure()
	# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

	# for i in range(len(pts)):
	# 	plt.plot(pts[i,0], pts[i,1], 'o')
	# plt.show()
	reader = get_datareader(1, 1)
	batch = reader.get_next()
	print(batch[0].shape, batch[1].shape, batch[2].shape)

	import matplotlib.pyplot as plt 
	img = batch[0][0]
	img = np.transpose(img, (1,2,0))
	img = np.uint8(img)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	

	hmapm = np.amax(batch[1][0], axis=0)
	plt.figure()
	plt.imshow(hmapm, cmap='jet')
	print(hmapm.max(), hmapm.min())

	hmap = np.amax(batch[2][0], axis=0)
	plt.figure()
	plt.imshow(hmap, cmap='jet')
	print(hmap.max(), hmap.min())
	plt.show()
