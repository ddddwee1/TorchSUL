import scipy.io as sio 
import numpy as np 
import pickle 
from tqdm import tqdm 

idmap = {}

def parse_vis(v):
	res = []
	for i in v:
		if isinstance(i, int):
			res.append(i)
		else:
			res.append(2)
	return np.int32(res)

def parse_inst(pt2):
	if pt2.dtype.names is None:
		return None
	else:
		pt = pt2['point'].item()
		x = np.float32(pt['x'])
		y = np.float32(pt['y'])
		ids = np.int32(pt['id'])
		if 'is_visible' in pt.dtype.names:
			vis = pt['is_visible']
			vis = parse_vis(vis)
		else:
			vis = None
		try:
			xy = np.stack([x,y], axis=1)
		except:
			return None
		# print(xy.shape)
		xyv = remap_pts([xy,ids,vis])
		return xyv

def remap_pts(bundle):
	xy, ids, vis = bundle
	res = np.zeros([16,3],dtype=np.float32)
	res[:,2] = -2
	for i in range(len(ids)):
		idx = ids[i]
		res[idx, :2] = xy[i]
		res[idx, 2] = vis[i]
	return res 


dt = sio.loadmat('mpii_human_pose_v1_u12_1', squeeze_me=True)
print('Loaded.')

dt = dt['RELEASE']
dt = dt['annolist']
dt = dt.item()

imglist = dt['image']
pts = dt['annorect']


results = []

for i in tqdm(range(len(pts)), ascii=True):
	# print('I', i)
	imgname = imglist[i].item()[0]
	annot = pts[i]
	if annot is not None:
		# print(annot.dtype)
		names = annot.dtype.names
		if i==24730:
			print(annot)
		if names is not None:
			if 'annopoints' in names:
				annot = annot['annopoints']
				# print(annot.size)
				if annot.size==1:
					res = parse_inst(annot.item())
					if res is not None:
						results.append([imgname, res])
				else:
					for j in range(annot.size):
						res = parse_inst(annot[j])
						if res is not None:
							results.append([imgname, res])


pickle.dump(results, open('MPII_kpts.pkl', 'wb'))
