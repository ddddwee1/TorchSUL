import numpy as np 

kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
oks_vars = (kpt_oks_sigmas * 2)**2

def compute_area(pts):
	x = pts[:,0]	
	y = pts[:,1]
	x1,x2 = x.min(), x.max()
	y1,y2 = y.min(), y.max()
	area = (x2-x1) * (y2-y1)
	return area

def compute_oks(p1, p2):
	# use box confidence to select the area 
	# conf = p1[:,2] * p2[:,2]
	conf = np.float32(p1[:,2]>0.1) * np.float32(p2[:,2]>0.1)
	# print(conf)
	x1 = p1[:,0]
	y1 = p1[:,1]
	x2 = p2[:,0]
	y2 = p2[:,1]
	dx = x1 - x2 
	dy = y1 - y2
	area = max(compute_area(p1), compute_area(p2))
	e = (dx**2 + dy**2) / oks_vars / (area+np.spacing(1)) / 2
	e = e[conf>0]
	if e.shape[0]==0:
		return 0.0
	iou = np.sum(np.exp(-e)) / e.shape[0]
	# print(iou)
	return iou

def nms(pts, scores_list):
	if len(scores_list)==0:
		return [], []
	explored = []
	scores = np.array(scores_list)
	groups = []
	while 1:
		idx = np.argmax(scores)
		gp = [idx,]
		explored.append(idx)
		scores[idx] = -9999
		for i in range(len(scores)):
			if i in explored:
				continue
			oks = compute_oks(pts[idx], pts[i])
			if oks>0.5:
				gp.append(i)
				explored.append(i)
				scores[i] = -9999
		groups.append(gp)
		if len(explored)==len(scores):
			break 
	# process the groups 
	# the first item should be the most confident one 
	# method 1: try only the most confident one 
	result_final = []
	scores_final = []
	for gp in groups:
		result_final.append(pts[gp[0]])
		scores_final.append(scores_list[gp[0]])
	return result_final, scores_final

def crop_images(img, ratio):
	h,w = img.shape[:2]
	h2,w2 = h//ratio, w//ratio
	hw = max(h2,w2)
	imgs = []
	metas = []
	for i in range(ratio):
		for j in range(ratio):
			h1 = hw*i 
			h2 = hw*i + hw 
			if h2>h:
				h1 = h-hw
				h2 = h  

			w1 = hw*j
			w2 = hw*j+hw 
			if w2>w:
				w1 = w-hw 
				w2 = w 
			piece = img[h1:h2, w1:w2]
			imgs.append(piece)
			metas.append([h1, w1])
	return imgs, metas

def restore_pts(pts, meta):
	result = []
	h1,w1 = meta
	for pt in pts:
		res = pt.copy()
		res[:,0] += w1 
		res[:,1] += h1 
		result.append(res)
	return result
