import numpy as np 
import pickle 

data = pickle.load(open('mpii.pkl', 'rb')) 

res = []
for i in data:
	img = i[0]
	pts = i[1]
	idx = np.where(pts[:,2]>0)[0]
	if len(idx) > 3:
		res.append(i)

pickle.dump(res, open('mpii_3pts.pkl', 'wb'))
