import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

def parse_kpts(buf):
	res = np.zeros([17,3]).astype(np.float32)
	res[0] = buf[6]
	res[1] = buf[2]
	res[2] = buf[1]
	res[3] = buf[0]
	res[4] = buf[3]
	res[5] = buf[4]
	res[6] = buf[5]
	res[7] = 0.5 * (buf[7] + buf[6])
	res[8] = buf[7]
	res[9] = 0.5 * (buf[9] + buf[8])
	res[10] = buf[9]
	res[11] = buf[13]
	res[12] = buf[14]
	res[13] = buf[15]
	res[14] = buf[12]
	res[15] = buf[11]
	res[16] = buf[10]
	res[7,2] = min(buf[7,2], buf[6,2])
	res[9,2] = min(buf[8,2], buf[9,2])
	return res 

data = pickle.load(open('MPII_kpts.pkl', 'rb'))
results = []
for i in data:
	pts = parse_kpts(i[1])
	results.append([i[0], pts])

pickle.dump(results, open('mpii.pkl', 'wb'))
