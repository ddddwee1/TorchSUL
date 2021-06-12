import json 
import pickle 
import glob
from tqdm import tqdm 
import numpy as np 

for i in tqdm(glob.glob('./masks2/*.pkl')):
	mask = pickle.load(open(i, 'rb'))
	# print(mask.dtype)
	mask[mask>0] = 1
	mask[mask<1] = 0
	mask = mask.astype(np.uint8)
	fname = i.split('/')[-1]
	pickle.dump(mask, open('./masks/%s'%fname, 'wb'))
