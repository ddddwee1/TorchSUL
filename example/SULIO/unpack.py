import sulio 
import os 
from tqdm import tqdm

ioin = sulio.ThreadedDataFrame('./pkl0430/', 'r', debug=True)

for i in tqdm(range(len(ioin))):
	data, fname = ioin.read(i)
	fdir, _ = os.path.split(fname)
	if not os.path.exists(fdir):
		os.makedirs(fdir)
	fout = open(fname, 'wb')
	fout.write(data)
	fout.close()
