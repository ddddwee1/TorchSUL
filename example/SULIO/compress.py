import sulio 
import glob 
from tqdm import tqdm 

ioout = sulio.ThreadedDataFrame('./pkl0430/', 'w', debug=True)

folders = glob.glob('./pickles_0430/*')
for fold in tqdm(folders):
	files = glob.glob(fold + '/*.*')
	for fname in files:
		f = open(fname, 'rb')
		f = f.read()
		ioout.write(f , metadata=fname)

ioout.close()
