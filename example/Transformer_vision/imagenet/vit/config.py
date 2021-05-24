import numpy as np 

# size
inp_size = 256 

# augmentation 
rotation = 30
min_scale = 0.7
max_scale = 1.5
max_translate = 50

blur_prob = 0.0
blur_size = [7, 11, 15, 21]
blur_type = ['vertical','horizontal','mean']

# extra 
distributed = True
data_root = '/home/ddwe_cy/imagenet/imagenet1k/'
