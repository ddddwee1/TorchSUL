import numpy as np 

# density map 
density_level = 4
density_num_layers = 1
density_channels = 32

# inst_seg map 
inst_num_layers = 4
inst_channels = 512
inst_feat_dim = 64

# affinity matrix 
affinity_sigma = 0.5
affinity_threshold = 0.1 

# gcn 
gcn_layers = 3
gcn_channels = 256
id_featdim = 1

# headnet 
head_layernum = 1
head_chn = 32

# upsmaple 
upsample_layers = 1
upsample_chn = 32

# size
inp_size = 512 
out_size = 256
base_sigma = 4.0
num_pts = 14

# augmentation 
rotation = 30
min_scale = 1
max_scale = 1
max_translate = 70

blur_prob = 0.0
blur_size = [7, 11, 15, 21]
blur_type = ['vertical','horizontal','mean']

# training 
data_root = '/data/pose3d/muco/'
max_iter = 50000
max_epoch = 100
init_lr = 0.001
iter_density_pretrain = 50
decay = 0.0001
momentum = 0.9
lr_step = 35000
lr_epoch = [70,90]
save_interval = 1

COCO_index = np.int64([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
COCO_reorder = np.int64([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

MPII_index = np.int64([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
MPII_reorder = np.int64([16,14,12,11,13,15,17,18,19,20,10,8,6,5,7,9])

PT_index = np.int64([0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21])
PT_reorder = np.int64([0,21,20,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

# extra 
max_inst = 30
distributed = True
depth_scaling = 2500
depth_mean = 1.45
rel_depth_scaling = 1000
tag_thresh = 1.0
tag_distance = 1.5
hmap_thresh = 0.1
nms_kernel = 5 
