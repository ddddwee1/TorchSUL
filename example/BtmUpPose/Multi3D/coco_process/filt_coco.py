import cv2 
import numpy as np 
import config 
import pickle 
from multiprocessing.pool import ThreadPool
import random 
import pycocotools._mask
import torch 

class RandomAffineTransform(object):
	# borrow from higher-hrnet
    def __init__(self,
                 input_size,
                 output_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 scale_type,
                 max_translate,
                 scale_aware_sigma=False):
        self.input_size = input_size
        self.output_size = output_size 

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate
        self.scale_aware_sigma = scale_aware_sigma

    def _get_affine_matrix(self, center, scale, res, rot=0):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def __call__(self, image, mask, joints, boxes):
        # assert isinstance(mask, list)
        # assert isinstance(joints, list)
        # assert len(mask) == len(joints)
        # assert len(mask) == len(self.output_size)

        height, width = image.shape[:2]

        center = np.array((width/2, height/2))
        if self.scale_type == 'long':
            scale = max(height, width)/200
        elif self.scale_type == 'short':
            scale = min(height, width)/200
        else:
            raise ValueError('Unkonw scale type: {}'.format(self.scale_type))
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale
        scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.max_translate > 0:
            dx = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            dy = np.random.randint(
                -self.max_translate*scale, self.max_translate*scale)
            center[0] += dx
            center[1] += dy

        mat_output = self._get_affine_matrix(center, scale, (self.output_size, self.output_size), aug_rot)[:2]
        # mask_out = np.zeros([self.output_size, self.output_size, mask.shape[-1]], dtype=np.uint8)
        # for i in range(mask.shape[-1]):
           #  mask_out[:,:,i] = cv2.warpAffine((mask[:,:,i]*255).astype(np.uint8), mat_output, (self.output_size, self.output_size)) 
        # print('L95',mask.shape)
        mask_out = cv2.warpAffine((mask*255).astype(np.uint8), mat_output, (self.output_size, self.output_size)) 
        # print('L97', mask_out.shape)
        mask_out = np.float32(mask_out) / 255
        mask_out = (mask_out > 0.5).astype(np.float32)

        joints[:, :, 0:2] = self._affine_joints(joints[:, :, 0:2], mat_output)
        if boxes is None:
        	whs = None
        else:
	        whs = boxes[:,2:] * (float(self.output_size) / (200 * scale))

        if self.scale_aware_sigma:
            joints[:, :, 3] = joints[:, :, 3] / aug_scale

        mat_input = self._get_affine_matrix(center, scale, (self.input_size, self.input_size), aug_rot)[:2]
        image = cv2.warpAffine(image, mat_input, (self.input_size, self.input_size))

        return image, mask_out, joints, whs

class DataReader():
	def __init__(self, bsize):
		self.pos = 0
		self.bsize = bsize
		self.aff = RandomAffineTransform(config.inp_size, config.out_size, config.rotation, config.min_scale, config.max_scale, 'short', config.max_translate, True)
		# load coco 
		coco_pts_data = pickle.load(open('imagewise_kpts.pkl' , 'rb'))
		self.seg_data = pickle.load(open('imagewise_segments.pkl', 'rb'))
		print('COCO images:', len(coco_pts_data))

		# combine
		self.pts_data = coco_pts_data
		self.img_indices = list(coco_pts_data.keys())
		print('Index pool:', len(self.img_indices))

	def _pre_process(self, img):
		img = np.float32(img)
		img = img / 255 
		img = img[:,:,::-1]
		img = img - np.float32([0.485, 0.456, 0.406])
		img = img / np.float32([0.229, 0.224, 0.225])

		img = np.transpose(img[None,...], [0,3,1,2])
		# img = torch.from_numpy(img)
		return img 

	def deprocess(self, img):
		img = np.transpose(img, [0, 2,3, 1])
		img = img * np.float32([0.229, 0.224, 0.225])
		img = img + np.float32([0.485, 0.456, 0.406])
		# img = img[:,:,:,::-1]
		img = img * 255 
		img = np.uint8(img[0])
		return img 

	def _generate_mask(self, seg, h, w):
		rle = pycocotools._mask.frPyObjects(seg, h, w)
		if not isinstance(rle, list):
			rle = [rle]
		m = pycocotools._mask.decode(rle)
		m = np.amax(m, axis=-1)
		return m 

	def _get_pts(self, pts_list, reorder):
		# ptsitem: [pts, box]
		res = []
		boxes = []
		for item in pts_list:
			if isinstance(item, list):
				pts = item[0]
				box = item[1]
			else:
				pts = item
				box = None
			# area = max(box[2] , box[3])
			# sigma = config.base_sigma * area / config.out_size
			# box size cannot reflect the person's scale, so dynamic sigma using box size is not appropriate 

			buff = np.zeros([config.num_pts,4], dtype=np.float32)
			buff[reorder,:3] = pts[:]
			# buff[:,3] = sigma
			# b = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
			res.append(buff)
			boxes.append(box)
		res = np.float32(res)
		if boxes[0] is None:
			boxes = None
		else:
			boxes = np.float32(boxes)
		return res, boxes

	def _get_density_map(self, segs, whs):
		assert segs.shape[-1] == len(whs)
		segs = segs.copy()
		segs = np.float32(segs)
		for i in range(len(whs)):
			longside = max(whs[i,0], whs[i,1])
			density = longside / config.out_size
			# print(density)
			density = np.floor(-np.log2(density))
			# print(density)
			density = np.clip(density, 1, config.density_level-1)
			# print(density)
			segs[:,:,i] *= density

			# print('lala', segs[])
		segs = np.amax(segs, axis=-1)
		return segs

	def get_next(self):
		# print('Fetch')
		if self.pos==len(self.img_indices):
			return None

		idx = self.img_indices[self.pos]
		print('\r %d / %d'%(self.pos, len(self.img_indices)), end='')
		if isinstance(idx, int):
			img = cv2.imread('./train2017/%012d.jpg'%(idx))
			h, w = img.shape[0], img.shape[1]
			#我日有两种mask
			mask = pickle.load(open('./masks/%012d.pkl'%(idx), 'rb'))
			segs = self.seg_data[idx]
			segs = [self._generate_mask(s, h, w) for s in segs]
			segs.append(mask)
			segs = np.stack(segs, axis=-1)

			pts = self.pts_data[idx]
			pts, boxes = self._get_pts(pts, config.COCO_reorder)

			# print(boxes)
			img, segs, pts, whs = self.aff(img, segs, pts, boxes)
			# print(whs)
			mask = segs[:,:,-1]
			segs = segs[:,:,:-1]

			density_map = self._get_density_map(segs, whs)

			img = self._pre_process(img)
			ptstype = 0
			self.pos += 1 

		return img, pts, density_map, mask, ptstype, idx

if __name__=='__main__':
	from tqdm import tqdm 
	
	reader = DataReader(1)
	invalid_idx = []
	while 1:
		batch = reader.get_next()
		if batch is None:
			break 
		pts = batch[1]
		if pts[:,:,2].sum() < 3.1:
			invalid_idx.append(batch[-1])
			print('No points: %d \n'%(len(invalid_idx)) )
	pickle.dump(invalid_idx, open('coco_no_kpts.pkl', 'wb'))

	data_imgwise = pickle.load(open('imagewise_kpts.pkl' , 'rb'))
	for inv in tqdm(invalid_idx):
		data_imgwise.pop(inv)
	pickle.dump(data_imgwise, open('filtered_coco_kpts.pkl', 'wb'))
