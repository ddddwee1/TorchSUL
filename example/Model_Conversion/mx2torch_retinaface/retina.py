import numpy as np 
import torch
from TorchSUL import Model as M 
import mnet 
import cv2 
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import cpu_nms_wrapper
from rcnn.processing.bbox_transform import clip_boxes

class RetinaFace():
	def __init__(self, modelpath, use_gpu, nms=0.4):
		self.use_gpu = use_gpu
		model = mnet.Detector()
		model = model.eval()
		x = torch.from_numpy(np.ones([1,3,640,640]).astype(np.float32))
		_ = model(x)
		M.Saver(model).restore('./model/')
		if self.use_gpu:
			model.cuda()
		self.model = model
		self.generate_anchors()
		self.nms = cpu_nms_wrapper(nms)

	def generate_anchors(self):
		_ratio = (1.,)
		anchor_cfg = {
			'32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
			'16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
			'8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
		}
		self.fpn_keys = [32,16,8]
		self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=False, cfg=anchor_cfg)))
		self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

	def pad_img_to_32(self, img):
		h,w = img.shape[:2]
		if h%32==0:
			canvas_h = h 
		else:
			canvas_h = (h//32+1)*32
		if w%32==0:
			canvas_w = w 
		else:
			canvas_w = (w//32+1)*32
		canvas = np.zeros([canvas_h, canvas_w, 3], dtype=np.uint8)
		canvas[:h, :w] = img 
		return canvas

	def preprocess(self, img):
		img = np.float32(img)
		img = img[:,:,::-1].copy()
		img = img[None, ...]
		img = np.transpose(img, [0,3,1,2])
		return img 

	def _clip_pad(self, x, shape):
		H,W = x.shape[2:]
		h,w = shape
		if h<H or w<W:
			x = x[:,:,:h,:w].copy()
		return x 

	@staticmethod
	def bbox_pred(boxes, box_deltas):
		"""
		Transform the set of class-agnostic boxes into class-specific boxes
		by applying the predicted offsets (box_deltas)
		:param boxes: !important [N 4]
		:param box_deltas: [N, 4 * num_classes]
		:return: [N 4 * num_classes]
		"""
		if boxes.shape[0] == 0:
			return np.zeros((0, box_deltas.shape[1]))

		boxes = boxes.astype(np.float, copy=False)
		widths = boxes[:, 2] - boxes[:, 0] + 1.0
		heights = boxes[:, 3] - boxes[:, 1] + 1.0
		ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
		ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

		dx = box_deltas[:, 0:1]
		dy = box_deltas[:, 1:2]
		dw = box_deltas[:, 2:3]
		dh = box_deltas[:, 3:4]

		pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
		pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
		pred_w = np.exp(dw) * widths[:, np.newaxis]
		pred_h = np.exp(dh) * heights[:, np.newaxis]

		pred_boxes = np.zeros(box_deltas.shape)
		# x1
		pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
		# y1
		pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
		# x2
		pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
		# y2
		pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

		if box_deltas.shape[1]>4:
			pred_boxes[:,4:] = box_deltas[:,4:]

		return pred_boxes

	@staticmethod
	def landmark_pred(boxes, landmark_deltas):
		if boxes.shape[0] == 0:
			return np.zeros((0, landmark_deltas.shape[1]))
		boxes = boxes.astype(np.float, copy=False)
		widths = boxes[:, 2] - boxes[:, 0] + 1.0
		heights = boxes[:, 3] - boxes[:, 1] + 1.0
		ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
		ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
		pred = landmark_deltas.copy()
		for i in range(5):
			pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
			pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
		return pred

	def detect(self, img, thresh, scales=[1.0], do_flip=False):
		proposal_list = []
		scores_list = []
		landmarks_list = []
		flips = [0,1] if do_flip else [0]
		for im_scale in scales:
			for flip in flips:
				if im_scale!=1.0:
					img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
				else:
					img = img.copy()
				if flip:
					img = img[:,::-1,:]

				# img = self.pad_img_to_32(img)
				imgshape = [img.shape[0], img.shape[1]]
				img = self.preprocess(img)
				img = torch.from_numpy(img)

				if self.use_gpu:
					img = img.cuda()

				net_out = self.model(img)

				for _idx,s in enumerate(self.fpn_keys):
					idx = _idx * 3
					scores = net_out[idx].detach().cpu().numpy()
					scores = scores[:, self._num_anchors[s]:]
					

					idx += 1
					bbox_deltas = net_out[idx].detach().cpu().numpy()

					h, w = bbox_deltas.shape[2], bbox_deltas.shape[3]

					A = self._num_anchors[s]
					K = h*w
					anchors_fpn = self._anchors_fpn[s]
					anchors_fpn = np.float32(anchors_fpn)
					anchors = anchors_plane(h, w, s, anchors_fpn)
					anchors = anchors.reshape((K*A, 4))

					scores = self._clip_pad(scores, (h, w))
					scores = scores.transpose([0,2,3,1]).reshape([-1,1])
					# print('SCR')
					# print(scores)
					# print(scores.shape)
					# input()

					bbox_deltas = self._clip_pad(bbox_deltas, (h,w))
					bbox_deltas = bbox_deltas.transpose([0,2,3,1])
					bbox_pred_len = bbox_deltas.shape[3]//A
					bbox_deltas = bbox_deltas.reshape([-1, bbox_pred_len])

					proposals = self.bbox_pred(anchors, bbox_deltas)
					proposals = clip_boxes(proposals, imgshape)
					

					scores_ravel = scores.ravel()
					order = np.where(scores_ravel>=thresh)[0]

					proposals = proposals[order]
					scores = scores[order]

					if flip:
						oldx1 = proposals[:, 0].copy()
						oldx2 = proposals[:, 2].copy()
						proposals[:, 0] = im.shape[1] - oldx2 - 1
						proposals[:, 2] = im.shape[1] - oldx1 - 1

					proposals[:,:4] /= im_scale
					# print('proposals')
					# print(proposals)
					# print(proposals.shape)
					# input()

					proposal_list.append(proposals)
					scores_list.append(scores)

					# landmarks 
					idx += 1 
					landmark_deltas = net_out[idx].detach().cpu().numpy()
					landmark_deltas = self._clip_pad(landmark_deltas, (h,w))
					landmark_pred_len = landmark_deltas.shape[1]//A 
					landmark_deltas = landmark_deltas.transpose((0,2,3,1)).reshape([-1,5,landmark_pred_len//5])
					landmarks = self.landmark_pred(anchors, landmark_deltas)
					landmarks = landmarks[order, :]

					if flip:
						landmarks[:,:,0] = imgshape[1] - landmarks[:,:,0] - 1 
						order = [1,0,2,4,3]
						flandmarks = landmarks[:,np.int32(order)]

					landmarks[:,:,:2] /= im_scale
					landmarks_list.append(landmarks)

		# print('PROPOSAL', proposal_list)
		proposals = np.vstack(proposal_list)
		landmarks = None 
		if proposals.shape[0]==0:
			return np.zeros([0,5]), np.zeros([0,5,2])
		scores = np.vstack(scores_list)
		scores_ravel = scores.ravel()
		order = scores_ravel.argsort()[::-1]
		proposals = proposals[order]
		scores = scores[order]
		landmarks = np.vstack(landmarks_list)
		landmarks = np.float32(landmarks[order])

		pre_det = np.hstack([proposals[:, 0:4], scores])
		pre_det = np.float32(pre_det)

		keep = self.nms(pre_det)
		det = np.hstack([pre_det, proposals[:,4:]])
		det = det[keep]
		landmarks = landmarks[keep]

		return det, landmarks
