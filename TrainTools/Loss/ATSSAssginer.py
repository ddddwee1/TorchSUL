import torch 

from torch import Tensor
from ..Util import BBOX


class ATSSAssigner():
    # https://github.com/sfzhang15/ATSS/blob/79dfb28bd18c931dd75a3ca2c63d32f5e4b1626a/atss_core/modeling/rpn/atss/loss.py#L131C58-L131C58
    def __init__(self, topk: int):
        self.topk = topk

    def _filter_level(self, ious: Tensor, det_idxs: Tensor, valid: Tensor):
        # if any element appears twice in det_idxs, it must be assigned to two GTs
        det_idx_sorted, ordi = det_idxs.flatten().sort()
        ious_sorted = ious.flatten()[ordi]
        valid_view = valid.view(-1)
        for i in range(len(det_idx_sorted)-1):
            if not valid_view[ordi[i]]:
                continue 
            n_total = 1
            iou_max = ious_sorted[i]
            idx_max = i
            for j in range(i+1, len(det_idx_sorted)):
                if det_idx_sorted[i] != det_idx_sorted[j]:
                    break 
                if valid_view[ordi[j]]:
                    if ious_sorted[j] > iou_max:
                        idx_max = j
                        iou_max = ious_sorted[j]
                    n_total += 1 
            if n_total==1:
                continue 
            for k in range(i, i+n_total):
                if k==idx_max:
                    valid_view[ordi[k]] = True 
                else:
                    valid_view[ordi[k]] = False

    def assign(self, assignment: Tensor, box_idx: Tensor) -> tuple[Tensor, Tensor]:
        idx_src, idx_gt = torch.where(assignment)
        idx_box = box_idx[idx_src, idx_gt]
        return idx_box, idx_gt

    def __call__(self, pred_boxes: list[BBOX], gt_boxes: BBOX) -> list[tuple[Tensor, Tensor]]:
        # gt_boxes = gt_boxes_in.convert('xcycwh')
        n_gt = len(gt_boxes)

        all_ious = []
        all_idxs = []
        all_boxes: list[BBOX] = []
        all_gts: list[BBOX] = []

        for pred_level in pred_boxes:
            # pred_level = pred_level.convert('xcycwh')
            
            # step 1,2: compute IOU, center distance 
            ious = pred_level ^ gt_boxes
            dist = pred_level | gt_boxes

            # step 3: select top k
            k = min(self.topk, dist.shape[0])
            _, top_idx = dist.topk(k, dim=0, largest=False)    # [k, N_label]
            label_idx = torch.arange(0, dist.shape[1]).unsqueeze(0).repeat(k, 1)
            
            iou_selected = ious[top_idx, label_idx]
            box_selected = pred_level[top_idx]    # [k*N_label, 4]
            gt_selected = gt_boxes[label_idx]

            all_ious.append(iou_selected)
            all_idxs.append(top_idx)
            all_boxes.append(box_selected)
            all_gts.append(gt_selected)
            
        # step 4: get iou threshold 
        # all_ious2 = [i.flatten() for i in all_ious]
        # all_ious2 = torch.cat(all_ious2)
        # thresh = all_ious2.mean() + all_ious2.std()
        all_ious2 = torch.cat(all_ious, dim=0)
        thresh = all_ious2.mean(0) + all_ious2.std(0)

        # step 5: thresholding boxes 
        valid_iou = [i>=thresh for i in all_ious]
        # step 6: limit the center to be inside target 
        # valid_inside_gt = [i.inside(gt).reshape(-1, n_gt)*valid for i,gt,valid in zip(all_boxes, all_gts, valid_iou)]
        valid_inside_gt = [i.inside(gt)*valid for i,gt,valid in zip(all_boxes, all_gts, valid_iou)]

        # step 7: each box should only be assigned to one gt. Otherwise, larger IOU should apply 
        [self._filter_level(iou, idx, valid) for iou,idx,valid in zip(all_ious, all_idxs, valid_inside_gt)]
        assignment = [self.assign(assign, idx) for idx,assign in zip(all_idxs, valid_inside_gt)]
        return assignment

