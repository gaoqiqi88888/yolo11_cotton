import torch
import torch.nn.functional as F
import math
import ultralytics.utils.loss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

# Original BboxLoss for reference/inheritance
OriginalBboxLoss = ultralytics.utils.loss.BboxLoss

class SDPBboxLoss(OriginalBboxLoss):
    def __init__(self, reg_max, alpha=1.9, delta=3.0):
        super().__init__(reg_max)
        self.alpha = alpha
        self.delta = delta
        # Initialize running mean of IoU loss
        self.iou_mean = 1.0 
        self.momentum = 0.937

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # Calculate basic IoU
        b1 = pred_bboxes[fg_mask]
        b2 = target_bboxes[fg_mask]
        iou = bbox_iou(b1, b2, xywh=False, CIoU=False)
        
        # WIoU calculations
        # 1. Calculate R (distance penalty)
        b1_x1, b1_y1, b1_x2, b1_y2 = b1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = b2.chunk(4, -1)
        
        # Convex hull (smallest enclosing box)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw.pow(2) + ch.pow(2) + 1e-7
        
        # Center distance
        center_x1 = (b1_x1 + b1_x2) / 2
        center_y1 = (b1_y1 + b1_y2) / 2
        center_x2 = (b2_x1 + b2_x2) / 2
        center_y2 = (b2_y1 + b2_y2) / 2
        rho2 = (center_x1 - center_x2).pow(2) + (center_y1 - center_y2).pow(2)
        
        # WIoU v1 term: R = exp(rho2 / c2)
        # R is the distance penalty.
        R = torch.exp(rho2 / c2)
        
        # Basic IoU loss (L_IoU)
        loss_iou_basic = 1.0 - iou
        
        # Update moving average of L_IoU
        loss_iou_detach = loss_iou_basic.detach()
        self.iou_mean = self.iou_mean * self.momentum + loss_iou_detach.mean() * (1 - self.momentum)
        
        # WIoU v3 terms
        # beta = L_IoU* / mean(L_IoU)
        beta = loss_iou_detach / (self.iou_mean + 1e-7)
        
        # r = beta / (delta * alpha^(beta - delta))
        # r is the gradient scaler (focusing coefficient)
        r = beta / (self.delta * torch.pow(self.alpha, beta - self.delta))
        
        # Final WIoU v3 loss: L = r * R * L_IoU
        loss_iou = (r * R * loss_iou_basic * weight).sum() / target_scores_sum

        # DFL loss (standard code from BboxLoss)
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

def patch_loss():
    print("Monkey-patching ultralytics.utils.loss.BboxLoss with SDPBboxLoss (WIoU)")
    ultralytics.utils.loss.BboxLoss = SDPBboxLoss


def restore_loss():
    print("Restoring ultralytics.utils.loss.BboxLoss to OriginalBboxLoss")
    ultralytics.utils.loss.BboxLoss = OriginalBboxLoss
