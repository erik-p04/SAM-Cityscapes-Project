import numpy as np
import torch
# IoU and mIoU metrics

def iou(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return None
    return intersection / union

def compute_iou(preds, targets, num_classes):
    ious = []

    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return sum(ious) / len(ious)
