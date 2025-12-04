import numpy as np

# IoU and mIoU metrics

def iou(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return None
    return intersection / union
