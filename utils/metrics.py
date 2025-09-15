# utils/metrics.py
import numpy as np
import torch

EPS = 1e-8

# ---------------- detection helpers ----------------
def iou(boxA, boxB):
    """
    Compute IoU between two boxes in [x1,y1,x2,y2] format.
    boxA, boxB: 1D arrays or tensors of length 4
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

    union = boxAArea + boxBArea - interArea + EPS
    return interArea / union

def precision_recall(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """
    Greedy matching (score-desc): compute TP, FP, FN then precision & recall.
    pred_boxes: Nx4 array (list or numpy), pred_scores: N, gt_boxes: Mx4
    """
    if len(pred_boxes) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        return prec, rec, tp, fp, fn

    pred_idx = np.argsort(-np.array(pred_scores))  # descending
    matched_gt = set()
    tp = 0
    fp = 0

    for i in pred_idx:
        pb = pred_boxes[i]
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt: continue
            cur_iou = iou(pb, gb)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_j = j
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    return float(prec), float(rec), int(tp), int(fp), int(fn)

def average_iou(pred_boxes, gt_boxes):
    """Compute average IoU across matched pairs (greedy match by IoU)."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    matched_gt = set()
    ious = []
    for pb in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt: continue
            cur_iou = iou(pb, gb)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_j = j
        if best_j >= 0:
            matched_gt.add(best_j)
            ious.append(best_iou)
    return float(np.mean(ious)) if len(ious) > 0 else 0.0

# ---------------- segmentation helpers ----------------
def boxes_to_mask(boxes, image_size):
    """
    Convert boxes (Nx4) to a binary mask of shape (H, W).
    boxes are in [x1,y1,x2,y2] float coordinates. image_size = (H,W)
    """
    H, W = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1 = max(0, min(W-1, x1))
        x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1))
        y2 = max(0, min(H-1, y2))
        if x2 >= x1 and y2 >= y1:
            mask[y1:y2+1, x1:x2+1] = 1
    return mask

def instance_to_mask(instance_np):
    """
    Combine instance mask (H,W) where each object has unique integer id >0
    into a binary mask (H,W) where foreground=1 else 0.
    """
    return (instance_np != 0).astype(np.uint8)

def mae(pred_mask, gt_mask):
    """Mean absolute error between predicted and GT masks (0..1)."""
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)
    return float(np.mean(np.abs(pred - gt)))

def s_alpha(pred, gt, alpha=0.5):
    """
    Structure measure Sα (S_alpha). Implementation is simplified:
    - Treats global mean and object-aware components as in some S-measure definitions.
    This is a simplified but commonly used variant.
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    fg = (pred * gt).sum() / (gt.sum() + EPS)
    bg = ((1 - pred) * (1 - gt)).sum() / ((1 - gt).sum() + EPS)
    return float(alpha * fg + (1 - alpha) * bg)

def e_phi(pred, gt, eps=1e-8):
    """
    Enhanced alignment measure E_phi. Use simplified version:
    - Binarize pred at threshold 0.5 and compute alignment map.
    """
    predf = pred.astype(np.float32)
    gtf = (gt > 0.5).astype(np.float32)
    th = min(2 * predf.mean(), 1.0)
    pb = (predf >= th).astype(np.float32)
    fg = gtf.sum()
    size = gtf.size
    if fg == 0:
        return float(1.0 - pb.mean())
    if fg == size:
        return float(pb.mean())
    pm = pb.mean()
    gm = gtf.mean()
    align = 2 * (pb - pm) * (gtf - gm) / (((pb - pm) ** 2 + (gtf - gm) ** 2) + eps)
    enhanced = ((align + 1) ** 2) / 4
    return float(enhanced.mean())

def fbeta_weighted(pred, gt, beta2=1.0):
    """
    F_beta weighted (Fβw). We compute precision/recall on binary masks.
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    tp = (pred * gt).sum()
    prec = tp / (pred.sum() + EPS)
    rec = tp / (gt.sum() + EPS)
    f = (1 + beta2) * prec * rec / (beta2 * prec + rec + EPS)
    return float(f)
