# utils/engine.py
import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import (boxes_to_mask, instance_to_mask,
                           precision_recall, average_iou,
                           mae, s_alpha, e_phi, fbeta_weighted)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    running_loss = 0.0
    iters = 0
    start = time.time()
    for images, targets in tqdm(data_loader, desc=f"Train epoch {epoch}"):
        images = [img.to(device) for img in images]
        targets_device = []
        for t in targets:
            t_on = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    t_on[k] = v.to(device)
                else:
                    t_on[k] = v
            targets_device.append(t_on)

        loss_dict = model(images, targets_device)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        iters += 1
        if iters % print_freq == 0:
            avg = running_loss / max(1, iters)
            print(f"[Epoch {epoch}] Iter {iters} AvgLoss: {avg:.4f}")

    avg_loss = running_loss / max(1, iters)
    elapsed = time.time() - start
    return {"epoch": epoch, "loss": avg_loss, "time_s": elapsed}

def evaluate(model, data_loader, device, cfg=None):
    """
    Evaluate model on data_loader.
    Returns dictionary with aggregated detection metrics and segmentation metrics (if requested).
    cfg: optional dict with keys:
        - seg_metric_mode: 'gt_instance' or 'gt_object' (which GT to use)
        - seg_approx_from_boxes: True/False (if True and model only outputs boxes, compute masks by painting boxes)
        - score_threshold: float
        - iou_thresh: float (for detection precision/recall)
    """
    model.eval()
    total_images = 0
    # detection accumulators
    all_precisions = []
    all_recalls = []
    all_avgious = []
    total_detections = 0
    # segmentation accumulators
    seg_mae_list = []
    seg_salpha_list = []
    seg_ephi_list = []
    seg_fbw_list = []

    cfg = cfg or {}
    seg_mode = cfg.get("seg_metric_mode", "gt_instance")  # or 'gt_object' (GT_Object mask)
    seg_approx = cfg.get("seg_approx_from_boxes", True)  # if True, paint boxes to mask
    score_thr = cfg.get("score_threshold", 0.5)
    iou_thresh = cfg.get("iou_threshold", 0.5)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt, img in zip(outputs, targets, images):
                total_images += 1
                # Gather predictions
                pred_boxes = out.get("boxes", torch.zeros((0,4))).cpu().numpy()
                pred_scores = out.get("scores", torch.zeros((0,))).cpu().numpy()
                # Filter by score
                keep_idx = pred_scores >= score_thr
                pred_boxes = pred_boxes[keep_idx]
                pred_scores = pred_scores[keep_idx]

                # Ground truth boxes
                gt_boxes = tgt.get("boxes", torch.zeros((0,4))).cpu().numpy()

                # Detection metrics: precision & recall (greedy), average IoU of matched
                prec, rec, tp, fp, fn = precision_recall(pred_boxes, pred_scores, gt_boxes, iou_thresh=iou_thresh)
                all_precisions.append(prec)
                all_recalls.append(rec)
                avg_iou = average_iou(pred_boxes, gt_boxes)
                all_avgious.append(avg_iou)
                total_detections += len(pred_boxes)

                # Segmentation metrics
                if seg_mode in ("gt_instance", "gt_object"):
                    # construct GT binary mask
                    # target may not contain instance mask array here; dataset stores only boxes/labels.
                    # We'll attempt to read GT instance file path from target if available, else skip seg metrics.
                    # Assumes dataset includes "inst_path" in target (optional)
                    gt_mask_np = None
                    if "inst_path" in tgt:
                        # If dataset provides path
                        try:
                            from PIL import Image
                            im = Image.open(tgt["inst_path"]).convert("L")
                            gt_mask_np = np.array(im)
                            if seg_mode == "gt_object":
                                # convert to binary (foreground)
                                gt_mask_np = (gt_mask_np != 0).astype(np.uint8)
                            else:
                                gt_mask_np = (gt_mask_np != 0).astype(np.uint8)
                        except Exception:
                            gt_mask_np = None

                    if gt_mask_np is None:
                        # fallback: if targets included masks (rare) or boxes->mask
                        if "masks" in tgt:
                            # tgt["masks"] expected as tensor [N,H,W]
                            m = tgt["masks"].cpu().numpy()
                            if m.size > 0:
                                gt_mask_np = np.sum(m, axis=0)
                                gt_mask_np = (gt_mask_np > 0).astype(np.uint8)
                        else:
                            # We cannot compute true segmentation metrics if no mask is present.
                            gt_mask_np = None

                    if gt_mask_np is not None:
                        H, W = gt_mask_np.shape
                        if seg_approx:
                            # build predicted mask from predicted boxes
                            pred_mask_np = boxes_to_mask(pred_boxes, (H, W))
                        else:
                            # try to use predicted masks (Mask R-CNN outputs 'masks')
                            if "masks" in out:
                                pm = out["masks"].cpu().numpy()  # [N,1,H,W] or [N,H,W]
                                if pm.ndim == 4: pm = pm[:,0]
                                if pm.shape[1:] == gt_mask_np.shape:
                                    pred_mask_np = (np.sum(pm, axis=0) > 0.5).astype(np.uint8)
                                else:
                                    pred_mask_np = boxes_to_mask(pred_boxes, (H, W))
                            else:
                                pred_mask_np = boxes_to_mask(pred_boxes, (H, W))

                        # compute mask metrics
                        seg_mae_list.append(mae(pred_mask_np, gt_mask_np))
                        seg_salpha_list.append(s_alpha(pred_mask_np, gt_mask_np))
                        seg_ephi_list.append(e_phi(pred_mask_np, gt_mask_np))
                        seg_fbw_list.append(fbeta_weighted(pred_mask_np, gt_mask_np))

    # Aggregate
    det_metrics = {
        "avg_precision": float(np.mean(all_precisions)) if len(all_precisions) else 0.0,
        "avg_recall": float(np.mean(all_recalls)) if len(all_recalls) else 0.0,
        "avg_iou": float(np.mean(all_avgious)) if len(all_avgious) else 0.0,
        "avg_detections_per_image": float(total_detections / max(1, total_images))
    }
    seg_metrics = {}
    if len(seg_mae_list) > 0:
        seg_metrics = {
            "mae": float(np.mean(seg_mae_list)),
            "s_alpha": float(np.mean(seg_salpha_list)),
            "e_phi": float(np.mean(seg_ephi_list)),
            "f_beta_w": float(np.mean(seg_fbw_list))
        }

    return {"detection": det_metrics, "segmentation": seg_metrics, "n_images": total_images}
