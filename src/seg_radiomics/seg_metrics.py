"""segmentation overlap metrics dice and iou (plus a confusion breakdown)

standard scores for how well the predicted mask matches the ground-truth mask pure
numpy work on binary masks of any dimensionality (2d slices or 3d volumes)
unit-tested for the values you'd expect (perfect = 1 disjoint = 0 half-overlap =
the known fraction)
"""
from __future__ import annotations

import numpy as np


def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """dice = 2*intersection / (|pred| + |gt|)"""
    p, g = _binarize(pred), _binarize(gt)
    inter = np.logical_and(p, g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """iou jaccard = intersection / union"""
    p, g = _binarize(pred), _binarize(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((inter + eps) / (union + eps))


def confusion(pred: np.ndarray, gt: np.ndarray) -> dict:
    """voxel-level tp/fp/fn/tn counts"""
    p, g = _binarize(pred), _binarize(gt)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sensitivity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """recall = tp / (tp + fn)"""
    c = confusion(pred, gt)
    return float((c["tp"] + eps) / (c["tp"] + c["fn"] + eps))


def precision(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """precision = tp / (tp + fp)"""
    c = confusion(pred, gt)
    return float((c["tp"] + eps) / (c["tp"] + c["fp"] + eps))


def all_seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "sensitivity": sensitivity(pred, gt),
        "precision": precision(pred, gt),
    }
