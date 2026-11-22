"""Shared helpers for the fetal-head detector: device choice, GT loading, prediction matching, sweep rule,
FP categorisation."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fetalvision.common.metrics import box_iou

log = logging.getLogger(__name__)


def pick_device(requested: str | None = "auto") -> str:
    """'auto' -> 'mps' if available, else 'cuda' if available, else 'cpu'."""
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def load_split(splits_csv: str | Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(splits_csv)
    df = df[df.split == split].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"no rows for split={split} in {splits_csv}")
    return df


def gt_box(row) -> np.ndarray:
    return np.array([row.x1, row.y1, row.x2, row.y2], dtype=float)


def match_single_gt(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt: np.ndarray, match_iou: float = 0.5) -> dict:
    """Greedy matching for ONE ground-truth box per image.

    Predictions are sorted by descending score; the first one with IoU >= match_iou is the TP, every other kept
    prediction is a FP; if no prediction matches, the GT is a FN.
    Returns dict(tp: 0/1, fp: int, fn: 0/1, tp_idx, ious (per pred, same order as input), fp_idx list).
    """
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=float).reshape(-1)
    ious = np.array([box_iou(b, gt) for b in pred_boxes]) if len(pred_boxes) else np.zeros(0)
    order = np.argsort(-pred_scores)
    tp_idx = None
    for i in order:
        if ious[i] >= match_iou:
            tp_idx = int(i)
            break
    fp_idx = [int(i) for i in order if i != tp_idx]
    return {"tp": int(tp_idx is not None), "fp": len(fp_idx), "fn": int(tp_idx is None),
            "tp_idx": tp_idx, "fp_idx": fp_idx, "ious": ious}


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def sweep_confidences(preds: dict[str, tuple[np.ndarray, np.ndarray]], gts: dict[str, np.ndarray],
                      conf_thresholds, match_iou: float = 0.5) -> pd.DataFrame:
    """preds[filename] = (boxes xyxy [N,4], scores [N]) at a very low conf; gts[filename] = box xyxy.
    Returns a table with precision/recall/F1 and counts for each confidence threshold."""
    rows = []
    for conf in conf_thresholds:
        tp = fp = fn = 0
        for fn_name, gt in gts.items():
            boxes, scores = preds.get(fn_name, (np.zeros((0, 4)), np.zeros(0)))
            keep = np.asarray(scores) >= conf
            m = match_single_gt(np.asarray(boxes)[keep], np.asarray(scores)[keep], gt, match_iou)
            tp += m["tp"]
            fp += m["fp"]
            fn += m["fn"]
        p, r, f1 = prf(tp, fp, fn)
        rows.append({"conf": float(conf), "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)


def choose_operating_point(sweep: pd.DataFrame, min_recall: float = 0.98) -> tuple[dict, str]:
    """Rule fixed a priori: maximise F1 subject to recall >= min_recall; if infeasible, maximise F1.
    Ties are broken towards the HIGHER confidence (fewer false positives). Returns (row as dict, rule_used)."""
    feasible = sweep[sweep.recall >= min_recall]
    rule = f"max F1 s.t. recall >= {min_recall}"
    if len(feasible) == 0:
        feasible = sweep
        rule = f"max F1 (recall >= {min_recall} infeasible)"
    best = feasible.sort_values(["f1", "conf"], ascending=[False, False]).iloc[0]
    return best.to_dict(), rule


def categorise_fp(iou: float, match_iou: float = 0.5, loc_low: float = 0.1) -> str:
    """FP taxonomy for a single-GT image: duplicate (second box on the GT, IoU >= match_iou),
    localisation (loc_low <= IoU < match_iou), background (IoU < loc_low)."""
    if iou >= match_iou:
        return "duplicate"
    if iou >= loc_low:
        return "localisation"
    return "background"


def run_predict(model, paths: list[Path], conf: float, iou: float, imgsz: int, device: str, max_det: int = 50,
                batch: int = 16) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run an ultralytics detector; returns {filename: (boxes xyxy px [N,4], scores [N])}."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    paths = [str(p) for p in paths]
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        results = model.predict(source=chunk, conf=conf, iou=iou, imgsz=imgsz, device=device, max_det=max_det,
                                verbose=False)
        for p, r in zip(chunk, results):
            if r.boxes is None or len(r.boxes) == 0:
                out[Path(p).name] = (np.zeros((0, 4)), np.zeros(0))
            else:
                out[Path(p).name] = (r.boxes.xyxy.cpu().numpy().astype(float), r.boxes.conf.cpu().numpy().astype(float))
    return out
