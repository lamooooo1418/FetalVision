"""Statistics shared across phases: confidence intervals, label-audit bounds, calibration (ECE), temperature scaling."""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion (used for test accuracy)."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper–Pearson upper bound on a rate given k events in n trials.
    For k = 0 this is 1 - (1-confidence)^(1/n) (e.g. 2.95 % for n = 100)."""
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(confidence, k + 1, n - k))


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    """ECE with equal-width confidence bins on the max-probability prediction. Returns ECE and the reliability table."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            acc, c, w = correct[mask].mean(), conf[mask].mean(), mask.mean()
            ece += w * abs(acc - c)
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": int(mask.sum()),
                         "accuracy": float(acc), "confidence": float(c)})
    return {"ece": float(ece), "n_bins": n_bins, "bins": rows}


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017) fitted on validation logits by minimising NLL. Returns T."""
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)

    def nll(t):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y_true)), y_true].mean()

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probs_to_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Softmax outputs are invariant to a constant shift, so log(p) is a valid logit vector for temperature scaling."""
    return np.log(np.clip(np.asarray(probs, dtype=float), eps, 1.0))


def top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == np.asarray(y_true)[:, None], axis=1)))


def dice_coefficient(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float((2.0 * np.logical_and(pred, gt).sum() + eps) / (pred.sum() + gt.sum() + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return float((np.logical_and(pred, gt).sum() + eps) / (union + eps))


def box_iou(a, b) -> float:
    """IoU of two boxes in xyxy pixel coordinates."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def average_precision_single_instance(scores: np.ndarray, ious: np.ndarray, n_gt: int, iou_thr: float) -> float:
    """AP for datasets with exactly one GT object per image and at most one prediction per image
    (COCO-style 101-point interpolation). `ious` is the IoU of each prediction with the GT of its image."""
    order = np.argsort(-np.asarray(scores))
    tp = (np.asarray(ious)[order] >= iou_thr).astype(float)
    fp = 1.0 - tp
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / max(n_gt, 1)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    rec_thrs = np.linspace(0, 1, 101)
    prec_interp = np.zeros_like(rec_thrs)
    for i, r in enumerate(rec_thrs):
        m = recall >= r
        prec_interp[i] = precision[m].max() if m.any() else 0.0
    return float(prec_interp.mean())


def map_50_95_single_instance(scores, ious, n_gt) -> dict:
    thrs = np.arange(0.5, 0.96, 0.05)
    aps = {f"AP{int(round(t*100))}": average_precision_single_instance(scores, ious, n_gt, t) for t in thrs}
    aps["mAP50-95"] = float(np.mean(list(aps.values())))
    aps["mAP50"] = aps["AP50"]
    return aps
