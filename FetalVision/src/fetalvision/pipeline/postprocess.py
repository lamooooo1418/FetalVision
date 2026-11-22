"""Numpy post-processing: YOLO raw-output decoding + NMS, mask -> ellipse -> head circumference."""
from __future__ import annotations
import logging
import numpy as np
import cv2

from .preprocess import LetterboxParams

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------- detector
def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by decreasing score."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo(raw: np.ndarray, conf_thr: float, iou_thr: float, params: LetterboxParams | None = None,
                max_det: int = 10, class_filter: int | None = None, class_agnostic: bool = False) -> dict:
    """Decode ultralytics raw output [1, 4+nc, N] (or [4+nc, N]) -> dict(boxes_xyxy [k,4] in ORIGINAL image pixels,
    scores [k], classes [k]). Generic over n_classes (read from the tensor shape). NMS is per class unless
    class_agnostic (ultralytics applies a per-class offset trick; we do the same)."""
    r = raw[0] if raw.ndim == 3 else raw
    nc = r.shape[0] - 4
    boxes = xywh2xyxy(r[:4].T.astype(np.float32))
    cls_scores = r[4:].T  # [N, nc]
    if class_filter is not None and nc > 1:
        cls_scores = cls_scores[:, class_filter:class_filter + 1]
        cls_ids = np.full((cls_scores.shape[0],), class_filter, dtype=np.int64)
        scores = cls_scores[:, 0]
    else:
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores.max(1)
    m = scores > conf_thr
    boxes, scores, cls_ids = boxes[m], scores[m], cls_ids[m]
    if len(boxes):
        offs = np.zeros_like(boxes) if class_agnostic else (cls_ids[:, None].astype(np.float32) * 7680.0) * np.array([1, 1, 1, 1], np.float32)
        keep = nms(boxes + offs, scores, iou_thr)[:max_det]
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if params is not None and len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - params.pad_left) / params.ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - params.pad_top) / params.ratio
        h, w = params.orig_hw
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return {"boxes_xyxy": boxes.astype(np.float32), "scores": scores.astype(np.float32), "classes": cls_ids.astype(np.int64),
            "n_classes": int(nc)}


# ---------------------------------------------------------------------------------------------------------- segmenter
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def paste_mask(prob_small: np.ndarray, crop_xyxy, full_hw: tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    """Resize crop-level probabilities to the crop size, threshold, paste into a full-resolution uint8 {0,1} mask."""
    x1, y1, x2, y2 = crop_xyxy
    full = np.zeros(full_hw, dtype=np.uint8)
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return full
    p = cv2.resize(prob_small.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
    full[y1:y2, x1:x2] = (p >= threshold).astype(np.uint8)
    return full


# -------------------------------------------------------------------------------------------------------- HC measure
def ramanujan_perimeter(a: float, b: float) -> float:
    """Ramanujan's first approximation of the ellipse perimeter with semi-axes a, b."""
    return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def _local_measure_hc(mask: np.ndarray, pixel_size_mm: float | None = None, min_area_px: int = 50) -> dict:
    """mask (H,W) {0,1}/bool -> largest contour -> cv2.fitEllipse -> HC via Ramanujan. Semi-axes a >= b in px,
    angle in degrees (OpenCV convention, rotation of the first axis)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    area = int(m.sum())
    out = {"mask_area_px": area, "ellipse": None, "hc_px": None, "hc_mm": None, "implementation": "pipeline.postprocess"}
    if area < min_area_px:
        return out
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return out
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return out
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(c)
    if d2 >= d1:  # canonical: a >= b semi-axes, angle = orientation of the major axis (same as segmentation.hc_measure)
        a, b, angle = d2 / 2.0, d1 / 2.0, angle + 90.0
    else:
        a, b = d1 / 2.0, d2 / 2.0
    hc_px = ramanujan_perimeter(a, b)
    out["ellipse"] = {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "angle": float(angle % 180.0)}
    out["hc_px"] = hc_px
    out["hc_mm"] = float(hc_px * pixel_size_mm) if pixel_size_mm else None
    return out


def _resolve_external_measure():
    """Prefer fetalvision.segmentation.hc_measure.measure_hc when the segmentation workstream has landed it.
    Its interface: measure_hc(mask, pixel_size_mm, original_size_wh=None) -> {valid, hc_px, hc_mm (nan if no pixel
    size), ellipse {cx, cy, a, b, angle_deg, area_px}} with a >= b semi-axes and angle_deg = major-axis orientation."""
    try:
        from fetalvision.segmentation import hc_measure as ext  # type: ignore
    except Exception:
        return None
    for name in ("measure_hc", "mask_to_hc", "hc_from_mask", "mask_to_ellipse_hc"):
        fn = getattr(ext, name, None)
        if callable(fn):
            log.info("using fetalvision.segmentation.hc_measure.%s for HC measurement", name)
            return fn
    return None


_EXTERNAL = _resolve_external_measure()


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _normalise_ellipse(ell) -> dict | None:
    """Accept {cx, cy, a, b, angle|angle_deg|theta} (semi-axes) and return {cx, cy, a, b, angle} with a >= b and
    angle (deg) = orientation of the major axis a (OpenCV cv2.ellipse convention for axes (2a, 2b))."""
    if not isinstance(ell, dict):
        return None
    try:
        cx, cy, a, b = float(ell["cx"]), float(ell["cy"]), float(ell["a"]), float(ell["b"])
    except (KeyError, TypeError, ValueError):
        return None
    ang = ell.get("angle", ell.get("angle_deg", ell.get("theta", 0.0)))
    ang = float(ang) if ang is not None else 0.0
    if b > a:
        a, b, ang = b, a, ang + 90.0
    return {"cx": cx, "cy": cy, "a": a, "b": b, "angle": ang % 180.0}


def measure_hc(mask: np.ndarray, pixel_size_mm: float | None = None, min_area_px: int = 50) -> dict:
    """mask -> {mask_area_px, ellipse{cx,cy,a,b,angle}, hc_px, hc_mm, implementation}. Uses the segmentation
    workstream's implementation when importable and well-formed, else the local cv2.fitEllipse + Ramanujan copy."""
    area = int((np.asarray(mask) > 0).sum())
    if _EXTERNAL is not None and area >= min_area_px:
        try:
            res = _EXTERNAL(mask, pixel_size_mm)
            if isinstance(res, dict) and ("hc_px" in res or "ellipse" in res):
                hc_px = _num(res.get("hc_px"))
                hc_mm = _num(res.get("hc_mm"))
                if hc_mm is None and hc_px is not None and pixel_size_mm:
                    hc_mm = hc_px * float(pixel_size_mm)
                return {"mask_area_px": area, "ellipse": _normalise_ellipse(res.get("ellipse")), "hc_px": hc_px,
                        "hc_mm": hc_mm, "implementation": "segmentation.hc_measure"}
        except Exception as e:  # fall back but log
            log.warning("external hc_measure failed (%s); using local implementation", e)
    return _local_measure_hc(mask, pixel_size_mm, min_area_px)
