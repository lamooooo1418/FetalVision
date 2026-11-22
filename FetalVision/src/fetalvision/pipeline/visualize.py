"""Draw the pipeline result (box, mask contour, fitted ellipse, label text) on a copy of the image."""
from __future__ import annotations
import numpy as np
import cv2

GREEN, BLUE, YELLOW, RED, WHITE = (60, 200, 60), (255, 160, 40), (0, 220, 255), (40, 40, 230), (255, 255, 255)


def annotate(image: np.ndarray, result: dict, mask: np.ndarray | None = None) -> np.ndarray:
    from .preprocess import to_bgr3
    img = to_bgr3(np.ascontiguousarray(image)).copy()
    h, w = img.shape[:2]
    scale = max(0.5, min(w, h) / 600.0)
    thick = max(1, int(round(2 * scale)))
    plane = result.get("plane", {})
    head = result.get("head") or {}
    seg = result.get("segmentation") or {}
    hc = result.get("hc") or {}
    mask = mask if mask is not None else result.get("mask")
    if mask is not None and np.any(mask):
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, BLUE, thick)
    if head.get("detected") and head.get("box_xyxy"):
        x1, y1, x2, y2 = [int(round(v)) for v in head["box_xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, thick)
        cv2.putText(img, f"head {head['score']:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, GREEN, thick)
    ell = seg.get("ellipse")
    if ell:
        cv2.ellipse(img, ((ell["cx"], ell["cy"]), (2 * ell["a"], 2 * ell["b"]), ell["angle"]), YELLOW, thick)
        cv2.circle(img, (int(round(ell["cx"])), int(round(ell["cy"]))), thick + 1, YELLOW, -1)
    lines = [f"{plane.get('label', '?')} p={plane.get('probability', 0):.3f}"]
    status = result.get("head_measurement", "")
    if hc.get("px") is not None:
        txt = f"HC = {hc['px']:.0f} px"
        if hc.get("mm") is not None:
            txt += f" = {hc['mm']:.1f} mm"
        lines.append(txt)
    elif status != "not_applicable":
        lines.append(f"head: {status}")
    t = result.get("timings_ms", {})
    if t:
        lines.append(f"{t.get('total', 0):.0f} ms")
    y = int(22 * scale)
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thick)
        cv2.rectangle(img, (6, y - th - 4), (10 + tw, y + 6), (0, 0, 0), -1)
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, WHITE, thick, cv2.LINE_AA)
        y += int(th + 14 * scale)
    return img
