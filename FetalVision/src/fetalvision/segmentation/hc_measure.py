"""Head-circumference measurement from a binary fetal-head mask.

mask -> largest connected component -> cv2.fitEllipse on its contour -> ellipse axes rescaled to the ORIGINAL image
resolution (800x540 for HC18) -> perimeter via Ramanujan's second approximation -> HC in px and mm (pixel size).
"""
from __future__ import annotations

import numpy as np
import cv2


def ramanujan_circumference(a: float, b: float) -> float:
    """Ramanujan's second approximation of an ellipse perimeter from semi-axes a and b (error < 1e-5 relative)."""
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    return float(np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected foreground component of a binary mask (uint8 0/1)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def fit_head_ellipse(mask: np.ndarray) -> dict | None:
    """Fit an ellipse to the external contour of the largest component. Returns dict(cx, cy, a, b, angle_deg) in the
    mask's pixel coordinates (a >= b are SEMI-axes), or None if the mask is empty / too small."""
    m = largest_component(mask)
    if m.sum() < 5:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(cnt)
    if not np.isfinite([cx, cy, d1, d2, ang]).all():
        return None
    return {**_canonical(cx, cy, d1, d2, ang), "area_px": float(m.sum())}


def _canonical(cx, cy, d1, d2, ang) -> dict:
    """cv2.fitEllipse gives (centre, (d1, d2), angle of the d1 axis). Return semi-axes a >= b with angle_deg = the
    orientation of the MAJOR axis a."""
    if d2 >= d1:
        a, b, ang = d2 / 2, d1 / 2, ang + 90.0
    else:
        a, b = d1 / 2, d2 / 2
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "angle_deg": float(ang % 180.0)}


def measure_hc(mask: np.ndarray, pixel_size_mm: float | None, original_size_wh: tuple[int, int] | None = None) -> dict:
    """HC from a binary mask. If `original_size_wh` is given and differs from the mask size, the ellipse is rescaled
    (anisotropically: x by W0/W, y by H0/H) to the original resolution before the perimeter is computed.
    Returns dict(hc_px, hc_mm, ellipse (in original coords), valid)."""
    mask = np.asarray(mask)
    h, w = mask.shape[:2]
    ell = fit_head_ellipse(mask)
    if ell is None:
        return {"valid": False, "hc_px": float("nan"), "hc_mm": float("nan"), "ellipse": None}
    sx = sy = 1.0
    if original_size_wh is not None:
        sx, sy = original_size_wh[0] / w, original_size_wh[1] / h
    ell_o = scale_ellipse(ell, sx, sy)
    hc_px = ramanujan_circumference(ell_o["a"], ell_o["b"])
    hc_mm = hc_px * float(pixel_size_mm) if pixel_size_mm is not None else float("nan")
    return {"valid": True, "hc_px": float(hc_px), "hc_mm": float(hc_mm), "ellipse": ell_o}


def scale_ellipse(ell: dict, sx: float, sy: float) -> dict:
    """Rescale an ellipse (centre, semi-axes, angle) by factors sx, sy. For isotropic scaling this is exact; for
    anisotropic scaling the ellipse is re-fitted to its transformed boundary points (exact up to sampling)."""
    if abs(sx - sy) < 1e-9:
        return {**ell, "cx": ell["cx"] * sx, "cy": ell["cy"] * sy, "a": ell["a"] * sx, "b": ell["b"] * sx}
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    th = np.deg2rad(ell["angle_deg"])  # orientation of the major axis a
    x = ell["cx"] + ell["a"] * np.cos(t) * np.cos(th) - ell["b"] * np.sin(t) * np.sin(th)
    y = ell["cy"] + ell["a"] * np.cos(t) * np.sin(th) + ell["b"] * np.sin(t) * np.cos(th)
    pts = np.stack([x * sx, y * sy], axis=1).astype(np.float32)
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts)
    return {**_canonical(cx, cy, d1, d2, ang), "area_px": ell.get("area_px", float("nan")) * sx * sy}


def ellipse_to_mask(ell: dict, size_wh: tuple[int, int]) -> np.ndarray:
    """Rasterise an ellipse dict (a >= b semi-axes; angle = orientation of the a-axis in degrees) into a 0/1 mask."""
    m = np.zeros((size_wh[1], size_wh[0]), np.uint8)
    cv2.ellipse(m, ((ell["cx"], ell["cy"]), (2 * ell["a"], 2 * ell["b"]), ell["angle_deg"]), 1, thickness=-1)
    return m
