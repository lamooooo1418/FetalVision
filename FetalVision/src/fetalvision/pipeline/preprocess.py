"""Pure-numpy/cv2 preprocessing shared by the ONNX pipeline, the parity checks and the benchmark."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_bgr3(image: np.ndarray) -> np.ndarray:
    """Accept gray (H,W), BGR (H,W,3) or BGRA (H,W,4) uint8 and return BGR uint8 (H,W,3)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image


def classifier_input(image_bgr: np.ndarray, size_hw=(224, 224)) -> np.ndarray:
    """NHWC float32 RGB in [0,255] (the Keras model contains its own rescaling), batch of 1."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (int(size_hw[1]), int(size_hw[0])), interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32)[None]


@dataclass
class LetterboxParams:
    ratio: float
    pad_left: int
    pad_top: int
    orig_hw: tuple[int, int]
    new_hw: tuple[int, int]


def letterbox(image_bgr: np.ndarray, new_shape: int | tuple[int, int] = 640, pad_value: int = 114) -> tuple[np.ndarray, LetterboxParams]:
    """Ultralytics-compatible LetterBox (auto=False, scaleup=True, center=True, INTER_LINEAR)."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h0, w0 = image_bgr.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (max(round(w0 * r), 1), max(round(h0 * r), 1))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = image_bgr
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return img, LetterboxParams(ratio=r, pad_left=left, pad_top=top, orig_hw=(h0, w0), new_hw=tuple(new_shape))


def detector_input(image_bgr: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, LetterboxParams]:
    """NCHW float32 RGB in [0,1] after letterboxing (same as ultralytics predict preprocess)."""
    lb, params = letterbox(image_bgr, imgsz)
    x = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None]), params


def expand_box(box_xyxy, margin: float, img_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Add `margin` x box size on each side and clip to the image; returns integer xyxy (x2,y2 exclusive)."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * bw, x2 + margin * bw
    y1, y2 = y1 - margin * bh, y2 + margin * bh
    h, w = img_hw
    x1i, y1i = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2i, y2i = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if x2i <= x1i + 1:
        x2i = min(w, x1i + 2)
    if y2i <= y1i + 1:
        y2i = min(h, y1i + 2)
    return x1i, y1i, x2i, y2i


def segmenter_input(crop_bgr: np.ndarray, size_wh=(512, 352), normalize: str = "imagenet", grayscale: bool = False) -> np.ndarray:
    """NCHW float32 resized to (W,H); RGB, or grayscale replicated to 3 channels (how the HC18 U-Net was trained);
    [0,1] then ImageNet normalisation when normalize == 'imagenet'."""
    if grayscale:
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        rgb = np.repeat(g[:, :, None], 3, axis=2)
    else:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (int(size_wh[0]), int(size_wh[1])), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if normalize == "imagenet":
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
