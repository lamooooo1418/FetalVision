"""U-Net construction (segmentation_models_pytorch), checkpoint export/import, and inference helpers
(full image and detector-box crop with paste-back)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fetalvision.segmentation.dataset import to_tensor


def build_model(mcfg: dict, pretrained: bool = True):
    import segmentation_models_pytorch as smp
    arch = getattr(smp, mcfg.get("arch", "Unet"))
    return arch(encoder_name=mcfg.get("encoder", "mobilenet_v2"),
                encoder_weights=mcfg.get("encoder_weights", "imagenet") if pretrained else None,
                in_channels=int(mcfg.get("in_channels", 3)), classes=int(mcfg.get("classes", 1)))


def write_meta(meta_path: str | Path, meta: dict) -> dict:
    """Write models/head_segmenter.json. Keys of an existing file (e.g. ONNX export metadata written by the export
    workstream) are preserved; our training metadata and flat compatibility keys (encoder, arch, input_size WxH,
    normalize, stand_in=false) are merged in."""
    meta_path = Path(meta_path)
    existing = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
        except Exception:  # pragma: no cover - corrupt file, overwrite
            existing = {}
    mcfg = meta.get("model", {})
    compat = {"model_name": "head_segmenter", "framework": "segmentation_models_pytorch", "arch": mcfg.get("arch", "Unet"),
              "encoder": mcfg.get("encoder"), "input_size": list(meta.get("input_size_wh", [])), "input_size_order": "WH",
              "normalize": "imagenet", "stand_in": False, "stand_in_kind": None,
              "weights_format": "torch state_dict of smp.Unet (load with fetalvision.segmentation.model.load_checkpoint)"}
    merged = {**existing, **compat, **meta}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(merged, indent=2))
    return merged


def save_checkpoint(model, weights_path: str | Path, meta_path: str | Path, meta: dict) -> None:
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    write_meta(meta_path, meta)


def load_checkpoint(weights_path: str | Path, meta_path: str | Path, device: str = "cpu"):
    """Load weights + metadata. Accepts our schema (`model` dict) or the flat export schema (`encoder`, `arch`)."""
    meta = json.loads(Path(meta_path).read_text())
    mcfg = meta.get("model") or {"arch": meta.get("arch", "Unet"), "encoder": meta.get("encoder", "mobilenet_v2"),
                                 "in_channels": 3, "classes": 1}
    model = build_model(mcfg, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.to(device).eval()
    return model, meta


class DiceBCELoss(torch.nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + dice.mean()


@torch.no_grad()
def predict_prob(model, img_gray: np.ndarray, input_size_wh: tuple[int, int], device: str) -> np.ndarray:
    """Full-image inference: returns a float32 probability map at the ORIGINAL image resolution."""
    h, w = img_gray.shape[:2]
    small = cv2.resize(img_gray, tuple(input_size_wh), interpolation=cv2.INTER_LINEAR)
    x = to_tensor(small)[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def expand_box_to_aspect(box, margin: float, aspect_wh: float, img_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    """Grow an xyxy box by `margin` on each side, then pad the shorter dimension so the crop has aspect W/H =
    aspect_wh (clipped to the image). Returns integer xyxy."""
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = bw * (1 + 2 * margin), bh * (1 + 2 * margin)
    if bw / bh < aspect_wh:
        bw = bh * aspect_wh
    else:
        bh = bw / aspect_wh
    W, H = img_wh
    nx1, ny1 = max(0, int(np.floor(cx - bw / 2))), max(0, int(np.floor(cy - bh / 2)))
    nx2, ny2 = min(W, int(np.ceil(cx + bw / 2))), min(H, int(np.ceil(cy + bh / 2)))
    return nx1, ny1, max(nx2, nx1 + 1), max(ny2, ny1 + 1)


@torch.no_grad()
def predict_prob_in_box(model, img_gray: np.ndarray, box, input_size_wh: tuple[int, int], device: str,
                        margin: float = 0.10) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Detector-box chain: crop the box (+margin, padded to the network aspect), predict, paste the probability map
    back into a zero map at the original resolution. Returns (prob map, crop xyxy used)."""
    h, w = img_gray.shape[:2]
    cx1, cy1, cx2, cy2 = expand_box_to_aspect(box, margin, input_size_wh[0] / input_size_wh[1], (w, h))
    crop = img_gray[cy1:cy2, cx1:cx2]
    small = cv2.resize(crop, tuple(input_size_wh), interpolation=cv2.INTER_LINEAR)
    x = to_tensor(small)[None].to(device)
    prob_small = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    prob = np.zeros((h, w), np.float32)
    prob[cy1:cy2, cx1:cx2] = cv2.resize(prob_small, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR)
    return prob, (cx1, cy1, cx2, cy2)
