"""HC18 segmentation dataset (image + filled head mask) with albumentations augmentation.

Images are 800x540 grayscale; they are resized to `input_size` (W x H, default 512x352), replicated to 3 channels
and normalised with ImageNet statistics so that ImageNet-pretrained encoders can be used.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_transforms(input_size_wh: tuple[int, int], augment: bool, aug_cfg: dict | None = None) -> A.Compose:
    w, h = input_size_wh
    aug_cfg = aug_cfg or {}
    tf = [A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)]
    if augment:
        tf += [
            A.HorizontalFlip(p=float(aug_cfg.get("hflip_p", 0.5))),
            A.Affine(translate_percent=float(aug_cfg.get("shift_limit", 0.05)),
                     scale=(1 - float(aug_cfg.get("scale_limit", 0.10)), 1 + float(aug_cfg.get("scale_limit", 0.10))),
                     rotate=(-float(aug_cfg.get("rotate_limit", 10)), float(aug_cfg.get("rotate_limit", 10))),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                       p=float(aug_cfg.get("brightness_contrast_p", 0.5))),
        ]
    return A.Compose(tf)


def to_tensor(img_gray: np.ndarray) -> torch.Tensor:
    """uint8 HxW grayscale -> float tensor 3xHxW normalised with ImageNet statistics."""
    x = img_gray.astype(np.float32) / 255.0
    x = np.repeat(x[..., None], 3, axis=-1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class HC18SegDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, raw_dir: str | Path, processed_dir: str | Path,
                 input_size_wh: tuple[int, int], augment: bool = False, aug_cfg: dict | None = None):
        self.rows = rows.reset_index(drop=True)
        self.raw_dir, self.processed_dir = Path(raw_dir), Path(processed_dir)
        self.input_size_wh = tuple(input_size_wh)
        self.tf = build_transforms(self.input_size_wh, augment, aug_cfg)

    def __len__(self) -> int:
        return len(self.rows)

    def load_raw(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        r = self.rows.iloc[i]
        img = cv2.imread(str(self.raw_dir / r.filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.processed_dir / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"missing image or mask for {r.filename}")
        return img, (mask > 127).astype(np.uint8)

    def __getitem__(self, i: int):
        img, mask = self.load_raw(i)
        t = self.tf(image=img, mask=mask)
        x = to_tensor(t["image"])
        y = torch.from_numpy(t["mask"].astype(np.float32))[None]
        return x, y, self.rows.iloc[i].filename
