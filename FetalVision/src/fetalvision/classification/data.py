"""tf.data loaders for the processed FETAL_PLANES_DB folders (Phase 1 settings: 224x224, batch 32, int labels).

`make_datasets()` returns (datasets, file_paths): the file paths are captured from `image_dataset_from_directory`
BEFORE cache/prefetch so predictions CSVs can carry the path of every image in dataset order (val/test are not
shuffled, so dataset order == file_paths order).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fetalvision.common.constants import CLASS_NAMES

log = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32


def configure_tf(cfg: dict | None = None) -> None:
    """Set CPU threading before any TF op is created (Apple M4 Pro: 10 performance cores)."""
    import tensorflow as tf
    hw = (cfg or {}).get("hardware", {}) if cfg else {}
    intra = int(hw.get("intra_op_threads", os.environ.get("FV_INTRA_THREADS", 10)))
    inter = int(hw.get("inter_op_threads", os.environ.get("FV_INTER_THREADS", 2)))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except RuntimeError:  # already initialised (e.g. in tests)
        pass
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:  # pragma: no cover
            pass


def build_augmentation(cfg: dict | None = None, seed: int = 42):
    """Phase 1 augmentation pipeline (training only). Factors are fractions as in Keras preprocessing layers."""
    from tensorflow import keras
    from keras import layers
    a = {"rotation": 0.014, "zoom": 0.05, "translation": 0.05, "contrast": 0.10, "fill_mode": "reflect"}
    a.update({k: v for k, v in (cfg or {}).items() if k in a})
    return keras.Sequential([
        layers.RandomRotation(a["rotation"], fill_mode=a["fill_mode"], seed=seed),
        layers.RandomZoom(a["zoom"], fill_mode=a["fill_mode"], seed=seed + 1),
        layers.RandomTranslation(a["translation"], a["translation"], fill_mode=a["fill_mode"], seed=seed + 2),
        layers.RandomContrast(a["contrast"], seed=seed + 3),
    ], name="phase1_augmentation")


def load_split(data_dir: str | Path, split: str, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE,
               shuffle: bool = False, seed: int = 42):
    """image_dataset_from_directory with the fixed class order. Returns (dataset, file_paths list)."""
    import tensorflow as tf
    ds = tf.keras.utils.image_dataset_from_directory(
        str(Path(data_dir) / split), image_size=tuple(image_size), batch_size=batch_size, label_mode="int",
        class_names=CLASS_NAMES, shuffle=shuffle, seed=seed if shuffle else None)
    paths = list(getattr(ds, "file_paths", []))
    return ds, paths


def make_datasets(cfg: dict) -> tuple[dict, dict]:
    """Build train/val/test datasets from a run config:
    data: {processed_dir, image_size, batch_size, cache}  augmentation: {enabled, ...}  debug: {max_steps}"""
    import tensorflow as tf
    d = cfg.get("data", {})
    seed = int(cfg.get("seed", 42))
    image_size = tuple(d.get("image_size", DEFAULT_IMAGE_SIZE))
    batch_size = int(d.get("batch_size", DEFAULT_BATCH_SIZE))
    processed = Path(d["processed_dir"])
    max_steps = int(cfg.get("debug", {}).get("max_steps", 0) or 0)
    datasets, paths = {}, {}
    for split in ["train", "val", "test"]:
        ds, p = load_split(processed, split, image_size, batch_size, shuffle=(split == "train"), seed=seed)
        if max_steps:
            ds = ds.take(max_steps)
            p = p[: max_steps * batch_size]
        if d.get("cache", True):
            ds = ds.cache()
        if split == "train":
            aug_cfg = cfg.get("augmentation", {})
            if aug_cfg.get("enabled", True):
                aug = build_augmentation(aug_cfg, seed)
                ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
                log.info("augmentation ON: %s", {k: v for k, v in aug_cfg.items() if k != "enabled"})
            else:
                log.info("augmentation OFF")
        datasets[split] = ds.prefetch(tf.data.AUTOTUNE)
        paths[split] = p
    return datasets, paths


def labels_from_paths(paths: list[str]) -> list[int]:
    """Class index from the parent folder name (processed layout <split>/<Class>/<file>)."""
    return [CLASS_NAMES.index(Path(p).parent.name) for p in paths]


def balanced_class_weights(paths: list[str]) -> dict[int, float]:
    """sklearn 'balanced' weights: n_samples / (n_classes * count_c)."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    y = np.asarray(labels_from_paths(paths))
    classes = np.arange(len(CLASS_NAMES))
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(x) for c, x in zip(classes, w)}
