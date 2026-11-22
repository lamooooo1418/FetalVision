"""Helpers for post-processing exported ONNX graphs and reading parity images."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)


def rename_io(onnx_path: str | Path, input_names: dict[str, str] | None = None,
              output_names: dict[str, str] | None = None) -> None:
    """Rename graph inputs/outputs in place (e.g. 'Identity:0' -> 'probabilities')."""
    import onnx
    model = onnx.load(str(onnx_path))
    mapping = {**(input_names or {}), **(output_names or {})}
    if not mapping:
        return
    for node in model.graph.node:
        node.input[:] = [mapping.get(n, n) for n in node.input]
        node.output[:] = [mapping.get(n, n) for n in node.output]
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if vi.name in mapping:
            vi.name = mapping[vi.name]
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def onnx_io_summary(onnx_path: str | Path) -> dict:
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)

    def dims(t):
        return [d.dim_value if d.dim_value > 0 else (d.dim_param or "N") for d in t.type.tensor_type.shape.dim]

    return {"inputs": [{"name": i.name, "shape": dims(i)} for i in m.graph.input],
            "outputs": [{"name": o.name, "shape": dims(o)} for o in m.graph.output],
            "opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "ir_version": m.ir_version, "n_nodes": len(m.graph.node)}


def list_images(folder: str | Path, n: int) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts and "Annotation" not in p.name) \
        if folder.exists() else []
    return files[:n]


def load_parity_images(images_dir: str | Path, n_real: int, n_noise: int, noise_hw: tuple[int, int] = (540, 800),
                       seed: int = 42) -> tuple[list[np.ndarray], list[str]]:
    """Real BGR images (uint8) from images_dir plus `n_noise` uniform-noise images of HC18 size."""
    import cv2
    paths = list_images(images_dir, n_real)
    imgs, names = [], []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
            names.append(p.name)
    if not imgs:
        log.warning("no real images found in %s; parity will use noise only", images_dir)
    rng = np.random.RandomState(seed)
    for i in range(n_noise):
        imgs.append(rng.randint(0, 256, (noise_hw[0], noise_hw[1], 3), dtype=np.uint8))
        names.append(f"noise_{i:02d}")
    return imgs, names


def timed(fn, *args, repeats: int = 1, **kwargs):
    """Return (result, mean_ms) of fn over `repeats` calls (perf_counter)."""
    import time
    t0 = time.perf_counter()
    for _ in range(repeats):
        res = fn(*args, **kwargs)
    return res, (time.perf_counter() - t0) * 1000.0 / repeats
