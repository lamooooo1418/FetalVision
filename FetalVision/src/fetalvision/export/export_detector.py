"""Export the ultralytics YOLO11n fetal-head detector to ONNX (imgsz 640, opset 17, simplified, static batch 1, no NMS).

    python -m fetalvision.export.export_detector --config configs/export.yaml [--weights models/head_detector.pt]

Raw output is [1, 4 + n_classes, 8400] (xywh in letterboxed pixels + class scores); decoding and NMS live in
fetalvision.pipeline.postprocess so the same numpy code serves any backend. When models/head_detector.pt does not
exist yet, the pretrained COCO `yolo11n.pt` (80 classes) is used as a STAND-IN and flagged in the metadata.
"""
from __future__ import annotations
import argparse, json, logging, shutil, sys
from pathlib import Path

from fetalvision.common.constants import HEAD_CLASS_NAMES
from fetalvision.common.config import load_config
from fetalvision.pipeline.utils import resolve_path, sha256_file, load_json_if_exists, write_json, setup_logging
from .onnx_utils import onnx_io_summary

log = logging.getLogger(__name__)


def resolve_detector_weights(cfg_det: dict) -> tuple[Path, bool]:
    real = resolve_path(cfg_det["weights"])
    if real.exists():
        return real, False
    si = resolve_path(cfg_det["stand_in_weights"])
    si.parent.mkdir(parents=True, exist_ok=True)
    if not si.exists():
        log.warning("real detector %s missing; downloading ultralytics %s as STAND-IN", real, si.name)
        from ultralytics import YOLO
        YOLO(si.name)  # downloads into CWD
        dl = Path.cwd() / si.name
        if dl.exists() and dl.resolve() != si.resolve():
            shutil.move(str(dl), str(si))
    log.warning("using STAND-IN detector weights %s", si)
    return si, True


def export_yolo_to_onnx(weights: str | Path, out_onnx: str | Path, imgsz: int = 640, opset: int = 17) -> dict:
    from ultralytics import YOLO
    weights, out_onnx = Path(weights), Path(out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    names = model.model.names if hasattr(model.model, "names") else {}
    produced = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False, nms=False, device="cpu")
    produced = Path(produced)
    if produced.resolve() != out_onnx.resolve():
        shutil.move(str(produced), str(out_onnx))
    io = onnx_io_summary(out_onnx)
    n_out_ch = io["outputs"][0]["shape"][1]
    n_classes = int(n_out_ch) - 4
    log.info("exported %s: %s -> %s (n_classes=%d)", out_onnx, io["inputs"][0]["shape"], io["outputs"][0]["shape"], n_classes)
    return {"io": io, "n_classes": n_classes, "class_names": [names[i] for i in sorted(names)] if names else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--stand-in", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    d = cfg["detector"]
    if a.weights:
        weights, stand_in = Path(a.weights), a.stand_in
    else:
        weights, stand_in = resolve_detector_weights(d)
        stand_in = stand_in or a.stand_in
    out_onnx = Path(a.out) if a.out else resolve_path(d["out_onnx"])
    out_meta = out_onnx.with_suffix(".json") if a.out else resolve_path(d["out_meta"])
    imgsz = int(a.imgsz or d.get("imgsz", 640))
    opset = int(cfg.get("opset", 17))
    info = export_yolo_to_onnx(weights, out_onnx, imgsz, opset)
    op = load_json_if_exists(d.get("operating_point_json"), "operating_point.json") or {}
    meta = {
        "model": "head_detector", "framework": "ultralytics", "arch": "yolo11n", "imgsz": imgsz, "opset": opset,
        "input_name": info["io"]["inputs"][0]["name"], "output_name": info["io"]["outputs"][0]["name"],
        "input_layout": "NCHW RGB float32 [0,1], letterbox 114", "output_layout": "[1, 4+n_classes, n_anchors] xywh (letterboxed px) + class scores",
        "n_classes": info["n_classes"],
        "class_names": info["class_names"] if stand_in else list(HEAD_CLASS_NAMES),
        "operating_point": {"conf": op.get("conf", 0.25), "iou": op.get("iou", 0.5), "source": "operating_point.json" if op else "default"},
        "source_weights": str(weights), "source_sha256": sha256_file(weights), "sha256": sha256_file(out_onnx),
        "stand_in": bool(stand_in), "stand_in_kind": ("pretrained COCO yolo11n (80 classes), NOT trained on fetal heads" if stand_in else None),
        "onnx_io": info["io"],
    }
    write_json(out_meta, meta)
    print(json.dumps({k: meta[k] for k in ["source_weights", "n_classes", "sha256", "stand_in", "operating_point"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
