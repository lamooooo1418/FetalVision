"""Export the segmentation_models_pytorch U-Net head segmenter to ONNX (opset 17, static input [1,3,H,W]).

    python -m fetalvision.export.export_segmenter --config configs/export.yaml [--weights models/head_segmenter.pt --meta models/head_segmenter.json]

models/head_segmenter.json (from the segmentation workstream) provides {encoder, input_size [W,H], ...}. The ONNX
graph outputs raw logits ('logits'); the pipeline applies the sigmoid. When the real weights are missing a
random-weight smp.Unet('mobilenet_v2') STAND-IN is created, saved as a state dict and flagged in the metadata.
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

from fetalvision.common.config import load_config
from fetalvision.common.seed import set_seed
from fetalvision.pipeline.utils import resolve_path, sha256_file, load_json_if_exists, write_json, setup_logging
from .onnx_utils import onnx_io_summary

log = logging.getLogger(__name__)
INPUT_NAME, OUTPUT_NAME = "image", "logits"


def build_unet(encoder: str, weights_path: str | Path | None):
    import torch, segmentation_models_pytorch as smp
    model = smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)
    if weights_path is not None:
        sd = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if not isinstance(sd, dict):  # a pickled full model
            model = sd
        else:
            model.load_state_dict(sd)
    return model.eval()


def normalise_source_meta(meta: dict, cfg_seg: dict) -> dict:
    """Accept the segmentation workstream's head_segmenter.json schema ({model:{arch,encoder,...}, input_size_wh,
    normalisation, threshold, original_size_wh, ...}) or a flat {encoder, input_size} and return a flat dict."""
    m = dict(meta or {})
    model = m.get("model") if isinstance(m.get("model"), dict) else {}
    out = {k: v for k, v in m.items() if k != "model"}
    out["encoder"] = m.get("encoder") or model.get("encoder") or cfg_seg.get("stand_in_encoder", "mobilenet_v2")
    out["arch"] = model.get("arch", m.get("arch", "Unet"))
    out["input_size"] = list(m.get("input_size") or m.get("input_size_wh") or cfg_seg.get("input_size", [512, 352]))
    norm = str(m.get("normalize") or m.get("normalisation") or "imagenet").lower()
    out["normalize"] = "imagenet" if "imagenet" in norm else ("none" if norm in ("none", "0-1", "unit") else norm)
    out["grayscale"] = bool(m.get("grayscale", "grayscale" in norm or "gray" in norm))
    out["normalisation_note"] = m.get("normalisation")
    out["mask_threshold"] = float(m.get("threshold", m.get("mask_threshold", 0.5)))
    out["training_meta"] = {k: m[k] for k in ("best_epoch", "best_val_dice", "seed", "git_commit", "original_size_wh") if k in m}
    return out


def resolve_segmenter_source(cfg_seg: dict, seed: int = 42) -> tuple[Path, dict, bool]:
    """(weights_path, meta, is_stand_in). Creates the random stand-in on first use."""
    real = resolve_path(cfg_seg["weights"])
    meta = normalise_source_meta(load_json_if_exists(cfg_seg.get("meta"), "head_segmenter.json") or {}, cfg_seg)
    if real.exists():
        return real, meta, False
    si = resolve_path(cfg_seg["stand_in_weights"])
    enc = cfg_seg.get("stand_in_encoder", "mobilenet_v2")
    if not si.exists():
        import torch
        set_seed(seed)
        log.warning("real segmenter %s missing; creating random-weight smp.Unet(%s) STAND-IN at %s", real, enc, si)
        si.parent.mkdir(parents=True, exist_ok=True)
        torch.save(build_unet(enc, None).state_dict(), str(si))
    log.warning("using STAND-IN segmenter weights %s", si)
    return si, normalise_source_meta({"encoder": enc, "input_size": cfg_seg.get("input_size", [512, 352])}, cfg_seg), True


def export_unet_to_onnx(model, out_onnx: str | Path, input_size_wh=(512, 352), opset: int = 17) -> dict:
    import torch, numpy as np
    out_onnx = Path(out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    w, h = int(input_size_wh[0]), int(input_size_wh[1])
    x = torch.randn(1, 3, h, w)
    torch.onnx.export(model, x, str(out_onnx), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                      opset_version=opset, dynamo=False, do_constant_folding=True)
    import onnxruntime as ort
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    got = sess.run(None, {INPUT_NAME: x.numpy()})[0]
    self_parity = float(np.abs(got - ref).max())
    log.info("exported %s; self-parity max|dlogit|=%.2e", out_onnx, self_parity)
    return {"io": onnx_io_summary(out_onnx), "self_parity_max_abs_diff": self_parity}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--meta", default=None, help="head_segmenter.json describing the weights")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stand-in", action="store_true")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    s = cfg["segmenter"]
    if a.weights:
        weights, stand_in = Path(a.weights), a.stand_in
        meta_in = normalise_source_meta(load_json_if_exists(a.meta or s.get("meta"), "head_segmenter.json") or {}, s)
    else:
        weights, meta_in, stand_in = resolve_segmenter_source(s, int(cfg.get("seed", 42)))
        stand_in = stand_in or a.stand_in
    out_onnx = Path(a.out) if a.out else resolve_path(s["out_onnx"])
    out_meta = out_onnx.with_suffix(".json") if a.out else resolve_path(s["out_meta"])
    opset = int(cfg.get("opset", 17))
    model = build_unet(meta_in["encoder"], weights)
    info = export_unet_to_onnx(model, out_onnx, meta_in["input_size"], opset)
    meta = {**meta_in,
            "model": "head_segmenter", "framework": "segmentation_models_pytorch", "arch": "Unet",
            "encoder": meta_in["encoder"], "input_size": [int(meta_in["input_size"][0]), int(meta_in["input_size"][1])],
            "input_size_order": "WH", "input_name": INPUT_NAME, "output_name": OUTPUT_NAME,
            "input_layout": ("NCHW grayscale replicated to 3 channels" if meta_in.get("grayscale") else "NCHW RGB") + ", [0,1] then ImageNet mean/std" if meta_in.get("normalize") == "imagenet" else "[0,1]",
            "output": "logits (apply sigmoid)", "normalize": meta_in.get("normalize", "imagenet"), "grayscale": bool(meta_in.get("grayscale", False)),
            "mask_threshold": meta_in.get("mask_threshold", 0.5), "opset": opset,
            "source_weights": str(weights), "source_sha256": sha256_file(weights), "sha256": sha256_file(out_onnx),
            "stand_in": bool(stand_in), "stand_in_kind": ("random-weight smp.Unet, NOT trained" if stand_in else None),
            "onnx_io": info["io"], "self_parity_max_abs_diff": info["self_parity_max_abs_diff"]}
    write_json(out_meta, meta)
    print(json.dumps({k: meta[k] for k in ["source_weights", "encoder", "input_size", "sha256", "stand_in"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
