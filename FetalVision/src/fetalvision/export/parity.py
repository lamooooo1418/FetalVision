"""Native framework vs ONNX Runtime parity on real ultrasound images (+ random noise images).

    python -m fetalvision.export.parity --kind classifier|detector|segmenter|all --config configs/export.yaml [--conf 0.001]

Writes results/phase2/export/parity_<kind>.json and parity_<kind>.md. Acceptance criteria (fixed in advance):
classifier max|dp| < 1e-4 and 100 % argmax agreement; detector best-box IoU >= 0.99 and |dscore| < 1e-3;
segmenter mask Dice >= 0.999.
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
import numpy as np

from fetalvision.common.config import load_config
from fetalvision.common.metrics import box_iou, dice_coefficient
from fetalvision.common.seed import set_seed
from fetalvision.pipeline.utils import resolve_path, load_json_if_exists, write_json, setup_logging, package_versions, hardware_description
from fetalvision.pipeline import preprocess as pre
from fetalvision.pipeline import postprocess as post
from .onnx_utils import load_parity_images

log = logging.getLogger(__name__)


def _images(cfg: dict) -> tuple[list[np.ndarray], list[str], str]:
    p = cfg["parity"]
    d = resolve_path(p["images_dir"])
    if d is None or not d.exists() or not any(d.iterdir()):
        alt = Path(os.environ.get("FETALVISION_DATA", "data")) / "hc18" / "raw" / "training_set"
        log.warning("parity images dir %s missing; falling back to %s", d, alt)
        d = alt
    imgs, names = load_parity_images(d, int(p.get("n_real", 32)), int(p.get("n_noise", 16)), seed=int(cfg.get("seed", 42)))
    return imgs, names, str(d)


def _ort_session(path: Path, providers=("CPUExecutionProvider",)):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=list(providers))


def _lat(fn, xs, warm: int = 3) -> list[float]:
    for x in xs[:warm]:
        fn(x)
    out = []
    for x in xs:
        t0 = time.perf_counter(); fn(x); out.append((time.perf_counter() - t0) * 1000)
    return out


def _pct(v):
    v = np.asarray(v, dtype=float)
    return {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)), "mean_ms": float(v.mean())}


# ---------------------------------------------------------------------------------------------------------------------
def parity_classifier(cfg: dict, providers) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    c = cfg["classifier"]
    onnx_path = resolve_path(c["out_onnx"]); meta = load_json_if_exists(c["out_meta"], "plane_classifier.json") or {}
    keras_path = Path(meta.get("source_keras") or resolve_path(c["stand_in_keras"]))
    model = keras.models.load_model(str(keras_path), compile=False)
    sess = _ort_session(onnx_path, providers)
    in_name = sess.get_inputs()[0].name
    imgs, names, src = _images(cfg)
    hw = tuple(meta.get("input_size", c.get("input_size", [224, 224])))
    xs = [pre.classifier_input(im, hw) for im in imgs]
    nat = np.concatenate([np.asarray(model(x, training=False)) for x in xs], 0)
    onx = np.concatenate([sess.run(None, {in_name: x})[0] for x in xs], 0)
    diff = np.abs(nat.astype(np.float64) - onx.astype(np.float64))
    per_img = diff.max(1)
    tol = cfg["parity"]["tolerances"]
    res = {
        "kind": "classifier", "onnx": str(onnx_path), "native": str(keras_path), "native_framework": f"keras {keras.__version__}",
        "images_dir": src, "n_images": len(imgs), "n_real": sum(not n.startswith("noise") for n in names), "n_noise": sum(n.startswith("noise") for n in names),
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "argmax_agreement": float((nat.argmax(1) == onx.argmax(1)).mean()),
        "max_abs_diff_real": float(per_img[[not n.startswith("noise") for n in names]].max()) if any(not n.startswith("noise") for n in names) else None,
        "max_abs_diff_noise": float(per_img[[n.startswith("noise") for n in names]].max()) if any(n.startswith("noise") for n in names) else None,
        "latency_native": _pct(_lat(lambda x: model(x, training=False), xs)),
        "latency_onnx": _pct(_lat(lambda x: sess.run(None, {in_name: x}), xs)),
        "providers": sess.get_providers(), "stand_in": bool(meta.get("stand_in", False)), "stand_in_kind": meta.get("stand_in_kind"),
        "criteria": {"max_abs_diff_lt": tol["classifier_max_abs_diff"], "argmax_agreement_eq": 1.0},
    }
    res["passed"] = bool(res["max_abs_diff"] < tol["classifier_max_abs_diff"] and res["argmax_agreement"] == 1.0)
    return res


def parity_detector(cfg: dict, providers, conf: float | None, iou: float | None) -> dict:
    import torch
    from ultralytics import YOLO
    d = cfg["detector"]
    onnx_path = resolve_path(d["out_onnx"]); meta = load_json_if_exists(d["out_meta"], "head_detector.json") or {}
    weights = Path(meta.get("source_weights") or resolve_path(d["stand_in_weights"]))
    imgsz = int(meta.get("imgsz", d.get("imgsz", 640)))
    op = meta.get("operating_point", {})
    conf = float(conf if conf is not None else (cfg["parity"].get("detector_parity_conf") or op.get("conf", 0.25)))
    iou = float(iou if iou is not None else op.get("iou", 0.5))
    model = YOLO(str(weights))
    sess = _ort_session(onnx_path, providers)
    in_name = sess.get_inputs()[0].name
    imgs, names, src = _images(cfg)
    max_det = 300
    # native: ultralytics predict with rect=False so it letterboxes to the static 640x640 of the ONNX graph (predict defaults to
    # rect=True -> 640x448 for HC18, which is a different network input and not a parity question)
    nat_boxes, nat_scores, nat_raw = [], [], []
    for im in imgs:
        r = model.predict(im, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device="cpu", verbose=False, rect=False)[0]
        nat_boxes.append(r.boxes.xyxy.cpu().numpy()); nat_scores.append(r.boxes.conf.cpu().numpy())
    # raw-tensor parity: torch forward vs ORT on the identical letterboxed tensor
    torch_model = model.model.eval()
    xs = [pre.detector_input(im, imgsz) for im in imgs]
    raw_diffs = []
    onx_boxes, onx_scores = [], []
    for x, params in xs:
        with torch.no_grad():
            y = torch_model(torch.from_numpy(x))
            y = y[0] if isinstance(y, (list, tuple)) else y
        raw_o = sess.run(None, {in_name: x})[0]
        raw_diffs.append(float(np.abs(y.numpy().astype(np.float64) - raw_o.astype(np.float64)).max()))
        dec = post.decode_yolo(raw_o, conf, iou, params, max_det=max_det)
        onx_boxes.append(dec["boxes_xyxy"]); onx_scores.append(dec["scores"])
    best_ious, best_score_diffs, matched_ious, matched_sdiffs, count_agree = [], [], [], [], 0
    n_with_boxes = 0
    for nb, ns, ob, os_ in zip(nat_boxes, nat_scores, onx_boxes, onx_scores):
        count_agree += int(len(nb) == len(ob))
        if len(nb) and len(ob):
            n_with_boxes += 1
            i, j = int(np.argmax(ns)), int(np.argmax(os_))
            best_ious.append(box_iou(nb[i], ob[j])); best_score_diffs.append(abs(float(ns[i]) - float(os_[j])))
            for k in range(len(nb)):
                ious = np.array([box_iou(nb[k], b) for b in ob]); m = int(ious.argmax())
                matched_ious.append(float(ious[m])); matched_sdiffs.append(abs(float(ns[k]) - float(os_[m])))
    tol = cfg["parity"]["tolerances"]
    res = {
        "kind": "detector", "onnx": str(onnx_path), "native": str(weights), "native_framework": f"ultralytics {__import__('ultralytics').__version__} / torch {torch.__version__}",
        "images_dir": src, "n_images": len(imgs), "imgsz": imgsz, "conf": conf, "iou": iou, "n_classes": int(meta.get("n_classes", -1)),
        "raw_output_max_abs_diff": float(max(raw_diffs)), "raw_output_mean_max_abs_diff": float(np.mean(raw_diffs)),
        "raw_output_max_abs_diff_relative_to_imgsz": float(max(raw_diffs)) / imgsz, "native_rect": False,
        "n_images_with_boxes_both": n_with_boxes, "n_images_native_boxes": int(sum(len(b) > 0 for b in nat_boxes)),
        "n_images_onnx_boxes": int(sum(len(b) > 0 for b in onx_boxes)), "box_count_agreement": count_agree / len(imgs),
        "best_box_iou_min": float(min(best_ious)) if best_ious else None, "best_box_iou_mean": float(np.mean(best_ious)) if best_ious else None,
        "best_box_score_max_abs_diff": float(max(best_score_diffs)) if best_score_diffs else None,
        "all_boxes_iou_min": float(min(matched_ious)) if matched_ious else None, "all_boxes_iou_mean": float(np.mean(matched_ious)) if matched_ious else None,
        "all_boxes_score_max_abs_diff": float(max(matched_sdiffs)) if matched_sdiffs else None, "n_boxes_native_total": int(sum(len(b) for b in nat_boxes)),
        "latency_native_predict": _pct(_lat(lambda im: model.predict(im, imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False, rect=False), imgs)),
        "latency_onnx_decode": _pct(_lat(lambda im: post.decode_yolo(sess.run(None, {in_name: pre.detector_input(im, imgsz)[0]})[0], conf, iou, pre.detector_input(im, imgsz)[1]), imgs)),
        "providers": sess.get_providers(), "stand_in": bool(meta.get("stand_in", False)), "stand_in_kind": meta.get("stand_in_kind"),
        "criteria": {"best_box_iou_ge": tol["detector_min_box_iou"], "best_box_score_diff_lt": tol["detector_max_score_diff"]},
    }
    res["passed"] = bool(best_ious and res["best_box_iou_min"] >= tol["detector_min_box_iou"] and res["best_box_score_max_abs_diff"] < tol["detector_max_score_diff"])
    if not best_ious:
        res["note"] = "no image produced a box in both backends at this conf; raw-tensor parity only"
    return res


def parity_segmenter(cfg: dict, providers) -> dict:
    import torch
    from .export_segmenter import build_unet
    s = cfg["segmenter"]
    onnx_path = resolve_path(s["out_onnx"]); meta = load_json_if_exists(s["out_meta"], "head_segmenter.json") or {}
    weights = Path(meta.get("source_weights") or resolve_path(s["stand_in_weights"]))
    wh = tuple(meta.get("input_size", s.get("input_size", [512, 352])))
    model = build_unet(meta.get("encoder", s.get("stand_in_encoder", "mobilenet_v2")), weights)
    sess = _ort_session(onnx_path, providers)
    in_name = sess.get_inputs()[0].name
    imgs, names, src = _images(cfg)
    # crops: a central box covering ~60 % of each image (what a head detection would hand over)
    xs = []
    for im in imgs:
        h, w = im.shape[:2]
        box = pre.expand_box([0.2 * w, 0.2 * h, 0.8 * w, 0.8 * h], 0.10, (h, w))
        xs.append(pre.segmenter_input(im[box[1]:box[3], box[0]:box[2]], wh, meta.get("normalize", "imagenet"), bool(meta.get("grayscale", False))))
    dices, pdiffs = [], []
    thr = 0.5
    for x in xs:
        with torch.no_grad():
            pn = torch.sigmoid(model(torch.from_numpy(x))).numpy()[0, 0]
        po = post.sigmoid(sess.run(None, {in_name: x})[0][0, 0])
        pdiffs.append(float(np.abs(pn.astype(np.float64) - po.astype(np.float64)).max()))
        dices.append(dice_coefficient(po >= thr, pn >= thr))
    tol = cfg["parity"]["tolerances"]
    res = {
        "kind": "segmenter", "onnx": str(onnx_path), "native": str(weights), "native_framework": f"segmentation_models_pytorch / torch {torch.__version__}",
        "images_dir": src, "n_images": len(imgs), "input_size_wh": list(wh), "mask_threshold": thr,
        "prob_max_abs_diff": float(max(pdiffs)), "prob_mean_max_abs_diff": float(np.mean(pdiffs)),
        "dice_min": float(min(dices)), "dice_mean": float(np.mean(dices)),
        "latency_native": _pct(_lat(lambda x: model(torch.from_numpy(x)), xs)) if True else None,
        "latency_onnx": _pct(_lat(lambda x: sess.run(None, {in_name: x}), xs)),
        "providers": sess.get_providers(), "stand_in": bool(meta.get("stand_in", False)), "stand_in_kind": meta.get("stand_in_kind"),
        "criteria": {"dice_ge": tol["segmenter_min_dice"]},
    }
    res["passed"] = bool(res["dice_min"] >= tol["segmenter_min_dice"])
    return res


def to_markdown(res: dict) -> str:
    k = res["kind"]
    rows = [("ONNX", Path(res["onnx"]).name), ("native", f"{Path(res['native']).name} ({res['native_framework']})"),
            ("images", f"{res['n_images']} ({res.get('n_real', res['n_images'] - 16)} real HC18 + noise)"),
            ("stand-in", f"{res['stand_in']} {res.get('stand_in_kind') or ''}"), ("providers", ", ".join(res["providers"]))]
    if k == "classifier":
        rows += [("max |Δprob|", f"{res['max_abs_diff']:.3e} (criterion < {res['criteria']['max_abs_diff_lt']:.0e})"),
                 ("mean |Δprob|", f"{res['mean_abs_diff']:.3e}"), ("argmax agreement", f"{100 * res['argmax_agreement']:.1f} %"),
                 ("latency native p50", f"{res['latency_native']['p50_ms']:.1f} ms"), ("latency ONNX p50", f"{res['latency_onnx']['p50_ms']:.1f} ms")]
    elif k == "detector":
        rows += [("conf / iou", f"{res['conf']} / {res['iou']}"), ("raw output max |Δ|", f"{res['raw_output_max_abs_diff']:.3e}"),
                 ("images with boxes (native/onnx)", f"{res['n_images_native_boxes']} / {res['n_images_onnx_boxes']}"),
                 ("best-box IoU min / mean", f"{res['best_box_iou_min']} / {res['best_box_iou_mean']} (criterion ≥ {res['criteria']['best_box_iou_ge']})"),
                 ("best-box |Δscore| max", f"{res['best_box_score_max_abs_diff']} (criterion < {res['criteria']['best_box_score_diff_lt']})"),
                 ("all boxes IoU min / |Δscore| max", f"{res['all_boxes_iou_min']} / {res['all_boxes_score_max_abs_diff']}"),
                 ("latency native predict p50", f"{res['latency_native_predict']['p50_ms']:.1f} ms"), ("latency ONNX+decode p50", f"{res['latency_onnx_decode']['p50_ms']:.1f} ms")]
    else:
        rows += [("prob max |Δ|", f"{res['prob_max_abs_diff']:.3e}"), ("Dice min / mean", f"{res['dice_min']:.6f} / {res['dice_mean']:.6f} (criterion ≥ {res['criteria']['dice_ge']})"),
                 ("latency native p50", f"{res['latency_native']['p50_ms']:.1f} ms"), ("latency ONNX p50", f"{res['latency_onnx']['p50_ms']:.1f} ms")]
    rows.append(("PASSED", str(res["passed"])))
    out = [f"## Parity: {k}", "", "| item | value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    if res.get("note"):
        out.append(f"\nNote: {res['note']}")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["classifier", "detector", "segmenter", "all"], default="all")
    ap.add_argument("--config", default="configs/export.yaml")
    ap.add_argument("--conf", type=float, default=None, help="detector conf for parity (stand-in yolo11n needs ~0.001 on ultrasound)")
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--providers", default="CPUExecutionProvider", help="comma-separated ORT providers")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config)
    set_seed(int(cfg.get("seed", 42)))
    providers = [p.strip() for p in a.providers.split(",") if p.strip()]
    out_dir = resolve_path(cfg["parity"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    kinds = ["classifier", "detector", "segmenter"] if a.kind == "all" else [a.kind]
    rc = 0
    for k in kinds:
        log.info("=== parity: %s ===", k)
        res = {"classifier": lambda: parity_classifier(cfg, providers), "detector": lambda: parity_detector(cfg, providers, a.conf, a.iou),
               "segmenter": lambda: parity_segmenter(cfg, providers)}[k]()
        res["versions"] = package_versions(); res["hardware"] = hardware_description()
        write_json(out_dir / f"parity_{k}.json", res)
        (out_dir / f"parity_{k}.md").write_text(to_markdown(res))
        print(to_markdown(res))
        rc |= 0 if res["passed"] else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
