"""Evaluate the fetal-head detector on the TEST split at the a-priori operating point, with an inference demo.

Usage:  python -m fetalvision.detection.evaluate --config configs/phase2_detection.yaml

Reports (a) ultralytics val metrics (standard protocol conf 0.001 / NMS 0.7, and at the operating point),
(b) our independent mAP50-95 via fetalvision.common.metrics.map_50_95_single_instance (top-1 box per image),
(c) P/R/F1 at the operating point with a FP taxonomy (duplicate / localisation / background) and recall by GT size,
(d) galleries of all FPs and FNs, (e) predictions_test.csv, metrics.json, 12 annotated demo images and a
10-second demo mp4 assembled from still test frames (HC18 has no video).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fetalvision.common.config import load_config, results_dir, write_metrics
from fetalvision.common.metrics import map_50_95_single_instance
from fetalvision.common.seed import set_seed
from fetalvision.common import wandb_utils
from fetalvision.detection.utils import (categorise_fp, gt_box, load_split, match_single_gt, pick_device, prf,
                                         run_predict)

log = logging.getLogger("fetalvision.detection.evaluate")


def draw_detections(img_gray: np.ndarray, gt, preds: list[tuple[np.ndarray, float, str]]) -> np.ndarray:
    """gt box in orange, predictions coloured by outcome (TP green, FP red). preds = [(box, score, outcome)]."""
    rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    if gt is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in gt]
        cv2.rectangle(rgb, (x1, y1), (x2 - 1, y2 - 1), (255, 140, 0), 2)
    for box, score, outcome in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = (0, 255, 0) if outcome == "TP" else (255, 0, 0)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rgb, f"{outcome} {score:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return rgb


def gallery(items: list[dict], img_dir: Path, out_png: Path, title: str, ncols: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(len(items), 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.1))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, it in zip(axes, items):
        img = cv2.imread(str(img_dir / it["filename"]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(draw_detections(img, it["gt"], it["preds"]))
        ax.set_title(it["title"], fontsize=8)
    fig.suptitle(title, fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(cfg: dict) -> dict:
    from ultralytics import YOLO

    set_seed(cfg["seed"])
    t_start = time.time()
    out = results_dir(cfg["phase"], cfg["run_name"], Path(cfg["results_dir"]).parent.parent)
    figs = out / "figures"
    device = pick_device(cfg.get("device", "auto"))
    op = json.loads((out / "operating_point.json").read_text())
    conf_op, iou_op, match_iou = float(op["conf"]), float(op["iou"]), float(op["match_iou"])
    ecfg = cfg["evaluate"]
    loc_low = float(ecfg.get("localisation_iou_low", 0.1))
    imgsz = int(cfg["imgsz"])

    all_rows = pd.read_csv(cfg["splits_csv"])
    test = load_split(cfg["splits_csv"], "test")
    img_dir = Path(cfg["data_yaml"]).parent / "images" / "test"
    paths = [img_dir / f for f in test.filename]
    model = YOLO(cfg["export_weights"])

    # ---- (a) ultralytics metrics --------------------------------------------------------------------------------
    project = Path(cfg["project_dir"]).resolve()  # absolute: ultralytics nests relative paths under runs/detect/
    v_std = model.val(data=cfg["data_yaml"], split="test", imgsz=imgsz, batch=int(cfg["batch"]), device=device,
                      conf=0.001, iou=0.7, plots=True, verbose=False, project=str(project), name="val_test_std",
                      exist_ok=True)
    v_op = model.val(data=cfg["data_yaml"], split="test", imgsz=imgsz, batch=int(cfg["batch"]), device=device,
                     conf=conf_op, iou=iou_op, plots=False, verbose=False, project=str(project), name="val_test_op",
                     exist_ok=True)
    ultra_std = {"mAP50": float(v_std.box.map50), "mAP50-95": float(v_std.box.map), "precision": float(v_std.box.mp),
                 "recall": float(v_std.box.mr), "per_class_AP50": {"fetal_head": float(v_std.box.ap50[0])},
                 "per_class_AP50-95": {"fetal_head": float(v_std.box.ap[0])}, "conf": 0.001, "nms_iou": 0.7}
    ultra_op = {"mAP50": float(v_op.box.map50), "mAP50-95": float(v_op.box.map), "precision": float(v_op.box.mp),
                "recall": float(v_op.box.mr), "conf": conf_op, "nms_iou": iou_op,
                "note": "mAP at the operating-point conf is computed on a truncated PR curve; use the standard protocol for mAP"}
    for p in (project / "val_test_std").glob("*.png"):
        shutil.copy2(p, figs / f"ultralytics_test_{p.name}")

    # ---- (b) independent mAP50-95 (top-1 box per image, raw conf 0.001, NMS = operating-point IoU) --------------
    preds_raw = run_predict(model, paths, conf=0.001, iou=iou_op, imgsz=imgsz, device=device)
    scores, ious = [], []
    for r in test.itertuples():
        boxes, sc = preds_raw[r.filename]
        if len(sc):
            k = int(np.argmax(sc))
            m = match_single_gt(boxes[k:k + 1], sc[k:k + 1], gt_box(r), match_iou)
            scores.append(float(sc[k]))
            ious.append(float(m["ious"][0]))
    own_map = map_50_95_single_instance(np.array(scores), np.array(ious), n_gt=len(test))
    own_map["n_predictions"] = len(scores)
    own_map["protocol"] = "top-1 box per image at conf 0.001 (NMS IoU = operating point), 101-point interpolation"

    # ---- (c) operating point: P/R/F1, FP taxonomy, recall by size ----------------------------------------------
    preds_op = run_predict(model, paths, conf=conf_op, iou=iou_op, imgsz=imgsz, device=device)
    tp = fp = fn = 0
    rows, fp_items, fn_items, tp_items = [], [], [], []
    fp_types = {"duplicate": 0, "localisation": 0, "background": 0}
    per_image = []
    for r in test.itertuples():
        gt = gt_box(r)
        boxes, sc = preds_op[r.filename]
        m = match_single_gt(boxes, sc, gt, match_iou)
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
        preds_draw = []
        if m["tp_idx"] is not None:
            i = m["tp_idx"]
            rows.append({"filename": r.filename, "gt_x1": gt[0], "gt_y1": gt[1], "gt_x2": gt[2], "gt_y2": gt[3],
                         "pred_x1": boxes[i, 0], "pred_y1": boxes[i, 1], "pred_x2": boxes[i, 2], "pred_y2": boxes[i, 3],
                         "score": sc[i], "iou": m["ious"][i], "outcome": "TP", "size_cat": r.size_cat, "hc_mm": r.hc_mm})
            preds_draw.append((boxes[i], float(sc[i]), "TP"))
            tp_items.append({"filename": r.filename, "gt": gt, "iou": float(m["ious"][i]), "score": float(sc[i]),
                             "preds": [(boxes[i], float(sc[i]), "TP")]})
        else:
            rows.append({"filename": r.filename, "gt_x1": gt[0], "gt_y1": gt[1], "gt_x2": gt[2], "gt_y2": gt[3],
                         "pred_x1": np.nan, "pred_y1": np.nan, "pred_x2": np.nan, "pred_y2": np.nan,
                         "score": np.nan, "iou": float(m["ious"].max()) if len(m["ious"]) else 0.0, "outcome": "FN",
                         "size_cat": r.size_cat, "hc_mm": r.hc_mm})
        for i in m["fp_idx"]:
            cat = categorise_fp(float(m["ious"][i]), match_iou, loc_low)
            fp_types[cat] += 1
            rows.append({"filename": r.filename, "gt_x1": gt[0], "gt_y1": gt[1], "gt_x2": gt[2], "gt_y2": gt[3],
                         "pred_x1": boxes[i, 0], "pred_y1": boxes[i, 1], "pred_x2": boxes[i, 2], "pred_y2": boxes[i, 3],
                         "score": sc[i], "iou": m["ious"][i], "outcome": f"FP_{cat}", "size_cat": r.size_cat,
                         "hc_mm": r.hc_mm})
            preds_draw.append((boxes[i], float(sc[i]), f"FP:{cat}"))
        if m["fp_idx"]:
            fp_items.append({"filename": r.filename, "gt": gt, "preds": preds_draw,
                             "title": f"{r.filename} – {len(m['fp_idx'])} FP ({', '.join(categorise_fp(float(m['ious'][i]), match_iou, loc_low) for i in m['fp_idx'])}), GT orange"})
        if m["fn"]:
            fn_items.append({"filename": r.filename, "gt": gt, "preds": preds_draw,
                             "title": f"{r.filename} – FN (no box with IoU>={match_iou} at conf>={conf_op:.2f}); best IoU {float(m['ious'].max()) if len(m['ious']) else 0:.2f}; GT orange"})
        per_image.append({"filename": r.filename, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "size_cat": r.size_cat,
                          "box_area_px": r.box_area_px, "hc_mm": r.hc_mm})
    p_op, r_op, f1_op = prf(tp, fp, fn)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out / "predictions_test.csv", index=False)
    pi = pd.DataFrame(per_image)

    # recall by GT size: dataset-wide area-fraction tertiles (size_cat from splits.csv) and absolute px-area tertiles
    frac_t = np.quantile(all_rows.box_area_frac.dropna(), [1 / 3, 2 / 3])
    px_t = np.quantile(all_rows.box_area_px.dropna(), [1 / 3, 2 / 3])
    pi["size_cat_px"] = np.where(pi.box_area_px < px_t[0], "small", np.where(pi.box_area_px < px_t[1], "medium", "large"))
    hc_t = np.quantile(all_rows.hc_mm.dropna(), [1 / 3, 2 / 3])
    pi["hc_cat"] = np.where(pi.hc_mm < hc_t[0], "small", np.where(pi.hc_mm < hc_t[1], "medium", "large"))

    def _by(col):
        d = {}
        for cat in ["small", "medium", "large"]:
            s = pi[pi[col] == cat]
            d[cat] = {"n_gt": int(len(s)), "tp": int(s.tp.sum()), "fn": int(s.fn.sum()), "fp": int(s.fp.sum()),
                      "recall": float(s.tp.sum() / len(s)) if len(s) else None}
        return d

    by_size = {"area_fraction_tertiles": {"boundaries": {"small_lt": float(frac_t[0]), "large_ge": float(frac_t[1])},
                                          "per_category": _by("size_cat")},
               "absolute_px_area_tertiles": {"boundaries_px": {"small_lt": float(px_t[0]), "large_ge": float(px_t[1])},
                                             "per_category": _by("size_cat_px")},
               "hc_mm_tertiles": {"boundaries_mm": {"small_lt": float(hc_t[0]), "large_ge": float(hc_t[1])},
                                  "per_category": _by("hc_cat")}}

    # ---- (d) galleries -----------------------------------------------------------------------------------------
    res_note = f"TEST split ({len(test)} images, 800x540 px, inference imgsz {imgsz}, conf >= {conf_op:.2f}, NMS IoU {iou_op}, match IoU {match_iou})"
    captions = {}
    if fp_items:
        t = f"All {len(fp_items)} test images with false positives – {res_note}. Orange = GT, green = TP, red = FP with category"
        gallery(fp_items, img_dir, figs / "gallery_false_positives.png", t)
        captions["gallery_false_positives.png"] = t
    if fn_items:
        t = f"All {len(fn_items)} false negatives – {res_note}. Orange = GT; red boxes (if any) are low-IoU predictions"
        gallery(fn_items, img_dir, figs / "gallery_false_negatives.png", t)
        captions["gallery_false_negatives.png"] = t
    if len(fp_items) + len(fn_items) < 6:
        worst = sorted(tp_items, key=lambda d: d["iou"])[:6]
        for w in worst:
            w["title"] = f"{w['filename']} – TP, IoU {w['iou']:.3f}, score {w['score']:.2f} (orange GT, green pred)"
        t = f"6 lowest-IoU true positives (fewer than 6 FP+FN) – {res_note}"
        gallery(worst, img_dir, figs / "gallery_lowest_iou_detections.png", t)
        captions["gallery_lowest_iou_detections.png"] = t

    # ---- (e) inference demo: 12 annotated images + synthetic video --------------------------------------------
    rng = np.random.RandomState(cfg["seed"])
    demo_idx = rng.choice(len(test), size=min(int(ecfg["n_demo_images"]), len(test)), replace=False)
    demo_dir = figs / "demo"
    demo_dir.mkdir(exist_ok=True)
    demo_items = []
    for k, i in enumerate(demo_idx):
        r = test.iloc[i]
        boxes, sc = preds_op[r.filename]
        m = match_single_gt(boxes, sc, gt_box(r), match_iou)
        preds_draw = [(boxes[j], float(sc[j]), "TP" if j == m["tp_idx"] else "FP") for j in range(len(sc))]
        img = cv2.imread(str(img_dir / r.filename), cv2.IMREAD_GRAYSCALE)
        vis = draw_detections(img, gt_box(r), preds_draw)
        cv2.putText(vis, f"{r.filename} test | imgsz {imgsz} conf>={conf_op:.2f} NMS {iou_op}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(demo_dir / f"demo_{k:02d}_{Path(r.filename).stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        demo_items.append({"filename": r.filename, "gt": gt_box(r), "preds": preds_draw,
                           "title": f"{r.filename} – {len(sc)} box(es), top score {sc.max() if len(sc) else 0:.2f}"})
    t = f"Inference demo: {len(demo_items)} random test images (seed {cfg['seed']}) – {res_note}; orange GT, green TP, red FP"
    gallery(demo_items, img_dir, figs / "demo_gallery.png", t, ncols=4)
    captions["demo_gallery.png"] = t

    n_frames, fps = int(ecfg["video_frames"]), int(ecfg["video_fps"])
    frame_idx = rng.choice(len(test), size=min(n_frames, len(test)), replace=False)
    video_path = out / "demo_video.mp4"
    h, w = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE).shape
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in frame_idx:
        r = test.iloc[i]
        frame = cv2.imread(str(img_dir / r.filename), cv2.IMREAD_GRAYSCALE)
        # pipeline-like inference on the raw frame (no GT used)
        res = model.predict(source=cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), conf=conf_op, iou=iou_op, imgsz=imgsz,
                            device=device, verbose=False)[0]
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for b, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"fetal_head {s:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"{r.filename} (test) | synthetic sequence of stills | conf>={conf_op:.2f}", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(vis)
    writer.release()
    (out / "demo_video_caption.txt").write_text(
        f"demo_video.mp4: {len(frame_idx)} TEST-split HC18 still frames at {fps} fps ({len(frame_idx) / fps:.0f} s), {w}x{h} px. "
        f"This is a SYNTHETIC sequence assembled from independent still images (HC18 contains no video); boxes and scores "
        f"come from the YOLO11n fetal-head detector at the operating point (imgsz {imgsz}, conf >= {conf_op:.2f}, NMS IoU {iou_op}). "
        f"Green = prediction; no ground truth is drawn. Seed {cfg['seed']}.\n")
    captions["demo_video.mp4"] = (out / "demo_video_caption.txt").read_text().strip()

    caps_path = figs / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    caps.update(captions)
    caps_path.write_text(json.dumps(caps, indent=2))

    # ---- metrics.json -------------------------------------------------------------------------------------------
    wall = time.time() - t_start
    metrics = {
        "operating_point": {"conf": conf_op, "nms_iou": iou_op, "match_iou": match_iou, "rule": op["rule"],
                            "val_precision": op["val_precision"], "val_recall": op["val_recall"], "val_f1": op["val_f1"]},
        "test_at_operating_point": {"precision": p_op, "recall": r_op, "f1": f1_op, "tp": tp, "fp": fp, "fn": fn,
                                    "n_images_with_fp": len(fp_items), "n_fn": len(fn_items)},
        "fp_breakdown": {**fp_types, "definitions": {"duplicate": f"IoU>={match_iou} with GT (second box on the same head)",
                                                      "localisation": f"{loc_low}<=IoU<{match_iou}",
                                                      "background": f"IoU<{loc_low}"}},
        "recall_by_gt_size": by_size,
        "ultralytics_val_standard": ultra_std,
        "ultralytics_val_at_operating_point": ultra_op,
        "independent_map": own_map,
        "acceptance": {"mAP50_target": 0.95, "recall_target": 0.98,
                       "mAP50_pass": bool(ultra_std["mAP50"] >= 0.95), "recall_pass": bool(r_op >= 0.98)},
    }
    extra = {"device": device, "wall_clock_seconds": wall, "weights": cfg["export_weights"], "imgsz": imgsz,
             "demo_video": str(video_path), "n_demo_images": len(demo_items)}
    write_metrics(out, cfg["run_name"], cfg["phase"], "test", [imgsz, imgsz], len(test), cfg["seed"], metrics, extra)
    run = wandb_utils.init_run(cfg, cfg["phase"], job_type="evaluate", tags=[cfg["phase"], "detection", "test"])
    flat = {"test/precision": p_op, "test/recall": r_op, "test/f1": f1_op, "test/mAP50": ultra_std["mAP50"],
            "test/mAP50-95": ultra_std["mAP50-95"], "test/own_mAP50-95": own_map["mAP50-95"], "test/fp": fp, "test/fn": fn}
    wandb_utils.log_metrics(run, flat)
    for name in ["gallery_false_positives.png", "gallery_false_negatives.png", "gallery_lowest_iou_detections.png",
                 "demo_gallery.png"]:
        if (figs / name).exists():
            wandb_utils.log_figure(run, name.replace(".png", ""), figs / name)
    wandb_utils.finish(run, flat)
    log.info("TEST @ op: P %.4f R %.4f F1 %.4f | ultralytics mAP50 %.4f mAP50-95 %.4f | own mAP50-95 %.4f | FP %s",
             p_op, r_op, f1_op, ultra_std["mAP50"], ultra_std["mAP50-95"], own_map["mAP50-95"], fp_types)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
