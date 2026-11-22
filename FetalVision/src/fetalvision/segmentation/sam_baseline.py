"""Zero-shot MobileSAM baseline for fetal-head segmentation, prompted with (i) the GT box and (ii) the detector box.

Usage:  python -m fetalvision.segmentation.sam_baseline --config configs/phase2_segmentation.yaml

"Masks generated with SAM and verified": every SAM mask is compared with the sonographer-derived filled-ellipse mask
(Dice / IoU / HC error); summary statistics, the share of images with Dice < 0.9 and the worst cases are reported
so the agreement between SAM and the human annotation is stated explicitly. SAM masks are saved to
$FETALVISION_DATA/hc18/processed/sam_masks/<split>/<prompt>/ for reuse by segmentation.evaluate.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from fetalvision.common.config import load_config, results_dir, save_resolved_config, write_metrics
from fetalvision.common.metrics import dice_coefficient, iou_score
from fetalvision.common.seed import set_seed
from fetalvision.common import wandb_utils
from fetalvision.detection.utils import load_split, pick_device, run_predict
from fetalvision.segmentation.hc_measure import measure_hc

log = logging.getLogger("fetalvision.segmentation.sam_baseline")


def sam_mask_from_box(sam, img_bgr: np.ndarray, box, device: str) -> np.ndarray:
    """Run SAM with a single box prompt; returns a uint8 0/1 mask at image resolution (empty if SAM returns none)."""
    h, w = img_bgr.shape[:2]
    res = sam.predict(source=img_bgr, bboxes=[[float(v) for v in box]], device=device, verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return np.zeros((h, w), np.uint8)
    m = res.masks.data[0].cpu().numpy().astype(np.uint8)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def summarise(df: pd.DataFrame, tol_mm: float) -> dict:
    d = df.dice.values
    e = df.abs_err_mm.dropna().values
    return {"n": int(len(df)), "dice_mean": float(d.mean()), "dice_std": float(d.std(ddof=0)), "dice_median": float(np.median(d)),
            "dice_min": float(d.min()), "iou_mean": float(df.iou.mean()), "share_dice_lt_0.9": float((d < 0.9).mean()),
            "n_dice_lt_0.9": int((d < 0.9).sum()), "n_empty_masks": int((df.pred_area_px == 0).sum()),
            "hc_mae_mm": float(e.mean()) if len(e) else None, "hc_median_ae_mm": float(np.median(e)) if len(e) else None,
            f"share_abs_err_le_{tol_mm:g}mm": float((e <= tol_mm).mean()) if len(e) else None,
            "n_hc_valid": int(len(e))}


def example_figure(items: list[dict], out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.9 * n))
    axes = np.atleast_2d(axes)
    for row, it in zip(axes, items):
        img = it["img"]
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for box, color in [(it["gt_box"], (255, 140, 0)), (it.get("det_box"), (0, 200, 255))]:
            if box is not None and np.isfinite(box).all():
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 2)
        row[0].imshow(rgb)
        row[0].set_title(f"{it['filename']} – image (orange GT box, cyan detector box)", fontsize=8)
        row[1].imshow(it["gt"], cmap="gray")
        row[1].set_title("GT mask (sonographer ellipse, filled)", fontsize=8)
        row[2].imshow(it["sam_gt"], cmap="gray")
        row[2].set_title(f"SAM | GT-box prompt – Dice {it['dice_gt']:.3f}", fontsize=8)
        row[3].imshow(it["sam_det"], cmap="gray")
        row[3].set_title(f"SAM | detector-box prompt – Dice {it['dice_det']:.3f}", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.suptitle(title, fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(cfg: dict) -> dict:
    from ultralytics import SAM, YOLO

    set_seed(cfg["seed"])
    t0 = time.time()
    out = Path(cfg["sam_results_dir"])
    (out / "figures").mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, out)
    scfg = cfg["sam"]
    device = pick_device(scfg.get("device", "auto"))  # MobileSAM is faster on CPU than MPS for single-image prompts
    sam = SAM(scfg["weights"])
    detector = YOLO(cfg["detector_weights"])
    op = json.loads(Path(cfg["operating_point_json"]).read_text())
    processed = Path(cfg["splits_csv"]).parent
    raw_dir = Path(cfg["raw_dir"])
    orig_wh = tuple(cfg["original_size"])
    tol = float(cfg["evaluate"].get("hc_tolerance_mm", 3.0))
    run = wandb_utils.init_run(cfg, cfg["phase"], job_type="sam_baseline", tags=[cfg["phase"], "segmentation", "sam"])

    all_metrics, captions = {}, {}
    for split in scfg["splits"]:
        rows = load_split(cfg["splits_csv"], split)
        img_dir = processed / "yolo" / "images" / split
        det = run_predict(detector, [img_dir / f for f in rows.filename], conf=float(op["conf"]), iou=float(op["iou"]),
                          imgsz=int(op["imgsz"]), device=device)
        recs, cache = [], {}
        for r in rows.itertuples():
            img = cv2.imread(str(raw_dir / r.filename), cv2.IMREAD_GRAYSCALE)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            gt = (cv2.imread(str(processed / r.mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            gt_box = np.array([r.x1, r.y1, r.x2, r.y2], float)
            boxes, scores = det[r.filename]
            det_box = boxes[int(np.argmax(scores))] if len(scores) else None
            masks = {"gt_box": sam_mask_from_box(sam, img_bgr, gt_box, device)}
            masks["det_box"] = sam_mask_from_box(sam, img_bgr, det_box, device) if det_box is not None else np.zeros_like(gt)
            for prompt, m in masks.items():
                mdir = processed / "sam_masks" / split / prompt
                mdir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mdir / r.filename), m * 255)
                hc = measure_hc(m, r.pixel_size_mm, orig_wh)
                recs.append({"filename": r.filename, "split": split, "prompt": prompt, "dice": dice_coefficient(m, gt),
                             "iou": iou_score(m, gt), "pred_area_px": int(m.sum()), "gt_area_px": int(gt.sum()),
                             "hc_pred_mm": hc["hc_mm"], "hc_gt_mm": r.hc_mm,
                             "abs_err_mm": abs(hc["hc_mm"] - r.hc_mm) if hc["valid"] else np.nan,
                             "det_box_found": det_box is not None, "det_score": float(scores.max()) if len(scores) else np.nan})
            cache[r.filename] = {"img": img, "gt": gt, "gt_box": gt_box, "det_box": det_box,
                                 "sam_gt": masks["gt_box"], "sam_det": masks["det_box"],
                                 "dice_gt": recs[-2]["dice"], "dice_det": recs[-1]["dice"], "filename": r.filename}
        df = pd.DataFrame(recs)
        df.to_csv(out / f"predictions_{split}.csv", index=False)
        all_metrics[split] = {p: summarise(df[df.prompt == p], tol) for p in ["gt_box", "det_box"]}
        all_metrics[split]["n_images"] = int(len(rows))
        all_metrics[split]["n_images_without_detector_box"] = int((~df[df.prompt == "det_box"].det_box_found).sum())
        # figures: 8 random examples + 3 worst (by det-box Dice)
        rng = np.random.RandomState(cfg["seed"])
        ex = [cache[f] for f in rows.filename.iloc[rng.choice(len(rows), size=min(int(scfg["n_examples"]), len(rows)), replace=False)]]
        title = (f"MobileSAM zero-shot – {int(scfg['n_examples'])} random {split.upper()} images (seed {cfg['seed']}), 800x540 px; "
                 f"prompts: GT box / detector box at operating point (conf>={op['conf']:.2f}, NMS {op['iou']}); "
                 f"Dice vs sonographer mask")
        example_figure(ex, out / "figures" / f"sam_examples_{split}.png", title)
        captions[f"sam_examples_{split}.png"] = title
        worst_names = df[df.prompt == "det_box"].nsmallest(3, "dice").filename.tolist()
        title_w = (f"MobileSAM 3 worst cases by detector-box Dice – {split.upper()} split, 800x540 px, same prompts as above; "
                   f"compare with GT (sonographer ellipse)")
        example_figure([cache[f] for f in worst_names], out / "figures" / f"sam_worst3_{split}.png", title_w)
        captions[f"sam_worst3_{split}.png"] = title_w
        all_metrics[split]["worst3_det_box"] = df[df.prompt == "det_box"].nsmallest(3, "dice")[["filename", "dice", "iou", "abs_err_mm"]].to_dict("records")
        log.info("%s: %s", split, json.dumps({p: all_metrics[split][p]["dice_mean"] for p in ["gt_box", "det_box"]}))

    (out / "figures" / "captions.json").write_text(json.dumps(captions, indent=2))
    te = all_metrics.get("test", {})
    agreement = {
        "statement": ("Zero-shot MobileSAM prompted with the GT box reaches mean Dice {:.3f} (SD {:.3f}, {:.1%} of test images "
                      "below 0.9) against the sonographer masks; with the detector box: Dice {:.3f} (SD {:.3f}, {:.1%} below 0.9). "
                      "SAM masks are therefore usable as a verification / fallback tool but are not pixel-exact ellipses."
                      ).format(te["gt_box"]["dice_mean"], te["gt_box"]["dice_std"], te["gt_box"]["share_dice_lt_0.9"],
                               te["det_box"]["dice_mean"], te["det_box"]["dice_std"], te["det_box"]["share_dice_lt_0.9"])
        if te else "test split not evaluated"}
    extra = {"device": device, "wall_clock_seconds": time.time() - t0, "sam_weights": scfg["weights"],
             "detector_weights": cfg["detector_weights"], "operating_point": {k: op[k] for k in ["conf", "iou", "imgsz"]},
             "sam_vs_sonographer_agreement": agreement, "mask_dir": str(processed / "sam_masks")}
    n_test = te.get("n_images", 0) if te else 0
    write_metrics(out, "segmentation_sam", cfg["phase"], "test" if te else scfg["splits"][-1], list(orig_wh), n_test,
                  cfg["seed"], all_metrics, extra)
    flat = {f"{s}/{p}/dice_mean": all_metrics[s][p]["dice_mean"] for s in scfg["splits"] for p in ["gt_box", "det_box"]}
    wandb_utils.log_metrics(run, flat)
    for name in captions:
        wandb_utils.log_figure(run, name.replace(".png", ""), out / "figures" / name)
    wandb_utils.finish(run, flat)
    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
