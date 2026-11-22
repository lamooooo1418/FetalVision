"""Evaluate the fetal-head U-Net on the TEST split: masks, HC measurement, full detector->U-Net chain, SAM side by side.

Usage:  python -m fetalvision.segmentation.evaluate --config configs/phase2_segmentation.yaml

Variants evaluated: (1) full-image U-Net; (2) detector-box chain: detector box at the operating point -> crop
(+margin, padded to the network aspect) -> U-Net -> paste back (falls back to full-image when no box). The SAM
baseline numbers (GT-box / detector-box prompts) are read from results/phase2/segmentation_sam/metrics.json.
Mask mAP50-95 uses fetalvision.common.metrics.map_50_95_single_instance with score = mean foreground probability
inside the predicted mask.
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
from fetalvision.common.metrics import dice_coefficient, iou_score, map_50_95_single_instance
from fetalvision.common.seed import set_seed
from fetalvision.common import wandb_utils
from fetalvision.detection.utils import load_split, pick_device, run_predict
from fetalvision.segmentation.hc_measure import ellipse_to_mask, largest_component, measure_hc
from fetalvision.segmentation.model import load_checkpoint, predict_prob, predict_prob_in_box

log = logging.getLogger("fetalvision.segmentation.evaluate")


def hausdorff95(a: np.ndarray, b: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance (px) between the contours of two binary masks."""
    from scipy.spatial.distance import cdist

    def contour_pts(m):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        return np.concatenate([c.reshape(-1, 2) for c in cs]).astype(np.float32)

    pa, pb = contour_pts(a), contour_pts(b)
    if pa is None or pb is None:
        return float("nan")
    if len(pa) > 1500:
        pa = pa[np.linspace(0, len(pa) - 1, 1500).astype(int)]
    if len(pb) > 1500:
        pb = pb[np.linspace(0, len(pb) - 1, 1500).astype(int)]
    d = cdist(pa, pb)
    return float(max(np.percentile(d.min(axis=1), 95), np.percentile(d.min(axis=0), 95)))


def failure_statement(pred: np.ndarray, gt: np.ndarray, dice: float) -> str:
    """Heuristic, explicit description of what went wrong for a low-Dice case (rule-based, reviewed by the team)."""
    pa, ga = pred.sum(), gt.sum()
    n_cc = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)[0] - 1
    parts = []
    if pa == 0:
        return "empty prediction: the network found no head (no foreground above threshold)"
    ratio = pa / max(ga, 1)
    if n_cc > 1:
        parts.append(f"fragmented mask with {n_cc} components (spurious blobs outside the skull)")
    if ratio > 1.10:
        parts.append(f"over-segmentation ({ratio:.2f}x GT area): mask leaks beyond the skull into adjacent tissue / shadow")
    elif ratio < 0.90:
        parts.append(f"under-segmentation ({ratio:.2f}x GT area): part of the head is missed (weak/low-contrast skull edge)")
    ys, xs = np.nonzero(pred)
    gys, gxs = np.nonzero(gt)
    shift = float(np.hypot(xs.mean() - gxs.mean(), ys.mean() - gys.mean()))
    if shift > 15:
        parts.append(f"mask displaced by {shift:.0f} px from the GT centroid")
    if not parts:
        parts.append("boundary inaccuracy only (shape/area roughly right)")
    return f"Dice {dice:.3f}: " + "; ".join(parts)


def summarise(df: pd.DataFrame, tol_mm: float) -> dict:
    d, i = df.dice.values, df.iou.values
    err = df.hc_pred_mm.values - df.hc_gt_mm.values
    ok = np.isfinite(err)
    ae = np.abs(err[ok])
    return {"n": int(len(df)), "dice": {"mean": float(d.mean()), "std": float(d.std(ddof=0)), "median": float(np.median(d)),
                                        "min": float(d.min())},
            "iou": {"mean": float(i.mean()), "std": float(i.std(ddof=0)), "median": float(np.median(i)), "min": float(i.min())},
            "pixel_accuracy_mean": float(df.pixel_acc.mean()) if "pixel_acc" in df else None,
            "hausdorff95_px": {"mean": float(np.nanmean(df.hd95_px)), "median": float(np.nanmedian(df.hd95_px)),
                               "max": float(np.nanmax(df.hd95_px))} if "hd95_px" in df else None,
            "share_dice_lt_0.9": float((d < 0.9).mean()), "share_dice_ge_0.95": float((d >= 0.95).mean()),
            "hc": {"mae_mm": float(ae.mean()), "median_ae_mm": float(np.median(ae)), "mean_signed_diff_mm": float(err[ok].mean()),
                   "std_signed_diff_mm": float(err[ok].std(ddof=0)), f"share_abs_err_le_{tol_mm:g}mm": float((ae <= tol_mm).mean()),
                   "max_ae_mm": float(ae.max()), "n_valid": int(ok.sum()), "n_invalid": int((~ok).sum())}}


def bland_altman(df: pd.DataFrame, out_png: Path, title: str, tol_mm: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt, pr = df.hc_gt_mm.values, df.hc_pred_mm.values
    ok = np.isfinite(pr)
    gt, pr = gt[ok], pr[ok]
    diff = pr - gt
    mean, sd = diff.mean(), diff.std(ddof=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(gt, pr, s=10, alpha=0.7)
    lo, hi = min(gt.min(), pr.min()) - 5, max(gt.max(), pr.max()) + 5
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    axes[0].set_xlabel("HC ground truth (mm)")
    axes[0].set_ylabel("HC predicted (mm)")
    axes[0].set_title(f"pred vs GT (MAE {np.abs(diff).mean():.2f} mm, n={len(gt)})", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].scatter((gt + pr) / 2, diff, s=10, alpha=0.7)
    axes[1].axhline(mean, color="red", lw=1, label=f"bias {mean:+.2f} mm")
    axes[1].axhline(mean + 1.96 * sd, color="grey", ls="--", lw=1, label=f"±1.96 SD ({1.96 * sd:.2f} mm)")
    axes[1].axhline(mean - 1.96 * sd, color="grey", ls="--", lw=1)
    axes[1].axhline(tol_mm, color="green", ls=":", lw=1, label=f"±{tol_mm:g} mm acceptance")
    axes[1].axhline(-tol_mm, color="green", ls=":", lw=1)
    axes[1].set_xlabel("mean of GT and predicted HC (mm)")
    axes[1].set_ylabel("predicted − GT (mm)")
    axes[1].set_title("Bland–Altman", fontsize=9)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return {"bias_mm": float(mean), "loa_low_mm": float(mean - 1.96 * sd), "loa_high_mm": float(mean + 1.96 * sd)}


def failure_gallery(items: list[dict], out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.9 * n))
    axes = np.atleast_2d(axes)
    for row, it in zip(axes, items):
        row[0].imshow(it["img"], cmap="gray")
        row[0].set_title(f"{it['filename']} – image", fontsize=8)
        row[1].imshow(it["gt"], cmap="gray")
        row[1].set_title("GT mask", fontsize=8)
        row[2].imshow(it["pred"], cmap="gray")
        row[2].set_title(f"prediction – Dice {it['dice']:.3f}", fontsize=8)
        rgb = cv2.cvtColor(it["img"], cv2.COLOR_GRAY2RGB)
        cs, _ = cv2.findContours(it["gt"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cs, -1, (255, 140, 0), 2)
        if it["ellipse"] is not None:
            e = it["ellipse"]
            cv2.ellipse(rgb, ((e["cx"], e["cy"]), (2 * e["a"], 2 * e["b"]), e["angle_deg"]), (0, 255, 0), 2)
        row[3].imshow(rgb)
        row[3].set_title(f"fitted ellipse (green) vs GT (orange) – HC err {it['hc_err']:+.1f} mm", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.suptitle(title, fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def evaluate_variant(name: str, rows: pd.DataFrame, predict_fn, raw_dir: Path, processed: Path, thr: float,
                     input_wh: tuple[int, int], orig_wh: tuple[int, int], tol_mm: float, keep_cache: bool = True):
    recs, cache, scores, ious = [], {}, [], []
    for r in rows.itertuples():
        img = cv2.imread(str(raw_dir / r.filename), cv2.IMREAD_GRAYSCALE)
        gt = (cv2.imread(str(processed / r.mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        prob, extra = predict_fn(img, r)
        pred = (prob > thr).astype(np.uint8)
        pred_lcc = largest_component(pred)
        # HC: ellipse fitted on the network-resolution mask, axes rescaled to the original 800x540 before measuring
        prob_small = cv2.resize(prob, tuple(input_wh), interpolation=cv2.INTER_LINEAR)
        hc = measure_hc((prob_small > thr).astype(np.uint8), r.pixel_size_mm, orig_wh)
        d, i = dice_coefficient(pred, gt), iou_score(pred, gt)
        if pred.sum() > 0:
            scores.append(float(prob[pred > 0].mean()))
            ious.append(i)
        recs.append({"filename": r.filename, "variant": name, "dice": d, "iou": i,
                     "pixel_acc": float((pred == gt).mean()), "hd95_px": hausdorff95(pred, gt),
                     "dice_largest_cc": dice_coefficient(pred_lcc, gt), "n_components": int(cv2.connectedComponentsWithStats(pred, connectivity=8)[0] - 1),
                     "pred_area_px": int(pred.sum()), "gt_area_px": int(gt.sum()), "hc_pred_mm": hc["hc_mm"],
                     "hc_gt_mm": r.hc_mm, "abs_err_mm": abs(hc["hc_mm"] - r.hc_mm) if hc["valid"] else np.nan,
                     "signed_err_mm": (hc["hc_mm"] - r.hc_mm) if hc["valid"] else np.nan, "pixel_size_mm": r.pixel_size_mm,
                     "size_cat": r.size_cat, **extra})
        if keep_cache:
            cache[r.filename] = {"img": img, "gt": gt, "pred": pred, "dice": d, "ellipse": hc["ellipse"],
                                 "hc_err": (hc["hc_mm"] - r.hc_mm) if hc["valid"] else float("nan"), "filename": r.filename}
    df = pd.DataFrame(recs)
    summary = summarise(df, tol_mm)
    mmap = map_50_95_single_instance(np.array(scores), np.array(ious), n_gt=len(rows))
    summary["mask_mAP"] = {k: v for k, v in mmap.items() if k in ("mAP50", "mAP50-95", "AP75")}
    summary["mask_mAP"]["score_definition"] = "mean foreground probability inside the predicted mask"
    summary["mask_mAP"]["n_predictions"] = len(scores)
    return df, summary, cache


def main(cfg: dict) -> dict:
    import torch
    from ultralytics import YOLO

    set_seed(cfg["seed"])
    t0 = time.time()
    out = results_dir(cfg["phase"], cfg["run_name"], Path(cfg["results_dir"]).parent.parent)
    figs = out / "figures"
    save_resolved_config(cfg, out)
    device = pick_device(cfg["train"].get("device", "auto"))
    ecfg = cfg["evaluate"]
    thr, tol, margin = float(ecfg["threshold"]), float(ecfg["hc_tolerance_mm"]), float(ecfg["crop_margin"])
    input_wh, orig_wh = tuple(cfg["input_size"]), tuple(cfg["original_size"])
    model, meta = load_checkpoint(cfg["export_weights"], cfg["export_meta"], device)
    processed, raw_dir = Path(cfg["splits_csv"]).parent, Path(cfg["raw_dir"])
    test = load_split(cfg["splits_csv"], "test")
    res_note = f"TEST split ({len(test)} images), network input {input_wh[0]}x{input_wh[1]}, metrics at 800x540 px, threshold {thr}"

    # ---- variant 1: full image --------------------------------------------------------------------------------
    def full_fn(img, r):
        return predict_prob(model, img, input_wh, device), {"det_box_found": None}

    df_full, sum_full, cache_full = evaluate_variant("unet_full", test, full_fn, raw_dir, processed, thr, input_wh, orig_wh, tol)
    log.info("U-Net full image: Dice %.4f, HC MAE %.2f mm", sum_full["dice"]["mean"], sum_full["hc"]["mae_mm"])

    # ---- variant 2: detector box -> crop -> U-Net -> paste back -------------------------------------------------
    op = json.loads(Path(cfg["operating_point_json"]).read_text())
    detector = YOLO(cfg["detector_weights"])
    det = run_predict(detector, [processed / "yolo" / "images" / "test" / f for f in test.filename], conf=float(op["conf"]),
                      iou=float(op["iou"]), imgsz=int(op["imgsz"]), device=device)

    def chain_fn(img, r):
        boxes, scores = det[r.filename]
        if len(scores) == 0:
            return predict_prob(model, img, input_wh, device), {"det_box_found": False, "det_score": np.nan}
        k = int(np.argmax(scores))
        prob, crop = predict_prob_in_box(model, img, boxes[k], input_wh, device, margin)
        return prob, {"det_box_found": True, "det_score": float(scores[k]), "crop_x1": crop[0], "crop_y1": crop[1],
                      "crop_x2": crop[2], "crop_y2": crop[3]}

    df_chain, sum_chain, cache_chain = evaluate_variant("unet_detector_crop", test, chain_fn, raw_dir, processed, thr,
                                                        input_wh, orig_wh, tol)
    sum_chain["n_fallback_full_image"] = int((df_chain.det_box_found == False).sum())  # noqa: E712
    sum_chain["crop_margin"] = margin
    sum_chain["note"] = ("the U-Net was trained on full images; crops are resized to the network input, so the head appears "
                         "larger than in training – reported as the integrated-pipeline number")
    log.info("U-Net detector crop: Dice %.4f, HC MAE %.2f mm", sum_chain["dice"]["mean"], sum_chain["hc"]["mae_mm"])

    pd.concat([df_full, df_chain]).to_csv(out / "predictions_test.csv", index=False)

    # ---- SAM side by side -------------------------------------------------------------------------------------
    sam_path = Path(cfg["sam_results_dir"]) / "metrics.json"
    sam = json.loads(sam_path.read_text())["metrics"].get("test") if sam_path.exists() else None
    sam_side = {}
    if sam:
        for p in ["gt_box", "det_box"]:
            s = sam[p]
            sam_side[f"sam_{p}"] = {"dice_mean": s["dice_mean"], "dice_std": s["dice_std"], "dice_median": s["dice_median"],
                                    "dice_min": s["dice_min"], "iou_mean": s["iou_mean"], "hc_mae_mm": s["hc_mae_mm"],
                                    "hc_median_ae_mm": s["hc_median_ae_mm"],
                                    f"share_abs_err_le_{tol:g}mm": s.get(f"share_abs_err_le_{tol:g}mm"),
                                    "share_dice_lt_0.9": s["share_dice_lt_0.9"]}

    # ---- figures ----------------------------------------------------------------------------------------------
    captions = {}
    t = f"U-Net (full image) head circumference: predicted vs GT and Bland–Altman – {res_note}; HC via ellipse fit + Ramanujan"
    ba_full = bland_altman(df_full, figs / "hc_bland_altman_unet_full.png", t, tol)
    captions["hc_bland_altman_unet_full.png"] = t
    t = f"Detector-box -> U-Net chain head circumference: predicted vs GT and Bland–Altman – {res_note}; crop margin {margin:.0%}"
    ba_chain = bland_altman(df_chain, figs / "hc_bland_altman_unet_detector_crop.png", t, tol)
    captions["hc_bland_altman_unet_detector_crop.png"] = t
    worst = df_full.nsmallest(int(ecfg["n_failures"]), "dice")
    items = [cache_full[f] for f in worst.filename]
    t = f"{len(items)} worst-Dice cases, U-Net full image – {res_note}; columns: image | GT | prediction | fitted ellipse (green) vs GT contour (orange)"
    failure_gallery(items, figs / "failure_gallery_unet_full.png", t)
    captions["failure_gallery_unet_full.png"] = t
    fail_caps = {}
    for it in items:
        fail_caps[it["filename"]] = failure_statement(it["pred"], it["gt"], it["dice"]) + f"; HC error {it['hc_err']:+.1f} mm (rule-based statement)"
    captions["failure_cases_unet_full"] = fail_caps
    # Dice histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.5, 1.0, 26)
    ax.hist(df_full.dice, bins=bins, alpha=0.6, label=f"U-Net full (mean {sum_full['dice']['mean']:.3f})")
    ax.hist(df_chain.dice, bins=bins, alpha=0.6, label=f"detector-crop U-Net (mean {sum_chain['dice']['mean']:.3f})")
    if sam_path.exists():
        sam_df = pd.read_csv(Path(cfg["sam_results_dir"]) / "predictions_test.csv")
        for p in ["gt_box", "det_box"]:
            s = sam_df[sam_df.prompt == p]
            ax.hist(s.dice, bins=bins, alpha=0.4, histtype="step", lw=2, label=f"SAM {p} (mean {s.dice.mean():.3f})")
    ax.axvline(0.95, color="grey", ls=":", label="target 0.95")
    ax.set_xlabel("Dice vs sonographer mask")
    ax.set_ylabel("images")
    ax.legend(fontsize=7)
    t = f"Dice distribution – {res_note}"
    ax.set_title(t, fontsize=8, wrap=True)
    fig.tight_layout()
    fig.savefig(figs / "dice_histogram.png", dpi=200)
    plt.close(fig)
    captions["dice_histogram.png"] = t
    caps_path = figs / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    caps.update(captions)
    caps_path.write_text(json.dumps(caps, indent=2))

    # ---- metrics.json -----------------------------------------------------------------------------------------
    sum_full["bland_altman"] = ba_full
    sum_chain["bland_altman"] = ba_chain
    metrics = {"unet_full_image": sum_full, "unet_detector_crop_chain": sum_chain, "sam_baseline_test": sam_side,
               "acceptance": {"dice_target": 0.95, "hc_mae_target_mm": tol,
                              "dice_pass_full": bool(sum_full["dice"]["mean"] >= 0.95),
                              "hc_mae_pass_full": bool(sum_full["hc"]["mae_mm"] <= tol),
                              "dice_pass_chain": bool(sum_chain["dice"]["mean"] >= 0.95),
                              "hc_mae_pass_chain": bool(sum_chain["hc"]["mae_mm"] <= tol)},
               "worst_cases_unet_full": worst[["filename", "dice", "iou", "hc_pred_mm", "hc_gt_mm", "abs_err_mm"]].to_dict("records")}
    extra = {"device": device, "wall_clock_seconds": time.time() - t0, "weights": cfg["export_weights"],
             "checkpoint_meta": meta, "detector_weights": cfg["detector_weights"],
             "operating_point": {k: op[k] for k in ["conf", "iou", "imgsz"]}, "threshold": thr,
             "hc_method": "largest component -> cv2.fitEllipse on the network-resolution mask -> axes rescaled to 800x540 -> Ramanujan perimeter x pixel size"}
    write_metrics(out, cfg["run_name"], cfg["phase"], "test", list(input_wh), len(test), cfg["seed"], metrics, extra)
    run = wandb_utils.init_run(cfg, cfg["phase"], job_type="evaluate", tags=[cfg["phase"], "segmentation", "test"])
    flat = {"test/full/dice": sum_full["dice"]["mean"], "test/full/iou": sum_full["iou"]["mean"],
            "test/full/hc_mae_mm": sum_full["hc"]["mae_mm"], "test/full/mask_mAP50-95": sum_full["mask_mAP"]["mAP50-95"],
            "test/chain/dice": sum_chain["dice"]["mean"], "test/chain/hc_mae_mm": sum_chain["hc"]["mae_mm"]}
    wandb_utils.log_metrics(run, flat)
    for name in [k for k in captions if k.endswith(".png")]:
        wandb_utils.log_figure(run, name.replace(".png", ""), figs / name)
    wandb_utils.finish(run, flat)
    log.info("done in %.1f min", (time.time() - t0) / 60)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
