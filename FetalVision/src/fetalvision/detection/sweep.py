"""Confidence / NMS-IoU sweep on the VAL split and a-priori operating-point selection.

Usage:  python -m fetalvision.detection.sweep --config configs/phase2_detection.yaml

The detector is run once per NMS IoU in `sweep.nms_ious` at conf 0.001; precision/recall/F1 are then derived for
conf thresholds conf_min..conf_max (one GT per image: TP if IoU >= match_iou, extra boxes FP, no box FN).
Operating-point rule (fixed before looking at the test split): maximise F1 subject to recall >= min_recall; if
infeasible, maximise F1. Writes sweep.csv, figures/sweep_prf_vs_conf.png and operating_point.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fetalvision.common.config import load_config, results_dir, git_commit
from fetalvision.common.seed import set_seed
from fetalvision.common import wandb_utils
from fetalvision.detection.utils import (choose_operating_point, gt_box, load_split, pick_device, run_predict,
                                         sweep_confidences)

log = logging.getLogger("fetalvision.detection.sweep")


def plot_sweep(table: pd.DataFrame, op: dict, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ious = sorted(table.nms_iou.unique())
    fig, axes = plt.subplots(1, len(ious), figsize=(5 * len(ious), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, iou in zip(axes, ious):
        t = table[table.nms_iou == iou]
        ax.plot(t.conf, t.precision, "o-", ms=3, label="precision")
        ax.plot(t.conf, t.recall, "s-", ms=3, label="recall")
        ax.plot(t.conf, t.f1, "^-", ms=3, label="F1")
        ax.axhline(op["min_recall"], color="grey", ls=":", lw=1, label=f"recall >= {op['min_recall']}")
        if abs(iou - op["iou"]) < 1e-9:
            ax.axvline(op["conf"], color="red", ls="--", lw=1, label=f"operating point conf={op['conf']:.2f}")
        ax.set_title(f"NMS IoU = {iou}")
        ax.set_xlabel("confidence threshold")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("metric")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(cfg: dict) -> dict:
    from ultralytics import YOLO

    set_seed(cfg["seed"])
    out = results_dir(cfg["phase"], cfg["run_name"], Path(cfg["results_dir"]).parent.parent)
    device = pick_device(cfg.get("device", "auto"))
    scfg = cfg["sweep"]
    val = load_split(cfg["splits_csv"], "val")
    img_dir = Path(cfg["data_yaml"]).parent / "images" / "val"
    paths = [img_dir / f for f in val.filename]
    gts = {r.filename: gt_box(r) for r in val.itertuples()}
    confs = np.round(np.arange(scfg["conf_min"], scfg["conf_max"] + 1e-9, scfg["conf_step"]), 4)
    model = YOLO(cfg["export_weights"])
    t0 = time.time()
    tables = []
    for nms_iou in scfg["nms_ious"]:
        preds = run_predict(model, paths, conf=0.001, iou=float(nms_iou), imgsz=int(cfg["imgsz"]), device=device)
        t = sweep_confidences(preds, gts, confs, scfg["match_iou"])
        t.insert(0, "nms_iou", float(nms_iou))
        tables.append(t)
        log.info("NMS IoU %.1f done (%d images)", nms_iou, len(paths))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "sweep.csv", index=False)
    best, rule = choose_operating_point(table, scfg["min_recall"])
    op = {"conf": float(best["conf"]), "iou": float(best["nms_iou"]), "val_precision": float(best["precision"]),
          "val_recall": float(best["recall"]), "val_f1": float(best["f1"]), "val_tp": int(best["tp"]),
          "val_fp": int(best["fp"]), "val_fn": int(best["fn"]), "rule": rule, "min_recall": scfg["min_recall"],
          "match_iou": scfg["match_iou"], "split": "val", "n_images": int(len(val)), "imgsz": int(cfg["imgsz"]),
          "device": device, "wall_clock_seconds": time.time() - t0, "git_commit": git_commit(),
          "weights": cfg["export_weights"]}
    (out / "operating_point.json").write_text(json.dumps(op, indent=2))
    title = (f"Fetal-head detector sweep – VAL split ({len(val)} images), imgsz {cfg['imgsz']}, raw conf 0.001, "
             f"match IoU {scfg['match_iou']}; rule: {rule} -> conf {op['conf']:.2f}, NMS IoU {op['iou']}")
    plot_sweep(table, op, out / "figures" / "sweep_prf_vs_conf.png", title)
    caps_path = out / "figures" / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    caps["sweep_prf_vs_conf.png"] = title
    caps_path.write_text(json.dumps(caps, indent=2))
    run = wandb_utils.init_run(cfg, cfg["phase"], job_type="sweep", tags=[cfg["phase"], "detection", "sweep"])
    wandb_utils.log_metrics(run, {f"op/{k}": v for k, v in op.items() if isinstance(v, (int, float))})
    wandb_utils.log_figure(run, "sweep", out / "figures" / "sweep_prf_vs_conf.png")
    wandb_utils.finish(run, {f"op/{k}": v for k, v in op.items()})
    log.info("operating point: %s", json.dumps(op))
    return op


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
