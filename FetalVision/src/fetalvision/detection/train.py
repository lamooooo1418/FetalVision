"""Fine-tune YOLO11n (COCO-pretrained) as a single-class fetal-head detector on the HC18 leakage-safe split.

Usage:  python -m fetalvision.detection.train --config configs/phase2_detection.yaml

A time budget (`max_train_minutes`) is enforced with an ultralytics callback: after each epoch the mean epoch time
is used to project the total; if the configured epochs would exceed the budget, the epoch count is reduced and the
decision is recorded in metrics.json (`epochs_planned`, `epochs_run`, `epochs_reduced_for_time`).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path

import pandas as pd

from fetalvision.common.config import load_config, results_dir, save_resolved_config, write_metrics
from fetalvision.common.seed import set_seed
from fetalvision.common import wandb_utils
from fetalvision.detection.utils import pick_device

log = logging.getLogger("fetalvision.detection.train")


def plot_learning_curves(results_csv: Path, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for k in ["train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "train/dfl_loss", "val/dfl_loss"]:
        if k in df:
            axes[0].plot(df["epoch"], df[k], label=k)
    axes[0].set_title("losses")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    for k in ["metrics/precision(B)", "metrics/recall(B)"]:
        if k in df:
            axes[1].plot(df["epoch"], df[k], label=k)
    axes[1].set_title("val precision / recall (ultralytics, conf 0.001)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7)
    for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if k in df:
            axes[2].plot(df["epoch"], df[k], label=k)
    axes[2].set_title("val mAP")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(cfg: dict) -> dict:
    from ultralytics import YOLO

    set_seed(cfg["seed"])
    out = results_dir(cfg["phase"], cfg["run_name"], Path(cfg["results_dir"]).parent.parent)
    save_resolved_config(cfg, out)
    device = pick_device(cfg.get("device", "auto"))
    amp = bool(cfg.get("amp", False)) and device == "cuda"
    log.info("device=%s amp=%s", device, amp)
    run = wandb_utils.init_run(cfg, cfg["phase"], job_type="train", tags=[cfg["phase"], "detection", "yolo11n"])

    budget_s = float(cfg.get("max_train_minutes", 60)) * 60.0
    epochs_planned = int(cfg["epochs"])
    state = {"epoch_times": [], "epochs_reduced_for_time": False, "epochs_final": epochs_planned, "t_epoch": None}

    def on_train_epoch_start(trainer):
        state["t_epoch"] = time.time()

    def on_fit_epoch_end(trainer):
        # record epoch time (train + val) and enforce the wall-clock budget
        state["epoch_times"].append(time.time() - state["t_epoch"])
        mean_t = sum(state["epoch_times"]) / len(state["epoch_times"])
        done = len(state["epoch_times"])
        projected = mean_t * trainer.epochs
        if projected > budget_s and trainer.epochs > done:
            new_epochs = max(done, int(math.floor(budget_s / mean_t)))
            log.warning("mean epoch %.1fs x %d epochs = %.1f min > budget %.1f min -> reducing epochs to %d",
                        mean_t, trainer.epochs, projected / 60, budget_s / 60, new_epochs)
            trainer.epochs = trainer.args.epochs = new_epochs
            state["epochs_reduced_for_time"] = True
            state["epochs_final"] = new_epochs
            if done >= new_epochs:
                trainer.stop = True
        # mirror per-epoch metrics to W&B
        if run is not None and trainer.metrics:
            wandb_utils.log_metrics(run, {k: float(v) for k, v in trainer.metrics.items()}, step=trainer.epoch + 1)

    model = YOLO(cfg["model"])
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    aug = cfg.get("augment", {})
    project = Path(cfg["project_dir"]).resolve()  # absolute: ultralytics nests relative paths under runs/detect/
    t0 = time.time()
    model.train(
        data=cfg["data_yaml"], epochs=epochs_planned, imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
        seed=int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", True)), patience=int(cfg.get("patience", 10)),
        cos_lr=bool(cfg.get("cos_lr", True)), optimizer=cfg.get("optimizer", "auto"), lr0=float(cfg.get("lr0", 0.01)),
        fliplr=float(aug.get("fliplr", 0.5)), flipud=float(aug.get("flipud", 0.0)), mosaic=float(aug.get("mosaic", 1.0)),
        degrees=float(aug.get("degrees", 0.0)), close_mosaic=int(aug.get("close_mosaic", 10)),
        workers=int(cfg.get("workers", 4)), device=device, amp=amp, project=str(project), name="train", exist_ok=True,
        plots=True, verbose=True,
    )
    wall = time.time() - t0
    return finalize(cfg, out, device, amp, wall, state, run, epochs_planned)


def finalize(cfg: dict, out: Path, device: str, amp: bool, wall: float, state: dict, run, epochs_planned: int) -> dict:
    """Post-training: export best.pt, copy artefacts, learning curves, val metrics of best.pt, metrics_train.json."""
    from ultralytics import YOLO

    project = Path(cfg["project_dir"]).resolve()
    run_dir = project / "train"
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found in {run_dir}")
    export = Path(cfg["export_weights"])
    export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export)
    log.info("best.pt -> %s (%.1f min wall-clock)", export, wall / 60)

    # copy training artefacts into the results contract
    figs = out / "figures"
    if (run_dir / "results.csv").exists():
        shutil.copy2(run_dir / "results.csv", out / "train_results.csv")
    for p in run_dir.glob("*.png"):
        shutil.copy2(p, figs / f"train_{p.name}")
    for p in run_dir.glob("*.jpg"):
        shutil.copy2(p, figs / f"train_{p.stem}.jpg")
    res_csv = out / "train_results.csv"
    df = pd.read_csv(res_csv)
    df.columns = [c.strip() for c in df.columns]
    epochs_run = int(len(df))
    curve_title = (f"YOLO11n fetal-head detector – learning curves (train split {epochs_run} epochs, val split metrics; "
                   f"imgsz {cfg['imgsz']}, batch {cfg['batch']}, seed {cfg['seed']}, device {device})")
    plot_learning_curves(res_csv, figs / "learning_curves.png", curve_title)

    # validation metrics of best.pt (standard ultralytics protocol: conf 0.001, NMS IoU 0.7)
    best_model = YOLO(str(best))
    v = best_model.val(data=cfg["data_yaml"], split="val", imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
                       device=device, conf=0.001, iou=0.7, plots=False, verbose=False, project=str(project),
                       name="val_best", exist_ok=True)
    val_metrics = {"mAP50": float(v.box.map50), "mAP50-95": float(v.box.map), "precision": float(v.box.mp),
                   "recall": float(v.box.mr), "conf": 0.001, "nms_iou": 0.7}
    best_row = df.iloc[df["metrics/mAP50-95(B)"].idxmax()] if "metrics/mAP50-95(B)" in df else df.iloc[-1]
    n_val = len(list(Path(cfg["data_yaml"]).parent.glob("images/val/*.png")))
    metrics = {"val_best_pt": val_metrics,
               "best_epoch_by_mAP50-95": int(best_row["epoch"]),
               "last_epoch_row": {k: float(best_row[k]) for k in df.columns if k != "epoch"}}
    extra = {"device": device, "amp": amp, "wall_clock_seconds": wall, "epochs_planned": epochs_planned,
             "epochs_run": epochs_run, "epochs_final_target": state["epochs_final"],
             "epochs_reduced_for_time": state["epochs_reduced_for_time"],
             "mean_epoch_seconds": (sum(state["epoch_times"]) / len(state["epoch_times"])) if state["epoch_times"]
             else (float(df["time"].iloc[-1]) / epochs_run if "time" in df else None),
             "early_stopped": epochs_run < state["epochs_final"], "model": cfg["model"], "imgsz": cfg["imgsz"],
             "batch": cfg["batch"], "weights": str(export), "yolo_run_dir": str(run_dir),
             "note": "val metrics use the ultralytics protocol (conf 0.001, NMS IoU 0.7) on the val split; the operating "
                     "point is chosen by detection.sweep and test metrics by detection.evaluate"}
    write_metrics(out, cfg["run_name"] + "_train", cfg["phase"], "val", [cfg["imgsz"], cfg["imgsz"]], n_val, cfg["seed"],
                  metrics, extra)
    (out / "metrics.json").rename(out / "metrics_train.json")
    wandb_utils.log_metrics(run, {f"val_best/{k}": v for k, v in val_metrics.items()})
    wandb_utils.log_figure(run, "learning_curves", figs / "learning_curves.png")
    wandb_utils.finish(run, {**{f"val_best/{k}": v for k, v in val_metrics.items()}, "wall_clock_seconds": wall,
                             "epochs_run": epochs_run, "device": device})
    log.info("val(best.pt): %s", json.dumps(val_metrics))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip training; export/evaluate an already completed run in <project_dir>/train")
    ap.add_argument("--wall-seconds", type=float, default=float("nan"),
                    help="with --finalize-only: wall-clock of the completed training to record in metrics_train.json")
    args = ap.parse_args()
    if args.finalize_only:
        _cfg = load_config(args.config)
        _out = results_dir(_cfg["phase"], _cfg["run_name"], Path(_cfg["results_dir"]).parent.parent)
        _dev = pick_device(_cfg.get("device", "auto"))
        _run = wandb_utils.init_run(_cfg, _cfg["phase"], job_type="train", tags=[_cfg["phase"], "detection", "finalize"])
        finalize(_cfg, _out, _dev, False, args.wall_seconds, {"epoch_times": [], "epochs_reduced_for_time": False,
                 "epochs_final": int(_cfg["epochs"])}, _run, int(_cfg["epochs"]))
    else:
        main(load_config(args.config))
