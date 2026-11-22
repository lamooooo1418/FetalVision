"""Train the fetal-head U-Net (smp.Unet, MobileNetV2 ImageNet encoder) on the HC18 leakage-safe split.

Usage:  python -m fetalvision.segmentation.train_unet --config configs/phase2_segmentation.yaml

Loss Dice+BCE, AdamW, cosine schedule, best checkpoint by val Dice. A wall-clock budget (`train.max_train_minutes`)
is enforced: after the first epoch the epoch count is reduced if the projection exceeds the budget (recorded in
metrics_train.json). Exports models/head_segmenter.pt (state dict) + models/head_segmenter.json (encoder, input size).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fetalvision.common.config import load_config, results_dir, save_resolved_config, write_metrics, git_commit
from fetalvision.common.seed import set_seed
from fetalvision.common import wandb_utils
from fetalvision.detection.utils import pick_device
from fetalvision.segmentation.dataset import HC18SegDataset
from fetalvision.segmentation.model import DiceBCELoss, build_model, save_checkpoint

log = logging.getLogger("fetalvision.segmentation.train_unet")


def batch_dice(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-7) -> np.ndarray:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).cpu().numpy()


def plot_curves(hist: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(hist.epoch, hist.train_loss, label="train Dice+BCE")
    axes[0].plot(hist.epoch, hist.val_loss, label="val Dice+BCE")
    axes[0].set_title("loss")
    axes[0].legend(fontsize=8)
    axes[1].plot(hist.epoch, hist.val_dice, label="val Dice (thr 0.5)")
    axes[1].axhline(0.95, color="grey", ls=":", label="target 0.95")
    axes[1].set_ylim(0.8, 1.0)
    axes[1].set_title("val Dice")
    axes[1].legend(fontsize=8)
    axes[2].plot(hist.epoch, hist.lr)
    axes[2].set_title("learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    tcfg, mcfg = cfg["train"], cfg["model"]
    out = results_dir(cfg["phase"], cfg["run_name"], Path(cfg["results_dir"]).parent.parent)
    save_resolved_config(cfg, out)
    device = pick_device(tcfg.get("device", "auto"))
    input_wh = tuple(int(v) for v in cfg["input_size"])
    log.info("device=%s input=%s", device, input_wh)
    run = wandb_utils.init_run(cfg, cfg["phase"], job_type="train", tags=[cfg["phase"], "segmentation", mcfg["encoder"]])

    df = pd.read_csv(cfg["splits_csv"])
    processed = Path(cfg["splits_csv"]).parent
    ds_tr = HC18SegDataset(df[df.split == "train"], cfg["raw_dir"], processed, input_wh, augment=True, aug_cfg=cfg["augment"])
    ds_va = HC18SegDataset(df[df.split == "val"], cfg["raw_dir"], processed, input_wh, augment=False)
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    dl_tr = DataLoader(ds_tr, batch_size=int(tcfg["batch_size"]), shuffle=True, num_workers=int(tcfg.get("num_workers", 0)),
                       drop_last=True, generator=g, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=int(tcfg["batch_size"]), shuffle=False, num_workers=int(tcfg.get("num_workers", 0)))

    model = build_model(mcfg, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    epochs_planned = int(tcfg["epochs"])
    epochs = epochs_planned
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=float(tcfg["lr"]) * 0.01)
    loss_fn = DiceBCELoss()
    budget_s = float(tcfg.get("max_train_minutes", 45)) * 60
    thr = float(tcfg.get("threshold", 0.5))

    hist, best_dice, best_epoch, reduced = [], -1.0, -1, False
    t0 = time.time()
    export_w, export_m = Path(cfg["export_weights"]), Path(cfg["export_meta"])
    meta = {"model": mcfg, "input_size_wh": list(input_wh), "original_size_wh": list(cfg["original_size"]),
            "normalisation": "imagenet mean/std on grayscale replicated to 3 channels", "threshold": thr,
            "seed": cfg["seed"], "git_commit": git_commit()}
    epoch = 0
    while epoch < epochs:
        te = time.time()
        model.train()
        tr_losses = []
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))
        sched.step()
        model.eval()
        va_losses, dices = [], []
        with torch.no_grad():
            for x, y, _ in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                va_losses.append(float(loss_fn(logits, y).item()))
                dices.extend(batch_dice(logits, y, thr).tolist())
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(tr_losses)), "val_loss": float(np.mean(va_losses)),
               "val_dice": float(np.mean(dices)), "lr": float(opt.param_groups[0]["lr"]), "epoch_seconds": time.time() - te}
        hist.append(row)
        wandb_utils.log_metrics(run, row, step=epoch + 1)
        if row["val_dice"] > best_dice:
            best_dice, best_epoch = row["val_dice"], epoch + 1
            save_checkpoint(model, export_w, export_m, {**meta, "best_epoch": best_epoch, "best_val_dice": best_dice})
        log.info("epoch %d/%d train %.4f val %.4f dice %.4f (%.0fs)%s", epoch + 1, epochs, row["train_loss"],
                 row["val_loss"], row["val_dice"], row["epoch_seconds"], " *" if best_epoch == epoch + 1 else "")
        # wall-clock budget: re-plan after the first epoch (and keep checking)
        mean_t = np.mean([h["epoch_seconds"] for h in hist])
        if mean_t * epochs > budget_s and epochs > epoch + 1:
            new_epochs = max(epoch + 1, int(math.floor(budget_s / mean_t)))
            log.warning("projected %.1f min > budget %.1f min -> epochs %d -> %d (cosine schedule rebuilt)",
                        mean_t * epochs / 60, budget_s / 60, epochs, new_epochs)
            epochs, reduced = new_epochs, True
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - (epoch + 1), 1),
                                                               eta_min=float(tcfg["lr"]) * 0.01)
        epoch += 1
    wall = time.time() - t0
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(out / "train_history.csv", index=False)
    title = (f"U-Net ({mcfg['encoder']}) fetal-head segmentation – learning curves; train split ({len(ds_tr)} imgs), "
             f"val split ({len(ds_va)} imgs), input {input_wh[0]}x{input_wh[1]}, threshold {thr}, batch {tcfg['batch_size']}, "
             f"seed {cfg['seed']}, device {device}; best val Dice {best_dice:.4f} @ epoch {best_epoch}")
    plot_curves(hist_df, out / "figures" / "learning_curves.png", title)
    caps_path = out / "figures" / "captions.json"
    caps = json.loads(caps_path.read_text()) if caps_path.exists() else {}
    caps["learning_curves.png"] = title
    caps_path.write_text(json.dumps(caps, indent=2))
    metrics = {"best_val_dice": best_dice, "best_epoch": best_epoch, "final_epoch": hist_df.iloc[-1].to_dict(),
               "n_params": n_params}
    extra = {"device": device, "amp": False, "wall_clock_seconds": wall, "epochs_planned": epochs_planned,
             "epochs_run": int(len(hist_df)), "epochs_reduced_for_time": reduced,
             "mean_epoch_seconds": float(hist_df.epoch_seconds.mean()), "weights": str(export_w), "meta": str(export_m),
             "encoder": mcfg["encoder"], "loss": "Dice + BCE", "optimizer": "AdamW", "lr": tcfg["lr"]}
    write_metrics(out, cfg["run_name"] + "_train", cfg["phase"], "val", list(input_wh), len(ds_va), cfg["seed"], metrics, extra)
    (out / "metrics.json").rename(out / "metrics_train.json")
    wandb_utils.log_figure(run, "learning_curves", out / "figures" / "learning_curves.png")
    wandb_utils.finish(run, {"best_val_dice": best_dice, "best_epoch": best_epoch, "wall_clock_seconds": wall})
    log.info("done: best val Dice %.4f at epoch %d, %.1f min", best_dice, best_epoch, wall / 60)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
