"""Train one Phase 1 classification run from a single YAML config, then evaluate on val and (once) on test.

python -m fetalvision.classification.train --config configs/phase1_<run>.yaml [--max-steps N]

Config keys (see configs/phase1_*.yaml):
  run_name, seed, model: {type: cnn_scratch|mobilenetv3s, mode: frozen|partial|full, dropout, partial_last_n}
  data: {processed_dir, image_size, batch_size, cache}, augmentation: {enabled, rotation, zoom, translation, contrast}
  training: scratch -> {optimizer: adam, lr, epochs, early_stopping_patience}
            transfer -> {stage1: {lr, epochs, early_stopping_patience}, stage2: {lr, epochs, early_stopping_patience}}
  class_weight: balanced|none, models_dir, results_root, debug: {max_steps}

Checkpoint rule (fixed): the checkpoint kept within a run is the epoch with the lowest VALIDATION loss
(ModelCheckpoint save_best_only + EarlyStopping restore_best_weights, both monitor val_loss).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from fetalvision.common.config import load_config, results_dir, save_resolved_config
from fetalvision.common.constants import CLASS_NAMES
from fetalvision.common.seed import set_seed
from fetalvision.common import wandb_utils

log = logging.getLogger("fetalvision.train")
PHASE = "phase1"


class _WandbEpochLogger:
    """Keras callback: logs per-epoch metrics to W&B with a stage prefix and a global epoch counter."""

    def __new__(cls, run, stage: str, offset: int):
        import keras

        class _CB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                payload = {f"{stage}/{k}": float(v) for k, v in logs.items()}
                payload.update({"epoch": offset + epoch + 1, f"{stage}/epoch": epoch + 1})
                lr = getattr(self.model.optimizer, "learning_rate", None)
                if lr is not None:
                    try:
                        payload[f"{stage}/lr"] = float(keras.ops.convert_to_numpy(lr))
                    except Exception:
                        pass
                wandb_utils.log_metrics(run, payload, step=offset + epoch + 1)
        return _CB()


def _callbacks(ckpt_path: Path, patience: int, wandb_run, stage: str, offset: int):
    import keras
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
        _WandbEpochLogger(wandb_run, stage, offset),
    ]


def _compile(model, lr: float):
    import keras
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def _fit(model, ds, cw, stage_cfg: dict, ckpt: Path, wandb_run, stage: str, offset: int):
    t0 = time.time()
    h = model.fit(ds["train"], validation_data=ds["val"], epochs=int(stage_cfg["epochs"]), class_weight=cw,
                  callbacks=_callbacks(ckpt, int(stage_cfg["early_stopping_patience"]), wandb_run, stage, offset), verbose=2)
    hist = {k: [float(x) for x in v] for k, v in h.history.items()}
    best = min(range(len(hist["val_loss"])), key=lambda i: hist["val_loss"][i])
    return {"stage": stage, "history": hist, "epochs_run": len(hist["val_loss"]), "best_epoch": best + 1,
            "best_val_loss": hist["val_loss"][best], "best_val_accuracy": hist["val_accuracy"][best],
            "lr": float(stage_cfg["lr"]), "max_epochs": int(stage_cfg["epochs"]),
            "early_stopping_patience": int(stage_cfg["early_stopping_patience"]), "wall_clock_seconds": time.time() - t0}


def plot_learning_curves(stages: list[dict], out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    offset = 0
    boundaries = []
    for s in stages:
        h = s["history"]
        ep = [offset + i + 1 for i in range(len(h["loss"]))]
        axes[0].plot(ep, h["accuracy"], "o-", ms=3, label=f"{s['stage']} train")
        axes[0].plot(ep, h["val_accuracy"], "s-", ms=3, label=f"{s['stage']} val")
        axes[1].plot(ep, h["loss"], "o-", ms=3, label=f"{s['stage']} train")
        axes[1].plot(ep, h["val_loss"], "s-", ms=3, label=f"{s['stage']} val")
        axes[1].axvline(offset + s["best_epoch"], color="green", ls=":", lw=1)
        offset += len(h["loss"])
        boundaries.append(offset)
    for ax, yl in zip(axes, ["accuracy", "loss (sparse CE)"]):
        for b in boundaries[:-1]:
            ax.axvline(b + 0.5, color="black", ls="--", lw=1)
        ax.set_xlabel("epoch (stages concatenated; dashed = stage boundary, dotted = best val_loss)")
        ax.set_ylabel(yl)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def train(cfg: dict) -> dict:
    from fetalvision.classification.data import configure_tf, make_datasets, balanced_class_weights, labels_from_paths
    from fetalvision.classification import models as M
    from fetalvision.classification.evaluate import evaluate_split, hardware_info
    import keras

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    configure_tf(cfg)
    run_name = cfg["run_name"]
    results_root = cfg.get("results_root")
    out = results_dir(PHASE, run_name, results_root)
    save_resolved_config(cfg, out)
    models_dir = Path(cfg.get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    debug = bool(cfg.get("debug", {}).get("max_steps"))
    t_start = time.time()

    ds, paths = make_datasets(cfg)
    train_labels = labels_from_paths(paths["train"])
    cw = balanced_class_weights(paths["train"]) if str(cfg.get("class_weight", "none")).lower() == "balanced" else None
    log.info("class weights: %s", cw)
    wandb_run = wandb_utils.init_run(cfg, PHASE, job_type="train", tags=[t for t in [PHASE, cfg["model"]["type"], cfg["model"].get("mode", "")] if t])

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    stages: list[dict] = []
    final_ckpt = models_dir / f"{run_name}.keras"
    image_size = tuple(cfg.get("data", {}).get("image_size", (224, 224)))
    if mcfg["type"] == "cnn_scratch":
        model = M.build_scratch_cnn(input_shape=(*image_size, 3), dropout=float(mcfg.get("dropout", 0.3)))
        _compile(model, float(tcfg["lr"]))
        stages.append(_fit(model, ds, cw, tcfg, final_ckpt, wandb_run, "scratch", 0))
        base = None
        trainable_backbone = None
    elif mcfg["type"] == "mobilenetv3s":
        mode = mcfg.get("mode", "frozen")
        model, base = M.build_mobilenetv3s(mode, input_shape=(*image_size, 3), dropout=float(mcfg.get("dropout", 0.3)),
                                           weights=mcfg.get("weights", "imagenet"), partial_last_n=int(mcfg.get("partial_last_n", 30)))
        head_ckpt = models_dir / f"{run_name}_head.keras" if mode != "frozen" else final_ckpt
        _compile(model, float(tcfg["stage1"]["lr"]))
        stages.append(_fit(model, ds, cw, tcfg["stage1"], head_ckpt, wandb_run, "stage1_head", 0))
        trainable_backbone = 0
        if mode != "frozen":
            model.load_weights(str(head_ckpt))  # restore best stage-1 weights before fine-tuning (as Phase 1)
            trainable_backbone = M.set_finetune_mode(base, mode, int(mcfg.get("partial_last_n", 30)))
            log.info("stage 2 (%s): %d trainable backbone layers", mode, trainable_backbone)
            _compile(model, float(tcfg["stage2"]["lr"]))  # recompile AFTER changing trainable flags
            stages.append(_fit(model, ds, cw, tcfg["stage2"], final_ckpt, wandb_run, f"stage2_{mode}", stages[0]["epochs_run"]))
    else:
        raise ValueError(f"unknown model type {mcfg['type']}")

    # The kept checkpoint = lowest val_loss epoch of the LAST stage (restore_best_weights already loaded it).
    model.load_weights(str(final_ckpt))
    model.save(str(final_ckpt))
    train_seconds = time.time() - t_start
    params = M.count_params(model)
    hist_payload = {"run_name": run_name, "stages": stages, "class_weight": cw, "train_wall_clock_seconds": train_seconds,
                    "n_train": len(paths["train"]), "n_val": len(paths["val"]), "n_test": len(paths["test"]),
                    "train_class_counts": {c: int(sum(1 for l in train_labels if l == i)) for i, c in enumerate(CLASS_NAMES)},
                    "params": params, "trainable_backbone_layers_stage2": trainable_backbone}
    (out / "history.json").write_text(json.dumps(hist_payload, indent=2))
    fig = out / "figures" / "learning_curves.png"
    plot_learning_curves(stages, fig, f"{run_name}: learning curves ({'+'.join(s['stage'] for s in stages)}, {image_size[0]}x{image_size[1]})")
    wandb_utils.log_figure(wandb_run, "learning_curves", fig)

    extra = {"training": {"stages": [{k: v for k, v in s.items() if k != "history"} for s in stages],
                          "wall_clock_seconds": train_seconds, "class_weight": cw, "params": params,
                          "trainable_backbone_layers_stage2": trainable_backbone,
                          "checkpoint_rule": "epoch with lowest validation loss (ModelCheckpoint save_best_only, EarlyStopping restore_best_weights)"},
             "hardware": hardware_info(), "model_path": str(final_ckpt.resolve()), "debug": debug,
             "data_dir": str(Path(cfg["data"]["processed_dir"]).resolve()), "augmentation": cfg.get("augmentation", {}),
             "selection_split": "val"}
    data_dir = Path(cfg["data"]["processed_dir"])
    m_val = evaluate_split(model, ds["val"], paths["val"], "val", run_name, data_dir, seed, results_root, extra, image_size)
    m_test = evaluate_split(model, ds["test"], paths["test"], "test", run_name, data_dir, seed, results_root,
                            {**extra, "val_metrics": m_val}, image_size)
    summary = {"val_accuracy": m_val["accuracy"], "val_macro_f1": m_val["macro_f1"], "val_log_loss": m_val["log_loss"],
               "test_accuracy": m_test["accuracy"], "test_macro_f1": m_test["macro_f1"], "test_log_loss": m_test["log_loss"],
               "test_ece": m_test["ece_10bins"], "train_wall_clock_seconds": train_seconds}
    wandb_utils.log_metrics(wandb_run, summary)
    for name in ["confusion_matrix_test.png", "reliability_test.png", "confusion_matrix_val.png"]:
        wandb_utils.log_figure(wandb_run, name.replace(".png", ""), out / "figures" / name)
    wandb_utils.finish(wandb_run, summary)
    log.info("DONE %s in %.1f min: %s", run_name, train_seconds / 60, json.dumps(summary))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="debug: use only N batches per split (smoke test)")
    ap.add_argument("--run-name", default=None, help="override run_name (e.g. smoke tests)")
    ap.add_argument("--models-dir", default=None, help="override models_dir (e.g. smoke tests)")
    ap.add_argument("--results-root", default=None, help="override results_root (e.g. smoke tests)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.max_steps:
        cfg.setdefault("debug", {})["max_steps"] = args.max_steps
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.models_dir:
        cfg["models_dir"] = args.models_dir
    if args.results_root:
        cfg["results_root"] = args.results_root
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
