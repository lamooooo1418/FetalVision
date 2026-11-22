"""Evaluate a Keras classifier on one processed split and write the Phase 1 results contract.

python -m fetalvision.classification.evaluate --model models/x.keras --data <processed dir> --split val|test --run-name X
                                                [--results-root results] [--max-steps N] [--seed 42]

Writes results/phase1/<run_name>/predictions_<split>.csv, figures/confusion_matrix_<split>.png,
figures/reliability_<split>.png, and metrics.json (split=test) or metrics_val.json (split=val). When the test split is
evaluated and metrics_val.json exists, its metrics are embedded as `val_metrics` so model selection reads one file.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score

from fetalvision.common.config import results_dir, write_metrics
from fetalvision.common.constants import CLASS_NAMES
from fetalvision.common.metrics import wilson_ci, expected_calibration_error, top_k_accuracy
from fetalvision.common.seed import set_seed

log = logging.getLogger("fetalvision.evaluate")
PHASE = "phase1"


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        info["device"] = "GPU" if info["gpus"] else "CPU"
    except Exception:  # pragma: no cover
        pass
    try:
        import keras
        info["keras"] = keras.__version__
    except Exception:  # pragma: no cover
        pass
    try:
        import subprocess
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["cpu"] = platform.processor()
    return info


def predict_dataset(model, ds) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a (non-shuffled) dataset; returns (probs [n, C], y_true [n])."""
    probs, ys = [], []
    for x, y in ds:
        probs.append(np.asarray(model(x, training=False)))
        ys.append(np.asarray(y))
    return np.concatenate(probs).astype(float), np.concatenate(ys).astype(int)


def nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """Multiclass log-loss (mean negative log-likelihood) with rows renormalised (float32 softmax rows != 1 exactly)."""
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_true)), np.asarray(y_true).astype(int)], eps, 1.0))))


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(axis=1)
    n = len(y_true)
    k = int((y_pred == y_true).sum())
    lo, hi = wilson_ci(k, n)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    ece = expected_calibration_error(probs, y_true, n_bins=10)
    return {
        "accuracy": k / n, "n_correct": k, "n_errors": int(n - k),
        "accuracy_ci95_wilson": [float(lo), float(hi)],
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASS_NAMES)), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": top_k_accuracy(probs, y_true, 2),
        "log_loss": nll(probs, y_true),
        "ece_10bins": ece["ece"], "reliability_bins": ece["bins"],
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": cm.tolist(), "confusion_matrix_labels": CLASS_NAMES,
        "confusion_pairs": {f"{CLASS_NAMES[i]}->{CLASS_NAMES[j]}": int(cm[i, j])
                            for i in range(len(CLASS_NAMES)) for j in range(len(CLASS_NAMES)) if i != j and cm[i, j] > 0},
    }


def predictions_frame(paths: list[str], y_true, probs, data_dir: Path | None) -> pd.DataFrame:
    probs = np.asarray(probs)
    df = pd.DataFrame({"path": paths, "y_true": np.asarray(y_true).astype(int), "y_pred": probs.argmax(axis=1)})
    for i, c in enumerate(CLASS_NAMES):
        df[f"p_{c}"] = probs[:, i]
    df["confidence"] = probs.max(axis=1)
    df["sha16"] = [Path(p).stem for p in paths]
    meta_path = Path(data_dir) / "all_metadata.csv" if data_dir else None
    if meta_path and meta_path.exists():
        meta = pd.read_csv(meta_path, dtype={"Patient_num": str})
        meta["sha16"] = meta["sha256"].str[:16]
        df = df.merge(meta[["sha16", "Patient_num"]].drop_duplicates("sha16"), on="sha16", how="left")
    else:
        df["Patient_num"] = ""
    df["group"] = df["Patient_num"]
    cols = ["path", "Patient_num", "group", "y_true", "y_pred"] + [f"p_{c}" for c in CLASS_NAMES] + ["confidence", "sha16"]
    return df[cols]


def plot_confusion_matrix(cm, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thr = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thr else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_reliability(bins: list[dict], ece: float, out_png: Path, title: str, bins2: list[dict] | None = None,
                     ece2: float | None = None, labels=("model", "after temperature scaling")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for b, e, lab, mk in [(bins, ece, labels[0], "o-"), (bins2, ece2, labels[1], "s-")]:
        if b is None:
            continue
        ax.plot([r["confidence"] for r in b], [r["accuracy"] for r in b], mk, label=f"{lab} (ECE={e:.4f})")
        for r in b:
            ax.annotate(str(r["count"]), (r["confidence"], r["accuracy"]), fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("mean predicted confidence (bin)")
    ax.set_ylabel("observed accuracy (bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def evaluate_split(model, ds, paths: list[str], split: str, run_name: str, data_dir: Path, seed: int,
                   results_root: str | None = None, extra: dict | None = None, image_size=(224, 224),
                   write_metrics_file: bool = True) -> dict:
    """Predict, compute metrics, write predictions CSV + figures (+ metrics file). Returns the metrics dict."""
    t0 = time.time()
    probs, y_true = predict_dataset(model, ds)
    assert len(probs) == len(paths), f"{len(probs)} predictions vs {len(paths)} paths: file_paths out of sync"
    infer_s = time.time() - t0
    m = classification_metrics(y_true, probs)
    m["inference_seconds_total"] = infer_s
    m["inference_ms_per_image"] = 1000 * infer_s / max(1, len(y_true))
    out = results_dir(PHASE, run_name, results_root)
    preds = predictions_frame(paths, y_true, probs, data_dir)
    preds.to_csv(out / f"predictions_{split}.csv", index=False)
    res = f"{image_size[0]}x{image_size[1]}"
    plot_confusion_matrix(m["confusion_matrix"], out / "figures" / f"confusion_matrix_{split}.png",
                          f"{run_name}\n{split} split (n={len(y_true)}, {res}) acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
    plot_reliability(m["reliability_bins"], m["ece_10bins"], out / "figures" / f"reliability_{split}.png",
                     f"{run_name} reliability, {split} split (n={len(y_true)}, {res}, 10 bins)")
    captions = {}
    cap_path = out / "figures" / "captions.json"
    if cap_path.exists():
        captions = json.loads(cap_path.read_text())
    captions[f"confusion_matrix_{split}.png"] = f"Confusion matrix on the {split} split (n={len(y_true)}, input {res})."
    captions[f"reliability_{split}.png"] = f"Reliability diagram on the {split} split (10 equal-width bins, input {res}); ECE={m['ece_10bins']:.4f}."
    cap_path.write_text(json.dumps(captions, indent=2))
    if write_metrics_file:
        write_split_metrics(out, run_name, split, m, len(y_true), seed, image_size, extra)
    log.info("[%s] %s: acc=%.4f macro-F1=%.4f logloss=%.4f errors=%d", run_name, split, m["accuracy"], m["macro_f1"],
             m["log_loss"], m["n_errors"])
    return m


def write_split_metrics(out: Path, run_name: str, split: str, m: dict, n: int, seed: int, image_size, extra: dict | None) -> Path:
    extra = dict(extra or {})
    extra.setdefault("selection_split", "val")
    extra.setdefault("hardware", hardware_info())
    if split == "val":
        # common.write_metrics always writes <dir>/metrics.json -> write into a scratch sub-dir and move to metrics_val.json
        tmp = Path(out) / "_val_tmp"
        tmp.mkdir(exist_ok=True)
        p = write_metrics(tmp, run_name, PHASE, "val", image_size, n, seed, m, extra)
        target = Path(out) / "metrics_val.json"
        p.replace(target)
        tmp.rmdir()
        return target
    # test: embed the validation metrics when available so select_model.py reads a single file
    val_file = out / "metrics_val.json"
    if "val_metrics" not in extra and val_file.exists():
        extra["val_metrics"] = json.loads(val_file.read_text())["metrics"]
    extra.setdefault("note", "Model/checkpoint selection used the validation split only; the test split was evaluated once.")
    return write_metrics(out, run_name, PHASE, "test", image_size, n, seed, m, extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="processed dir with train/ val/ test/ and all_metadata.csv")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--note", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(args.seed)
    from fetalvision.classification.data import configure_tf, load_split
    configure_tf()
    import keras
    model = keras.models.load_model(args.model, compile=False)
    ds, paths = load_split(args.data, args.split, batch_size=args.batch_size, shuffle=False)
    if args.max_steps:
        ds, paths = ds.take(args.max_steps), paths[: args.max_steps * args.batch_size]
    extra = {"model_path": str(Path(args.model).resolve()), "data_dir": str(Path(args.data).resolve())}
    if args.note:
        extra["note"] = args.note
    evaluate_split(model, ds, paths, args.split, args.run_name, Path(args.data), args.seed, args.results_root, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
