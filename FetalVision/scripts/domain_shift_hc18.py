#!/usr/bin/env python
"""Measure the Phase 1 plane classifier on HC18 brain images (domain shift): share routed as Brain, probability stats.
Writes results/phase2/domain_shift/metrics.json. Uses the selected Keras model from results/phase1/model_selection.json."""
from __future__ import annotations
import json, os, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fetalvision.common.constants import CLASS_NAMES, BRAIN_IDX
from fetalvision.common.config import write_metrics, data_root
from fetalvision.common.metrics import probs_to_logits, apply_temperature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--review-threshold", type=float, default=0.6)
    a = ap.parse_args()
    sel = json.loads((ROOT / "results/phase1/model_selection.json").read_text())["selected_run"]
    model_path = ROOT / "models" / f"{sel}.keras"
    cal = ROOT / "results/phase1" / sel / "calibration.json"
    T = json.loads(cal.read_text())["temperature"] if cal.exists() else 1.0
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    splits = pd.read_csv(data_root() / "hc18/processed/splits.csv")
    rows = splits[splits["split"] == a.split]
    img_dir = data_root() / "hc18/raw/training_set"
    X = np.stack([np.asarray(Image.open(img_dir / f).convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32)
                  for f in rows["filename"]])
    probs = model.predict(X, batch_size=32, verbose=0)
    probs_T = apply_temperature(probs_to_logits(probs), T)
    pred = probs_T.argmax(1)
    p_brain = probs_T[:, BRAIN_IDX]
    counts = {c: int((pred == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    out_dir = ROOT / "results/phase2/domain_shift"; (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    df = rows[["filename", "hc_mm"]].copy()
    df["pred"] = [CLASS_NAMES[i] for i in pred]; df["p_brain"] = p_brain; df["confidence"] = probs_T.max(1)
    df.to_csv(out_dir / f"predictions_{a.split}.csv", index=False)
    # HC tertiles vs routing
    q = df["hc_mm"].quantile([1/3, 2/3]).values
    cat = np.where(df["hc_mm"] < q[0], "small", np.where(df["hc_mm"] < q[1], "medium", "large"))
    by_size = {k: {"n": int((cat == k).sum()), "routed_brain": int(((cat == k) & (df["pred"] == "Brain")).sum())} for k in ["small", "medium", "large"]}
    for k in by_size: by_size[k]["share"] = by_size[k]["routed_brain"] / max(by_size[k]["n"], 1)
    metrics = {
        "classifier_run": sel, "temperature": T, "n_images": int(len(df)),
        "predicted_class_counts": counts,
        "share_routed_brain": float((pred == BRAIN_IDX).sum() / len(df)),
        "share_routed_brain_or_flagged_review": float(((pred == BRAIN_IDX) | (probs_T.max(1) < a.review_threshold)).sum() / len(df)),
        "mean_p_brain": float(p_brain.mean()), "median_p_brain": float(np.median(p_brain)),
        "share_confidence_below_review_threshold": float((probs_T.max(1) < a.review_threshold).mean()),
        "review_threshold": a.review_threshold,
        "routed_brain_by_hc_tertile": by_size,
        "note": "HC18 images are all trans-thalamic brain planes; any non-Brain prediction is a routing miss caused by domain shift (different centre, machines, presets, field of view) between FETAL_PLANES_DB and HC18.",
    }
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(list(counts), list(counts.values())); ax[0].set_title(f"Plane predicted on HC18 {a.split} ({len(df)} brain images)"); ax[0].set_ylabel("images")
    ax[1].hist(p_brain, bins=20); ax[1].set_title(f"P(Brain) after temperature scaling (T={T:.2f})"); ax[1].set_xlabel("P(Brain)")
    fig.suptitle(f"Domain shift: classifier {sel} (224x224) on HC18 {a.split} split", fontsize=10); fig.tight_layout()
    fig.savefig(out_dir / "figures/domain_shift_hc18.png", dpi=200); plt.close(fig)
    write_metrics(out_dir, "domain_shift_hc18", "phase2", a.split, [224, 224], len(df), 42, metrics)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
