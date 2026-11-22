#!/usr/bin/env python
"""Compose the cross-stage failure gallery for the Phase 2 report: one explicit failure per stage.
Panels: (1) classifier error (selected Phase 1 run, highest-confidence error), (2) detector lowest-IoU detection,
(3) U-Net worst-Dice case with GT/pred contours, (4) domain-shift: HC18 brain image routed as non-Brain.
Writes results/phase2/failure_gallery.png and results/phase2/failure_gallery.json (statements)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fetalvision.common.config import data_root

R = ROOT / "results"
sel = json.loads((R / "phase1/model_selection.json").read_text())["selected_run"]
statements = {}
fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
for ax in axes: ax.axis("off")

# 1 classifier: highest-confidence error of the selected run (test split)
pred = pd.read_csv(R / "phase1" / sel / "predictions_test.csv")
cls = [c[2:] for c in pred.columns if c.startswith("p_")]
err = pred[pred.y_true != pred.y_pred].copy()
err["conf"] = err[[f"p_{c}" for c in cls]].max(axis=1)
e = err.sort_values("conf", ascending=False).iloc[0]
im = cv2.imread(str(e.path)); axes[0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
t_true = cls[int(e.y_true)] if str(e.y_true).isdigit() else e.y_true
t_pred = cls[int(e.y_pred)] if str(e.y_pred).isdigit() else e.y_pred
axes[0].set_title(f"1. Classifier ({sel}, test, 224x224)\ntrue {t_true} -> predicted {t_pred}, confidence {e.conf:.2f}", fontsize=9)
statements["classifier"] = (f"Highest-confidence classifier error on the test split: a {t_true} plane predicted as {t_pred} with confidence "
                            f"{e.conf:.2f}. Because it is confident, the review flag (probability < 0.6) does not catch it; confident errors on images whose "
                            "salient structure resembles another plane (here a curved bone next to a round soft-tissue mass) are the residual classifier failure.")

# 2 detector: lowest-IoU detection
det = pd.read_csv(R / "phase2/detection/predictions_test.csv")
iou_col = [c for c in det.columns if "iou" in c.lower()][0]
d = det.sort_values(iou_col).iloc[0]
img_dir = data_root() / "hc18/raw/training_set"
fn = d.filename if "filename" in det.columns else d.iloc[0]
im = cv2.imread(str(img_dir / fn)); im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
def box(prefix):
    cols = [c for c in det.columns if c.startswith(prefix)]
    return [float(d[c]) for c in cols[:4]] if len(cols) >= 4 else None
gt = box("gt_"); pr = box("pred_")
if gt: cv2.rectangle(im, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (255, 140, 0), 2)
if pr: cv2.rectangle(im, (int(pr[0]), int(pr[1])), (int(pr[2]), int(pr[3])), (0, 255, 0), 2)
axes[1].imshow(im); axes[1].set_title(f"2. Detector (test, 640 px, conf>=0.85, NMS 0.3)\n{fn}: lowest IoU {float(d[iou_col]):.3f} (orange GT, green pred)", fontsize=9)
statements["detector"] = (f"Lowest-IoU detection on the test split ({fn}, IoU {float(d[iou_col]):.3f}): the box is confidently placed but extends beyond the skull "
                          "into adjacent echogenic tissue; no FP/FN occurred at the operating point, so mild mislocalisation is the residual detector failure.")

# 3 segmentation: worst Dice
seg = json.loads((R / "phase2/segmentation/metrics.json").read_text())
w = seg["metrics"]["worst_cases_unet_full"][0]
im = cv2.imread(str(img_dir / w["filename"]), cv2.IMREAD_GRAYSCALE); im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
gtm = cv2.imread(str(data_root() / "hc18/processed/masks" / w["filename"]), cv2.IMREAD_GRAYSCALE)
if gtm is not None:
    cs, _ = cv2.findContours((gtm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (255, 140, 0), 2)
onnx_path = ROOT / "models/head_segmenter.onnx"
if onnx_path.exists():
    import onnxruntime as ort
    g = cv2.imread(str(img_dir / w["filename"]), cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(g, (512, 352), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = np.stack([(x - m_) / s_ for m_, s_ in zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))], 0)[None]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    prob = 1 / (1 + np.exp(-out)) if out.min() < 0 or out.max() > 1 else out
    pm = cv2.resize((prob > 0.5).astype(np.uint8) * 255, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours((pm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (0, 255, 0), 2)
axes[2].imshow(im); axes[2].set_title(f"3. Segmentation (U-Net, test, 512x352 in, thr 0.5)\n{w['filename']}: Dice {w['dice']:.3f}, HC {w['hc_pred_mm']:.1f} vs {w['hc_gt_mm']:.1f} mm", fontsize=9)
statements["segmentation"] = (f"Worst-Dice segmentation on the test split ({w['filename']}, Dice {w['dice']:.3f}): a small early-gestation head with a faint lower skull edge; "
                              f"the mask misses part of the head and the HC is under-estimated by {w['hc_gt_mm']-w['hc_pred_mm']:.1f} mm, outside the 3 mm acceptance band.")

# 4 domain shift
ds_path = R / "phase2/domain_shift/predictions_test.csv"
if ds_path.exists():
    ds = pd.read_csv(ds_path); nb = ds[ds.pred != "Brain"].sort_values("p_brain")
    if len(nb):
        r0 = nb.iloc[0]; im = cv2.imread(str(img_dir / r0.filename)); im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        axes[3].imshow(im); axes[3].set_title(f"4. Domain shift (classifier on HC18 test, 224x224)\n{r0.filename}: brain plane routed as {r0.pred} (P(Brain)={r0.p_brain:.2f})", fontsize=9)
        m = json.loads((R / "phase2/domain_shift/metrics.json").read_text())["metrics"]
        statements["domain_shift"] = (f"HC18 brain image {r0.filename} routed as {r0.pred} (P(Brain) {r0.p_brain:.2f}); overall {m['share_routed_brain']*100:.1f}% of HC18 test images are routed as Brain by a classifier trained on FETAL_PLANES_DB, "
                                      "so cross-centre deployment needs the review flag / force_head option or classifier fine-tuning on the target centre.")
else:
    axes[3].set_title("4. Domain shift: run scripts/domain_shift_hc18.py first", fontsize=9)
fig.suptitle("FetalVision failure gallery: one explicit failure per stage (all on held-out test splits)", fontsize=11)
fig.tight_layout(); fig.savefig(R / "phase2/failure_gallery.png", dpi=170); plt.close(fig)
(R / "phase2/failure_gallery.json").write_text(json.dumps({"selected_classifier": sel, "statements": statements}, indent=2))
print(json.dumps(statements, indent=1))
