"""Post-hoc temperature scaling for one run (Guo et al., 2017).

T is fitted on the run's VALIDATION predictions (predictions_val.csv -> log-probabilities, which are valid logits up to
a per-row constant) and applied to its TEST predictions. Reports T, ECE/NLL before and after on both splits, checks
that the argmax (and therefore accuracy / confusion matrix) is unchanged, saves reliability diagrams before/after and
writes results/phase1/<run>/calibration.json (+ predictions_test_calibrated.csv).

python -m fetalvision.classification.calibrate --run-name X [--results-root results]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fetalvision.common.constants import CLASS_NAMES
from fetalvision.common.metrics import fit_temperature, apply_temperature, probs_to_logits, expected_calibration_error

log = logging.getLogger("fetalvision.calibrate")
PROB_COLS = [f"p_{c}" for c in CLASS_NAMES]


def _nll(probs, y):
    p = np.asarray(probs, float)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def calibrate_run(run_dir: Path, n_bins: int = 10) -> dict:
    from fetalvision.classification.evaluate import plot_reliability
    val = pd.read_csv(run_dir / "predictions_val.csv")
    test = pd.read_csv(run_dir / "predictions_test.csv")
    yv, yt = val["y_true"].values.astype(int), test["y_true"].values.astype(int)
    pv, pt = val[PROB_COLS].values, test[PROB_COLS].values
    lv, lt = probs_to_logits(pv), probs_to_logits(pt)
    T = fit_temperature(lv, yv)
    pv2, pt2 = apply_temperature(lv, T), apply_temperature(lt, T)
    out = {"run_name": run_dir.name, "temperature": T, "fitted_on": "val", "n_val": int(len(yv)), "n_test": int(len(yt)),
           "method": "temperature scaling on log-probabilities (softmax-invariant shift), T minimises validation NLL"}
    for split, y, p0, p1 in [("val", yv, pv, pv2), ("test", yt, pt, pt2)]:
        e0, e1 = expected_calibration_error(p0, y, n_bins), expected_calibration_error(p1, y, n_bins)
        argmax_same = bool(np.array_equal(p0.argmax(1), p1.argmax(1)))
        out[split] = {"ece_before": e0["ece"], "ece_after": e1["ece"], "nll_before": _nll(p0, y), "nll_after": _nll(p1, y),
                      "accuracy": float((p0.argmax(1) == y).mean()), "argmax_unchanged": argmax_same,
                      "mean_confidence_before": float(p0.max(1).mean()), "mean_confidence_after": float(p1.max(1).mean()),
                      "bins_before": e0["bins"], "bins_after": e1["bins"]}
        plot_reliability(e0["bins"], e0["ece"], run_dir / "figures" / f"reliability_{split}_calibrated.png",
                         f"{run_dir.name}: reliability on {split} (n={len(y)}, 224x224, {n_bins} bins), T={T:.3f}",
                         e1["bins"], e1["ece"], labels=("before temperature scaling", "after temperature scaling"))
        assert argmax_same, "temperature scaling must not change the argmax"
    cal = test.copy()
    for i, c in enumerate(CLASS_NAMES):
        cal[f"p_{c}"] = pt2[:, i]
    cal["confidence"] = pt2.max(1)
    cal.to_csv(run_dir / "predictions_test_calibrated.csv", index=False)
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=2))
    cap = run_dir / "figures" / "captions.json"
    captions = json.loads(cap.read_text()) if cap.exists() else {}
    for split in ["val", "test"]:
        captions[f"reliability_{split}_calibrated.png"] = (f"Reliability diagram on {split} before/after temperature scaling "
                                                           f"(T={T:.3f} fitted on val; ECE {out[split]['ece_before']:.4f} -> {out[split]['ece_after']:.4f}).")
    cap.write_text(json.dumps(captions, indent=2))
    log.info("%s: T=%.3f  val ECE %.4f->%.4f  test ECE %.4f->%.4f  test NLL %.4f->%.4f", run_dir.name, T,
             out["val"]["ece_before"], out["val"]["ece_after"], out["test"]["ece_before"], out["test"]["ece_after"],
             out["test"]["nll_before"], out["test"]["nll_after"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    calibrate_run(Path(args.results_root) / "phase1" / args.run_name, args.bins)
    return 0


if __name__ == "__main__":
    sys.exit(main())
