"""Error analysis covering ALL test errors of one run, with testable cause checks (numbers, not guesses).

python -m fetalvision.classification.error_analysis --run-name X [--results-root results] [--data <processed dir>]

Inputs : results/phase1/<run>/predictions_test.csv (+ metrics.json for the confusion-matrix cross-check) and the
         processed dir's all_metadata.csv (Patient_num, Operator, US_Machine, phash, split) joined on sha256[:16].
Outputs: error_analysis.json, error_review.csv (template for manual cause labelling), figures/errors_page<k>.png
         (contact sheets of every error, 20 per page, 200 dpi), figures/error_confidence_hist.png,
         figures/error_rate_by_machine.png.

Cause checks (each produces a number in error_analysis.json):
  (a) confidence of errors vs correct predictions (median, share < 0.6)       -> "model knows it does not know?"
  (b) error rate by US_Machine and by Operator with counts + chi-square test     -> acquisition-dependent?
  (c) errors per patient (concentration; share of errors from the top-3 patients) -> patient-level clusters?
  (d) nearest TRAIN image by pHash for every error (<= 6 bits = near duplicate)  -> leakage / label-noise suspects
  (e) second-choice analysis: is the true class the 2nd-ranked prediction?        -> top-2 recovery
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

log = logging.getLogger("fetalvision.error_analysis")
PROB_COLS = [f"p_{c}" for c in CLASS_NAMES]


def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def load_predictions(run_dir: Path, data_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    preds = pd.read_csv(run_dir / "predictions_test.csv", dtype={"Patient_num": str})
    preds["sha16"] = [Path(p).stem for p in preds["path"]]
    if data_dir is None:
        data_dir = Path(preds["path"].iloc[0]).parents[2]
    meta_path = Path(data_dir) / "all_metadata.csv"
    meta = None
    if meta_path.exists():
        meta = pd.read_csv(meta_path, dtype={"Patient_num": str})
        meta["sha16"] = meta["sha256"].str[:16]
        cols = [c for c in ["sha16", "Patient_num", "Operator", "US_Machine", "phash", "Image_name", "Brain_plane"] if c in meta]
        preds = preds.drop(columns=[c for c in ["Patient_num"] if c in preds]).merge(meta[cols].drop_duplicates("sha16"), on="sha16", how="left")
    preds["correct"] = preds["y_true"] == preds["y_pred"]
    return preds, meta


def check_consistency(preds: pd.DataFrame, run_dir: Path) -> dict:
    n = len(preds)
    n_err = int((~preds["correct"]).sum())
    probs = preds[PROB_COLS].values
    assert np.array_equal(probs.argmax(1), preds["y_pred"].values), "y_pred must equal argmax of the probabilities"
    out = {"n_test": n, "n_correct": int(preds["correct"].sum()), "n_errors": n_err, "accuracy": float(preds["correct"].mean())}
    assert out["n_correct"] + n_err == n
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for t, p in zip(preds["y_true"], preds["y_pred"]):
        cm[t, p] += 1
    off_diag = int(cm.sum() - np.trace(cm))
    assert off_diag == n_err, f"off-diagonal sum {off_diag} != number of errors {n_err}"
    out["confusion_matrix_from_predictions"] = cm.tolist()
    mfile = run_dir / "metrics.json"
    if mfile.exists():
        m = json.loads(mfile.read_text())
        cm_m = np.asarray(m["metrics"]["confusion_matrix"])
        out["matches_metrics_json_confusion_matrix"] = bool(np.array_equal(cm, cm_m))
        assert out["matches_metrics_json_confusion_matrix"], "confusion matrix differs from metrics.json"
    return out


def confusion_pair_table(err: pd.DataFrame) -> list[dict]:
    rows = []
    for (t, p), g in err.groupby(["y_true", "y_pred"]):
        rows.append({"true": CLASS_NAMES[t], "pred": CLASS_NAMES[p], "count": int(len(g)),
                     "mean_confidence": float(g["confidence"].mean())})
    return sorted(rows, key=lambda r: -r["count"])


def check_confidence(preds: pd.DataFrame, thr: float = 0.6) -> dict:
    err, ok = preds[~preds["correct"]], preds[preds["correct"]]
    return {"median_confidence_errors": float(err["confidence"].median()) if len(err) else None,
            "median_confidence_correct": float(ok["confidence"].median()),
            "mean_confidence_errors": float(err["confidence"].mean()) if len(err) else None,
            "mean_confidence_correct": float(ok["confidence"].mean()),
            f"share_errors_below_{thr}": float((err["confidence"] < thr).mean()) if len(err) else None,
            f"share_correct_below_{thr}": float((ok["confidence"] < thr).mean()),
            f"n_errors_below_{thr}": int((err["confidence"] < thr).sum()),
            "interpretation": "errors with low confidence can be caught by an abstention threshold; high-confidence errors cannot"}


def check_by_factor(preds: pd.DataFrame, col: str) -> dict:
    if col not in preds or preds[col].isna().all():
        return {"available": False}
    from scipy.stats import chi2_contingency
    tab = preds.groupby(col)["correct"].agg(n="size", correct="sum")
    tab["errors"] = tab["n"] - tab["correct"]
    tab["error_rate"] = tab["errors"] / tab["n"]
    rows = {str(k): {"n": int(r.n), "errors": int(r.errors), "error_rate": float(r.error_rate)} for k, r in tab.iterrows()}
    out = {"available": True, "by_level": rows}
    ct = tab[["correct", "errors"]].values
    if ct.shape[0] > 1 and (ct.sum(axis=1) > 0).all():
        chi2, p, dof, exp = chi2_contingency(ct)
        out["chi_square"] = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "min_expected_count": float(exp.min()),
                             "note": "expected counts < 5 make the chi-square approximation unreliable" if exp.min() < 5 else ""}
    return out


def check_patients(preds: pd.DataFrame) -> dict:
    err = preds[~preds["correct"]]
    if "Patient_num" not in preds or len(err) == 0:
        return {"available": False}
    per_pat = err.groupby("Patient_num").size().sort_values(ascending=False)
    n_pat_test = int(preds["Patient_num"].nunique())
    imgs_per_pat = preds.groupby("Patient_num").size()
    top3 = per_pat.head(3)
    return {"available": True, "n_patients_test": n_pat_test, "n_patients_with_errors": int(len(per_pat)),
            "share_patients_with_errors": float(len(per_pat) / n_pat_test),
            "errors_per_patient_distribution": {str(k): int(v) for k, v in per_pat.value_counts().sort_index().items()},
            "top3_patients": [{"Patient_num": str(k), "errors": int(v), "images_in_test": int(imgs_per_pat[k])} for k, v in top3.items()],
            "share_of_errors_from_top3_patients": float(top3.sum() / len(err)),
            "max_errors_single_patient": int(per_pat.iloc[0]),
            "mean_error_rate_patients_with_errors": float((per_pat / imgs_per_pat[per_pat.index]).mean())}


def check_train_near_duplicates(err: pd.DataFrame, meta: pd.DataFrame | None, max_dist: int = 6) -> dict:
    if meta is None or "phash" not in err or err["phash"].isna().all():
        return {"available": False}
    train = meta[meta["split"] == "train"][["phash", "label", "Patient_num", "path"]].dropna(subset=["phash"]).reset_index(drop=True)
    tr = np.array([int(h, 16) for h in train["phash"]], dtype=np.uint64)
    rows, n_near, n_near_label_mismatch = [], 0, 0
    for r in err.itertuples():
        if not isinstance(r.phash, str):
            continue
        x = np.bitwise_xor(tr, np.uint64(int(r.phash, 16)))
        d = np.array([bin(int(v)).count("1") for v in x])
        j = int(d.argmin())
        near = bool(d[j] <= max_dist)
        n_near += near
        mismatch = near and train.loc[j, "label"] != CLASS_NAMES[r.y_true]
        n_near_label_mismatch += mismatch
        rows.append({"error_id": int(r.error_id), "min_phash_dist_to_train": int(d[j]), "nearest_train_label": train.loc[j, "label"],
                     "nearest_train_patient": str(train.loc[j, "Patient_num"]), "nearest_train_path": train.loc[j, "path"],
                     "is_near_duplicate": near, "near_duplicate_label_mismatch": bool(mismatch),
                     "nearest_train_label_equals_prediction": bool(train.loc[j, "label"] == CLASS_NAMES[r.y_pred])})
    return {"available": True, "max_phash_dist": max_dist, "n_errors_with_train_near_duplicate": int(n_near),
            "n_near_duplicates_with_different_label": int(n_near_label_mismatch),
            "n_near_duplicate_label_equals_prediction": int(sum(r["nearest_train_label_equals_prediction"] and r["is_near_duplicate"] for r in rows)),
            "median_min_phash_dist": float(np.median([r["min_phash_dist_to_train"] for r in rows])) if rows else None,
            "per_error": rows}


def check_second_choice(err: pd.DataFrame) -> dict:
    if len(err) == 0:
        return {"n_errors": 0}
    probs = err[PROB_COLS].values
    order = np.argsort(-probs, axis=1)
    second = order[:, 1]
    rec = second == err["y_true"].values
    p_true = probs[np.arange(len(err)), err["y_true"].values]
    return {"n_errors": int(len(err)), "n_true_class_is_second_choice": int(rec.sum()), "share_recovered_by_top2": float(rec.mean()),
            "median_p_true": float(np.median(p_true)), "n_p_true_below_0.1": int((p_true < 0.1).sum()),
            "margin_median": float(np.median(probs.max(1) - p_true))}


def contact_sheets(err: pd.DataFrame, fig_dir: Path, run_name: str, per_page: int = 20) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    files = []
    n_pages = int(np.ceil(len(err) / per_page)) if len(err) else 0
    for page in range(n_pages):
        sub = err.iloc[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(4, 5, figsize=(15, 12.5))
        for ax in axes.flat:
            ax.axis("off")
        for ax, r in zip(axes.flat, sub.itertuples()):
            try:
                with Image.open(r.path) as im:
                    ax.imshow(im.convert("L"), cmap="gray")
            except Exception:
                ax.text(0.5, 0.5, "missing", ha="center")
            extra = f"\n{getattr(r, 'US_Machine', '')} / pat {getattr(r, 'Patient_num', '')}" if hasattr(r, "US_Machine") else ""
            ax.set_title(f"#{r.error_id}  {CLASS_NAMES[r.y_true]}->{CLASS_NAMES[r.y_pred]}  conf={r.confidence:.2f}{extra}", fontsize=8)
        fig.suptitle(f"{run_name}: ALL test errors, page {page + 1}/{n_pages} (errors {sub.error_id.min()}-{sub.error_id.max()} of {len(err)}; 224x224 model input)", fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"errors_page{page + 1}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        files.append(out.name)
    return files


def plots(preds: pd.DataFrame, by_machine: dict, fig_dir: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    err, ok = preds[~preds["correct"]], preds[preds["correct"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.25, 1.0, 16)
    ax.hist(ok["confidence"], bins=bins, alpha=.6, label=f"correct (n={len(ok)})", density=True)
    ax.hist(err["confidence"], bins=bins, alpha=.6, label=f"errors (n={len(err)})", density=True)
    ax.axvline(0.6, color="k", ls="--", lw=1)
    ax.set_xlabel("max softmax probability")
    ax.set_ylabel("density")
    ax.set_title(f"{run_name}: confidence of correct vs wrong test predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "error_confidence_hist.png", dpi=200)
    plt.close(fig)
    if by_machine.get("available"):
        lv = by_machine["by_level"]
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(lv)
        ax.bar(names, [100 * lv[k]["error_rate"] for k in names])
        for i, k in enumerate(names):
            ax.text(i, 100 * lv[k]["error_rate"], f"{lv[k]['errors']}/{lv[k]['n']}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("test error rate (%)")
        p = by_machine.get("chi_square", {}).get("p_value")
        ax.set_title(f"{run_name}: error rate by US machine" + (f" (chi-square p={p:.3f})" if p is not None else ""))
        fig.tight_layout()
        fig.savefig(fig_dir / "error_rate_by_machine.png", dpi=200)
        plt.close(fig)


def analyse(run_dir: Path, data_dir: Path | None = None) -> dict:
    preds, meta = load_predictions(run_dir, data_dir)
    consistency = check_consistency(preds, run_dir)
    err = preds[~preds["correct"]].copy().reset_index(drop=True)
    err["error_id"] = np.arange(1, len(err) + 1)
    err["p_true"] = [row[PROB_COLS[int(t)]] for row, t in zip(err.to_dict("records"), err["y_true"])]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    by_machine = check_by_factor(preds, "US_Machine")
    out = {"run_name": run_dir.name, "consistency": consistency,
           "errors_by_true_class": {c: int((err["y_true"] == i).sum()) for i, c in enumerate(CLASS_NAMES)},
           "recall_by_class": {c: float(preds.loc[preds.y_true == i, "correct"].mean()) for i, c in enumerate(CLASS_NAMES)},
           "confusion_pairs": confusion_pair_table(err),
           "a_confidence": check_confidence(preds),
           "b_by_us_machine": by_machine, "b_by_operator": check_by_factor(preds, "Operator"),
           "c_patients": check_patients(preds),
           "d_train_near_duplicates": check_train_near_duplicates(err, meta),
           "e_second_choice": check_second_choice(err)}
    out["contact_sheets"] = contact_sheets(err, fig_dir, run_dir.name)
    plots(preds, by_machine, fig_dir, run_dir.name)
    review = err[["error_id", "path", "Patient_num", "y_true", "y_pred", "confidence", "p_true"]].copy()
    review["true"] = [CLASS_NAMES[i] for i in review.pop("y_true")]
    review["pred"] = [CLASS_NAMES[i] for i in review.pop("y_pred")]
    review = review.rename(columns={"confidence": "conf"})
    for c in ["US_Machine", "Operator"]:
        if c in err:
            review[c] = err[c].values
    nd = out["d_train_near_duplicates"]
    if nd.get("available"):
        dmap = {r["error_id"]: r["min_phash_dist_to_train"] for r in nd["per_error"]}
        review["min_phash_dist_to_train"] = review["error_id"].map(dmap)
    sec = np.argsort(-err[PROB_COLS].values, axis=1)[:, 1] if len(err) else []
    review["second_choice"] = [CLASS_NAMES[i] for i in sec]
    review["cause_category"] = ""
    review["note"] = ""
    review = review[["error_id", "path", "Patient_num", "true", "pred", "conf", "p_true", "second_choice"]
                    + [c for c in ["US_Machine", "Operator", "min_phash_dist_to_train"] if c in review] + ["cause_category", "note"]]
    review.to_csv(run_dir / "error_review.csv", index=False)
    out["error_review_csv"] = str(run_dir / "error_review.csv")
    (run_dir / "error_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    cap = fig_dir / "captions.json"
    captions = json.loads(cap.read_text()) if cap.exists() else {}
    for f in out["contact_sheets"]:
        captions[f] = f"Contact sheet of all test errors of {run_dir.name} (tile title: error id, true->pred, confidence, US machine, patient)."
    captions["error_confidence_hist.png"] = "Distribution of the max softmax probability for correct vs wrong test predictions."
    captions["error_rate_by_machine.png"] = "Test error rate per ultrasound machine with counts and chi-square p-value."
    cap.write_text(json.dumps(captions, indent=2))
    log.info("%s: %d errors; conf median err=%.3f ok=%.3f; top2 recovers %d; train near-dups %s; patients with errors %s",
             run_dir.name, len(err), out["a_confidence"]["median_confidence_errors"] or 0, out["a_confidence"]["median_confidence_correct"],
             out["e_second_choice"].get("n_true_class_is_second_choice", 0), nd.get("n_errors_with_train_near_duplicate"),
             out["c_patients"].get("n_patients_with_errors"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--data", default=None, help="processed dir (default: inferred from the prediction paths)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    analyse(Path(args.results_root) / "phase1" / args.run_name, Path(args.data) if args.data else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
