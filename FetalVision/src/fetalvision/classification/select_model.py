"""Model selection on the VALIDATION split (stated rule, applied mechanically).

Rule: kept model = run with the highest validation macro-F1; tie (|delta| < 1e-6) -> lower validation log-loss.
Within a run the checkpoint is the epoch with the lowest validation loss (train.py). The test split is only reported.

python -m fetalvision.classification.select_model [--results-root results] [--pattern 'p1v2-*'] [--include-original]
Writes results/phase1/model_selection.json with the ranked table.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("fetalvision.select_model")
RULE = ("Selection on the validation split only: highest validation macro-F1; ties (< 1e-6) broken by lower validation "
        "log-loss. Checkpoint within a run: epoch with the lowest validation loss. Test metrics are reported, never used "
        "for selection. Debug/smoke runs are excluded.")


def collect_runs(results_root: Path, pattern: str = "p1v2-*", include: tuple[str, ...] = ()) -> list[dict]:
    rows = []
    for mpath in sorted((results_root / "phase1").glob("*/metrics.json")):
        run = mpath.parent.name
        if not (fnmatch.fnmatch(run, pattern) or run in include):
            continue
        m = json.loads(mpath.read_text())
        if m.get("debug"):
            continue
        val = m.get("val_metrics")
        if val is None:
            vfile = mpath.parent / "metrics_val.json"
            if vfile.exists():
                val = json.loads(vfile.read_text())["metrics"]
        if val is None:
            log.warning("%s has no validation metrics; skipped", run)
            continue
        test = m.get("metrics", {}) if m.get("split_evaluated") == "test" else {}
        rows.append({"run_name": run, "val_macro_f1": float(val["macro_f1"]), "val_log_loss": float(val["log_loss"]),
                     "val_accuracy": float(val["accuracy"]), "val_abdomen_recall": float(val["per_class"]["Abdomen"]["recall"]),
                     "test_accuracy": test.get("accuracy"), "test_macro_f1": test.get("macro_f1"), "test_log_loss": test.get("log_loss"),
                     "test_abdomen_recall": (test.get("per_class") or {}).get("Abdomen", {}).get("recall"),
                     "test_abdomen_thorax_confusions": ((test.get("confusion_pairs") or {}).get("Abdomen->Thorax", 0)
                                                        + (test.get("confusion_pairs") or {}).get("Thorax->Abdomen", 0)) if test else None,
                     "n_test": m.get("n_images"), "model_path": m.get("model_path"), "data_dir": m.get("data_dir"),
                     "class_weight": (m.get("training") or {}).get("class_weight"),
                     "train_wall_clock_seconds": (m.get("training") or {}).get("wall_clock_seconds")})
    return rows


def rank_runs(rows: list[dict], tol: float = 1e-6) -> list[dict]:
    """Sort by (-val_macro_f1, val_log_loss); ties in macro-F1 within `tol` are decided by log-loss."""
    def key(r):
        return (-round(r["val_macro_f1"] / tol) * tol, r["val_log_loss"])
    ranked = sorted(rows, key=key)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked


def select(results_root: Path, pattern: str = "p1v2-*", include: tuple[str, ...] = (), eligible_pattern: str | None = None) -> dict:
    rows = collect_runs(results_root, pattern, include)
    if not rows:
        raise SystemExit("no completed runs found")
    ranked = rank_runs(rows)
    eligible = [r for r in ranked if fnmatch.fnmatch(r["run_name"], eligible_pattern or pattern)]
    selected = eligible[0] if eligible else ranked[0]
    out = {"rule": RULE, "selected_run": selected["run_name"], "selected": selected, "ranked": ranked,
           "n_runs": len(ranked), "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    path = results_root / "phase1" / "model_selection.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{'rank':>4} {'run':34s} {'val mF1':>8s} {'val NLL':>8s} {'val acc':>8s} {'test acc':>8s} {'test mF1':>8s}")
    for r in ranked:
        ta = r["test_accuracy"]
        tf_ = r["test_macro_f1"]
        print(f"{r['rank']:>4} {r['run_name']:34s} {r['val_macro_f1']:8.4f} {r['val_log_loss']:8.4f} {r['val_accuracy']:8.4f} "
              f"{(ta if ta is not None else float('nan')):8.4f} {(tf_ if tf_ is not None else float('nan')):8.4f}")
    print("SELECTED:", selected["run_name"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--pattern", default="p1v2-*", help="glob of run names to rank")
    ap.add_argument("--include", nargs="*", default=["p1-mnv3s-partial-ft-original"],
                    help="extra runs shown in the table (not eligible for selection unless they match --pattern)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    select(Path(args.results_root), args.pattern, tuple(args.include), eligible_pattern=args.pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main())
