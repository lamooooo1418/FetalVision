# FetalVision

Fetal ultrasound analysis in three stages: **standard-plane classification** (Abdomen / Brain / Femur / Thorax),
**fetal-head detection**, **head segmentation** and **head-circumference (HC) measurement**, packaged as an ONNX
pipeline behind an HTTP service.

KAUST Academy AI251P Applied Computer Vision, capstone project. Team FetalVision: Baneen Al-Bahrani (225459),
Dalal Al-Muget (225460), Lama Al-Askar (225470), Rahaf Al-Osaimi (225484).

| Deliverable | File |
|---|---|
| Phase 1 report (revised after feedback) | `reports/FetalVision_phase1_v2_report.pdf` |
| Phase 2 report (M4 detection, M5 segmentation, M8 integration) | `reports/FetalVision_phase2_report.pdf` |
| Dataset card | `DATASET_CARD.md` |
| Model card | `MODEL_CARD.md` |
| Response to the Phase 1 feedback, item by item | `PHASE1_FEEDBACK_RESPONSE.md` |

## Repository layout

```
configs/            one YAML per run or component (all hyper-parameters, seeds, paths)
src/fetalvision/    Python package
  common/           config loading, seeding, metrics (CIs, ECE, temperature scaling, Dice/IoU/mAP), hashing, W&B helpers
  data/             prepare_planes.py (FETAL_PLANES_DB cleaning, near-duplicates, patient-grouped splits, audit)
                    prepare_hc18.py (HC18 ellipse -> mask -> box conversion, grouped split, YOLO export, audit)
  classification/   models, training, evaluation, model selection, calibration, error analysis
  detection/        YOLO11n training, threshold sweep, evaluation and error breakdown
  segmentation/     MobileSAM baseline, U-Net training and evaluation, HC measurement
  pipeline/         end-to-end ONNX pipeline with per-stage timing and manifest
  export/           ONNX export and numerical parity checks
  service/          FastAPI service (/health, /manifest, /metrics, /predict, /predict/annotated)
  bench/            latency / throughput / memory benchmark
notebooks/          Colab notebooks 01-07 (thin wrappers around the package commands)
scripts/            run_phase1_all.sh, run_reproduction.sh, export_all.sh, serve.sh, domain_shift_hc18.py, make_failure_gallery.py
models/             exported ONNX models and their metadata (training weights are not committed)
results/            metrics.json, predictions and figures of every run; every number in the reports comes from here
reports/            the two PDF reports
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
export FETALVISION_DATA=/path/to/data        # datasets are downloaded here and never committed
export WANDB_API_KEY=...                     # optional: without it runs are logged offline (wandb sync later)
```

Datasets (both CC BY 4.0, not redistributed):

* FETAL_PLANES_DB — https://doi.org/10.5281/zenodo.3904280 → `$FETALVISION_DATA/fetal_planes/raw/`
* HC18 — https://doi.org/10.5281/zenodo.1327317 → `$FETALVISION_DATA/hc18/raw/`

## Phase 1 (M1-M3): data foundations and classification

```bash
python -m fetalvision.data.prepare_planes --config configs/phase1_data.yaml --verify-phase1   # reproduces the Phase 1 split exactly, then applies near-duplicate removal
bash scripts/run_phase1_all.sh      # 12 runs (scratch, frozen, partial/full fine-tuning, class weights, stratified split, longer schedules)
                                    # then validation-based selection, temperature scaling and error analysis of the selected run
bash scripts/run_reproduction.sh configs/phase1_mnv3s_partial_ft_e40_rerun.yaml p1v2-mnv3s-partial-ft-e40   # independent rerun
```

Colab: `notebooks/01_phase1_data_audit.ipynb`, `02_phase1_classification.ipynb`, `07_independent_reproduction.ipynb`.

Selection rule (fixed before the runs): highest validation macro-F1, ties broken by lower validation log-loss;
checkpoint = epoch with the lowest validation loss; the test split is evaluated once per run after selection
(`results/phase1/model_selection.json`).

## Phase 2 (M4, M5, M8)

```bash
python -m fetalvision.data.prepare_hc18 --config configs/phase2_hc18_data.yaml
python -m fetalvision.detection.train --config configs/phase2_detection.yaml
python -m fetalvision.detection.sweep --config configs/phase2_detection.yaml        # operating point on the validation split
python -m fetalvision.detection.evaluate --config configs/phase2_detection.yaml     # test metrics, FP/FN breakdown, galleries, demo video
python -m fetalvision.segmentation.sam_baseline --config configs/phase2_segmentation.yaml
python -m fetalvision.segmentation.train_unet --config configs/phase2_segmentation.yaml
python -m fetalvision.segmentation.evaluate --config configs/phase2_segmentation.yaml
python scripts/domain_shift_hc18.py --split test                                     # classifier behaviour on HC18
bash scripts/export_all.sh                                                           # ONNX export + parity checks + manifest
python -m fetalvision.bench.benchmark --config configs/service.yaml --out results/phase2/benchmark/local --backends ort_cpu --http
python scripts/make_failure_gallery.py
bash scripts/serve.sh                                                                # FastAPI on :8000
curl -F "file=@image.png" -F "pixel_size_mm=0.1" http://localhost:8000/predict
docker build -t fetalvision . && docker run -p 8000:8000 fetalvision
```

Colab: `notebooks/03_hc18_prepare.ipynb`, `04_detection.ipynb`, `05_segmentation.ipynb`,
`06_integration_benchmark.ipynb` (T4 benchmark; its results are in `results/phase2/benchmark/colab_t4/`).

## Acceptance criteria (fixed before the Phase 2 results) and outcome

| Component | Criterion | Measured (test split) |
|---|---|---|
| Plane classifier | macro-F1 >= 0.85 | 0.961 (accuracy 97.05 %) |
| Head detector | mAP50 >= 0.95; recall >= 0.98 at the operating point | mAP50 0.995; recall 1.000 |
| Head segmentation | Dice >= 0.95; HC mean absolute error <= 3.0 mm | Dice 0.972; HC MAE 2.10 mm |
| Service | p95 <= 200 ms and >= 5 img/s on a T4; p95 <= 1.5 s on CPU; ONNX parity | T4: p95 42 ms, 28.6 img/s; 2-vCPU CPU: p95 449 ms; parity passed |

## Reproducibility

* One YAML per run; the resolved configuration is saved next to every result.
* `set_seed(42)` in every entry point; the seed is recorded in every `metrics.json`.
* Pinned `requirements.txt` (training and evaluation) and `requirements-service.txt` (Docker image).
* All runs are tracked in Weights & Biases (project `FetalVision`, groups `phase1` and `phase2`).
* No data, no personal information and no hard-coded paths are committed (`$FETALVISION_DATA`).

## Licences and data policy

FETAL_PLANES_DB (Burgos-Artizzu et al., 2020) and HC18 (van den Heuvel et al., 2018) are CC BY 4.0. The system is a
research prototype for plane routing and head-circumference measurement support; it makes no diagnostic claims.
