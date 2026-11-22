# Model card – FetalVision integrated pipeline (v2.0.0)

Structure follows Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). Numbers in `double braces` are
taken from `results/**/metrics.json`;
**no number in this card is typed by hand**.

## 1. Model details

| item | value |
|---|---|
| Developed by | FetalVision capstone team, KAUST Academy AI251P (2026) |
| Version | fetalvision `2.0.0`; model manifest id `68985665657f` |
| System | 4-stage pipeline: (1) standard-plane classifier -> (2) gate (Brain or `force_head`) -> (3) fetal-head detector -> (4) head segmentation on the box crop -> ellipse fit -> head circumference (HC) in px (and mm when the caller supplies the pixel size) |
| Stage 1 – plane classifier | MobileNetV3-Small (Keras 3 / TensorFlow), 224x224x3 RGB input in [0,255] (rescaling inside the graph), softmax over `["Abdomen","Brain","Femur","Thorax"]`, temperature-scaled (T = `0.827`), exported to ONNX opset 17 (`models/plane_classifier.onnx`) |
| Stage 3 – head detector | Ultralytics YOLO11n, single class `fetal_head`, imgsz 640, letterbox 114, raw output `[1,5,8400]`, numpy decode + NMS at the operating point conf = `0.85`, IoU = `0.3` (`models/head_detector.onnx`) |
| Stage 4 – head segmenter | segmentation_models_pytorch U-Net (`mobilenet_v2` encoder), 3-channel ImageNet-normalised input 512x352 (WxH) of the box crop with a 10 % margin, sigmoid output thresholded at 0.5, pasted back at full resolution (`models/head_segmenter.onnx`) |
| HC measurement | largest component -> `cv2.fitEllipse` -> Ramanujan perimeter; HC_mm = HC_px x pixel_size_mm (caller-supplied) |
| Runtime | ONNX Runtime `1.31.1`; providers CPU / CUDA / CoreML selectable in `configs/service.yaml`; FastAPI service (`/predict`, `/predict/annotated`, `/health`, `/manifest`, `/metrics`); Docker image `python:3.12-slim` + CPU onnxruntime |
| Licence of the code | project licence (see repository) |
| Contact / feedback | repository issues |

### Training data
* **FETAL_PLANES_DB** (Burgos-Artizzu et al., 2020; Zenodo 3904280; **CC BY 4.0**) – 4-class subset (Abdomen, Brain, Femur, Thorax): 6 440 images from 1 158 patients, two hospitals, six ultrasound machines. Patient-grouped train/val/test split (no patient in two splits); near-duplicates removed with pHash+dHash before splitting (`results/phase1/*/audit.json`).
* **HC18** (van den Heuvel et al., 2018; Zenodo 1327317; **CC BY 4.0**) – 999 training images (800x540) of the standard trans-thalamic plane with expert HC ellipse annotations and the pixel size; used for the head detector (boxes derived from the ellipse annotations) and the segmenter (filled-ellipse masks). HC18 ships **no patient identifiers**, so its split is image-grouped only (see Caveats).

## 2. Intended use
* **Primary use**: research prototype that (a) routes a 2-D fetal ultrasound frame to one of four standard planes and (b) for brain planes proposes a fetal-head ellipse and an HC value to *support* (not replace) a human measurement.
* **Intended users**: medical-imaging researchers and developers evaluating automated plane recognition / biometry; course assessors.
* **Out-of-scope uses** (the model must **not** be used for):
  * clinical diagnosis, screening or any patient-management decision;
  * gestational-age dating or growth-chart decisions derived from the HC value;
  * detection of anomalies (ventriculomegaly, skull defects, ...) – the segmenter only outlines the skull contour;
  * non-standard planes, maternal cervix, placenta, other fetal structures (the classifier has only four classes and no "other" class: every input is forced into one of them);
  * other imaging modalities (MRI, CT, photographs of screens) or 3-D / 4-D volumes;
  * video streams / cine loops – frame-by-frame use is **unvalidated** (no temporal smoothing, no plane-quality check);
  * images with burned-in patient data or text overlays (not seen in training; also a privacy risk);
  * paediatric, neonatal or adult heads; animal imaging;
  * any setting where the pixel size is unknown or unreliable (HC in mm is only as good as the supplied pixel size).

## 3. Factors
* **Acquisition**: ultrasound vendor/machine (6 machines in FETAL_PLANES_DB, 3 in HC18), probe, depth and gain settings, presence of callipers or annotations, cropping of the fan.
* **Operator**: plane quality and standardisation vary between operators; HC18 contains only expert-approved planes.
* **Fetal factors**: gestational age (HC18 spans the 2nd–3rd trimester; head size and shadowing change with GA), fetal position, twins, oligohydramnios.
* **Image quality**: acoustic shadowing of the far skull, low contrast, motion.
* **Class imbalance**: Brain is the majority class in FETAL_PLANES_DB; Femur/Thorax are minority classes – per-class recall is reported, not only accuracy.
* **Dataset shift between stages**: stage 1 is trained on FETAL_PLANES_DB, stages 3–4 on HC18; the two datasets differ in vendor, cropping and resolution.

## 4. Metrics and acceptance criteria (fixed in advance)

| stage / property | metric | acceptance criterion | result | source |
|---|---|---|---|---|
| Plane classifier (test, patient-grouped) | accuracy (95 % Wilson CI) | >= `macro-F1 >= 0.85` | `97.05 %` (`95.74-97.96 %`) | results/phase1/p1v2-mnv3s-partial-ft-e40/metrics.json |
| Plane classifier | macro F1 / Brain recall | report | `0.9605` / `0.989` | idem |
| Plane classifier | ECE after temperature scaling | report | `0.0074 (after temperature scaling; 0.0096 before)` | results/phase1/p1v2-mnv3s-partial-ft-e40/calibration.json |
| Head detector (HC18 test) | mAP50 / mAP50-95; recall at operating point | mAP50 >= `0.95` | `0.995` / `0.945`; `1.000` | results/phase2/detection/metrics.json |
| Head segmenter (HC18 test) | Dice; HC abs. error (mm) | Dice >= `0.95`; HC MAE <= `3.0 mm` mm | `0.9716`; `2.10` | results/phase2/segmentation/metrics.json |
| End-to-end HC (HC18 test, detector -> segmenter) | HC MAE / median abs. error (mm), % within 5 mm | report | `2.71` / `2.29`, `64.9 % within 3 mm` | results/phase2/segmentation/metrics.json |
| ONNX parity – classifier | max abs diff of probabilities; argmax agreement | < 1e-4; 100 % | `2.3e-06`; `100 %` | results/phase2/export/parity_classifier.json |
| ONNX parity – detector | best-box IoU (min); max abs score diff | >= 0.99; < 1e-3 | `0.9989`; `1.8e-07` | results/phase2/export/parity_detector.json |
| ONNX parity – segmenter | mask Dice (min) | >= 0.999 | `1.0000` | results/phase2/export/parity_segmenter.json |
| Latency – T4 GPU (ORT CUDA) | full-pipeline p95 (ms); throughput (img/s) | <= 200 ms; >= 5 img/s | `42 ms`; `28.6 img/s` | results/phase2/benchmark/colab_t4/metrics.json |
| Latency – 2-vCPU CPU (ORT CPU) | full-pipeline p95 (ms) | <= 1500 ms | `449 ms` | results/phase2/benchmark/colab_t4/metrics.json (ort_cpu backend) |
| Latency – Apple M4 Pro (ORT CPU, dev machine) | full-pipeline p50 / p95 (ms) | report | `46 ms` / `52 ms` | results/phase2/benchmark/m4pro_cpu/metrics.json |

Latency conditions: batch 1, 800x540 HC18 inputs, 20 warm-up + 200 timed iterations, per-stage input resolutions
224x224 / 640x640 / 512x352, single process; hardware, versions and thread settings are recorded in each
`metrics.json` under `conditions`. Benchmarks run with `force_head=true` so that all four stages execute on every image.

## 5. Evaluation data
* Classifier: FETAL_PLANES_DB test split, **patient-grouped** (no patient overlap with train/val), `914` images; class distribution reported in `results/phase1/p1v2-mnv3s-partial-ft-e40/metrics.json`.
* Detector / segmenter / end-to-end HC: HC18 test split (`154` images) defined in `data/hc18/processed/splits.csv`; HC ground truth from the official `training_set_pixel_size_and_HC.csv`.
* Parity and latency: 32 HC18 images (every 25th file of the training set) + 16 uniform-noise images; the noise images probe numerical robustness only.

## 6. Ethical considerations
* **No personal data**: both datasets are public, anonymised research datasets released under CC BY 4.0; raw images are never committed to the repository; the service stores no uploaded images (it keeps only counters and latency statistics in memory).
* **Licences**: FETAL_PLANES_DB and HC18 are CC BY 4.0 – attribution is given in `DATASET_CARD.md`; the pretrained backbones (ImageNet / COCO) are used under their respective licences (Ultralytics YOLO11: AGPL-3.0 – a commercial deployment would require an Ultralytics licence).
* **No diagnostic claim**: outputs are probabilistic suggestions; the HC value is a geometric fit, not a validated biometric measurement. Any misuse for clinical decisions could cause harm (wrong dating, missed anomaly). The service returns `head_measurement: not_applicable` for non-brain planes rather than guessing.
* **Fairness**: the training data come from two Spanish hospitals (FETAL_PLANES_DB) and one Dutch hospital (HC18); performance on other populations, machines and protocols is unknown.

## 7. Caveats and recommendations
* **HC18 has no patient IDs** – multiple images of the same fetus may sit in both train and test; detector/segmenter scores may be optimistic.
* **Domain shift between the two datasets** – the classifier is trained on FETAL_PLANES_DB (Spain, several machines) while HC18 comes from one Dutch centre. On the HC18 test split (all trans-thalamic brain planes) the selected classifier routes 90.9 % of the images as Brain (`results/phase2/domain_shift/metrics.json`); the remainder would be rejected by the gate. This is why the service exposes `force_head`, flags low-confidence planes for review, and why a plane-quality / out-of-distribution check and a small target-centre fine-tuning are recommended before any cross-centre deployment.
* **HC depends on the caller-supplied pixel size** – no DICOM parsing is performed; wrong units silently produce wrong millimetres.
* **Stand-ins** – until the trained detector and segmenter are exported, the manifest lists `stand_in_models`; results produced with stand-ins are engineering measurements (latency, parity), not accuracy claims. The current manifest: `none`.
* Temperature scaling is fitted on the validation split; calibration on HC18-like images is not guaranteed.
* The detector returns only the highest-scoring box; twins or two heads in one frame are not handled.
* Recommended next steps: an "other/non-standard" class or OOD detector in stage 1, patient-level HC18 splitting if IDs become available, external validation set from a third vendor, uncertainty reporting on the ellipse fit (e.g. mask-boundary agreement).

