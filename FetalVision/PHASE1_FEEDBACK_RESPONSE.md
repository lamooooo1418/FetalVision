# Response to the Phase 1 feedback (29 / 40)

Each deduction from the instructors' Phase 1 feedback sheet is listed with the
action taken in this revision and the evidence that a grader can open. "Evidence" paths are inside this repository.

| # | Part | Deduction (points) | What was wrong | Action taken in v2 | Evidence |
|---|------|-------------------|----------------|--------------------|----------|
| 1 | Part 1 Cleaning | −1 | No near-duplicate check and no reason given | Perceptual-hash near-duplicate detection (pHash ≤ 4 bits **and** dHash ≤ 6 bits on 64-bit hashes), run within each patient and across patients; counts and the removal policy are reported; the check is a config switch so the original split can still be reproduced exactly | `src/fetalvision/common/hashing.py`, `src/fetalvision/data/prepare_planes.py`, `results/phase1/data_audit/audit.json`, report §2.2 |
| 2 | Part 3 Label audit | −1.5 | Error rate and 95 % bound "not established" | 0 confirmed errors in 100 audited images → point estimate 0 %, one-sided 95 % Clopper–Pearson upper bound 2.95 %; second-round audit list generated (balanced sample + every test error of the kept model) | `fetalvision.common.metrics.clopper_pearson_upper`, `audit.json → label_audit`, report §2.4 |
| 3 | Part 5 Models compared | −1 | Only partial fine-tuning (last 30 layers); full fine-tuning missing | New run with **all** backbone layers trainable (`p1v2-mnv3s-full-ft`), compared with scratch, frozen and partial runs on validation in W&B | `configs/phase1_mnv3s_full_ft.yaml`, `results/phase1/p1v2-mnv3s-full-ft/`, report §3.1 |
| 4 | Part 6 Protocol | −0.5 −0.5 | Kept model argued with test results; checkpoint rule not stated; validation macro-F1 not shown | Selection rule fixed before the runs: kept model = highest **validation** macro-F1 (tie → lower validation log-loss); checkpoint = epoch with lowest validation loss; validation macro-F1 is reported for every run and the test set is evaluated once per run after selection | `src/fetalvision/classification/select_model.py`, `results/phase1/model_selection.json`, report §3.1 |
| 5 | Part 9 Error analysis and plan | −0.5 −0.5 −0.5 | Gallery (12 images) contradicted the confusion matrix; causes were guesses; plan actions generic and untested | Gallery and confusion matrix are now produced from the **same** `predictions_test.csv`; all test errors are shown and tabulated; each hypothesised cause is tested with a measurement (confidence distribution, error rate by ultrasound machine and operator, clustering by patient, near-duplicates to training images, top-2 recovery); the plan actions (class-weighted loss, stratified-group split, augmentation during training, temperature scaling) were implemented and their effect measured | `src/fetalvision/classification/error_analysis.py`, `results/phase1/<run>/error_analysis.json`, report §3.5–3.6 |
| 6 | Part 10 Repository | −1.5 | Repository held only an HC18 labelling notebook | Full Phase 1 code (data preparation, training, evaluation, calibration, error analysis), reproduction notebook, augmentation config, pinned `requirements.txt` committed | this repository |
| 7 | Part 11 Seeds | −1 | Seed stated but code absent | `set_seed(42)` in every entry point; seed recorded in every `metrics.json` | `src/fetalvision/common/seed.py` |
| 8 | Part 12 Pinned requirements | −1.5 | No requirements file | `requirements.txt` with exact versions used for the runs | `requirements.txt` |
| 9 | Part 13 Independent reproduction | −1 | Rerun numbers could not be traced to a run | Reproduction notebook writes `results/phase1/reproduction/<run>/metrics.json` and the comparison table; W&B run of the rerun | `notebooks/07_independent_reproduction.ipynb`, `results/phase1/reproduction/` |
| — | W&B project not accessible | — | Instructors could not open the project | Project visibility set to public / instructors added as viewers; all runs (offline runs synced with `wandb sync`) | W&B link in the report header |
| — | Class imbalance only monitored | — | Brain 47 %, Abdomen 11 % | Class-weighted loss (`balanced`) and StratifiedGroupKFold split runs; Abdomen recall and Abdomen↔Thorax confusion compared before/after | `results/phase1/p1v2-mnv3s-*-cw*/`, report §3.6 |

## Differences between the Phase 1 submission and v2 that change numbers
* Near-duplicates are removed (within-patient clusters keep one image). Patients assigned to each split are unchanged
  because the patient-level `GroupShuffleSplit` depends only on the set of patient IDs; image counts per split shrink.
* All v2 runs train **with** the training-only augmentation pipeline that Phase 1 only verified.
* Runs were executed on the hardware stated in each `metrics.json` (CPU runs locally or T4 in Colab); results may
  differ slightly from Colab T4 numbers.
