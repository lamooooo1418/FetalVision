# FetalVision dataset card (v2 — Phase 1 revision and Phase 2)

Two public datasets are used. Neither is redistributed; both are downloaded from Zenodo by the preparation scripts.
No data are committed to this repository. Both datasets are released under **CC BY 4.0** (verified through the Zenodo
API records 3904280 and 1327317 on 8 November 2026).

## 1. FETAL_PLANES_DB — plane classification (Phase 1, M1–M3)

| Attribute | Value |
|---|---|
| Source | Burgos-Artizzu et al., *Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes*, Sci. Rep. 10, 10200 (2020). Zenodo DOI 10.5281/zenodo.3904280 |
| Licence | CC BY 4.0 (attribution required; redistribution of the processed images is not done) |
| Acquisition | Two hospitals in Barcelona (BCNatal), routine screening, six ultrasound machines, several operators; 12,400 images from 1,792 patients, labelled under the supervision of an expert maternal–fetal clinician |
| Metadata | `Image_name, Patient_num, Plane, Brain_plane, Operator, US_Machine, Train` |
| Subset used | Four fetal anatomical planes: Abdomen, Brain, Femur, Thorax (Maternal cervix and Other excluded) → 6,561 records before cleaning |
| Cleaning (Phase 1, reproduced exactly) | 0 unreadable files; 117 exact duplicates (SHA-256) removed; 4 records on 2 identical images with conflicting labels excluded → 6,440 images, 1,158 patients |
| Cleaning added in v2 | Perceptual-hash near-duplicate detection (pHash ≤ 4 bits and dHash ≤ 6 bits, 64-bit hashes), evaluated within and across patients; counts and policy in `results/phase1/data_audit/audit.json` and in the Phase 1 v2 report §2.2 |
| Grouping key | `Patient_num` — all images of a patient fall in one split (images of one patient share anatomy, machine and settings) |
| Splits | `group_shuffle`: GroupShuffleSplit(seed 42) 70/15/15 — identical patient assignment to the Phase 1 submission; `stratified_group`: StratifiedGroupKFold-based 70/15/15 keeping the class mix per split (used for the class-imbalance experiment) |
| Class balance | Brain 47.1 %, Thorax 26.4 %, Femur 15.7 %, Abdomen 10.8 % (whole subset); per-split percentages in `audit.json` |
| Label audit | 100 images (25 per class) reviewed by the team against the dataset labels: 0 confirmed errors → error rate 0 %, one-sided 95 % Clopper–Pearson upper bound 2.95 %. Reviewers are not clinicians; the bound describes agreement with the published labels, not clinical truth |
| Known limitations | Strong class imbalance; images from a single centre and country; burned-in machine text/scale markers present in images; labels are plane categories, not diagnoses |
| Privacy | No names or identifiers in the images; `Patient_num` is a pseudonymous integer used only for grouping and never exported with predictions |

## 2. HC18 — fetal head detection, segmentation and head-circumference measurement (Phase 2, M4–M5)

| Attribute | Value |
|---|---|
| Source | van den Heuvel, de Bruijn, de Korte, van Ginneken, *Automated measurement of fetal head circumference using 2D ultrasound images*, PLoS ONE 13(8): e0200412 (2018). Grand Challenge HC18. Zenodo DOI 10.5281/zenodo.1327317 |
| Licence | CC BY 4.0 |
| Content | 999 training images (800 × 540 px, grayscale PNG) with a sonographer-drawn head ellipse outline (`*_Annotation.png`), per-image pixel size (0.052–0.326 mm) and head circumference in mm; 335 test images without annotations (not used) |
| Acquisition | Radboud University Medical Center (Nijmegen, NL); standard trans-thalamic plane; images from 551 pregnant women across trimesters |
| Derived annotations | Filled ellipse → binary head **mask** (segmentation ground truth); mask bounding rectangle → **box** (detection ground truth, class `fetal_head`); conversion verified visually on a 40-image sample and automatically for every image (`results/phase2/hc18_prep/`) |
| Grouping key | HC18 publishes **no patient IDs**. Proxy: examination id = numeric filename prefix (`010_HC.png` and `010_2HC.png` are the same examination) merged with perceptual-hash near-duplicate clusters. This is a limitation: two examinations of the same woman cannot be linked |
| Splits | GroupShuffleSplit on the proxy group, seed 42, 70/15/15; HC and pixel-size distributions per split in `audit.json` |
| Known limitations | Single centre; every image contains exactly one head (no negatives → false positives can only be studied as duplicate or mislocalised boxes); annotation is an ellipse, so the "mask" is a geometric approximation of the skull outline |
| Privacy | Grayscale images without burned-in identifiers; no personal data |

## 3. Domain shift between the datasets
The classifier is trained on FETAL_PLANES_DB (Spain, several machines) and gates HC18-style images (Netherlands,
one centre). The integration tests therefore report the classifier's behaviour on HC18 brain images separately
(Phase 2 report §4), and the model card lists this shift as a caveat.

## 4. Intended use and out-of-scope use
Research and education on plane classification, head detection/segmentation and biometric measurement support.
Not a medical device; not for clinical diagnosis, gestational-age dating decisions or anomaly detection.
