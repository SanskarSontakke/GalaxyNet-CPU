# GalaxyNet-CPU (Kaggle-ready TensorFlow Pipeline)

Production-style Galaxy Zoo 3-class classifier (`spiral`, `elliptical`, `irregular`) using:

- Corrected Galaxy Zoo hierarchical label logic.
- `tf.data` path-based loading (no full-RAM image arrays).
- EfficientNetV2B0 transfer learning (warm-up + fine-tuning).
- Focal Loss with inverse-frequency alpha.
- TTA, confusion matrices, ROC, kappa/MCC, and CV stability check.

## Repository Layout

```text
galaxynet/
├── config.py
├── dataset.py
├── model.py
├── losses.py
├── train.py
├── evaluate.py
├── utils.py
├── make_labels.py
├── requirements.txt
├── kernel-metadata.json
└── README.md
```

## Install

```bash
pip install -r galaxynet/requirements.txt
```

## Kaggle Run

Use a Kaggle notebook/script with dataset `galaxy-zoo-the-galaxy-challenge` attached.

```bash
python -m galaxynet.train
```

The training script will:
1. unzip `training_solutions_rev1.zip` and `images_training_rev1.zip` to `/kaggle/temp`
2. generate cleaned 3-class labels with hierarchical thresholds
3. stratify split (70/15/15)
4. train in two phases
5. save full metrics and plots to `/kaggle/working/outputs`

## Local Run

```bash
python -m galaxynet.train \
  --solutions-csv /path/to/training_solutions_rev1.csv \
  --image-dir /path/to/images_training_rev1
```

Or if you already generated labels:

```bash
python -m galaxynet.train \
  --labels-csv /path/to/labels.csv \
  --image-dir /path/to/images_training_rev1
```

## Standalone Label Generation

```bash
python -m galaxynet.make_labels \
  --solutions-csv /path/to/training_solutions_rev1.csv \
  --image-dir /path/to/images_training_rev1 \
  --output-csv labels.csv
```

## Correct Label Rules (Implemented)

- `elliptical`: `Class1.1 >= 0.469`
- `spiral`: `Class1.2 >= 0.430` **and** `Class4.1 >= 0.430`
- `irregular`: `Class6.1 >= 0.469` and not elliptical and not spiral
- else dropped as `unknown`

## Outputs

Saved under `outputs/`:

- `labels.csv`
- `best_model.keras`
- `warmup_best.keras`
- `training_log.csv`
- `training_curves_warmup.png`
- `training_curves_finetune.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `class_metrics_bar.png`
- `roc_curves.png`
- `prediction_confidence_histogram.png`
- `metrics.json`

## Notes

- No `RandomOverSampler` is used.
- All splits are path-based and non-overlapping.
- EfficientNet preprocessing is left internal (images passed as float32 in `[0,255]`).
