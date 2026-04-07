# GalaxyNet-CPU

CPU-first machine learning pipeline for Galaxy Zoo image classification (spiral, elliptical, irregular).

## Features

- End-to-end training/evaluation script for **CPU-only** systems.
- Image preprocessing (resize, normalize, label encoding).
- Dataset split with target ratios: **70% train / 15% validation / 15% test**.
- Lightweight CNN baseline (<2M parameters).
- Training with class weighting, early stopping, and reproducibility controls.
- Outputs:
  - Trained model
  - Accuracy and loss curves
  - Confusion matrix
  - JSON metrics report

## Project Structure

```text
GalaxyNet-CPU/
├── README.md
├── requirements.txt
├── train_galaxy_classifier.py
└── research_paper_template.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Layout

Put your dataset into class folders:

```text
data/
├── spiral/
│   ├── img1.jpg
│   └── ...
├── elliptical/
│   ├── img2.jpg
│   └── ...
└── irregular/
    ├── img3.jpg
    └── ...
```

Supported image types: `.jpg`, `.jpeg`, `.png`.

## Train

```bash
python train_galaxy_classifier.py \
  --data-dir data \
  --output-dir outputs \
  --image-size 64 \
  --epochs 20 \
  --batch-size 32
```

## Recommended CPU Settings

- `--image-size 96` (96×96 is still comfortable on a modern 4-core CPU)
- `--batch-size 16`
- `--epochs 40`
- `--max-images 15000` (optional cap)

## Outputs

The script creates:

- `outputs/galaxy_cnn.keras`
- `outputs/training_curves.png`
- `outputs/confusion_matrix.png`
- `outputs/metrics.json`

`metrics.json` includes:

- train/val/test sizes
- test accuracy and loss
- training time
- inference latency per image
- class mapping and model parameter count

## Reproducibility

Use `--seed` (default: `42`) for deterministic shuffling/splits and consistent initialization.

## Notes

- This baseline is intentionally lightweight for student-friendly CPU training.
- For better accuracy, consider careful augmentation, class balancing, and hyperparameter tuning.
