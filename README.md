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
├── research_paper_template.md
└── scripts/
    └── organize_dataset.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Where to Get Galaxy Data (Bulk)

You can get galaxy morphology data in two practical ways:

1. **Kaggle (easiest bulk download)**
   - Recommended for a student project because images are typically pre-packaged.
   - Install the Kaggle CLI and set your API token (`~/.kaggle/kaggle.json`).
   - Example bulk download:

   ```bash
   kaggle competitions download -c galaxy-zoo-the-galaxy-challenge -p data/raw
   unzip data/raw/galaxy-zoo-the-galaxy-challenge.zip -d data/raw
   ```

2. **Galaxy Zoo official data releases (research-grade catalogs)**
   - Best if you want original catalog-level labels and publication-grade provenance.
   - Download images/catalogs from official Galaxy Zoo data pages and then organize them into class folders.

## Data Layout Required by This Project

The training script expects class folders exactly like this:

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

## Organize Data Automatically

If you have:
- a flat image folder (for example `data/raw/images/`), and
- a CSV labels file with columns `filename,label`

you can organize your dataset in bulk:

```bash
python scripts/organize_dataset.py \
  --images-dir data/raw/images \
  --labels-csv data/raw/labels.csv \
  --output-dir data \
  --mode copy
```

Valid labels are:
- `spiral`
- `elliptical`
- `irregular`

## Train

```bash
python train_galaxy_classifier.py \
  --data-dir data \
  --output-dir outputs \
  --image-size 64 \
  --epochs 20 \
  --batch-size 32
```

### Recommended CPU Settings

- `--image-size 64`
- `--batch-size 16` or `32`
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
