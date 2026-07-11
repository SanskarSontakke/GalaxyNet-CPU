# GalaxyNet

> A TensorFlow reimplementation of a rotation-invariant convolutional network that predicts the 37 Galaxy Zoo crowd-vote fractions for a galaxy image, built for the Kaggle *Galaxy Zoo – The Galaxy Challenge*.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.12%E2%80%932.17-FF6F00.svg)](https://www.tensorflow.org/)

## What it does

Given a 424×424 galaxy image, GalaxyNet predicts the 37 probability values that the
Galaxy Zoo competition asks for — the fraction of volunteers who would give each answer
across the project's 11-question morphology decision tree. It is trained as a direct
regression against those fractions using RMSE, the same metric the Kaggle leaderboard uses,
so there is no separate classification step. A custom output layer normalizes each question
block and re-applies the decision-tree weights so the predictions stay internally consistent.

## Why I built it

I wanted to reproduce Sander Dieleman's (benanne) winning Galaxy Zoo approach in TensorFlow
to learn how rotation invariance, viewpoint averaging, and test-time augmentation are built
into a network, and how a hierarchical multi-label target can be predicted directly instead
of through a cascade of classifiers.

## Tech stack

- Python 3.11
- TensorFlow / Keras (model, training loop, data pipeline)
- NumPy and pandas (label handling, arrays)
- scikit-learn (train/validation splitting)
- OpenCV and matplotlib (image and plotting utilities)
- tqdm (progress bars)

Exact version ranges are pinned in [`requirements.txt`](requirements.txt).

## Getting started

```bash
# 1. Clone
git clone https://github.com/SanskarSontakke/GalaxyNet-CPU.git
cd GalaxyNet-CPU

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide the data
#    Download the "Galaxy Zoo - The Galaxy Challenge" competition data from Kaggle
#    and place it under ./galaxy_raw/ (the loader also accepts the zipped archives
#    and extracts them). It looks for training_solutions_rev1.csv and the
#    images_training_rev1 / images_test_rev1 folders.

# 4. Train + evaluate locally
python train.py
```

### Running on Kaggle

`train.py` and the other modules are merged into a single script so they can run as a
Kaggle kernel:

```bash
python scripts/consolidate.py   # regenerates train_kaggle_bundle.py
kaggle kernels push -p .        # uses kernel-metadata.json
```

## How it works

The default architecture (`BenanneNetTF` in `config.py`) follows the benanne design:

```text
Input (424×424×3)
      ↓ rescale to [0,1]
Two views (regular + 45° rotated), each resized to 69×69, plus flips
      ↓
16 aligned 45×45 parts (4 corner rotations × 4 views)
      ↓
Shared 4-layer convnet  (Conv → BatchNorm → ReLU, ×4)
      ↓
Merge the 16 part-features back per galaxy
      ↓
2× Maxout dense (2048 units) with dropout 0.5
      ↓
Dense(37) logits
      ↓
GalaxyOutputLayer: per-question softmax-style normalization + decision-tree weighting
      ↓
37 tree-weighted probabilities
```

Training runs in three phases that act as a manual step-decay of the SGD learning rate
(`4e-2 → 4e-3 → 4e-4`), each phase starting from the best validation weights of the
previous one. Early stopping and best-checkpoint saving are applied in every phase, so the
epoch counts in `config.py` are generous caps rather than fixed durations. Evaluation and
submission use test-time augmentation (10 rotations × 3 zooms × 2 flips = 60 passes) and
report the global RMSE.

An alternative path in `model.py` swaps the from-scratch convnet for an ImageNet backbone
(`EfficientNetV2B0/B1/B2`, `ConvNeXtTiny`) with progressive unfreezing; select it by setting
`architecture` in `config.py`.

For a deeper walkthrough of the architecture and the decision-tree math, see
[`TECHNICAL_GUIDE.md`](TECHNICAL_GUIDE.md).

### Repository layout

```text
config.py                 # Central configuration dataclass
dataset.py                # tf.data pipeline: loading, augmentation, splits
model.py                  # build_regression_model() + custom layers
losses.py                 # RMSE loss and metric (aligned to the Kaggle metric)
train.py                  # 3-phase training orchestrator
evaluate.py               # TTA evaluation + competition submission writer
utils.py                  # Environment/strategy setup and helpers
scripts/consolidate.py    # Merges modules into train_kaggle_bundle.py for Kaggle
train_kaggle_bundle.py    # Auto-generated single-file bundle (do not edit by hand)
kernel-metadata.json      # Kaggle kernel definition
```

## Results / status

Working training pipeline and a learning project — still a work in progress. The pipeline
runs end to end (train → validate → TTA → submission file), and the architecture reproduces
the benanne design in TensorFlow.

I have not yet completed a full, verified training run of the current `BenanneNetTF`
configuration, so I am not quoting a leaderboard or validation number here.

<!-- TODO: after a full run, record the measured global validation RMSE and TTA RMSE here. -->

> Note: `REPORT.md` contains earlier RMSE figures, but those were produced by a different
> (EfficientNet-based) configuration and do not describe the current default network. They
> need to be re-measured before being treated as results for this model.

## License

MIT © 2026 Sanskar Sontakke
