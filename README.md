# GalaxyNet — Unified 37-Node Regression Pipeline

Production-grade deep learning pipeline designed to predict the exact fractional human consensus distributions for the **Galaxy Zoo Challenge** on Kaggle. This model mathematically bypasses discrete classification, directly optimizing **Root Mean Squared Error (RMSE)** across the complete 37-dimensional spatial-attribute array.

> [!TIP]
> **Curious about the math?** Start by reading the [Technical Deep Dive Guide](TECHNICAL_GUIDE.md) for architectural block diagrams and progressive resizing strategies.

## Architecture: Benanne-Inspired Regression

```text
Input (424×424) → Benanne Affine Augmentation
                        ↓
       69×69 View + 45° Rotated View
                        ↓
        16 Aligned 45×45 Shared Parts
                        ↓
              Shared 4-Layer ConvNet
                        ↓
            2× Maxout Dense(2048)
                        ↓
                Dense(37 logits)
                        ↓
      GalaxyOutputLayer (tree normalization + weighting)
                        ↓
           TTA Averaged Output Float32
```

### Why Regression instead of Classification?
- **Native Objective Alignment**: Kaggle evaluates this competition using *Root Mean Squared Error (RMSE)* on the raw voting fractions, not categorical cross-entropy. Hard-thresholding classes destroys perfectly viable partial-confidence data.
- **Solves the "Irregular" Imbalance**: By treating "Irregularity" as a fluid percentage (e.g., 60% irregular, 40% spiral) rather than a rigid boundary, we completely avoid the need to oversample or apply artificial class weights.
- **Architectural Elegance**: Collapses complex multi-stage cascading pipelines down into a single massive, highly unified prediction head capable of mapping interdependent variables.

## Key Features

- **Benanne-Inspired View Logic**: Regular and 45° rotated views are split into 16 aligned 45×45 parts before feature extraction.
- **Ordered 90/10 Validation Split**: Matches the legacy benanne validation protocol more closely than the earlier random split.
- **Direct RMSE Optimization**: Native `RMSELoss` is applied after tree-aware normalization so validation and inference score the same object.
- **Benanne Affine Augmentations**: Random rotation, zoom, translation, flip, and color perturbation are ported from the winning solution logic.
- **60-TTA Optimization**: Final evaluation and submission use 10 rotations × 3 zoom scales × 2 flips.

## Repository Layout

```text
galaxynet/
├── config.py              # Central unified configuration dataclass
├── dataset.py             # Data loader streaming '37-node target coordinates' directly
├── model.py               # Unified build_regression_model() logic
├── losses.py              # Custom RMSE metric and Loss objects
├── train.py               # Linear 3-Phase orchestrator (runs without multiprocessing logic)
├── evaluate.py            # Global Multi-Target TTA RMSE Evaluation 
├── utils.py               # Runtime setup, environment resolution, helper utilities
├── scripts/
│   ├── consolidate.py     # Compiles runtime codebase into a single Kaggle script
│   └── pack_dataset.py
├── requirements.txt
├── TECHNICAL_GUIDE.md     # In-depth architectural methodology 
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Kaggle Run Environment

1. Generate the bundle (compiles all local scripts into a single Kaggle-valid executing string):
   ```bash
   python scripts/consolidate.py
   ```

2. Push to Kaggle servers:
   ```bash
   kaggle kernels push -p .
   ```

### Execution Flow:
1. Extracts Kaggle *Galaxy Zoo - The Galaxy Challenge* data cleanly into the local environment.
2. Formats all 37 targets into `float32` bounding targets natively.
3. Warmup Phase (128x128 bounding boxes, frozen core).
4. Mid-Tune Phase (same architecture, lower SGD learning rate).
5. Fine-Tune Phase (continued full-resolution training with lower LR).
6. Evaluates on the held-out validation split and exports a competition-format `submission.csv.gz`.

## Expected Results

A successful full single-pass training run is expected to yield:

| Set | Expected Leaderboard Target | Validation |
|-------|-----------|--------|
| **Kaggle Objective** | **~0.100 RMSE** | **~0.098 Val_RMSE** |

> Achieving ~0.100 RMSE implies that, on average across all 37 distinct biological and functional target classes, your model is perfectly mimicking human crowdsourced astronomists to within 10% tolerance universally.

## Outputs

```text
outputs/
├── phase1.keras                       # Frozen core weights
├── phase2.keras                       # 192px tuned core
├── unified_best.keras                 # Fully optimized model (Final)
├── evaluation_results.json            # Final metrics dictionary
├── submission.csv.gz                  # Competition-format submission file
└── split_info.json
```

## Hardware Allocation

| Component | GPU P100 16GB Time | GPU T4x2 Time |
|-----------|---------------|---------------|
| Phase 1 (128px Warmup) | ~30 min | ~18 min |
| Phase 2 (192px Mid-Tune) | ~45 min | ~25 min |
| BenanneNetTF Full-Resolution Train | ~2.5 hrs | ~1.5 hrs |
| **Total** | **~3.9 hours** | **~2.2 hours** |

_Runs comfortably within the standard Kaggle 12-hour session quota limits without Out-Of-Memory interrupts due to the un-looped batch scaling protocol._
