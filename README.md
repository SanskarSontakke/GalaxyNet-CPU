# GalaxyNet V25 — Two-Stage Cascade Pipeline

Production-grade Galaxy Zoo 3-class morphology classifier (`spiral`, `elliptical`, `irregular`) targeting **≥95% overall accuracy** and **≥0.80 irregular F1**.

> [!TIP]
> **New to the project?** Start with our [Technical Deep Dive Guide](TECHNICAL_GUIDE.md) for architectural diagrams and flowcharts.

## Architecture: Two-Stage Cascade

```
Input (224×224) → Stage 1: EfficientNetV2B1 (Elliptical vs Non-Elliptical)
                    ├── P(ell) ≥ t₁ → Predict: Elliptical
                    └── P(ell) < t₁ → Stage 2: EfficientNetV2B2+B1 Ensemble
                                        ├── P(spiral) ≥ t₂ → Predict: Spiral
                                        └── P(spiral) < t₂ → Predict: Irregular
```

### Why Cascade?
- **Decouples easy from hard**: Elliptical galaxies are highly separable (smooth, round). A dedicated binary classifier achieves >99% accuracy, removing contamination from the harder Stage 2.
- **Focuses capacity**: Stage 2 dedicates all model capacity to the hard spiral/irregular boundary without elliptical interference.
- **Optimal thresholds**: Each stage has an independently calibrated threshold via 2D grid search.

## Key Features

- **3-Phase Resolution Curriculum**: 128→192→224px progressive resizing
- **Cosine Annealing with Warm Restarts (SGDR)**: Escapes local minima in complex loss landscapes
- **Stochastic Weight Averaging (SWA)**: +0.5-1.5% generalization boost
- **Binary Focal Loss**: Separate gamma per stage (1.5 easy / 2.5 hard)
- **Class-Differentiated Augmentation**: Aggressive Mixup + Cutout for irregular class
- **Stage 2 Ensemble**: 2 models (EfficientNetV2B2 + B1) averaged before threshold
- **16-TTA**: 4 rotations × 2 flips × 2 crops = systematic coverage
- **Auto-Fallback**: If irregular F1 < 0.75, automatically re-trains Stage 2 with relaxed thresholds

## Repository Layout

```text
galaxynet/
├── config.py              # V25 Config dataclass with all hyperparameters
├── dataset.py             # Label gen, Mixup, Cutout, per-class augmentation, binary datasets
├── model.py               # build_stage1_model(), build_stage2_model()
├── losses.py              # BinaryFocalLoss + CategoricalFocalLoss
├── train.py               # Cascade training orchestrator (subprocess-isolated)
├── evaluate.py            # Cascade eval: TTA-16, threshold calibration, ECE, error analysis
├── utils.py               # SWA, gradient accumulation, LLRD, SGDR schedule
├── make_labels.py         # Standalone label generation with validation
├── scripts/
│   ├── consolidate.py     # Merges modules into train_kaggle_bundle.py
│   └── organize_dataset.py
├── requirements.txt
├── kernel-metadata.json   # Kaggle GPU + Internet enabled
└── README.md
```

## Label Rules (V25 — Tightened)

| Class | Threshold Rule |
|-------|---------------|
| Elliptical | `Class1.1 ≥ 0.469` |
| Spiral | `Class1.2 ≥ 0.450` AND `Class4.1 ≥ 0.450` |
| Irregular | `Class6.1 ≥ 0.500` AND `(Class6.1−Class6.2) ≥ 0.15` AND not elliptical AND not spiral |

## Install

```bash
pip install -r requirements.txt
```

## Kaggle Run

1. Generate the bundle:
   ```bash
   cd GalaxyNet-CPU
   python scripts/consolidate.py
   ```

2. Push to Kaggle:
   ```bash
   kaggle kernels push -p .
   ```

The training pipeline will:
1. Extract Galaxy Zoo data from competition source
2. Generate 3-class labels with V25 tightened thresholds
3. Train Stage 1 (elliptical binary) — 3 progressive phases + SWA
4. Train Stage 2 ensemble (spiral/irregular binary) — 2 models × 3 phases + SWA
5. Calibrate cascade thresholds via 2D grid search on validation set
6. Evaluate with 16-TTA and generate comprehensive metrics + plots
7. Auto-fallback if irregular F1 < 0.75

## Expected Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Spiral | ~88% | ~93% | ≥ 0.90 |
| Elliptical | ~96% | ~96% | ≥ 0.94 |
| Irregular | ~75% | ~85% | ≥ 0.80 |
| **Overall** | | | **≥ 0.95 accuracy** |

## Outputs

```text
outputs/
├── stage1_best.keras                    # Stage 1 model
├── stage2_EfficientNetV2B2_seed42_best.keras
├── stage2_EfficientNetV2B1_seed123_best.keras
├── training_curves_stage1_phase[1-3].png
├── training_curves_stage2_*_phase[1-3].png
├── confusion_matrix_cascade.png
├── confusion_matrix_normalized.png
├── confusion_matrix_stage2_binary.png
├── roc_curves_cascade.png
├── threshold_sensitivity_stage1.png
├── threshold_sensitivity_stage2.png
├── confidence_calibration.png
├── class_metrics_bar.png
├── prediction_confidence_histogram.png
└── metrics.json
```

## Training Time

~4 hours on Kaggle P100 (16GB VRAM), well within the 9-hour session limit.

| Component | Estimated Time |
|-----------|---------------|
| Stage 1 (3 phases + SWA) | ~60 min |
| Stage 2 Model A (3 phases + SWA) | ~70 min |
| Stage 2 Model B (3 phases + SWA) | ~70 min |
| Evaluation + Plots | ~15 min |
| **Total** | **~215 min** |

## Previous Version (V24)

V24 used a flat EfficientNetV2B0 + ConvNeXtTiny ensemble with 3-way softmax at 160px resolution, achieving 88.03% accuracy with 56.32% irregular F1. The V25 cascade replaces this entirely.
