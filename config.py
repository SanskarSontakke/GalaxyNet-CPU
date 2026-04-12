from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────
    kaggle_input_dir: Path = Path('/kaggle/input/competitions/galaxy-zoo-the-galaxy-challenge')
    kaggle_temp_dir: Path = Path('/kaggle/temp')
    output_dir: Path = Path('/kaggle/working/outputs')

    # ── Image resolution curriculum ────────────────────────────
    image_size_phase1: int = 128
    image_size_phase2: int = 192
    image_size_phase3: int = 288   # Upgraded to 288px for final precision
    center_crop_ratio: float = 0.75

    # ── Batch sizes (per phase, tuned for P100 16GB) ───────────
    batch_size_phase1: int = 64    # 128px fits large batches
    batch_size_phase2: int = 32    # 192px
    batch_size_phase3: int = 16    # 224px (use grad accumulation if needed)
    grad_accumulation_steps: int = 2  # effective batch = 32 at phase3

    # ── Data split ─────────────────────────────────────────────
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15

    # ── Label thresholds (TIGHTENED from V24) ──────────────────
    elliptical_threshold: float = 0.469  # unchanged (high consensus)
    spiral_disk_threshold: float = 0.450  # tightened from 0.430
    spiral_arms_threshold: float = 0.450  # tightened from 0.430
    irregular_threshold: float = 0.500   # tightened from 0.469
    irregular_margin: float = 0.15       # tightened from 0.10

    # ── Soft Labels (V26) ──────────────────────────────────────
    use_soft_labels: bool = True    # Regress crowd-sourced voting fractions
    # When True, Stage 1 label = Class1.1 (P(elliptical)), 
    # Stage 2 label = normalized spiral confidence

    # ── Stage 1 training (elliptical binary) ───────────────────
    stage1_architecture: str = 'EfficientNetV2B1'
    stage1_warmup_epochs: int = 12
    stage1_midtune_epochs: int = 10
    stage1_finetune_epochs: int = 20
    stage1_warmup_lr: float = 3e-4
    stage1_midtune_lr: float = 3e-5
    stage1_finetune_lr: float = 8e-6
    stage1_unfreeze_phase2: int = 50  # unfreeze last 50 layers for phase 2
    stage1_unfreeze_phase3: int = 80  # unfreeze last 80 layers for phase 3

    # ── Stage 2 training (spiral vs irregular binary) ──────────
    stage2_architectures: Tuple[str, ...] = ('EfficientNetV2B2', 'EfficientNetV2B1', 'ConvNeXtTiny')
    stage2_seeds: Tuple[int, ...] = (42, 123, 77)
    stage2_warmup_epochs: int = 12
    stage2_midtune_epochs: int = 10
    stage2_finetune_epochs: int = 25
    stage2_warmup_lr: float = 3e-4
    stage2_midtune_lr: float = 3e-5
    stage2_finetune_lr: float = 8e-6
    stage2_unfreeze_phase2: int = 50
    stage2_unfreeze_phase3: int = 80
    # Stage 1 confidence threshold for filtering training data into Stage 2
    stage1_filter_confidence: float = 0.85

    # ── Focal loss ─────────────────────────────────────────────
    stage1_focal_gamma: float = 2.0   # Standard gamma for stable discrimination
    stage2_focal_gamma: float = 1.5   # moderate focus on hard examples
    label_smoothing: float = 0.05

    # ── OHEM (V26) ─────────────────────────────────────────────
    ohem_keep_ratio: float = 0.70     # Keep top 70% hardest examples
    ohem_start_phase: int = 2         # Only enable in Phase 2+ (not warmup)

    # ── Augmentation ───────────────────────────────────────────
    mixup_alpha: float = 0.4          # Beta distribution parameter
    mixup_prob_irregular: float = 0.5  # apply mixup to 50% of irregular batches
    cutout_n_holes_irregular: int = 3
    cutout_n_holes_other: int = 1
    cutout_max_size_ratio: float = 0.20

    # ── Astronomy-Specific Augmentations (V26) ─────────────────
    augment_poisson_scale: float = 25.0   # Poisson noise intensity (higher = less noise)
    augment_poisson_prob: float = 0.3     # Probability of applying Poisson noise
    augment_blur_prob: float = 0.2        # Probability of applying PSF blur
    augment_blur_sigma_range: Tuple[float, float] = (0.5, 2.0)  # Gaussian blur sigma range

    # ── SWA ────────────────────────────────────────────────────
    swa_epochs: int = 5
    swa_lr_high: float = 1e-5
    swa_lr_low: float = 5e-6

    # ── TTA ────────────────────────────────────────────────────
    tta_n_augments: int = 16

    # ── Runtime ────────────────────────────────────────────────
    enable_mixed_precision: bool = True
    use_sample_weighting: bool = True

    # ── XGBoost Stacking (V26) ─────────────────────────────────
    use_xgboost_stacking: bool = True
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05

    # ── Minimum class counts after threshold tightening ────────
    min_irregular_count: int = 3000
    min_spiral_count: int = 10000
    min_elliptical_count: int = 18000

    # ── Fallback params (triggered if irregular_f1 < 0.75) ────
    fallback_irregular_threshold: float = 0.469
    fallback_irregular_margin: float = 0.10
    fallback_stage2_focal_gamma: float = 3.0
    fallback_pos_weight_multiplier: float = 1.5
