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
    image_size_phase3: int = 288
    center_crop_ratio: float = 0.75

    # ── Batch sizes (per phase, tuned for P100 16GB) ───────────
    batch_size_phase1: int = 64
    batch_size_phase2: int = 32
    batch_size_phase3: int = 16
    grad_accumulation_steps: int = 2  # effective batch = 32 at phase3

    # ── Data split ─────────────────────────────────────────────
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15

    # ── Model Architecture ─────────────────────────────────────
    architecture: str = 'EfficientNetV2B2'
    
    # ── Training Schedule ──────────────────────────────────────
    warmup_epochs: int = 12
    midtune_epochs: int = 10
    finetune_epochs: int = 25
    
    warmup_lr: float = 3e-4
    midtune_lr: float = 3e-5
    finetune_lr: float = 8e-6
    
    unfreeze_phase2: int = 50
    unfreeze_phase3: int = 80

    # ── Augmentation ───────────────────────────────────────────
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    cutout_n_holes: int = 2
    cutout_max_size_ratio: float = 0.20

    # ── Astronomy-Specific Augmentations (V26) ─────────────────
    augment_poisson_scale: float = 25.0   # higher = less noise
    augment_poisson_prob: float = 0.3
    augment_blur_prob: float = 0.2
    augment_blur_sigma_range: Tuple[float, float] = (0.5, 2.0)

    # ── SWA ────────────────────────────────────────────────────
    swa_epochs: int = 5
    swa_lr_high: float = 1e-5
    swa_lr_low: float = 5e-6

    # ── TTA ────────────────────────────────────────────────────
    tta_n_augments: int = 16

    # ── Runtime ────────────────────────────────────────────────
    enable_mixed_precision: bool = True
