from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import tensorflow as tf

@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────
    kaggle_input_dir: Path = Path('/kaggle/input/galaxy-zoo-the-galaxy-challenge')
    kaggle_temp_dir: Path = Path('/kaggle/temp')
    output_dir: Path = Path('outputs')

    # ── Image resolution curriculum ────────────────────────────
    image_size_phase1: int = 424
    image_size_phase2: int = 424
    image_size_phase3: int = 424
    center_crop_ratio: float = 1.0

    # ── Batch sizes (BASE per device - Highly conservative for dual T4 16GB GPUs to avoid System OOM) ──
    batch_size_phase1: int = 4
    batch_size_phase2: int = 4
    batch_size_phase3: int = 4
    grad_accumulation_steps: int = 1  # Disabled for MirroredStrategy stability

    # ── Data split ─────────────────────────────────────────────
    seed: int = 42
    val_split: float = 0.10
    test_split: float = 0.00
    legacy_ordered_split: bool = True

    # ── Model Architecture ─────────────────────────────────────
    architecture: str = 'BenanneNetTF'
    multi_view: bool = False
    
    # ── Training Schedule (TEST RUN) ───────────────────────────
    warmup_epochs: int = 1
    midtune_epochs: int = 1
    finetune_epochs: int = 1
    
    warmup_lr: float = 4e-2
    midtune_lr: float = 4e-3
    finetune_lr: float = 4e-4
    
    unfreeze_phase2: int = 50
    unfreeze_phase3: int = 100 
    
    # ── Augmentation ───────────────────────────────────────────
    mixup_alpha: float = 0.4
    mixup_prob: float = 0.5
    cutout_n_holes: int = 2
    cutout_max_size_ratio: float = 0.20

    # ── Astronomy-Specific Augmentations (V26) ─────────────────
    augment_poisson_scale: float = 25.0   
    augment_poisson_prob: float = 0.3
    augment_blur_prob: float = 0.2
    augment_blur_sigma_range: Tuple[float, float] = (0.5, 2.0)

    # ── SWA ────────────────────────────────────────────────────
    swa_epochs: int = 0
    swa_lr_high: float = 5e-6
    swa_lr_low: float = 1e-6

    # ── TTA ────────────────────────────────────────────────────
    tta_n_augments: int = 60
    submission_filename: str = 'submission.csv.gz'

    # ── Runtime ────────────────────────────────────────────────
    enable_mixed_precision: bool = True

    def get_scaled_batch_size(self, base_size: int, strategy: tf.distribute.Strategy) -> int:
        return base_size * strategy.num_replicas_in_sync
