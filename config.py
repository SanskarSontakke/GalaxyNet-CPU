from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import tensorflow as tf


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────
    kaggle_input_dir: Path = Path('/kaggle/input/galaxy-zoo-the-galaxy-challenge')
    kaggle_temp_dir: Path = Path('/kaggle/temp')
    output_dir: Path = Path('/kaggle/working/outputs')

    # ── Image resolution curriculum ────────────────────────────
    image_size_phase1: int = 128
    image_size_phase2: int = 192
    image_size_phase3: int = 384  
    center_crop_ratio: float = 0.75

    # ── Batch sizes (BASE per device) ──────────────────────────
    batch_size_phase1: int = 16
    batch_size_phase2: int = 8
    batch_size_phase3: int = 4
    grad_accumulation_steps: int = 1 

    # ── Data split ─────────────────────────────────────────────
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.10

    # ── Model Architecture ─────────────────────────────────────
    architecture: str = 'EfficientNetV2B2'
    multi_view: bool = True  
    
    # ── Training Schedule (MINIMUM FOR TESTING) ────────────────
    warmup_epochs: int = 1   
    midtune_epochs: int = 1   
    finetune_epochs: int = 1  
    
    warmup_lr: float = 2e-4   
    midtune_lr: float = 2e-5
    finetune_lr: float = 5e-6
    
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
    swa_epochs: int = 1  # Minimum
    swa_lr_high: float = 1e-5
    swa_lr_low: float = 2e-6

    # ── TTA ────────────────────────────────────────────────────
    tta_n_augments: int = 2 # Minimum 

    # ── Runtime ────────────────────────────────────────────────
    enable_mixed_precision: bool = True

    def get_scaled_batch_size(self, base_size: int, strategy: tf.distribute.Strategy) -> int:
        return base_size * strategy.num_replicas_in_sync
