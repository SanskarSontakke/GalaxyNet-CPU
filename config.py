from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    kaggle_input_dir: Path = Path('/kaggle/input/competitions/galaxy-zoo-the-galaxy-challenge')
    kaggle_temp_dir: Path = Path('/kaggle/temp')
    output_dir: Path = Path('/kaggle/working/outputs')

    # Data
    image_size_warmup: int = 128
    image_size_finetune: int = 160
    # Current active size (updated at runtime by curriculum)
    image_size: int = 128
    
    batch_size: int = 64
    batch_size_finetune: int = 32 # Safety reduction for 160px
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15

    # Label curation & Thresholds
    elliptical_threshold: float = 0.469
    spiral_disk_threshold: float = 0.430
    spiral_arms_threshold: float = 0.430
    irregular_threshold: float = 0.469
    # New: minimum probability margin for irregular odd features
    irregular_margin: float = 0.10

    # Training phase 1 (Warmup)
    warmup_epochs: int = 10
    warmup_lr: float = 1e-3

    # Training phase 2 (Fine-tune)
    finetune_epochs: int = 30
    finetune_lr: float = 5e-5
    finetune_unfreeze_last_n: int = 30
    
    # Advanced Regimes
    label_smoothing: float = 0.05
    use_sample_weighting: bool = True
    use_oversampling: bool = False  # Replaced by weighting
    ensemble_architectures: tuple[str, ...] = ('EfficientNetV2B0', 'ConvNeXtTiny')

    # Focal Loss
    focal_gamma: float = 2.0

    # TTA - Only orientation-preserving transforms
    tta_n_augments: int = 8

    # Cross validation
    cv_folds: int = 5
    cv_finetune_epochs: int = 5

    # Runtime flags
    enable_mixed_precision: bool = True
    cache_val_test: bool = True
