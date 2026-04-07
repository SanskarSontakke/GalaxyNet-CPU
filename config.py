from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    kaggle_input_dir: Path = Path('/kaggle/input/galaxy-zoo-the-galaxy-challenge')
    kaggle_temp_dir: Path = Path('/kaggle/temp')
    output_dir: Path = Path('/kaggle/working/outputs')

    # Local override paths
    local_image_dir: Path | None = None
    local_solutions_csv: Path | None = None
    local_labels_csv: Path | None = None

    # Data
    image_size: int = 128
    batch_size: int = 32
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15

    # Label thresholds
    elliptical_threshold: float = 0.469
    spiral_disk_threshold: float = 0.430
    spiral_arms_threshold: float = 0.430
    irregular_threshold: float = 0.469

    # Training phase 1
    warmup_epochs: int = 10
    warmup_lr: float = 1e-3

    # Training phase 2
    finetune_epochs: int = 50
    finetune_lr: float = 1e-5
    finetune_unfreeze_last_n: int = 30

    # Focal Loss
    focal_gamma: float = 2.0

    # TTA
    tta_n_augments: int = 16

    # Cross validation
    cv_folds: int = 5
    cv_finetune_epochs: int = 5

    # Runtime flags
    enable_mixed_precision: bool = True
    cache_val_test: bool = True
