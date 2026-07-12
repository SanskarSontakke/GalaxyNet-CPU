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

    # ── Batch sizes (BASE per device) ──
    # BenanneNetTF is a small from-scratch convnet that immediately downsamples
    # each image to 69x69 views, so its memory footprint is tiny and a batch of
    # 4 leaves the accelerator badly underused with very noisy gradients. 32 is
    # comfortable here. Lower these back toward 4-8 if you switch `architecture`
    # to a heavy ImageNet backbone (EfficientNet/ConvNeXt) at full resolution.
    batch_size_phase1: int = 32
    batch_size_phase2: int = 32
    batch_size_phase3: int = 32

    # ── Data split ─────────────────────────────────────────────
    seed: int = 42
    val_split: float = 0.10
    test_split: float = 0.00
    legacy_ordered_split: bool = True

    # ── Model Architecture ─────────────────────────────────────
    architecture: str = 'BenanneNetTF'
    multi_view: bool = False
    
    # ── Training Schedule ──────────────────────────────────────
    # These are epoch CAPS, not fixed counts: EarlyStopping(restore_best_weights)
    # in train.py stops each phase once validation RMSE stops improving, so
    # setting them generously is safe. The three phases form a manual step-decay
    # of the SGD learning rate (4e-2 -> 4e-3 -> 4e-4), each starting from the
    # best weights of the previous phase.
    #
    # NOTE: the original benanne solution trained for ~67 GPU-hours. A single
    # Kaggle session is capped at 12h with no resume logic here, so treat these
    # as the main dial to trade run time against accuracy and watch the clock.
    warmup_epochs: int = 25
    midtune_epochs: int = 15
    finetune_epochs: int = 15

    # EarlyStopping patience (epochs without val-RMSE improvement) per phase.
    early_stop_patience: int = 6

    warmup_lr: float = 4e-2
    midtune_lr: float = 4e-3
    finetune_lr: float = 4e-4
    
    unfreeze_phase2: int = 50
    unfreeze_phase3: int = 100 
    
    # ── Augmentation ───────────────────────────────────────────
    # Affine transform + colour perturbation always run during training. The
    # switch below adds the optional sensor-noise / seeing / occlusion style
    # augmentations (Poisson noise, Gaussian blur, cutout), each gated by its
    # own probability. Set to False to train with the plain benanne pipeline.
    enable_extra_augment: bool = True

    cutout_prob: float = 0.3
    cutout_n_holes: int = 2
    cutout_max_size_ratio: float = 0.20

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
