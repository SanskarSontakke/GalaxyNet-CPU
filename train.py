from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import sys
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import (
    build_dataset,
    build_datasets,
    generate_labels_df,
)
from evaluate import evaluate_regression
from losses import RMSELoss, rmse_metric
from model import build_regression_model, freeze_base, unfreeze_top_layers
from utils import (
    GradientAccumulationModel,
    build_cosine_restart_schedule,
    is_kaggle_runtime,
    run_swa,
    save_json,
    set_global_seed,
    setup_environment,
    setup_kaggle_environment,
)


class ConciseLogging(tf.keras.callbacks.Callback):
    """Minimal epoch-end logging to keep Kaggle notebook output clean."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch + 1:03d} | "
            f"loss: {logs.get('loss', 0):.4f} | "
            f"rmse: {logs.get('rmse_metric', 0):.4f} | "
            f"val_loss: {logs.get('val_loss', 0):.4f} | "
            f"val_rmse: {logs.get('val_rmse_metric', 0):.4f}"
        )
        print(msg)


def train_unified_regression(config: Config, solutions_csv: Path, image_dir: Path):
    """Train single 37-node regression model using progressive resizing."""
    
    set_global_seed(config.seed)
    if config.enable_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    df, target_cols = generate_labels_df(solutions_csv, image_dir)
    print(f"Loaded {len(df)} images with {len(target_cols)} regression targets.")
    
    split_info, (train_df, val_df, test_df) = build_datasets(df, config)
    
    train_paths = train_df['image_path'].values
    train_labels = train_df[target_cols].values.astype(np.float32)
    val_paths = val_df['image_path'].values
    val_labels = val_df[target_cols].values.astype(np.float32)
    
    # 2. Build Model
    model, base_model = build_regression_model(architecture=config.architecture)
    
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print('\n' + '=' * 60)
    print(f'TRAINING REGRESSION MODEL ({config.architecture})')
    print('=' * 60)

    loss_fn = RMSELoss()

    # ── Phase 1: Warmup @ 128px ──
    print(f'\n[Phase 1] Warmup @ {config.image_size_phase1}px (frozen backbone)')
    freeze_base(base_model)

    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase1, config.batch_size_phase1,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase1, config.batch_size_phase1,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.warmup_lr, clipnorm=1.0),
        loss=loss_fn,
        metrics=[rmse_metric],
    )

    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.warmup_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'phase1.keras'),
                monitor='val_rmse_metric', save_best_only=True, mode='min',
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── Phase 2: Mid-tune @ 192px ──
    print(f'\n[Phase 2] Mid-tune @ {config.image_size_phase2}px (unfreeze {config.unfreeze_phase2} layers)')
    n_unfrozen = unfreeze_top_layers(base_model, config.unfreeze_phase2)
    print(f'  Unfrozen layers: {n_unfrozen}')

    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase2, config.batch_size_phase2,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase2, config.batch_size_phase2,
        center_crop_ratio=config.center_crop_ratio,
    )

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        config.midtune_lr, decay_steps=config.midtune_epochs * (len(train_paths) // config.batch_size_phase2)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
        loss=loss_fn, metrics=[rmse_metric],
    )

    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.midtune_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'phase2.keras'),
                monitor='val_rmse_metric', save_best_only=True, mode='min',
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── Phase 3: Full fine-tune @ 288px ──
    print(f'\n[Phase 3] Full fine-tune @ {config.image_size_phase3}px (unfreeze {config.unfreeze_phase3} layers)')
    n_unfrozen = unfreeze_top_layers(base_model, config.unfreeze_phase3)
    
    # Use Gradient Accumulation if batch size is small
    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase3, config.batch_size_phase3,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase3, config.batch_size_phase3,
        center_crop_ratio=config.center_crop_ratio,
    )

    steps_per_epoch = len(train_paths) // config.batch_size_phase3
    lr_schedule_p3 = build_cosine_restart_schedule(
        config.finetune_lr, steps_per_epoch, restart_epochs=5, t_mul=1.5, m_mul=0.9
    )
    
    active_model = model
    if config.grad_accumulation_steps > 1:
        active_model = GradientAccumulationModel(model, config.grad_accumulation_steps)

    active_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule_p3, clipnorm=1.0),
        loss=loss_fn, metrics=[rmse_metric],
    )

    h3 = active_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.finetune_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'unified_best.keras'),
                monitor='val_rmse_metric', save_best_only=True, mode='min',
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── SWA ──
    print('\n[SWA Phase]')
    train_ds_swa = build_dataset(
        train_paths, train_labels, config.image_size_phase3, config.batch_size_phase3,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=True,
    )
    model = run_swa(model, train_ds_swa, config.swa_epochs, config.swa_lr_high, config.swa_lr_low)
    model.save(str(output_dir / 'unified_best.keras'))

    # Save details
    save_json(split_info, output_dir / 'split_info.json')

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()
    
    return test_df, target_cols


def main():
    t0 = time.time()
    config = Config()

    if is_kaggle_runtime():
        print("Running in Kaggle environment.")
        solutions_csv, image_dir = setup_kaggle_environment(config)
    else:
        print("Running in local/VM environment.")
        setup_environment(config)
        solutions_csv = Path("training_solutions_rev1.csv")
        image_dir = Path("images_training_rev1")

    # Pipeline
    test_df, target_cols = train_unified_regression(config, solutions_csv, image_dir)
    
    print(f"\nTraining pipeline completed in {(time.time() - t0) / 60:.1f} minutes.")
    
    # Run evaluation
    evaluate_regression(config, test_df, target_cols)


if __name__ == '__main__':
    main()
