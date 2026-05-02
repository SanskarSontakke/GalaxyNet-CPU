from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from dataset import (
    build_dataset,
    build_datasets,
    generate_labels_df,
)
from evaluate import evaluate_regression
from losses import HierarchicalRMSELoss, RMSELoss, rmse_metric
from model import (
    build_regression_model,
    freeze_base,
    unfreeze_top_layers,
)
from utils import (
    build_cosine_restart_schedule,
    GradientAccumulationModel,
    is_kaggle_runtime,
    run_swa,
    save_json,
    setup_environment,
    setup_kaggle_environment,
    get_strategy,
)


class ConciseLogging(tf.keras.callbacks.Callback):
    """Minimal logging to prevent log flooding during long regression runs."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr_val = float(lr(self.model.optimizer.iterations))
        else:
            # Handle standard variables or distributed MirroredVariables
            lr_val = float(tf.convert_to_tensor(lr))
        
        print(f"Epoch {epoch+1:02d} | Loss: {logs.get('loss', 0):.5f} | "
              f"Val RMSE: {logs.get('val_rmse_metric', 0):.5f} | LR: {lr_val:.2e}")


def train_unified_regression(config: Config, solutions_csv: Path, image_dir: Path):
    """Unified progressive resizing training loop for 37-node regression on TPU."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Strategy Initialization ──
    strategy = get_strategy()
    
    # ── Dataset Preparation ──
    df, target_cols = generate_labels_df(solutions_csv, image_dir)
    split_info, (train_df, val_df, test_df) = build_datasets(df, config)

    train_paths = train_df['image_path'].values[:100]
    train_labels = train_df[target_cols].values[:100].astype(np.float32)
    val_paths = val_df['image_path'].values[:20]
    val_labels = val_df[target_cols].values[:20].astype(np.float32)

    # Scale batch sizes by number of TPU replicas
    bs1 = config.get_scaled_batch_size(config.batch_size_phase1, strategy)
    bs2 = config.get_scaled_batch_size(config.batch_size_phase2, strategy)
    bs3 = config.get_scaled_batch_size(config.batch_size_phase3, strategy)
    
    print(f"Global Batch Sizes (all cores): P1={bs1}, P2={bs2}, P3={bs3}")

    # ── Model & Strategy Scope ──
    with strategy.scope():
        model, base_model = build_regression_model(config.architecture, multi_view=config.multi_view)
        loss_fn = HierarchicalRMSELoss()

    # ── Phase 1: Warmup (Fixed layers, 128px) ──
    print('\n[Phase 1] Warmup (Frozen Backbone, 128px)')
    freeze_base(base_model)
    
    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase1, bs1,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase1, bs1,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False,
    )

    with strategy.scope():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.warmup_lr),
            loss=loss_fn, metrics=[rmse_metric],
        )
        train_model = model

    train_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.warmup_epochs,
        callbacks=[ConciseLogging()],
        verbose=0,
    )

    # ── Phase 2: Mid-tune (Partial unfreeze, 192px) ──
    print(f'\n[Phase 2] Mid-tune (Unfreeze {config.unfreeze_phase2} layers, 192px)')
    unfreeze_top_layers(base_model, config.unfreeze_phase2)

    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase2, bs2,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase2, bs2,
        center_crop_ratio=config.center_crop_ratio,
    )

    with strategy.scope():
        steps_per_epoch = len(train_paths) // bs2
        lr_schedule = build_cosine_restart_schedule(
            config.midtune_lr, steps_per_epoch, restart_epochs=5
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
            loss=loss_fn, metrics=[rmse_metric],
        )
        train_model = model

    train_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.midtune_epochs,
        callbacks=[ConciseLogging()],
        verbose=0,
    )

    # ── Phase 3: Fine-tune (More unfreeze, 384px) ──
    print(f'\n[Phase 3] Fine-tune (Unfreeze {config.unfreeze_phase3} layers, {config.image_size_phase3}px)')
    unfreeze_top_layers(base_model, config.unfreeze_phase3)

    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase3, bs3,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase3, bs3,
        center_crop_ratio=config.center_crop_ratio,
    )

    with strategy.scope():
        steps_per_epoch = len(train_paths) // bs3
        lr_schedule_p3 = build_cosine_restart_schedule(
            config.finetune_lr, steps_per_epoch, restart_epochs=5
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_schedule_p3, clipnorm=1.0),
            loss=loss_fn, metrics=[rmse_metric],
        )
        train_model = model

    train_model.fit(
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
        train_paths, train_labels, config.image_size_phase3, bs3,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=True,
    )
    model = run_swa(model, train_ds_swa, config.swa_epochs, config.swa_lr_high, config.swa_lr_low)
    model.save(str(output_dir / 'unified_best.keras'))

    # Save details
    save_json(output_dir / 'split_info.json', split_info)

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()
    
    return test_df, target_cols


def main():
    t0 = time.time()
    config = Config()

    setup_environment(config)
    
    if is_kaggle_runtime():
        solutions_csv, image_dir = setup_kaggle_environment(config)
    else:
        solutions_csv = Path("galaxy_raw/training_solutions_rev1.csv")
        image_dir = Path("galaxy_raw/training_images/images_training_rev1")

    # Run pipeline
    test_df, target_cols = train_unified_regression(config, solutions_csv, image_dir)

    print(f"\nTraining pipeline completed in {(time.time() - t0) / 60:.1f} minutes.")

    # Run evaluation
    evaluate_regression(config, test_df, target_cols)


if __name__ == '__main__':
    main()
