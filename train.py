from __future__ import annotations

import gc
import os
import time
import warnings
from pathlib import Path

# Suppress noisy TF/Keras/JAX logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_LEVEL'] = 'error'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress oneDNN notice

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*layout failed: INVALID_ARGUMENT.*')

import numpy as np
import tensorflow as tf

tf.keras.utils.disable_interactive_logging()

from config import Config
from dataset import (
    build_dataset,
    build_datasets,
    generate_labels_df,
)
from evaluate import evaluate_regression
from losses import RMSELoss, rmse_metric
from model import (
    build_regression_model,
    freeze_base,
    unfreeze_top_layers,
)
from utils import (
    build_cosine_restart_schedule,
    is_kaggle_runtime,
    run_swa,
    save_json,
    setup_environment,
    setup_kaggle_environment,
    setup_local_environment,
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
        
        print(f"Epoch {epoch+1:02d} | Loss: {logs.get('loss', 0):.4f} | "
              f"Val RMSE: {logs.get('val_rmse_metric', 0):.4f} | LR: {lr_val:.1e}")


def train_unified_regression(config: Config, solutions_csv: Path, image_dir: Path):
    """Unified progressive resizing training loop for 37-node regression."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Strategy Initialization ──
    strategy = get_strategy()
    
    # ── Dataset Preparation ──
    df, target_cols = generate_labels_df(solutions_csv, image_dir)
    split_info, (train_df, val_df, test_df) = build_datasets(df, config)

    train_paths = train_df['image_path'].values
    train_labels = train_df[target_cols].values.astype(np.float32)
    val_paths = val_df['image_path'].values
    val_labels = val_df[target_cols].values.astype(np.float32)
    uses_benanne_schedule = config.architecture == 'BenanneNetTF'

    # Scale batch sizes by number of TPU replicas
    bs1 = config.get_scaled_batch_size(config.batch_size_phase1, strategy)
    bs2 = config.get_scaled_batch_size(config.batch_size_phase2, strategy)
    bs3 = config.get_scaled_batch_size(config.batch_size_phase3, strategy)
    

    # ── Model & Strategy Scope ──
    with strategy.scope():
        model, base_model = build_regression_model(config.architecture, multi_view=config.multi_view)
        loss_fn = RMSELoss()

    def compile_for_phase(phase_lr: float, steps_per_epoch: int | None = None):
        with strategy.scope():
            if uses_benanne_schedule:
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=phase_lr,
                    momentum=0.9,
                    nesterov=True,
                )
            elif steps_per_epoch is None:
                optimizer = tf.keras.optimizers.Adam(phase_lr)
            else:
                optimizer = tf.keras.optimizers.Adam(
                    build_cosine_restart_schedule(
                        phase_lr,
                        steps_per_epoch,
                        restart_epochs=5,
                    ),
                    clipnorm=1.0,
                )
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=[rmse_metric])
            return model

    # ── Callbacks shared across phases ──
    # We keep a single best checkpoint on disk across all three phases. The
    # running best validation RMSE is threaded into each phase's checkpoint via
    # `initial_value_threshold`, so a poor early epoch of a later phase can never
    # overwrite a better model saved by an earlier phase. EarlyStopping restores
    # each phase's best weights before the next phase continues from them.
    checkpoint_path = str(output_dir / 'unified_best.keras')
    best_val_rmse = float('inf')

    def phase_callbacks():
        return [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_rmse_metric', save_best_only=True, mode='min',
                initial_value_threshold=best_val_rmse, verbose=0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_rmse_metric', mode='min',
                patience=config.early_stop_patience,
                restore_best_weights=True, verbose=0,
            ),
            ConciseLogging(),
        ]

    def update_best(history):
        nonlocal best_val_rmse
        vals = history.history.get('val_rmse_metric', [])
        if vals:
            best_val_rmse = min(best_val_rmse, min(vals))

    # ── Phase 1: Warmup (Fixed layers, 128px) ──
    print(f'\n[Phase 1] Warmup (Frozen Backbone, {config.image_size_phase1}px)')
    freeze_base(base_model)
    
    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase1, bs1,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
        drop_remainder=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase1, bs1,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False,
    )

    train_model = compile_for_phase(config.warmup_lr)

    history = train_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.warmup_epochs,
        callbacks=phase_callbacks(),
        verbose=0,
    )
    update_best(history)

    # ── Phase 2: Mid-tune (Partial unfreeze, 192px) ──
    print(f'\n[Phase 2] Mid-tune (Unfreeze {config.unfreeze_phase2} layers, {config.image_size_phase2}px)')
    unfreeze_top_layers(base_model, config.unfreeze_phase2)

    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase2, bs2,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
        drop_remainder=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase2, bs2,
        center_crop_ratio=config.center_crop_ratio,
    )

    steps_per_epoch = len(train_paths) // bs2
    train_model = compile_for_phase(
        config.midtune_lr,
        None if uses_benanne_schedule else steps_per_epoch,
    )

    history = train_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.midtune_epochs,
        callbacks=phase_callbacks(),
        verbose=0,
    )
    update_best(history)

    # ── Phase 3: Fine-tune (More unfreeze, 384px) ──
    print(f'\n[Phase 3] Fine-tune (Unfreeze {config.unfreeze_phase3} layers, {config.image_size_phase3}px)')
    unfreeze_top_layers(base_model, config.unfreeze_phase3)

    train_ds = build_dataset(
        train_paths, train_labels, config.image_size_phase3, bs3,
        center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=True,
        drop_remainder=True,
    )
    val_ds = build_dataset(
        val_paths, val_labels, config.image_size_phase3, bs3,
        center_crop_ratio=config.center_crop_ratio,
    )

    steps_per_epoch = len(train_paths) // bs3
    train_model = compile_for_phase(
        config.finetune_lr,
        None if uses_benanne_schedule else steps_per_epoch,
    )

    history = train_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.finetune_epochs,
        callbacks=phase_callbacks(),
        verbose=0,
    )
    update_best(history)

    # ── SWA ──
    if config.swa_epochs > 0 and not uses_benanne_schedule:
        print('\n[SWA Phase]')
        train_ds_swa = build_dataset(
            train_paths, train_labels, config.image_size_phase3, bs3,
            center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=True,
            drop_remainder=True,
        )
        with strategy.scope():
            model = run_swa(model, train_ds_swa, config.swa_epochs, config.swa_lr_high, config.swa_lr_low)
            model.save(str(output_dir / 'unified_best.keras'))

    # Save details
    save_json(output_dir / 'split_info.json', split_info)

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()
    
    eval_df = test_df if len(test_df) > 0 else val_df
    eval_name = 'test' if len(test_df) > 0 else 'validation'
    return eval_df, target_cols, eval_name


def main():
    t0 = time.time()
    config = Config()

    setup_environment(config)
    
    if is_kaggle_runtime():
        solutions_csv, image_dir, test_image_dir, submission_template_path = setup_kaggle_environment(config)
    else:
        solutions_csv, image_dir, test_image_dir, submission_template_path = setup_local_environment(config)

    # Run pipeline
    eval_df, target_cols, eval_name = train_unified_regression(config, solutions_csv, image_dir)

    print(f"\nTraining pipeline completed in {(time.time() - t0) / 60:.1f} minutes.")

    evaluate_regression(config, eval_df, target_cols, split_name=eval_name)

    if test_image_dir.exists():
        from evaluate import generate_competition_submission

        generate_competition_submission(
            config=config,
            test_image_dir=test_image_dir,
            target_cols=target_cols,
            submission_template_path=submission_template_path,
        )


if __name__ == '__main__':
    main()
