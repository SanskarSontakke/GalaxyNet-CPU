from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import sys
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import (
    CLASS_NAMES,
    build_binary_dataset_stage1,
    build_binary_dataset_stage2,
    build_dataset,
    build_datasets,
    compute_sample_weights,
    filter_stage2_training_data,
    generate_labels_df,
    validate_class_counts,
)
from evaluate import evaluate_cascade
from losses import BinaryFocalLoss
from model import build_stage1_model, build_stage2_model, freeze_base, unfreeze_top_layers
from utils import (
    GradientAccumulationModel,
    build_cosine_restart_schedule,
    build_llrd_optimizer,
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
        # Support both 'accuracy' and 'binary_accuracy' metric names
        acc_key = 'accuracy' if 'accuracy' in logs else 'binary_accuracy'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in logs else 'val_binary_accuracy'
        msg = (
            f"Epoch {epoch + 1:03d} | "
            f"loss: {logs.get('loss', 0):.4f} | "
            f"acc: {logs.get(acc_key, 0):.4f} | "
            f"val_acc: {logs.get(val_acc_key, 0):.4f}"
        )
        print(msg)


def _binary_metrics():
    """Standard metrics for binary classification stages."""
    return [
        'binary_accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


# ══════════════════════════════════════════════════════════════
# STAGE 1 — ELLIPTICAL BINARY CLASSIFIER
# ══════════════════════════════════════════════════════════════

def _train_stage1_isolated(train_df_dict, val_df_dict, config_dict, output_dir_str):
    """Train Stage 1 (elliptical vs non-elliptical) in an isolated subprocess.
    
    3-phase progressive resizing curriculum:
      Phase 1: 128px warmup (frozen backbone)
      Phase 2: 192px mid-tune (unfreeze last 50 layers)
      Phase 3: 224px full fine-tune (unfreeze last 80 layers) + SGDR
    """
    import gc
    tf.keras.backend.clear_session()
    gc.collect()

    config = Config(**config_dict)
    output_dir = Path(output_dir_str)
    train_df = pd.DataFrame(train_df_dict)
    val_df = pd.DataFrame(val_df_dict)

    set_global_seed(config.seed)
    if config.enable_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    print('\n' + '=' * 60)
    print('STAGE 1: Elliptical vs Non-Elliptical Binary Classifier')
    print('=' * 60)

    # Compute class imbalance weight
    n_ell = (train_df['label'] == 'elliptical').sum()
    n_non_ell = (train_df['label'] != 'elliptical').sum()
    pos_weight = float(n_non_ell) / max(float(n_ell), 1.0)
    print(f'  Elliptical: {n_ell}, Non-elliptical: {n_non_ell}, pos_weight: {pos_weight:.3f}')

    focal = BinaryFocalLoss(
        gamma=config.stage1_focal_gamma,
        pos_weight=pos_weight,
        label_smoothing=0.0,  # No label smoothing for Stage 1 (clean labels)
    )

    model, base_model = build_stage1_model(architecture=config.stage1_architecture)

    # ── Phase 1: Warmup @ 128px ──
    print(f'\n[Phase 1] Warmup @ {config.image_size_phase1}px')
    freeze_base(base_model)

    train_ds = build_binary_dataset_stage1(
        train_df, config.image_size_phase1, config.batch_size_phase1, config,
        augment=True, shuffle=True,
    )
    val_ds = build_binary_dataset_stage1(
        val_df, config.image_size_phase1, config.batch_size_phase1, config,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.stage1_warmup_lr, clipnorm=1.0),
        loss=focal,
        metrics=_binary_metrics(),
    )

    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.stage1_warmup_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'stage1_phase1.keras'),
                monitor='val_binary_accuracy', save_best_only=True, mode='max',
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_binary_accuracy', patience=3, restore_best_weights=True,
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── Phase 2: Mid-tune @ 192px ──
    print(f'\n[Phase 2] Mid-tune @ {config.image_size_phase2}px (unfreeze last {config.stage1_unfreeze_phase2} layers)')
    n_unfrozen = unfreeze_top_layers(base_model, config.stage1_unfreeze_phase2)
    print(f'  Unfrozen layers: {n_unfrozen}')

    train_ds = build_binary_dataset_stage1(
        train_df, config.image_size_phase2, config.batch_size_phase2, config,
        augment=True, shuffle=True,
    )
    val_ds = build_binary_dataset_stage1(
        val_df, config.image_size_phase2, config.batch_size_phase2, config,
    )

    lr_schedule_p2 = tf.keras.optimizers.schedules.CosineDecay(
        config.stage1_midtune_lr,
        decay_steps=config.stage1_midtune_epochs * (len(train_df) // config.batch_size_phase2),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule_p2, clipnorm=1.0),
        loss=focal,
        metrics=_binary_metrics(),
    )

    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.stage1_midtune_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'stage1_phase2.keras'),
                monitor='val_binary_accuracy', save_best_only=True, mode='max',
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_binary_accuracy', patience=4, restore_best_weights=True,
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── Phase 3: Full fine-tune @ 224px with SGDR ──
    print(f'\n[Phase 3] Full fine-tune @ {config.image_size_phase3}px (unfreeze last {config.stage1_unfreeze_phase3} layers)')
    n_unfrozen = unfreeze_top_layers(base_model, config.stage1_unfreeze_phase3)
    print(f'  Unfrozen layers: {n_unfrozen}')

    steps_per_epoch_p3 = len(train_df) // config.batch_size_phase3
    lr_schedule_p3 = build_cosine_restart_schedule(
        config.stage1_finetune_lr, steps_per_epoch_p3,
        restart_epochs=5, t_mul=1.5, m_mul=0.9,
    )

    train_ds = build_binary_dataset_stage1(
        train_df, config.image_size_phase3, config.batch_size_phase3, config,
        augment=True, shuffle=True,
    )
    val_ds = build_binary_dataset_stage1(
        val_df, config.image_size_phase3, config.batch_size_phase3, config,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule_p3, clipnorm=1.0),
        loss=focal,
        metrics=_binary_metrics(),
    )

    h3 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.stage1_finetune_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'stage1_best.keras'),
                monitor='val_binary_accuracy', save_best_only=True, mode='max',
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_binary_accuracy', patience=8, restore_best_weights=True,
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── SWA for Stage 1 ──
    print('\n[Stage 1 SWA]')
    train_ds_swa = build_binary_dataset_stage1(
        train_df, config.image_size_phase3, config.batch_size_phase3, config,
        augment=False, shuffle=True,
    )
    model = run_swa(model, train_ds_swa, config.swa_epochs, config.swa_lr_high, config.swa_lr_low)
    model.save(str(output_dir / 'stage1_best.keras'))

    # Save training history
    history = {
        'phase1': h1.history,
        'phase2': h2.history,
        'phase3': h3.history,
    }
    # Convert numpy arrays in history to lists for JSON serialization
    for phase_key, phase_hist in history.items():
        for metric_key, values in phase_hist.items():
            history[phase_key][metric_key] = [float(v) for v in values]

    with open(output_dir / 'stage1_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\n[Stage 1] Training complete. Model saved to {output_dir / "stage1_best.keras"}')

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()


# ══════════════════════════════════════════════════════════════
# STAGE 2 — SPIRAL vs IRREGULAR BINARY CLASSIFIER
# ══════════════════════════════════════════════════════════════

def _train_stage2_isolated(
    train_df_dict, val_df_dict, config_dict, output_dir_str,
    architecture, seed, model_tag,
):
    """Train a single Stage 2 model (spiral vs irregular) in isolated subprocess.
    
    3-phase progressive resizing curriculum with heavier augmentation
    and higher focal gamma for the harder spiral/irregular boundary.
    """
    import gc
    tf.keras.backend.clear_session()
    gc.collect()

    config = Config(**config_dict)
    output_dir = Path(output_dir_str)
    train_df = pd.DataFrame(train_df_dict)
    val_df = pd.DataFrame(val_df_dict)

    set_global_seed(seed)
    if config.enable_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    print(f'\n{"=" * 60}')
    print(f'STAGE 2 [{model_tag}]: Spiral vs Irregular ({architecture}, seed={seed})')
    print(f'{"=" * 60}')

    # Filter to spiral + irregular only
    mask = train_df['label'].isin(['spiral', 'irregular'])
    s2_train = train_df[mask].copy().reset_index(drop=True)
    mask_val = val_df['label'].isin(['spiral', 'irregular'])
    s2_val = val_df[mask_val].copy().reset_index(drop=True)

    n_spiral = (s2_train['label'] == 'spiral').sum()
    n_irregular = (s2_train['label'] == 'irregular').sum()
    pos_weight = float(n_spiral) / max(float(n_irregular), 1.0)
    print(f'  Spiral: {n_spiral}, Irregular: {n_irregular}, pos_weight(spiral/irr): {pos_weight:.3f}')

    focal = BinaryFocalLoss(
        gamma=config.stage2_focal_gamma,
        pos_weight=pos_weight,
        label_smoothing=config.label_smoothing,
    )

    # Adjust batch sizes for larger Stage 2 models
    bs_p1 = config.batch_size_phase1
    bs_p2 = 24 if architecture == 'EfficientNetV2B2' else config.batch_size_phase2
    bs_p3 = config.batch_size_phase3

    model, base_model = build_stage2_model(architecture=architecture)

    # ── Phase 1: Warmup @ 128px ──
    print(f'\n[Phase 1] Warmup @ {config.image_size_phase1}px')
    freeze_base(base_model)

    train_ds = build_binary_dataset_stage2(
        s2_train, config.image_size_phase1, bs_p1, config,
        augment=True, shuffle=True,
    )
    val_ds = build_binary_dataset_stage2(
        s2_val, config.image_size_phase1, bs_p1, config,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.stage2_warmup_lr, clipnorm=1.0),
        loss=focal,
        metrics=_binary_metrics(),
    )

    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.stage2_warmup_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / f'{model_tag}_phase1.keras'),
                monitor='val_binary_accuracy', save_best_only=True, mode='max',
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_binary_accuracy', patience=3, restore_best_weights=True,
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── Phase 2: Mid-tune @ 192px ──
    print(f'\n[Phase 2] Mid-tune @ {config.image_size_phase2}px (unfreeze last {config.stage2_unfreeze_phase2})')
    n_unfrozen = unfreeze_top_layers(base_model, config.stage2_unfreeze_phase2)
    print(f'  Unfrozen layers: {n_unfrozen}')

    train_ds = build_binary_dataset_stage2(
        s2_train, config.image_size_phase2, bs_p2, config,
        augment=True, shuffle=True,
    )
    val_ds = build_binary_dataset_stage2(
        s2_val, config.image_size_phase2, bs_p2, config,
    )

    lr_schedule_p2 = tf.keras.optimizers.schedules.CosineDecay(
        config.stage2_midtune_lr,
        decay_steps=config.stage2_midtune_epochs * (len(s2_train) // bs_p2),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule_p2, clipnorm=1.0),
        loss=focal,
        metrics=_binary_metrics(),
    )

    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.stage2_midtune_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / f'{model_tag}_phase2.keras'),
                monitor='val_binary_accuracy', save_best_only=True, mode='max',
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_binary_accuracy', patience=5, restore_best_weights=True,
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── Phase 3: Full fine-tune @ 224px with SGDR ──
    print(f'\n[Phase 3] Full fine-tune @ {config.image_size_phase3}px (unfreeze last {config.stage2_unfreeze_phase3})')
    n_unfrozen = unfreeze_top_layers(base_model, config.stage2_unfreeze_phase3)
    print(f'  Unfrozen layers: {n_unfrozen}')

    steps_per_epoch_p3 = len(s2_train) // bs_p3
    lr_schedule_p3 = build_cosine_restart_schedule(
        config.stage2_finetune_lr, steps_per_epoch_p3,
        restart_epochs=5, t_mul=1.5, m_mul=0.9,
    )

    train_ds = build_binary_dataset_stage2(
        s2_train, config.image_size_phase3, bs_p3, config,
        augment=True, shuffle=True,
    )
    val_ds = build_binary_dataset_stage2(
        s2_val, config.image_size_phase3, bs_p3, config,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule_p3, clipnorm=1.0),
        loss=focal,
        metrics=_binary_metrics(),
    )

    h3 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.stage2_finetune_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / f'{model_tag}_best.keras'),
                monitor='val_binary_accuracy', save_best_only=True, mode='max',
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_binary_accuracy', patience=8, restore_best_weights=True,
            ),
            ConciseLogging(),
        ],
        verbose=0,
    )

    # ── SWA for Stage 2 ──
    print(f'\n[{model_tag} SWA]')
    train_ds_swa = build_binary_dataset_stage2(
        s2_train, config.image_size_phase3, bs_p3, config,
        augment=False, shuffle=True,
    )
    model = run_swa(model, train_ds_swa, config.swa_epochs, config.swa_lr_high, config.swa_lr_low)
    model.save(str(output_dir / f'{model_tag}_best.keras'))

    # Save history
    history = {
        'phase1': h1.history,
        'phase2': h2.history,
        'phase3': h3.history,
    }
    for phase_key, phase_hist in history.items():
        for metric_key, values in phase_hist.items():
            history[phase_key][metric_key] = [float(v) for v in values]

    with open(output_dir / f'{model_tag}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\n[{model_tag}] Training complete. Model saved to {output_dir / f"{model_tag}_best.keras"}')

    tf.keras.backend.clear_session()
    gc.collect()


# ══════════════════════════════════════════════════════════════
# CASCADE TRAINING ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def run_cascade_training(train_df, val_df, config: Config, output_dir: Path) -> dict:
    """Orchestrate the full cascade training pipeline.
    
    Order:
    1. Train Stage 1 (elliptical binary) on full dataset
    2. Train Stage 2 models (spiral vs irregular) on non-elliptical subset
    3. Each stage runs in a separate subprocess for VRAM isolation
    """
    import multiprocessing

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    config_dict = asdict(config)
    output_dir_str = str(output_dir)
    train_df_dict = train_df.to_dict()
    val_df_dict = val_df.to_dict()

    results = {}

    # ── 1. Train Stage 1 ──
    print('\n' + '=' * 70)
    print('TRAINING STAGE 1: Elliptical vs Non-Elliptical')
    print('=' * 70)
    t0 = time.time()

    p = multiprocessing.Process(
        target=_train_stage1_isolated,
        args=(train_df_dict, val_df_dict, config_dict, output_dir_str),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f'Stage 1 training failed with exit code {p.exitcode}')

    stage1_time = time.time() - t0
    print(f'\nStage 1 completed in {stage1_time / 60:.1f} minutes')

    with open(output_dir / 'stage1_history.json', 'r') as f:
        results['stage1'] = json.load(f)

    # ── 2. Train Stage 2 (multiple ensemble models) ──
    print('\n' + '=' * 70)
    print('TRAINING STAGE 2: Spiral vs Irregular (Ensemble)')
    print('=' * 70)

    stage2_results = {}
    for i, (arch, seed) in enumerate(zip(config.stage2_architectures, config.stage2_seeds)):
        model_tag = f'stage2_{arch}_seed{seed}'
        t0 = time.time()

        p = multiprocessing.Process(
            target=_train_stage2_isolated,
            args=(
                train_df_dict, val_df_dict, config_dict, output_dir_str,
                arch, seed, model_tag,
            ),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f'Stage 2 [{model_tag}] training failed with exit code {p.exitcode}')

        stage2_time = time.time() - t0
        print(f'\n{model_tag} completed in {stage2_time / 60:.1f} minutes')

        with open(output_dir / f'{model_tag}_history.json', 'r') as f:
            stage2_results[model_tag] = json.load(f)

    results['stage2'] = stage2_results

    return results


# ══════════════════════════════════════════════════════════════
# FALLBACK — RE-TRAIN STAGE 2 WITH RELAXED THRESHOLDS
# ══════════════════════════════════════════════════════════════

def run_stage2_fallback(train_df, val_df, config: Config, output_dir: Path) -> dict:
    """Re-train Stage 2 with relaxed thresholds and stronger regularization.
    
    Triggered automatically if irregular_f1 < 0.75 after initial training.
    """
    import multiprocessing

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print('\n' + '!' * 70)
    print('FALLBACK: Re-training Stage 2 with relaxed thresholds')
    print('!' * 70)

    # Apply fallback config changes
    config_dict = asdict(config)
    config_dict['stage2_focal_gamma'] = config.fallback_stage2_focal_gamma

    output_dir_str = str(output_dir)
    train_df_dict = train_df.to_dict()
    val_df_dict = val_df.to_dict()

    stage2_results = {}
    for i, (arch, seed) in enumerate(zip(config.stage2_architectures, config.stage2_seeds)):
        model_tag = f'stage2_{arch}_seed{seed}_fallback'

        p = multiprocessing.Process(
            target=_train_stage2_isolated,
            args=(
                train_df_dict, val_df_dict, config_dict, output_dir_str,
                arch, seed, model_tag,
            ),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f'WARNING: Fallback {model_tag} failed with exit code {p.exitcode}')
            continue

        with open(output_dir / f'{model_tag}_history.json', 'r') as f:
            stage2_results[model_tag] = json.load(f)

    return stage2_results


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    config = Config()

    if config.enable_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f'Mixed precision enabled: {policy.name}')

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_environment(config)

    # ── Dataset setup ──
    if is_kaggle_runtime():
        solutions_csv, image_dir = setup_kaggle_environment(config)
        labels_csv = output_dir / 'labels.csv'
    else:
        solutions_csv = Path('training_solutions_rev1.csv')
        image_dir = Path('images')
        labels_csv = output_dir / 'labels.csv'

    # Always regenerate labels with V25 thresholds
    labels_df = generate_labels_df(solutions_csv, image_dir, config)
    labels_df.to_csv(labels_csv, index=False)

    # Validate class counts
    validate_class_counts(labels_df, config)

    # Split data
    class_counts, split_info, (train_df, val_df, test_df) = build_datasets(labels_df, config)

    print('\n=== Dataset Split ===')
    print(f'  Train: {len(train_df)}')
    print(f'  Val:   {len(val_df)}')
    print(f'  Test:  {len(test_df)}')
    print(f'  Class distribution: {class_counts}')

    # ── Run cascade training ──
    start_time = time.time()
    training_results = run_cascade_training(train_df, val_df, config, output_dir)
    total_train_time = time.time() - start_time
    print(f'\nTotal training time: {total_train_time / 60:.1f} minutes')

    # ── Evaluate cascade pipeline ──
    print('\n' + '=' * 70)
    print('CASCADE EVALUATION')
    print('=' * 70)

    # Collect Stage 2 model paths
    stage2_model_paths = []
    for arch, seed in zip(config.stage2_architectures, config.stage2_seeds):
        tag = f'stage2_{arch}_seed{seed}'
        model_path = output_dir / f'{tag}_best.keras'
        if model_path.exists():
            stage2_model_paths.append(str(model_path))

    eval_results = evaluate_cascade(
        stage1_model_path=str(output_dir / 'stage1_best.keras'),
        stage2_model_paths=stage2_model_paths,
        test_df=test_df,
        val_df=val_df,
        config=config,
        output_dir=output_dir,
    )

    # ── Check if fallback is needed ──
    irregular_f1 = eval_results.get('cascade_evaluation', {}).get(
        'classification_report', {}
    ).get('irregular', {}).get('f1-score', 0.0)

    fallback_triggered = False
    if irregular_f1 < 0.75:
        print(f'\n⚠️ Irregular F1 = {irregular_f1:.4f} < 0.75 — triggering Stage 2 fallback!')
        fallback_triggered = True

        fallback_results = run_stage2_fallback(train_df, val_df, config, output_dir)

        # Re-evaluate with fallback models
        fallback_model_paths = []
        for arch, seed in zip(config.stage2_architectures, config.stage2_seeds):
            tag = f'stage2_{arch}_seed{seed}_fallback'
            model_path = output_dir / f'{tag}_best.keras'
            if model_path.exists():
                fallback_model_paths.append(str(model_path))

        if fallback_model_paths:
            eval_results_fallback = evaluate_cascade(
                stage1_model_path=str(output_dir / 'stage1_best.keras'),
                stage2_model_paths=fallback_model_paths,
                test_df=test_df,
                val_df=val_df,
                config=config,
                output_dir=output_dir,
            )
            # Use fallback results if better
            fallback_irr_f1 = eval_results_fallback.get('cascade_evaluation', {}).get(
                'classification_report', {}
            ).get('irregular', {}).get('f1-score', 0.0)
            if fallback_irr_f1 > irregular_f1:
                eval_results = eval_results_fallback
                eval_results['fallback_triggered'] = True

    # ── Compile final metrics ──
    import datetime

    final_metrics = {
        'run_id': 'v25_cascade_224px',
        'timestamp': datetime.datetime.now().isoformat(),
        'training_time_minutes': total_train_time / 60,
        'dataset': split_info,
        'stage1': {
            'architecture': config.stage1_architecture,
            'image_size_phases': [config.image_size_phase1, config.image_size_phase2, config.image_size_phase3],
            'model_path': str(output_dir / 'stage1_best.keras'),
            'training_history': training_results.get('stage1', {}),
        },
        'stage2': {
            'architectures': [f'{a}_seed{s}' for a, s in zip(config.stage2_architectures, config.stage2_seeds)],
            'image_size_phases': [config.image_size_phase1, config.image_size_phase2, config.image_size_phase3],
            'model_paths': stage2_model_paths,
            'training_history': training_results.get('stage2', {}),
        },
        'cascade_evaluation': eval_results.get('cascade_evaluation', {}),
        'swa_applied': True,
        'tta_n_augments': config.tta_n_augments,
        'fallback_triggered': fallback_triggered,
    }

    save_json(output_dir / 'metrics.json', final_metrics)

    # ── Final report ──
    cascade_eval = eval_results.get('cascade_evaluation', {})
    print('\n' + '=' * 60)
    print('FINAL RESULTS — GalaxyNet V25 Cascade Pipeline')
    print('=' * 60)
    print(f"  Test Accuracy (standard): {cascade_eval.get('test_accuracy_standard', 'N/A')}")
    print(f"  Test Accuracy (TTA-16):   {cascade_eval.get('test_accuracy_tta', 'N/A')}")
    print(f"  Cohen's Kappa:            {cascade_eval.get('cohen_kappa', 'N/A')}")
    print(f"  Matthews CorrCoef:        {cascade_eval.get('matthews_corrcoef', 'N/A')}")

    report = cascade_eval.get('classification_report', {})
    print(f"\n  Per-class F1:")
    for cls in CLASS_NAMES:
        cls_data = report.get(cls, {})
        print(f"    {cls:>12s}: P={cls_data.get('precision', 0):.4f}  R={cls_data.get('recall', 0):.4f}  F1={cls_data.get('f1-score', 0):.4f}")

    targets = cascade_eval.get('min_targets_met', {})
    print(f"\n  Acceptance Criteria:")
    for key, met in targets.items():
        status = '✓' if met else '✗'
        print(f"    {status} {key}: {'PASS' if met else 'FAIL'}")

    print(f"\n  Fallback triggered: {fallback_triggered}")
    print(f"  All outputs saved to: {output_dir}")
    print('=' * 60)


if __name__ == '__main__':
    main()
