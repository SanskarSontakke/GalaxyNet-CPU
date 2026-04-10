from __future__ import annotations

import argparse
import os
import time
import gc
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

import sys
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import build_datasets, build_dataset, compute_alpha_from_counts, generate_labels_df, compute_sample_weights
from evaluate import evaluate_model, plot_training_curves
from losses import FocalLoss
from model import build_model, unfreeze_top_layers
from utils import is_kaggle_runtime, save_json, setup_environment, setup_kaggle_environment


class ConciseLogging(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (f"Epoch {epoch+1:03d} | loss: {logs.get('loss', 0):.4f} | "
               f"acc: {logs.get('accuracy', 0):.4f} | val_acc: {logs.get('val_accuracy', 0):.4f}")
        print(msg)


def _metrics():
    return [
        'accuracy',
        tf.keras.metrics.AUC(name='auc', multi_label=False),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


def _train_head_isolated(arch, train_df, val_df, class_counts, config_dict, output_dir_str):
    import gc
    import json
    
    # VRAM Isolated profile batch sizing
    config = Config(**config_dict)
    output_dir = Path(output_dir_str)
    bs_finetune = 16 if arch == 'ConvNeXtTiny' else config.batch_size_finetune

    tf.keras.backend.clear_session()
    
    alpha = compute_alpha_from_counts(class_counts)
    focal = FocalLoss(gamma=config.focal_gamma, alpha=alpha, label_smoothing=config.label_smoothing)
    current_size = config.image_size_warmup

    print(f"\n[Training Ensemble Head: {arch}]")
    
    # 1. Warmup
    tr_x = train_df['image_path'].astype(str).to_numpy()
    tr_y = tf.keras.utils.to_categorical(train_df['label_id'].to_numpy(), 3)
    train_weights = compute_sample_weights(train_df) if config.use_sample_weighting else None
    
    va_x = val_df['image_path'].astype(str).to_numpy()
    va_y = tf.keras.utils.to_categorical(val_df['label_id'].to_numpy(), 3)
    
    train_ds = build_dataset(tr_x, tr_y, current_size, config.batch_size, weights=train_weights, augment=True, shuffle=True)
    val_ds = build_dataset(va_x, va_y, current_size, config.batch_size, augment=False, shuffle=False)

    model, base_model = build_model(current_size, num_classes=3, architecture=arch)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.warmup_lr), loss=focal, metrics=_metrics())
    
    warmup_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / f'warmup_{arch}.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ConciseLogging()
    ]

    print(f"Warmup @ {current_size}px...")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.warmup_epochs, callbacks=warmup_callbacks, verbose=0)

    # 2. Fine-tuning
    new_size = config.image_size_finetune
    print(f"Transitioning to Fine-Tuning @ {new_size}px... (Batch Size: {bs_finetune})")
    
    tr_w = (train_df['label_id'].map({cid: 2.0 if cid == 2 else 1.0 for cid in range(3)})).to_numpy()
    
    train_ds_hf = build_dataset(tr_x, tr_y, new_size, bs_finetune, weights=tr_w, augment=True, shuffle=True, cache=False)
    val_ds_hf = build_dataset(va_x, va_y, new_size, bs_finetune, augment=False, shuffle=False, cache=False)
    
    unfreeze_top_layers(base_model, config.finetune_unfreeze_last_n)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        config.finetune_lr, decay_steps=config.finetune_epochs * (len(train_df) // bs_finetune)
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=focal, metrics=_metrics())

    finetune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / f'best_{arch}.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        ConciseLogging()
    ]
    
    h2 = model.fit(train_ds_hf, validation_data=val_ds_hf, epochs=config.finetune_epochs, callbacks=finetune_callbacks, verbose=0)
    
    result = { 'h1': h1.history, 'h2': h2.history }
    with open(output_dir / f'{arch}_history.json', 'w') as f:
        json.dump(result, f)


def run_ensemble_training(train_df, val_df, class_counts, config: Config, output_dir: Path):
    import multiprocessing
    import json
    from dataclasses import asdict
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    ensemble_results = {}
    config_dict = asdict(config)
    output_dir_str = str(output_dir)

    for arch in config.ensemble_architectures:
        p = multiprocessing.Process(target=_train_head_isolated, args=(arch, train_df, val_df, class_counts, config_dict, output_dir_str))
        p.start()
        p.join()
        
        if p.exitcode != 0:
            raise RuntimeError(f"Training failed for {arch} with exit code {p.exitcode}")
            
        with open(output_dir / f'{arch}_history.json', 'r') as f:
            ensemble_results[arch] = json.load(f)
            
    return ensemble_results


def main():
    config = Config()
    if config.enable_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled: {policy.name}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_environment(config)

    if is_kaggle_runtime():
        solutions_csv, image_dir = setup_kaggle_environment(config)
        labels_csv = output_dir / 'labels.csv'
    else:
        # standard fallback
        solutions_csv = Path('training_solutions_rev1.csv')
        image_dir = Path('images')
        labels_csv = output_dir / 'labels.csv'

    if not labels_csv.exists():
        labels_df = generate_labels_df(solutions_csv, image_dir, config)
        labels_df.to_csv(labels_csv, index=False)
    else:
        labels_df = pd.read_csv(labels_csv)

    train_ds, val_ds, test_ds, class_counts, split_info, (train_df, val_df, test_df) = build_datasets(labels_df, config)
    
    # Assert is now robust for any processing changes
    assert split_info['train_size'] + split_info['val_size'] + split_info['test_size'] >= split_info['total_labeled_images']

    results = run_ensemble_training(train_df, val_df, class_counts, config, output_dir)
    
    # Save ensemble results and select best model for summary report
    best_arch = config.ensemble_architectures[0]
    best_path = output_dir / f'best_{best_arch}.keras'
    model = tf.keras.models.load_model(best_path, custom_objects={'FocalLoss': FocalLoss})

    eval_results = evaluate_model(model, test_df, labels_df, build_model, config, output_dir)
    save_json(output_dir / 'metrics.json', { 'ensemble': results, 'final_eval': eval_results })

    print('\n' + '=' * 50)
    print(f"ELITE ENSEMBLE TEST ACCURACY (TTA): {eval_results['test_accuracy_tta']:.4f}")
    print('=' * 50)


if __name__ == '__main__':
    main()
