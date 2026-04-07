from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf

from galaxynet.config import Config
from galaxynet.dataset import build_datasets, compute_alpha_from_counts, generate_labels_df
from galaxynet.evaluate import evaluate_model, plot_training_curves
from galaxynet.losses import FocalLoss
from galaxynet.model import build_model, unfreeze_top_layers
from galaxynet.utils import is_kaggle_runtime, save_json, setup_environment, setup_kaggle_environment


def _metrics():
    return [
        'accuracy',
        tf.keras.metrics.AUC(name='auc', multi_label=False),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]


def run_training(train_ds, val_ds, class_counts: dict[str, int], config: Config, output_dir: Path):
    alpha = compute_alpha_from_counts(class_counts)
    focal = FocalLoss(gamma=config.focal_gamma, alpha=alpha)

    model, base_model = build_model(config.image_size, num_classes=3)

    model.compile(optimizer=tf.keras.optimizers.Adam(config.warmup_lr), loss=focal, metrics=_metrics())
    warmup_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / 'warmup_best.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ]

    warmup_start = time.perf_counter()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.warmup_epochs, callbacks=warmup_callbacks)
    warmup_seconds = time.perf_counter() - warmup_start

    unfrozen = unfreeze_top_layers(base_model, config.finetune_unfreeze_last_n)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.finetune_lr), loss=focal, metrics=_metrics())

    finetune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / 'best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(output_dir / 'training_log.csv'),
    ]

    finetune_start = time.perf_counter()
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=config.finetune_epochs, callbacks=finetune_callbacks)
    finetune_seconds = time.perf_counter() - finetune_start

    best_path = output_dir / 'best_model.keras'
    model = tf.keras.models.load_model(best_path, custom_objects={'FocalLoss': FocalLoss})

    return model, h1.history, h2.history, {
        'alpha': alpha,
        'warmup_seconds': warmup_seconds,
        'finetune_seconds': finetune_seconds,
        'unfrozen_layers': unfrozen,
    }


def parse_local_args() -> tuple[Path, Path, Path | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--solutions-csv', type=Path, required=False)
    parser.add_argument('--image-dir', type=Path, required=False)
    parser.add_argument('--labels-csv', type=Path, required=False)
    args = parser.parse_args()

    if args.labels_csv is not None and args.image_dir is not None:
        return args.labels_csv, args.image_dir, args.labels_csv
    if args.solutions_csv is None or args.image_dir is None:
        raise ValueError('Local mode requires --solutions-csv and --image-dir (or --labels-csv and --image-dir).')
    return args.solutions_csv, args.image_dir, None


def main():
    config = Config()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_environment(config)

    if is_kaggle_runtime():
        solutions_csv, image_dir = setup_kaggle_environment(config)
        labels_csv = output_dir / 'labels.csv'
    else:
        csv_path, image_dir, provided_labels = parse_local_args()
        solutions_csv = csv_path
        labels_csv = provided_labels or (output_dir / 'labels.csv')

    if labels_csv.exists() and labels_csv.suffix == '.csv' and labels_csv.name == 'labels.csv':
        labels_df = pd.read_csv(labels_csv)
    elif labels_csv.exists() and labels_csv.name != 'training_solutions_rev1.csv':
        labels_df = pd.read_csv(labels_csv)
    else:
        labels_df = generate_labels_df(solutions_csv, image_dir, config)
        labels_df.to_csv(output_dir / 'labels.csv', index=False)

    train_ds, val_ds, test_ds, class_counts, split_info, (_, _, test_df) = build_datasets(labels_df, config)
    assert split_info['train_size'] + split_info['val_size'] + split_info['test_size'] == split_info['total_labeled_images']

    model, h1, h2, train_meta = run_training(train_ds, val_ds, class_counts, config, output_dir)

    plot_training_curves(h1, output_dir / 'training_curves_warmup.png', 'Warm-up')
    plot_training_curves(h2, output_dir / 'training_curves_finetune.png', 'Fine-tune')

    eval_results = evaluate_model(model, test_df, labels_df, build_model, config, output_dir)

    metrics = {
        'dataset': {
            'total_labeled_images': split_info['total_labeled_images'],
            'class_distribution': split_info['class_distribution'],
            'train_size': split_info['train_size'],
            'val_size': split_info['val_size'],
            'test_size': split_info['test_size'],
            'image_size': config.image_size,
            'label_thresholds': {
                'elliptical': config.elliptical_threshold,
                'spiral_disk': config.spiral_disk_threshold,
                'spiral_arms': config.spiral_arms_threshold,
                'irregular': config.irregular_threshold,
            },
        },
        'training': {
            'warmup_epochs_ran': len(h1.get('loss', [])),
            'finetune_epochs_ran': len(h2.get('loss', [])),
            'total_train_time_seconds': train_meta['warmup_seconds'] + train_meta['finetune_seconds'],
            'base_model': 'EfficientNetV2B0',
            'pretrained_weights': 'imagenet',
            'unfrozen_layers': train_meta['unfrozen_layers'],
            'focal_alpha': train_meta['alpha'],
        },
        'evaluation': {
            'test_accuracy_standard': eval_results['test_accuracy_standard'],
            'test_accuracy_tta': eval_results['test_accuracy_tta'],
            'test_loss': eval_results['test_loss'],
            'cohen_kappa': eval_results['cohen_kappa'],
            'matthews_corrcoef': eval_results['matthews_corrcoef'],
            'classification_report': eval_results['classification_report'],
            'confusion_matrix': eval_results['confusion_matrix'],
            'cross_val_accuracy_mean': eval_results['cross_val_accuracy_mean'],
            'cross_val_accuracy_std': eval_results['cross_val_accuracy_std'],
            'roc_auc_ovr': eval_results['roc_auc_ovr'],
        },
        'inference': {
            'ms_per_image_standard': eval_results['ms_per_image_standard'],
            'ms_per_image_tta_16x': eval_results['ms_per_image_tta_16x'],
        },
        'model': {
            'total_parameters': int(model.count_params()),
            'model_path': str(output_dir / 'best_model.keras'),
        },
    }

    save_json(output_dir / 'metrics.json', metrics)
    print('\n' + '=' * 50)
    print(f"FINAL TEST ACCURACY (TTA): {metrics['evaluation']['test_accuracy_tta']:.4f}")
    print(f"Cohen Kappa:              {metrics['evaluation']['cohen_kappa']:.4f}")
    print(
        'Cross-val mean±std:       '
        f"{metrics['evaluation']['cross_val_accuracy_mean']:.4f} ± {metrics['evaluation']['cross_val_accuracy_std']:.4f}"
    )
    print('=' * 50)


if __name__ == '__main__':
    main()
