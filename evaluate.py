from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from dataset import build_dataset
from losses import HierarchicalRMSELoss, RMSELoss, rmse_metric
from model import (
    BenannePartExtractor,
    GalaxyOutputLayer,
    MaxoutDense,
    MultiViewAverage,
    MultiViewLayer,
    PartFeatureMerge,
)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the global Root Mean Squared Error over all targets."""
    mse = np.mean(np.square(y_true - y_pred))
    return np.sqrt(mse)


def load_inference_model(model_path: Path) -> tf.keras.Model:
    custom_objects = {
        'RMSELoss': RMSELoss,
        'HierarchicalRMSELoss': HierarchicalRMSELoss,
        'rmse_metric': rmse_metric,
        'MultiViewLayer': MultiViewLayer,
        'MultiViewAverage': MultiViewAverage,
        'GalaxyOutputLayer': GalaxyOutputLayer,
        'BenannePartExtractor': BenannePartExtractor,
        'PartFeatureMerge': PartFeatureMerge,
        'MaxoutDense': MaxoutDense,
    }
    return tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )


def predict_tta(model: tf.keras.Model, test_paths: np.ndarray, config: Config) -> np.ndarray:
    """Run Test-Time Augmentation (TTA) and return averaged predictions."""
    print(f"\n[Evaluation] Running TTA ({config.tta_n_augments} passes)...")
    
    # Dummy labels for the dataset builder
    dummy_labels = np.zeros((len(test_paths), 37), dtype=np.float32)
    
    tta_preds = []
    
    # Pass 1: Original (unaugmented) center crop
    ds_base = build_dataset(
        test_paths, dummy_labels, config.image_size_phase3, config.batch_size_phase3,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False,
        drop_remainder=False, tta_index=0,
    )
    base_preds = model.predict(ds_base, verbose=1)
    tta_preds.append(base_preds)
    
    # Passes 2 to N: Augmented crops/rotations
    for i in range(config.tta_n_augments - 1):
        ds_aug = build_dataset(
            test_paths, dummy_labels, config.image_size_phase3, config.batch_size_phase3,
            center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False,
            drop_remainder=False, tta_index=i + 1,
        )
        preds = model.predict(ds_aug, verbose=0)
        tta_preds.append(preds)
        print(f"  TTA pass {i+2}/{config.tta_n_augments} completed.")
        
    return np.mean(tta_preds, axis=0)


def _load_submission_template(submission_template_path: Path) -> tuple[list[str], list[int]]:
    if submission_template_path.suffix == '.zip':
        with zipfile.ZipFile(submission_template_path, 'r') as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name, 'r') as handle:
                lines = [line.decode('utf-8').strip() for line in handle]
    else:
        with submission_template_path.open('r', encoding='utf-8') as handle:
            lines = [line.strip() for line in handle]

    reader = csv.reader(lines)
    rows = list(reader)
    header = rows[0]
    galaxy_ids = [int(row[0]) for row in rows[1:] if row]
    return header, galaxy_ids


def generate_competition_submission(
    config: Config,
    test_image_dir: Path,
    target_cols: list[str],
    submission_template_path: Path,
) -> Path | None:
    output_dir = config.output_dir
    model_path = output_dir / 'unified_best.keras'
    if not model_path.exists():
        print(f"Error: Unified model not found at {model_path}")
        return None

    header, galaxy_ids = _load_submission_template(submission_template_path)
    expected_header = ['GalaxyID'] + target_cols
    if header != expected_header:
        raise ValueError('Submission template header does not match expected competition columns.')

    test_paths = np.array([str(test_image_dir / f'{galaxy_id}.jpg') for galaxy_id in galaxy_ids])
    missing_paths = [path for path in test_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(f'Missing {len(missing_paths)} competition test images.')

    print('\n' + '=' * 60)
    print('COMPETITION SUBMISSION')
    print('=' * 60)
    print(f'Predicting {len(test_paths)} competition test images...')

    model = load_inference_model(model_path)
    predictions = predict_tta(model, test_paths, config)
    if predictions.shape != (len(galaxy_ids), len(target_cols)):
        raise ValueError(
            f'Unexpected prediction shape {predictions.shape}, expected {(len(galaxy_ids), len(target_cols))}.'
        )

    submission_df = pd.DataFrame(predictions, columns=target_cols)
    submission_df.insert(0, 'GalaxyID', galaxy_ids)

    submission_path = output_dir / config.submission_filename
    submission_df.to_csv(
        submission_path,
        index=False,
        compression='gzip',
        float_format='%.6f',
    )
    print(f'Submission written to {submission_path}')
    return submission_path


def evaluate_regression(config: Config, test_df: pd.DataFrame, target_cols: list[str], split_name: str = 'test'):
    """Evaluate the single unified regression model using RMSE."""
    output_dir = config.output_dir
    model_path = output_dir / 'unified_best.keras'
    
    if not model_path.exists():
        print(f"Error: Unified model not found at {model_path}")
        return

    print('\n' + '=' * 60)
    print('EVALUATION: REGRESSION RMSE')
    print('=' * 60)
    
    # Load model
    print("Loading unified model...")
    model = load_inference_model(model_path)
    
    test_paths = test_df['image_path'].values
    y_true = test_df[target_cols].values.astype(np.float32)
    
    # Standard Prediction (No TTA)
    dummy_labels = np.zeros_like(y_true)
    ds_test = build_dataset(
        test_paths, dummy_labels, config.image_size_phase3, config.batch_size_phase3,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False,
        drop_remainder=False,
    )
    
    print("\nRunning standard evaluation (1-pass)...")
    y_pred_std = model.predict(ds_test, verbose=1)
    rmse_std = calculate_rmse(y_true, y_pred_std)
    
    # TTA Prediction
    y_pred_tta = predict_tta(model, test_paths, config)
    rmse_tta = calculate_rmse(y_true, y_pred_tta)
    
    # Report
    print('\n' + '-' * 40)
    print(f"{split_name.upper()} METRICS")
    print('-' * 40)
    print(f"  RMSE (Single Pass): {rmse_std:.5f}")
    print(f"  RMSE (TTA {config.tta_n_augments}x):     {rmse_tta:.5f}")
    print('-' * 40)
    
    # Save results
    results = {
        'split_name': split_name,
        'split_size': len(test_df),
        'rmse_standard': float(rmse_std),
        'rmse_tta': float(rmse_tta),
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {output_dir / 'evaluation_results.json'}")
