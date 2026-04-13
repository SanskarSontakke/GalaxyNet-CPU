from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from config import Config
from dataset import build_dataset
from losses import RMSELoss, rmse_metric


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the global Root Mean Squared Error over all targets."""
    mse = np.mean(np.square(y_true - y_pred))
    return np.sqrt(mse)


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
    )
    base_preds = model.predict(ds_base, verbose=1)
    tta_preds.append(base_preds)
    
    # Passes 2 to N: Augmented crops/rotations
    for i in range(config.tta_n_augments - 1):
        ds_aug = build_dataset(
            test_paths, dummy_labels, config.image_size_phase3, config.batch_size_phase3,
            center_crop_ratio=config.center_crop_ratio, augment=True, shuffle=False,
        )
        preds = model.predict(ds_aug, verbose=0)
        tta_preds.append(preds)
        print(f"  TTA pass {i+2}/{config.tta_n_augments} completed.")
        
    return np.mean(tta_preds, axis=0)


def evaluate_regression(config: Config, test_df: pd.DataFrame, target_cols: list[str]):
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
    custom_objects = {
        'RMSELoss': RMSELoss,
        'rmse_metric': rmse_metric,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    test_paths = test_df['image_path'].values
    y_true = test_df[target_cols].values.astype(np.float32)
    
    # Standard Prediction (No TTA)
    dummy_labels = np.zeros_like(y_true)
    ds_test = build_dataset(
        test_paths, dummy_labels, config.image_size_phase3, config.batch_size_phase3,
        center_crop_ratio=config.center_crop_ratio, augment=False, shuffle=False,
    )
    
    print("\nRunning standard evaluation (1-pass)...")
    y_pred_std = model.predict(ds_test, verbose=1)
    rmse_std = calculate_rmse(y_true, y_pred_std)
    
    # TTA Prediction
    y_pred_tta = predict_tta(model, test_paths, config)
    rmse_tta = calculate_rmse(y_true, y_pred_tta)
    
    # Report
    print('\n' + '-' * 40)
    print("TEST METRICS (Kaggle Leaderboard Objective)")
    print('-' * 40)
    print(f"  RMSE (Single Pass): {rmse_std:.5f}")
    print(f"  RMSE (TTA {config.tta_n_augments}x):     {rmse_tta:.5f}")
    print('-' * 40)
    
    # Save results
    results = {
        'test_size': len(test_df),
        'rmse_standard': float(rmse_std),
        'rmse_tta': float(rmse_tta),
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {output_dir / 'evaluation_results.json'}")
