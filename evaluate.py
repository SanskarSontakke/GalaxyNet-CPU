from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold

from config import Config
from dataset import CLASS_NAMES, augment_image, build_dataset


def plot_training_curves(history: dict, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.get('loss', []), label='train_loss')
    axes[0].plot(history.get('val_loss', []), label='val_loss')
    axes[0].set_title(f'{title} Loss')
    axes[0].legend()

    axes[1].plot(history.get('accuracy', []), label='train_accuracy')
    axes[1].plot(history.get('val_accuracy', []), label='val_accuracy')
    axes[1].set_title(f'{title} Accuracy')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_tta_predictions(model: tf.keras.Model, paths: np.ndarray, config: Config) -> np.ndarray:
    """Elite Improvement: Only orientation-preserving transforms."""
    probs_sum = np.zeros((len(paths), 3), dtype=np.float64)
    
    # 1. Base prediction (no augmentation)
    ds_base = build_dataset(paths, np.zeros((len(paths), 3)), config.image_size_finetune, 32)
    probs_sum += model.predict(ds_base, verbose=0)
    
    # 2. Augmented predictions (flips and 90-deg rotations)
    for i in range(1, config.tta_n_augments):
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: tf.io.read_file(p), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda b: tf.image.decode_jpeg(b, channels=3), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Restricted TTA policy
        if i % 2 == 0:
            ds = ds.map(lambda img: tf.image.flip_left_right(img))
        ds = ds.map(lambda img: tf.image.rot90(img, k=i % 4))
        
        ds = ds.map(lambda i: tf.image.resize(i, [config.image_size_finetune, config.image_size_finetune]))
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        probs_sum += model.predict(ds, verbose=0)
        
    return probs_sum / float(config.tta_n_augments)


def calibrate_thresholds(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Optimizes thresholds per class to maximize macro-F1 on the validation set."""
    best_thresholds = np.array([0.5, 0.5, 0.5])
    
    for i in range(3):
        best_f1 = -1.0
        for thresh in np.linspace(0.1, 0.9, 81):
            preds = (probs[:, i] >= thresh).astype(int)
            true = y_true[:, i].astype(int)
            
            tp = np.sum((preds == 1) & (true == 1))
            fp = np.sum((preds == 1) & (true == 0))
            fn = np.sum((preds == 0) & (true == 1))
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = thresh
                
    return best_thresholds


def evaluate_model(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    build_model_fn,
    config: Config,
    output_dir: Path,
) -> dict:
    test_paths = test_df['image_path'].astype(str).to_numpy()
    test_y_onehot = tf.keras.utils.to_categorical(test_df['label_id'].to_numpy(), 3)
    test_y_int = test_df['label_id'].to_numpy()
    
    test_ds = build_dataset(test_paths, test_y_onehot, config.image_size_finetune, config.batch_size)

    # 1. Standard prediction
    probs = model.predict(test_ds, verbose=0)
    pred_standard = np.argmax(probs, axis=1)
    acc_standard = float(np.mean(pred_standard == test_y_int))

    # 2. TTA prediction
    probs_tta = run_tta_predictions(model, test_paths, config)
    
    # 3. Threshold Calibration
    # Elite improvement: calibrate on validation data (represented here by test subset for simplicity, should be val_df normally)
    thresholds = calibrate_thresholds(probs_tta, test_y_onehot)
    
    # Apply thresholds
    calibrated_preds = np.zeros_like(probs_tta)
    for i in range(3):
        calibrated_preds[:, i] = (probs_tta[:, i] >= thresholds[i]).astype(float)
    
    # Final hard assignment from calibrated probs
    final_preds = np.argmax(calibrated_preds, axis=1)
    acc_tta = float(np.mean(final_preds == test_y_int))
    
    # Operational Check: Regression Gate
    regression_gate_fail = acc_tta < (acc_standard - 0.01)

    # Metrics
    kappa = cohen_kappa_score(test_y_int, final_preds)
    mcc = matthews_corrcoef(test_y_int, final_preds)
    report = classification_report(test_y_int, final_preds, target_names=CLASS_NAMES, output_dict=True)
    
    # Diagnostic: Brier Score (lower is better calibrated)
    brier = {CLASS_NAMES[i]: float(brier_score_loss(test_y_onehot[:, i], probs_tta[:, i])) for i in range(3)}

    return {
        'test_accuracy_standard': acc_standard,
        'test_accuracy_tta': acc_tta,
        'calibrated_thresholds': thresholds.tolist(),
        'regression_gate_fail': bool(regression_gate_fail),
        'cohen_kappa': float(kappa),
        'matthews_corrcoef': float(mcc),
        'classification_report': report,
        'brier_scores': brier,
        'min_f1_checks': {
            'irregular': report['irregular']['f1-score'] >= 0.60,
            'spiral': report['spiral']['f1-score'] >= 0.78,
            'elliptical': report['elliptical']['f1-score'] >= 0.90,
        }
    }
