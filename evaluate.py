from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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
    f1_score,
)

from config import Config
from dataset import CLASS_NAMES, build_dataset
from losses import BinaryFocalLoss


# ══════════════════════════════════════════════════════════════
# TTA — 16 STRUCTURED AUGMENTS
# ══════════════════════════════════════════════════════════════

def _build_tta_dataset(paths: np.ndarray, image_size: int, center_crop_ratio: float,
                        rotation_k: int, flip_lr: bool, crop_ratio: float) -> tf.data.Dataset:
    """Build a single TTA view dataset with specific augmentation."""
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def load_and_augment(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)

        # Center crop with specified ratio
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        ch = tf.cast(tf.cast(h, tf.float32) * crop_ratio, tf.int32)
        cw = tf.cast(tf.cast(w, tf.float32) * crop_ratio, tf.int32)
        offset_h = (h - ch) // 2
        offset_w = (w - cw) // 2
        img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, ch, cw)

        # Rotation (0, 90, 180, 270)
        img = tf.image.rot90(img, k=rotation_k)

        # Flip
        if flip_lr:
            img = tf.image.flip_left_right(img)

        img = tf.image.resize(img, [image_size, image_size])
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img

    ds = ds.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return ds


def run_tta_predictions_binary(
    model: tf.keras.Model,
    paths: np.ndarray,
    image_size: int,
    center_crop_ratio: float = 0.75,
    n_augments: int = 16,
) -> np.ndarray:
    """Run TTA with 16 structured augments for binary models.
    
    16 combinations: 4 rotations × 2 flips × 2 center crops (0.75, 0.85)
    Returns averaged sigmoid probabilities.
    """
    probs_sum = np.zeros(len(paths), dtype=np.float64)
    count = 0

    crop_ratios = [0.75, 0.85]
    rotations = [0, 1, 2, 3]
    flips = [False, True]

    for crop_r in crop_ratios:
        for rot_k in rotations:
            for flip_lr in flips:
                if count >= n_augments:
                    break
                ds = _build_tta_dataset(paths, image_size, center_crop_ratio, rot_k, flip_lr, crop_r)
                preds = model.predict(ds, verbose=0).flatten()
                probs_sum += preds
                count += 1
            if count >= n_augments:
                break
        if count >= n_augments:
            break

    return probs_sum / float(count)


# ══════════════════════════════════════════════════════════════
# CASCADE INFERENCE
# ══════════════════════════════════════════════════════════════

def cascade_predict(
    stage1_model: tf.keras.Model,
    stage2_models: list[tf.keras.Model],
    paths: np.ndarray,
    config: Config,
    stage1_threshold: float = 0.5,
    stage2_threshold: float = 0.5,
    use_tta: bool = True,
) -> tuple[np.ndarray, dict]:
    """Run full cascade inference pipeline.
    
    Stage 1: P(elliptical) >= stage1_threshold → predict elliptical
    Stage 2: Average ensemble → P(spiral) >= stage2_threshold → spiral, else irregular
    
    Returns:
        predictions: int array of class IDs (0=spiral, 1=elliptical, 2=irregular)
        details: dict with per-stage probabilities for analysis
    """
    image_size = config.image_size_phase3
    n_samples = len(paths)

    # Stage 1 predictions
    if use_tta:
        stage1_probs = run_tta_predictions_binary(
            stage1_model, paths, image_size,
            center_crop_ratio=config.center_crop_ratio,
            n_augments=config.tta_n_augments,
        )
    else:
        dummy_labels = np.zeros(n_samples, dtype=np.float32)
        ds = build_dataset(paths, dummy_labels, image_size, 32, center_crop_ratio=config.center_crop_ratio)
        stage1_probs = stage1_model.predict(ds, verbose=0).flatten()

    # Classify Stage 1
    is_elliptical = stage1_probs >= stage1_threshold
    non_elliptical_indices = np.where(~is_elliptical)[0]

    # Stage 2 predictions (only for non-elliptical samples)
    stage2_probs_avg = np.zeros(n_samples, dtype=np.float64)
    if len(non_elliptical_indices) > 0:
        ne_paths = paths[non_elliptical_indices]

        for s2_model in stage2_models:
            if use_tta:
                s2_probs = run_tta_predictions_binary(
                    s2_model, ne_paths, image_size,
                    center_crop_ratio=config.center_crop_ratio,
                    n_augments=config.tta_n_augments,
                )
            else:
                dummy_labels = np.zeros(len(ne_paths), dtype=np.float32)
                ds = build_dataset(ne_paths, dummy_labels, image_size, 32,
                                   center_crop_ratio=config.center_crop_ratio)
                s2_probs = s2_model.predict(ds, verbose=0).flatten()

            stage2_probs_avg[non_elliptical_indices] += s2_probs

        stage2_probs_avg[non_elliptical_indices] /= len(stage2_models)

    # Final classification
    predictions = np.full(n_samples, -1, dtype=np.int32)
    predictions[is_elliptical] = 1  # elliptical

    # Among non-elliptical:
    is_spiral = stage2_probs_avg >= stage2_threshold
    for idx in non_elliptical_indices:
        if is_spiral[idx]:
            predictions[idx] = 0  # spiral
        else:
            predictions[idx] = 2  # irregular

    details = {
        'stage1_probs': stage1_probs,
        'stage2_probs': stage2_probs_avg,
        'n_elliptical_predicted': int(is_elliptical.sum()),
        'n_non_elliptical': len(non_elliptical_indices),
    }

    return predictions, details


# ══════════════════════════════════════════════════════════════
# THRESHOLD CALIBRATION — 2D GRID SEARCH
# ══════════════════════════════════════════════════════════════

def calibrate_cascade_thresholds(
    stage1_model: tf.keras.Model,
    stage2_models: list[tf.keras.Model],
    val_df: pd.DataFrame,
    config: Config,
) -> tuple[float, float]:
    """Calibrate both cascade thresholds using 2D grid search on validation set.
    
    Optimizes for macro-F1 score across all 3 classes.
    """
    paths = val_df['image_path'].astype(str).to_numpy()
    y_true = val_df['label_id'].to_numpy()
    image_size = config.image_size_phase3

    # Get raw probabilities (without TTA for speed during calibration)
    dummy_labels = np.zeros(len(paths), dtype=np.float32)
    ds = build_dataset(paths, dummy_labels, image_size, 32, center_crop_ratio=config.center_crop_ratio)
    stage1_probs = stage1_model.predict(ds, verbose=0).flatten()

    # For Stage 2, predict all non-elliptical candidates
    stage2_probs_all = np.zeros(len(paths), dtype=np.float64)
    for s2_model in stage2_models:
        ds = build_dataset(paths, dummy_labels, image_size, 32, center_crop_ratio=config.center_crop_ratio)
        s2_preds = s2_model.predict(ds, verbose=0).flatten()
        stage2_probs_all += s2_preds
    stage2_probs_all /= len(stage2_models)

    best_macro_f1 = -1.0
    best_t1, best_t2 = 0.5, 0.5

    t1_range = np.linspace(0.3, 0.7, 41)
    t2_range = np.linspace(0.3, 0.7, 41)

    print('\n[Threshold Calibration] 2D grid search...')

    for t1 in t1_range:
        for t2 in t2_range:
            preds = np.full(len(paths), -1, dtype=np.int32)
            is_ell = stage1_probs >= t1
            preds[is_ell] = 1  # elliptical
            ne_mask = ~is_ell
            ne_spiral = stage2_probs_all >= t2
            preds[ne_mask & ne_spiral] = 0  # spiral
            preds[ne_mask & ~ne_spiral] = 2  # irregular

            if np.any(preds == -1):
                continue

            macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_t1 = float(t1)
                best_t2 = float(t2)

    print(f'  Best thresholds: stage1={best_t1:.3f}, stage2={best_t2:.3f}')
    print(f'  Best macro-F1:   {best_macro_f1:.4f}')

    return best_t1, best_t2


# ══════════════════════════════════════════════════════════════
# EXPECTED CALIBRATION ERROR (ECE)
# ══════════════════════════════════════════════════════════════

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error for binary predictions.
    
    |mean_confidence - mean_accuracy| per bin, weighted by bin count.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return float(ece)


# ══════════════════════════════════════════════════════════════
# CASCADE ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════

def cascade_error_analysis(
    y_true: np.ndarray,
    predictions: np.ndarray,
    details: dict,
    stage1_threshold: float,
) -> dict:
    """Decompose errors by cascade stage.
    
    Reports:
    - Stage 1 errors: ellipticals missed + non-ellipticals blocked
    - Stage 2 errors: spiral↔irregular misclassifications
    - Percentage of total errors from each stage
    """
    total_errors = int(np.sum(y_true != predictions))
    if total_errors == 0:
        return {'total_errors': 0, 'stage1_error_pct': 0.0, 'stage2_error_pct': 0.0}

    stage1_probs = details['stage1_probs']
    is_ell_pred = stage1_probs >= stage1_threshold
    is_ell_true = y_true == 1  # elliptical = 1

    # Stage 1 errors
    ell_missed = int(np.sum(is_ell_true & ~is_ell_pred))         # FN (missed ellipticals)
    non_ell_blocked = int(np.sum(~is_ell_true & is_ell_pred))     # FP (non-ell called ell)
    stage1_errors = ell_missed + non_ell_blocked

    # Stage 2 errors (among non-elliptical predictions)
    ne_mask = ~is_ell_pred
    ne_true = y_true[ne_mask]
    ne_pred = predictions[ne_mask]
    stage2_errors = int(np.sum(ne_true != ne_pred)) - ell_missed  # Subtract Stage 1 leaks
    stage2_errors = max(0, stage2_errors)

    # Specific Stage 2 confusions
    spiral_as_irregular = int(np.sum((ne_true == 0) & (ne_pred == 2)))
    irregular_as_spiral = int(np.sum((ne_true == 2) & (ne_pred == 0)))

    stage1_pct = stage1_errors / max(total_errors, 1) * 100
    stage2_pct = stage2_errors / max(total_errors, 1) * 100

    return {
        'total_errors': total_errors,
        'stage1_errors': stage1_errors,
        'stage1_elliptical_missed': ell_missed,
        'stage1_nonelliptical_blocked': non_ell_blocked,
        'stage1_false_negative_rate': float(ell_missed / max(is_ell_true.sum(), 1)),
        'stage2_errors': stage2_errors,
        'stage2_spiral_as_irregular': spiral_as_irregular,
        'stage2_irregular_as_spiral': irregular_as_spiral,
        'stage1_error_pct': float(stage1_pct),
        'stage2_error_pct': float(stage2_pct),
    }


# ══════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, out_path: Path, title: str) -> None:
    """Plot loss and accuracy curves for a training phase."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get('loss', []), label='train_loss', linewidth=2)
    axes[0].plot(history.get('val_loss', []), label='val_loss', linewidth=2)
    axes[0].set_title(f'{title} — Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    acc_key = 'binary_accuracy' if 'binary_accuracy' in history else 'accuracy'
    val_acc_key = f'val_{acc_key}'
    axes[1].plot(history.get(acc_key, []), label='train_acc', linewidth=2)
    axes[1].plot(history.get(val_acc_key, []), label='val_acc', linewidth=2)
    axes[1].set_title(f'{title} — Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, title='Confusion Matrix',
                          normalize=False):
    """Plot confusion matrix with optional normalization."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Text annotations
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = f'{cm[i, j]:{fmt}}' if not normalize else f'{cm[i, j]:.2%}'
            ax.text(j, i, val, ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_roc_curves(y_true_onehot, y_probs_3class, class_names, out_path):
    """Plot one-vs-rest ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3', '#4CAF50', '#FF5722']

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if y_true_onehot.shape[1] > i:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs_3class[:, i])
            auc = roc_auc_score(y_true_onehot[:, i], y_probs_3class[:, i])
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{cls} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    stage1_probs: np.ndarray,
    stage2_probs: np.ndarray,
    output_dir: Path,
):
    """Plot threshold sensitivity curves for both stages."""
    # Stage 1
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = []
    for t in thresholds:
        preds = np.where(stage1_probs >= t, 1, 0)
        is_ell = y_true == 1
        f1 = f1_score(is_ell.astype(int), preds, zero_division=0)
        f1s.append(f1)

    ax.plot(thresholds, f1s, linewidth=2, color='#2196F3')
    best_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Best: {thresholds[best_idx]:.2f} (F1={f1s[best_idx]:.3f})')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Elliptical F1', fontsize=12)
    ax.set_title('Stage 1 Threshold Sensitivity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'threshold_sensitivity_stage1.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Stage 2
    fig, ax = plt.subplots(figsize=(8, 5))
    f1s = []
    non_ell_mask = y_true != 1
    y_true_s2 = y_true[non_ell_mask]
    s2_probs_ne = stage2_probs[non_ell_mask]

    for t in thresholds:
        preds = np.where(s2_probs_ne >= t, 0, 2)  # spiral=0 if >= t, else irregular=2
        # Compute irregular F1
        is_irr_true = (y_true_s2 == 2).astype(int)
        is_irr_pred = (preds == 2).astype(int)
        f1 = f1_score(is_irr_true, is_irr_pred, zero_division=0)
        f1s.append(f1)

    ax.plot(thresholds, f1s, linewidth=2, color='#FF5722')
    best_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Best: {thresholds[best_idx]:.2f} (F1={f1s[best_idx]:.3f})')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Irregular F1', fontsize=12)
    ax.set_title('Stage 2 Threshold Sensitivity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'threshold_sensitivity_stage2.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_confidence_calibration(y_true, y_probs_3class, class_names, out_path, n_bins=10):
    """Plot Expected Calibration Error diagram per class."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#2196F3', '#4CAF50', '#FF5722']

    for i, (cls, color, ax) in enumerate(zip(class_names, colors, axes)):
        y_binary = (y_true == i).astype(int)
        probs = y_probs_3class[:, i]

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for b in range(n_bins):
            mask = (probs >= bin_boundaries[b]) & (probs < bin_boundaries[b + 1])
            if mask.sum() > 0:
                bin_accs.append(y_binary[mask].mean())
                bin_confs.append(probs[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(0)
                bin_confs.append((bin_boundaries[b] + bin_boundaries[b + 1]) / 2)
                bin_counts.append(0)

        ece = compute_ece(y_binary, probs, n_bins)

        bin_centers = [(bin_boundaries[b] + bin_boundaries[b + 1]) / 2 for b in range(n_bins)]
        ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6, color=color, edgecolor='white')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_title(f'{cls} (ECE={ece:.4f})', fontsize=12)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Confidence Calibration (ECE)', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_class_metrics_bar(report: dict, class_names: list, out_path: Path):
    """Plot P/R/F1 bar chart per class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.25

    precisions = [report.get(c, {}).get('precision', 0) for c in class_names]
    recalls = [report.get(c, {}).get('recall', 0) for c in class_names]
    f1s = [report.get(c, {}).get('f1-score', 0) for c in class_names]

    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#4CAF50', alpha=0.8)
    bars3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#FF5722', alpha=0.8)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_confidence_histogram(predictions: np.ndarray, details: dict,
                               class_names: list, out_path: Path):
    """Plot prediction confidence distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stage 1 confidence
    axes[0].hist(details['stage1_probs'], bins=50, alpha=0.7, color='#2196F3', edgecolor='white')
    axes[0].set_title('Stage 1 Confidence: P(Elliptical)', fontsize=12)
    axes[0].set_xlabel('Probability')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # Stage 2 confidence (non-elliptical only)
    ne_probs = details['stage2_probs'][details['stage2_probs'] > 0]
    if len(ne_probs) > 0:
        axes[1].hist(ne_probs, bins=50, alpha=0.7, color='#FF5722', edgecolor='white')
    axes[1].set_title('Stage 2 Confidence: P(Spiral)', fontsize=12)
    axes[1].set_xlabel('Probability')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# MAIN EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════

def evaluate_cascade(
    stage1_model_path: str,
    stage2_model_paths: list[str],
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Config,
    output_dir: Path,
) -> dict:
    """Comprehensive cascade pipeline evaluation.
    
    1. Load models
    2. Calibrate thresholds on validation set
    3. Run standard + TTA predictions on test set
    4. Compute all metrics
    5. Generate all plots
    """
    print('\n[Evaluation] Loading models...')
    from losses import (
        OHEMBinaryLoss, 
        CategoricalFocalLoss,
        SoftBinaryAccuracy,
        SoftAUC,
        SoftPrecision,
        SoftRecall
    )
    custom_objects = {
        'BinaryFocalLoss': BinaryFocalLoss,
        'OHEMBinaryLoss': OHEMBinaryLoss,
        'CategoricalFocalLoss': CategoricalFocalLoss,
        'FocalLoss': CategoricalFocalLoss,
        'SoftBinaryAccuracy': SoftBinaryAccuracy,
        'SoftAUC': SoftAUC,
        'SoftPrecision': SoftPrecision,
        'SoftRecall': SoftRecall,
    }

    stage1_model = tf.keras.models.load_model(stage1_model_path, custom_objects=custom_objects)
    stage2_models = [
        tf.keras.models.load_model(p, custom_objects=custom_objects)
        for p in stage2_model_paths
    ]

    print(f'  Stage 1: {stage1_model_path}')
    for p in stage2_model_paths:
        print(f'  Stage 2: {p}')

    # ── Calibrate thresholds ──
    t1, t2 = calibrate_cascade_thresholds(stage1_model, stage2_models, val_df, config)

    # ── Test set evaluation ──
    test_paths = test_df['image_path'].astype(str).to_numpy()
    y_true = test_df['label_id'].to_numpy()
    y_true_onehot = np.eye(3)[y_true]

    # Standard predictions
    print('\n[Evaluation] Standard predictions...')
    preds_standard, details_standard = cascade_predict(
        stage1_model, stage2_models, test_paths, config,
        stage1_threshold=t1, stage2_threshold=t2, use_tta=False,
    )
    acc_standard = float(np.mean(preds_standard == y_true))

    # TTA predictions
    print('[Evaluation] TTA-16 predictions...')
    preds_tta, details_tta = cascade_predict(
        stage1_model, stage2_models, test_paths, config,
        stage1_threshold=t1, stage2_threshold=t2, use_tta=True,
    )
    acc_tta = float(np.mean(preds_tta == y_true))

    # ── Metrics ──
    report = classification_report(y_true, preds_tta, target_names=CLASS_NAMES, output_dict=True)
    kappa = cohen_kappa_score(y_true, preds_tta)
    mcc = matthews_corrcoef(y_true, preds_tta)

    # Brier scores (construct 3-class probabilities from cascade)
    probs_3class = np.zeros((len(test_paths), 3), dtype=np.float64)
    probs_3class[:, 1] = details_tta['stage1_probs']  # P(elliptical)
    probs_3class[:, 0] = (1 - details_tta['stage1_probs']) * details_tta['stage2_probs']  # P(spiral)
    probs_3class[:, 2] = (1 - details_tta['stage1_probs']) * (1 - details_tta['stage2_probs'])  # P(irregular)
    # Normalize
    row_sums = probs_3class.sum(axis=1, keepdims=True)
    probs_3class = probs_3class / np.maximum(row_sums, 1e-8)

    brier = {
        CLASS_NAMES[i]: float(brier_score_loss(y_true_onehot[:, i], probs_3class[:, i]))
        for i in range(3)
    }

    # ECE per class
    ece_scores = {}
    for i, cls in enumerate(CLASS_NAMES):
        y_binary = (y_true == i).astype(int)
        ece_scores[cls] = compute_ece(y_binary, probs_3class[:, i])

    # Error analysis
    error_analysis = cascade_error_analysis(y_true, preds_tta, details_tta, t1)

    # Acceptance criteria
    min_targets = {
        'overall_accuracy_95': acc_tta >= 0.95,
        'irregular_f1_80': report.get('irregular', {}).get('f1-score', 0) >= 0.80,
        'irregular_recall_80': report.get('irregular', {}).get('recall', 0) >= 0.80,
        'spiral_f1_90': report.get('spiral', {}).get('f1-score', 0) >= 0.90,
        'elliptical_f1_93': report.get('elliptical', {}).get('f1-score', 0) >= 0.93,
        'cohen_kappa_90': kappa >= 0.90,
        'stage1_fnr_03': error_analysis.get('stage1_false_negative_rate', 1.0) <= 0.03,
    }

    # ── Generate all plots ──
    print('\n[Evaluation] Generating plots...')

    # Training curves (from saved histories)
    for stage_prefix in ['stage1']:
        hist_path = output_dir / f'{stage_prefix}_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            for phase_key, phase_hist in hist.items():
                plot_training_curves(
                    phase_hist,
                    output_dir / f'training_curves_{stage_prefix}_{phase_key}.png',
                    f'{stage_prefix.upper()} {phase_key}',
                )

    # Stage 2 training curves
    for arch, seed in zip(config.stage2_architectures, config.stage2_seeds):
        tag = f'stage2_{arch}_seed{seed}'
        hist_path = output_dir / f'{tag}_history.json'
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            for phase_key, phase_hist in hist.items():
                plot_training_curves(
                    phase_hist,
                    output_dir / f'training_curves_{tag}_{phase_key}.png',
                    f'{tag} {phase_key}',
                )

    # Confusion matrices
    plot_confusion_matrix(y_true, preds_tta, CLASS_NAMES,
                          output_dir / 'confusion_matrix_cascade.png',
                          'Cascade Confusion Matrix')
    plot_confusion_matrix(y_true, preds_tta, CLASS_NAMES,
                          output_dir / 'confusion_matrix_normalized.png',
                          'Cascade Confusion Matrix (Normalized)', normalize=True)

    # Stage 2 binary confusion (spiral vs irregular only)
    ne_mask = y_true != 1
    if ne_mask.sum() > 0:
        s2_true = y_true[ne_mask]
        s2_pred = preds_tta[ne_mask]
        s2_names = ['spiral', 'irregular']
        # Remap: spiral=0, irregular=2 → binary 0, 1
        s2_true_binary = np.where(s2_true == 0, 0, 1)
        s2_pred_binary = np.where(s2_pred == 0, 0, 1)
        plot_confusion_matrix(s2_true_binary, s2_pred_binary, s2_names,
                              output_dir / 'confusion_matrix_stage2_binary.png',
                              'Stage 2 Binary: Spiral vs Irregular')

    # ROC curves
    try:
        plot_roc_curves(y_true_onehot, probs_3class, CLASS_NAMES,
                        output_dir / 'roc_curves_cascade.png')
    except Exception as e:
        print(f'  Warning: ROC plot failed: {e}')

    # Threshold sensitivity
    plot_threshold_sensitivity(y_true, details_tta['stage1_probs'],
                               details_tta['stage2_probs'], output_dir)

    # Confidence calibration
    plot_confidence_calibration(y_true, probs_3class, CLASS_NAMES,
                                output_dir / 'confidence_calibration.png')

    # Class metrics bar
    plot_class_metrics_bar(report, CLASS_NAMES, output_dir / 'class_metrics_bar.png')

    # Confidence histogram
    plot_confidence_histogram(preds_tta, details_tta, CLASS_NAMES,
                              output_dir / 'prediction_confidence_histogram.png')

    # ── Compile results ──
    cascade_eval = {
        'test_accuracy_standard': acc_standard,
        'test_accuracy_tta': acc_tta,
        'calibrated_thresholds': {'stage1': t1, 'stage2': t2},
        'cohen_kappa': float(kappa),
        'matthews_corrcoef': float(mcc),
        'classification_report': report,
        'brier_scores': brier,
        'ece_scores': ece_scores,
        'error_analysis': error_analysis,
        'min_targets_met': min_targets,
    }

    # Print summary
    print('\n=== Cascade Evaluation Summary ===')
    print(f'  Standard Accuracy: {acc_standard:.4f}')
    print(f'  TTA-16 Accuracy:   {acc_tta:.4f}')
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    print(f'  Matthews MCC:      {mcc:.4f}')
    print(f'  Thresholds: t1={t1:.3f}, t2={t2:.3f}')
    print(f'\n  Error decomposition:')
    print(f"    Stage 1: {error_analysis.get('stage1_error_pct', 0):.1f}% of errors")
    print(f"    Stage 2: {error_analysis.get('stage2_error_pct', 0):.1f}% of errors")
    print(f"    Stage 1 FNR: {error_analysis.get('stage1_false_negative_rate', 0):.4f}")

    # Cleanup
    del stage1_model
    for m in stage2_models:
        del m
    tf.keras.backend.clear_session()

    return {'cascade_evaluation': cascade_eval}
