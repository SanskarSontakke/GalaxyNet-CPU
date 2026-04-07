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
)
from sklearn.model_selection import StratifiedKFold

from galaxynet.config import Config
from galaxynet.dataset import CLASS_NAMES, augment_image, build_dataset


def _dataset_to_arrays(df: pd.DataFrame):
    paths = df['image_path'].astype(str).to_numpy()
    y = tf.keras.utils.to_categorical(df['label_id'].to_numpy(), num_classes=3)
    y_int = df['label_id'].to_numpy()
    return paths, y, y_int


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


def plot_confusions(cm: np.ndarray, out_raw: Path, out_norm: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center')
    fig.tight_layout(); fig.savefig(out_raw, dpi=160); plt.close(fig)

    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title('Normalized Confusion Matrix')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center')
    fig.tight_layout(); fig.savefig(out_norm, dpi=160); plt.close(fig)


def plot_roc(y_true_onehot: np.ndarray, probs: np.ndarray, out_path: Path) -> dict[str, float]:
    aucs: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], probs[:, i])
        auc = roc_auc_score(y_true_onehot[:, i], probs[:, i])
        aucs[name] = float(auc)
        ax.plot(fpr, tpr, label=f'{name} AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(); ax.set_title('One-vs-Rest ROC')
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
    return aucs


def run_tta_predictions(model: tf.keras.Model, paths: np.ndarray, config: Config) -> np.ndarray:
    probs_sum = np.zeros((len(paths), 3), dtype=np.float64)
    for _ in range(config.tta_n_augments):
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: tf.io.read_file(p), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda b: tf.image.decode_jpeg(b, channels=3), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda i: (tf.cast(i, tf.float32), tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)))
        ds = ds.map(lambda i, l: augment_image(i, l)[0], num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda i: tf.image.resize(i, [config.image_size, config.image_size]), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
        probs_sum += model.predict(ds, verbose=0)
    return probs_sum / float(config.tta_n_augments)


def cross_val_stability_check(labels_df: pd.DataFrame, build_model_fn, config: Config) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    y = labels_df['label_id'].to_numpy()
    paths = labels_df['image_path'].astype(str).to_numpy()
    accs: list[float] = []

    for train_idx, val_idx in skf.split(paths, y):
        fold_train = pd.DataFrame({'image_path': paths[train_idx], 'label_id': y[train_idx]})
        fold_val = pd.DataFrame({'image_path': paths[val_idx], 'label_id': y[val_idx]})

        tr_x, tr_y = fold_train['image_path'].to_numpy(), tf.keras.utils.to_categorical(fold_train['label_id'], 3)
        va_x, va_y = fold_val['image_path'].to_numpy(), tf.keras.utils.to_categorical(fold_val['label_id'], 3)

        train_ds = build_dataset(tr_x, tr_y, config.image_size, config.batch_size, augment=True, shuffle=True)
        val_ds = build_dataset(va_x, va_y, config.image_size, config.batch_size, augment=False)

        model, base_model = build_model_fn(config.image_size, 3)
        base_model.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=config.cv_finetune_epochs, verbose=0)
        _, val_acc = model.evaluate(val_ds, verbose=0)
        accs.append(float(val_acc))

    return float(np.mean(accs)), float(np.std(accs))


def evaluate_model(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    build_model_fn,
    config: Config,
    output_dir: Path,
) -> dict:
    test_paths, test_y_onehot, test_y_int = _dataset_to_arrays(test_df)
    test_ds = build_dataset(test_paths, test_y_onehot, config.image_size, config.batch_size)

    start = time.perf_counter()
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0)
    infer_ms = (time.perf_counter() - start) * 1000.0 / len(test_df)

    pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(test_y_int, pred)
    report = classification_report(test_y_int, pred, target_names=CLASS_NAMES, output_dict=True, digits=4)
    kappa = cohen_kappa_score(test_y_int, pred)
    mcc = matthews_corrcoef(test_y_int, pred)

    tta_start = time.perf_counter()
    tta_probs = run_tta_predictions(model, test_paths, config)
    tta_pred = np.argmax(tta_probs, axis=1)
    tta_acc = float(np.mean(tta_pred == test_y_int))
    tta_ms = (time.perf_counter() - tta_start) * 1000.0 / len(test_df)

    plot_confusions(cm, output_dir / 'confusion_matrix.png', output_dir / 'confusion_matrix_normalized.png')

    class_df = pd.DataFrame({
        'class': CLASS_NAMES,
        'precision': [report[c]['precision'] for c in CLASS_NAMES],
        'recall': [report[c]['recall'] for c in CLASS_NAMES],
        'f1-score': [report[c]['f1-score'] for c in CLASS_NAMES],
    })
    class_df.set_index('class').plot(kind='bar', figsize=(8, 5))
    plt.tight_layout(); plt.savefig(output_dir / 'class_metrics_bar.png', dpi=160); plt.close()

    aucs = plot_roc(test_y_onehot, probs, output_dir / 'roc_curves.png')

    confidence = np.max(probs, axis=1)
    plt.figure(figsize=(7, 4))
    plt.hist(confidence, bins=20)
    plt.title('Prediction Confidence Histogram')
    plt.xlabel('max softmax probability'); plt.ylabel('count')
    plt.tight_layout(); plt.savefig(output_dir / 'prediction_confidence_histogram.png', dpi=160); plt.close()

    cv_mean, cv_std = cross_val_stability_check(labels_df, build_model_fn, config)

    return {
        'test_loss': float(test_loss),
        'test_accuracy_standard': float(test_acc),
        'test_accuracy_tta': float(tta_acc),
        'cohen_kappa': float(kappa),
        'matthews_corrcoef': float(mcc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc_ovr': aucs,
        'cross_val_accuracy_mean': float(cv_mean),
        'cross_val_accuracy_std': float(cv_std),
        'ms_per_image_standard': float(infer_ms),
        'ms_per_image_tta_16x': float(tta_ms),
    }
