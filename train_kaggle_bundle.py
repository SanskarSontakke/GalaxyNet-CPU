# AUTO-GENERATED KAGGLE BUNDLE
# This file merges modular components for Kaggle compatibility.
from __future__ import annotations
import os
import sys
import time
import argparse
import json
import zipfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ====================
# START OF config.py
# ====================


from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    kaggle_input_dir: Path = Path('/kaggle/input/competitions/galaxy-zoo-the-galaxy-challenge')
    kaggle_temp_dir: Path = Path('/kaggle/temp')
    output_dir: Path = Path('/kaggle/working/outputs')

    # Local override paths
    local_image_dir: Path | None = None
    local_solutions_csv: Path | None = None
    local_labels_csv: Path | None = None

    # Data
    image_size: int = 128
    batch_size: int = 64
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15

    # Label thresholds
    elliptical_threshold: float = 0.469
    spiral_disk_threshold: float = 0.430
    spiral_arms_threshold: float = 0.430
    irregular_threshold: float = 0.469

    # Training phase 1
    warmup_epochs: int = 10
    warmup_lr: float = 1e-3

    # Training phase 2
    finetune_epochs: int = 30
    finetune_lr: float = 1e-5
    finetune_unfreeze_last_n: int = 30
    use_oversampling: bool = True

    # Focal Loss
    focal_gamma: float = 2.0

    # TTA
    tta_n_augments: int = 16

    # Cross validation
    cv_folds: int = 5
    cv_finetune_epochs: int = 5

    # Runtime flags
    enable_mixed_precision: bool = True
    cache_val_test: bool = True

# ====================
# START OF utils.py
# ====================


import json
import os
import random
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf



def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_environment(config: Config) -> None:
    set_global_seed(config.seed)

    if config.enable_mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'GPU available: {gpus}')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print('No GPU detected — running on CPU')


def find_kaggle_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f'Could not find {name} under {root}')
    return matches[0]


def _extract_zip_if_needed(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)


def setup_kaggle_environment(config: Config) -> Tuple[Path, Path]:
    train_csv_zip = find_kaggle_file(config.kaggle_input_dir, 'training_solutions_rev1.zip')
    image_zip = find_kaggle_file(config.kaggle_input_dir, 'images_training_rev1.zip')

    _extract_zip_if_needed(train_csv_zip, config.kaggle_temp_dir)
    _extract_zip_if_needed(image_zip, config.kaggle_temp_dir)

    csv_path = find_kaggle_file(config.kaggle_temp_dir, 'training_solutions_rev1.csv')
    image_dir = find_kaggle_file(config.kaggle_temp_dir, 'images_training_rev1')

    jpg_count = len(list(Path(image_dir).glob('*.jpg')))
    if jpg_count < 1000:
        raise RuntimeError(f'Expected >=1000 jpg files, found {jpg_count} in {image_dir}')

    return Path(csv_path), Path(image_dir)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def is_kaggle_runtime() -> bool:
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

# ====================
# START OF losses.py
# ====================


import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None, name: str = 'focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = tf.constant(alpha if alpha is not None else [1.0, 1.0, 1.0], dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
        focal_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = -alpha_t * focal_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        return {
            'gamma': self.gamma,
            'alpha': self.alpha.numpy().tolist(),
            'name': self.name,
        }

# ====================
# START OF model.py
# ====================


import tensorflow as tf
from tensorflow.keras import layers


def build_model(image_size: int = 128, num_classes: int = 3):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name='image')

    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='galaxynet_efficientnetv2b0')
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, last_n: int) -> int:
    base_model.trainable = True
    if last_n <= 0:
        for layer in base_model.layers:
            layer.trainable = False
        return 0

    for layer in base_model.layers[:-last_n]:
        layer.trainable = False
    for layer in base_model.layers[-last_n:]:
        layer.trainable = True
    return last_n

# ====================
# START OF dataset.py
# ====================


import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


try:
    import tensorflow_addons as tfa
except Exception:  # fallback handled at runtime
    tfa = None

CLASS_NAMES = ['spiral', 'elliptical', 'irregular']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def determine_class(row: pd.Series, config: Config) -> str:
    is_elliptical = row['Class1.1'] >= config.elliptical_threshold
    is_spiral = (row['Class1.2'] >= config.spiral_disk_threshold) and (
        row['Class4.1'] >= config.spiral_arms_threshold
    )
    is_irregular = (
        row['Class6.1'] >= config.irregular_threshold
        and not is_elliptical
        and not is_spiral
    )

    if is_elliptical:
        return 'elliptical'
    if is_spiral:
        return 'spiral'
    if is_irregular:
        return 'irregular'
    return 'unknown'


def generate_labels_df(solutions_csv: Path, image_dir: Path, config: Config) -> pd.DataFrame:
    df = pd.read_csv(solutions_csv)
    required_cols = {'GalaxyID', 'Class1.1', 'Class1.2', 'Class4.1', 'Class6.1'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {solutions_csv}: {sorted(missing)}')

    df['label'] = df.apply(lambda r: determine_class(r, config), axis=1)
    df = df[df['label'] != 'unknown'].copy()
    df['image_path'] = df['GalaxyID'].astype(str).apply(lambda gid: str(image_dir / f'{gid}.jpg'))
    df = df[df['image_path'].apply(lambda p: Path(p).exists())].copy()
    df['label_id'] = df['label'].map(CLASS_TO_ID)

    counts = df['label'].value_counts().to_dict()
    for cls in CLASS_NAMES:
        if counts.get(cls, 0) < 1000:
            raise ValueError(f'Class {cls} has only {counts.get(cls, 0)} samples (<1000).')

    return df[['GalaxyID', 'image_path', 'label', 'label_id']]


def compute_alpha_from_counts(class_counts: dict[str, int]) -> list[float]:
    counts = np.array([class_counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float64)
    total = counts.sum()
    alpha = total / (len(CLASS_NAMES) * counts)
    alpha = alpha / alpha.sum()
    return alpha.tolist()


def _center_crop(image: tf.Tensor, ratio: float = 0.8) -> tf.Tensor:
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    ch = tf.cast(tf.cast(h, tf.float32) * ratio, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * ratio, tf.int32)
    offset_h = (h - ch) // 2
    offset_w = (w - cw) // 2
    return tf.image.crop_to_bounding_box(image, offset_h, offset_w, ch, cw)


def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor, image_size: int):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = _center_crop(img, ratio=0.8)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


def augment_image(image: tf.Tensor, label: tf.Tensor):
    if tfa is not None:
        angle = tf.random.uniform([], 0.0, 2.0 * math.pi)
        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
    else:
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2 * 255.0)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    image = tf.image.random_crop(image, [112, 112, 3])
    image = tf.image.resize(image, [128, 128])
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_dataset(
    image_paths: np.ndarray,
    labels_onehot: np.ndarray,
    image_size: int,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_onehot))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    ds = ds.map(lambda p, y: load_and_preprocess_image(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _split_dataframe(df: pd.DataFrame, config: Config):
    train_df, temp_df = train_test_split(
        df,
        test_size=config.val_split + config.test_split,
        random_state=config.seed,
        stratify=df['label_id'],
    )
    test_ratio_of_temp = config.test_split / (config.val_split + config.test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio_of_temp,
        random_state=config.seed,
        stratify=temp_df['label_id'],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _assert_non_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_set, val_set, test_set = set(train_df['image_path']), set(val_df['image_path']), set(test_df['image_path'])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def build_datasets(labels_df: pd.DataFrame, config: Config):
    train_df, val_df, test_df = _split_dataframe(labels_df, config)
    _assert_non_overlap(train_df, val_df, test_df)

    if config.use_oversampling:
        counts = train_df['label'].value_counts().to_dict()
        max_c = max(counts.values())
        to_concat = []
        for cls, count in counts.items():
            if count < max_c:
                ratio = int(max_c / count) - 1
                if ratio > 0:
                    to_concat.append(train_df[train_df['label'] == cls].sample(n=ratio*count, replace=True, random_state=config.seed))
        if to_concat:
            train_df = pd.concat([train_df] + to_concat).sample(frac=1, random_state=config.seed).reset_index(drop=True)

    class_counts = labels_df['label'].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()

    def to_xy(df: pd.DataFrame):
        paths = df['image_path'].astype(str).to_numpy()
        y = tf.keras.utils.to_categorical(df['label_id'].to_numpy(), num_classes=3)
        return paths, y

    train_paths, train_y = to_xy(train_df)
    val_paths, val_y = to_xy(val_df)
    test_paths, test_y = to_xy(test_df)

    train_ds = build_dataset(train_paths, train_y, config.image_size, config.batch_size, augment=True, shuffle=True)
    val_ds = build_dataset(val_paths, val_y, config.image_size, config.batch_size, augment=False, shuffle=False, cache=config.cache_val_test)
    test_ds = build_dataset(test_paths, test_y, config.image_size, config.batch_size, augment=False, shuffle=False, cache=config.cache_val_test)

    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_labeled_images': len(labels_df),
        'class_distribution': class_counts,
        'config': asdict(config),
        'train_distribution': train_df['label'].value_counts().to_dict(),
        'val_distribution': val_df['label'].value_counts().to_dict(),
        'test_distribution': test_df['label'].value_counts().to_dict(),
    }

    return train_ds, val_ds, test_ds, class_counts, split_info, (train_df, val_df, test_df)

# ====================
# START OF evaluate.py
# ====================


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
        eval_out = model.evaluate(val_ds, verbose=0)
        val_acc = eval_out[1] # [loss, accuracy, ...]
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
    eval_out = model.evaluate(test_ds, verbose=0)
    test_loss, test_acc = eval_out[0], eval_out[1]
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

# ====================
# START OF train.py
# ====================


import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf

import sys
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



class ConciseLogging(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch+1:03d} | loss: {logs.get('loss', 0):.4f} | acc: {logs.get('accuracy', 0):.4f} | val_loss: {logs.get('val_loss', 0):.4f} | val_acc: {logs.get('val_accuracy', 0):.4f}"
        print(msg)


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
        ConciseLogging(),
    ]

    warmup_start = time.perf_counter()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.warmup_epochs, callbacks=warmup_callbacks, verbose=0)
    warmup_seconds = time.perf_counter() - warmup_start

    unfrozen = unfreeze_top_layers(base_model, config.finetune_unfreeze_last_n)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.finetune_lr), loss=focal, metrics=_metrics())

    finetune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(output_dir / 'best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.CSVLogger(output_dir / 'training_log.csv'),
        ConciseLogging(),
    ]

    finetune_start = time.perf_counter()
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=config.finetune_epochs, callbacks=finetune_callbacks, verbose=0)
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
    # With oversampling, the total images used for training may exceed the original dataset count.
    assert split_info['train_size'] + split_info['val_size'] + split_info['test_size'] >= split_info['total_labeled_images']

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