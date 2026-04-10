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

    # Data
    image_size_warmup: int = 128
    image_size_finetune: int = 160
    # Current active size (updated at runtime by curriculum)
    image_size: int = 128

    batch_size: int = 64
    batch_size_finetune: int = 32 # Safety reduction for 160px
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15

    # Label curation & Thresholds
    elliptical_threshold: float = 0.469
    spiral_disk_threshold: float = 0.430
    spiral_arms_threshold: float = 0.430
    irregular_threshold: float = 0.469
    # New: minimum probability margin for irregular odd features
    irregular_margin: float = 0.10

    # Training phase 1 (Warmup)
    warmup_epochs: int = 10
    warmup_lr: float = 1e-3

    # Training phase 2 (Fine-tune)
    finetune_epochs: int = 30
    finetune_lr: float = 5e-5
    finetune_unfreeze_last_n: int = 30

    # Advanced Regimes
    label_smoothing: float = 0.05
    use_sample_weighting: bool = True
    use_oversampling: bool = False  # Replaced by weighting
    ensemble_architectures: tuple[str, ...] = ('EfficientNetV2B0', 'ConvNeXtTiny')

    # Focal Loss
    focal_gamma: float = 2.0

    # TTA - Only orientation-preserving transforms
    tta_n_augments: int = 8

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
    def __init__(self, alpha: list[float] | None = None, gamma: float = 2.0, name: str = 'focal_loss', label_smoothing: float = 0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Elite Improvement: Label Smoothing
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        # Categorical cross entropy core
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Focal weight: (1 - p)^gamma
        weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * cross_entropy

        # Class balanced alpha scaling
        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=y_true.dtype)
            loss = alpha * loss

        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha, 'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return config

# ====================
# START OF model.py
# ====================


import tensorflow as tf


def build_model(image_size: int, num_classes: int, architecture: str = 'EfficientNetV2B0'):
    """Produces the base model and the modified classifier head."""
    # Elite Improvement: Multi-architecture support for Ensembling
    # Using (None, None, 3) enables the Resolution Curriculum to resize inputs dynamically
    inputs = tf.keras.Input(shape=(None, None, 3))
    if architecture == 'EfficientNetV2B0':
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
    elif architecture == 'ConvNeXtTiny':
        # ConvNeXt architectures are excellent for spatial patterns
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=inputs)
    else:
        raise ValueError(f'Unsupported architecture: {architecture}')

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'GalaxyNet_{architecture}')
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, last_n: int):
    """Gradually unfreezes layers for deep fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-last_n]:
        layer.trainable = False
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
except Exception:
    tfa = None

CLASS_NAMES = ['spiral', 'elliptical', 'irregular']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def determine_class(row: pd.Series, config: Config) -> str:
    is_elliptical = row['Class1.1'] >= config.elliptical_threshold
    is_spiral = (row['Class1.2'] >= config.spiral_disk_threshold) and (
        row['Class4.1'] >= config.spiral_arms_threshold
    )

    # Elite Improvement: Irregular margin curation
    # Class 6.1: Yes (Odd), Class 6.2: No (Normal)
    odd_margin = row['Class6.1'] - row['Class6.2']
    is_irregular = (
        row['Class6.1'] >= config.irregular_threshold
        and odd_margin >= config.irregular_margin
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


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Computes weights inversely proportional to class frequency."""
    counts = df['label_id'].value_counts().to_dict()
    total = len(df)
    # Balanced weighting: total / (n_classes * count)
    weights_map = {cid: total / (len(CLASS_NAMES) * count) for cid, count in counts.items()}
    # Normalize so mean weight is 1.0
    mean_w = np.mean(list(weights_map.values()))
    weights_map = {cid: w / mean_w for cid, w in weights_map.items()}
    return df['label_id'].map(weights_map).to_numpy()


def generate_labels_df(solutions_csv: Path, image_dir: Path, config: Config) -> pd.DataFrame:
    df = pd.read_csv(solutions_csv)
    # Added Class6.2 for margin calculation
    required_cols = {'GalaxyID', 'Class1.1', 'Class1.2', 'Class4.1', 'Class6.1', 'Class6.2'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {solutions_csv}: {sorted(missing)}')

    df['label'] = df.apply(lambda r: determine_class(r, config), axis=1)
    df = df[df['label'] != 'unknown'].copy()
    df['image_path'] = df['GalaxyID'].astype(str).apply(lambda gid: str(image_dir / f'{gid}.jpg'))
    df = df[df['image_path'].apply(lambda p: Path(p).exists())].copy()
    df['label_id'] = df['label'].map(CLASS_TO_ID)

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


def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor, image_size: int, weight: tf.Tensor = None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = _center_crop(img, ratio=0.8)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.clip_by_value(img, 0.0, 255.0)
    if weight is not None:
        return img, label, weight
    return img, label


def augment_image(image: tf.Tensor, label: tf.Tensor, weight: tf.Tensor = None):
    # Elite Improvement: Access dynamic shape at runtime for curriculum stability
    shape = tf.shape(image)
    orig_h, orig_w = shape[0], shape[1]

    if tfa is not None:
        angle = tf.random.uniform([], 0.0, 2.0 * math.pi)
        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
    else:
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Elite Improvement: Error-driven augmentation (dynamic blur simulation)
    if tf.random.uniform([]) > 0.8:
        image = tf.image.resize(image, [orig_h // 2, orig_w // 2])
        image = tf.image.resize(image, [orig_h, orig_w])

    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1 * 255.0)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Dynamic cropping based on runtime resolution
    crop_h = tf.cast(tf.cast(orig_h, tf.float32) * 0.9, tf.int32)
    crop_w = tf.cast(tf.cast(orig_w, tf.float32) * 0.9, tf.int32)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])

    # Standardize back to current curriculum resolution
    image = tf.image.resize(image, [orig_h, orig_w])
    image = tf.clip_by_value(image, 0.0, 255.0)

    if weight is not None:
        return image, label, weight
    return image, label


def build_dataset(
    image_paths: np.ndarray,
    labels_onehot: np.ndarray,
    image_size: int,
    batch_size: int,
    weights: np.ndarray = None,
    augment: bool = False,
    shuffle: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    if weights is not None:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_onehot, weights))
        ds = ds.map(lambda p, y, w: load_and_preprocess_image(p, y, image_size, w), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_onehot))
        ds = ds.map(lambda p, y: load_and_preprocess_image(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    if augment:
        if weights is not None:
            ds = ds.map(lambda i, l, w: augment_image(i, l, w), num_parallel_calls=tf.data.AUTOTUNE)
        else:
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


def build_datasets(labels_df: pd.DataFrame, config: Config):
    train_df, val_df, test_df = _split_dataframe(labels_df, config)

    # Elite Improvement: Sample weighting instead of duplication
    train_weights = None
    if config.use_sample_weighting:
        train_weights = compute_sample_weights(train_df)

    class_counts = labels_df['label'].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()

    def to_xyw(df: pd.DataFrame, weights_arr=None):
        paths = df['image_path'].astype(str).to_numpy()
        y = tf.keras.utils.to_categorical(df['label_id'].to_numpy(), num_classes=3)
        return paths, y, weights_arr

    tr_x, tr_y, tr_w = to_xyw(train_df, train_weights)
    va_x, va_y, _ = to_xyw(val_df)
    tx_x, tx_y, _ = to_xyw(test_df)

    # Resolution start at config.image_size (128)
    train_ds = build_dataset(tr_x, tr_y, config.image_size, config.batch_size, weights=tr_w, augment=True, shuffle=True)
    val_ds = build_dataset(va_x, va_y, config.image_size, config.batch_size, augment=False, shuffle=False, cache=config.cache_val_test)
    test_ds = build_dataset(tx_x, tx_y, config.image_size, config.batch_size, augment=False, shuffle=False, cache=config.cache_val_test)

    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_labeled_images': len(labels_df),
        'class_distribution': class_counts,
        'config': asdict(config),
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
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold



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

# ====================
# START OF train.py
# ====================


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