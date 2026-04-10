from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import Config

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
