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
