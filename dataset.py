from __future__ import annotations

import math
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


# ══════════════════════════════════════════════════════════════
# LABEL DETERMINATION (Regression - 37 Targets)
# ══════════════════════════════════════════════════════════════

def generate_labels_df(solutions_csv: Path, image_dir: Path) -> pd.DataFrame:
    """Generate 37-target DataFrame from Galaxy Zoo solutions CSV for Regression."""
    df = pd.read_csv(solutions_csv)

    # Automatically extract all 37 Class probability columns
    target_cols = [col for col in df.columns if col.startswith('Class')]
    if len(target_cols) != 37:
        print(f"WARNING: Expected 37 Class columns, found {len(target_cols)}")

    df['image_path'] = df['GalaxyID'].astype(str).apply(lambda gid: str(image_dir / f'{gid}.jpg'))
    
    # Check physical file existence to prune missing downloads
    df = df[df['image_path'].apply(lambda p: Path(p).exists())].copy()
    
    keep_cols = ['image_path'] + target_cols
    return df[keep_cols], target_cols


# ══════════════════════════════════════════════════════════════
# IMAGE LOADING & BASE PREPROCESSING
# ══════════════════════════════════════════════════════════════

def _smart_crop(image: tf.Tensor, ratio: float = 0.75) -> tf.Tensor:
    """Crops the image around its brightness centroid instead of the static center.
    
    Uses a Gaussian center prior to avoid latching onto background stars.
    """
    shape = tf.shape(image)
    h_int, w_int = shape[0], shape[1]
    h, w = tf.cast(h_int, tf.float32), tf.cast(w_int, tf.float32)
    
    # Calculate centroid using green channel (brightness approximation)
    img_green = image[..., 1]
    
    # Apply Gaussian center prior (sigma^2 = 5000 as in benanne solution)
    yy_grid, xx_grid = tf.meshgrid(tf.range(h_int), tf.range(w_int), indexing='ij')
    yy_grid = tf.cast(yy_grid, tf.float32)
    xx_grid = tf.cast(xx_grid, tf.float32)
    
    prior = tf.exp(-((yy_grid - h/2.0)**2 + (xx_grid - w/2.0)**2) / 5000.0)
    img_weighted = img_green * prior
    
    total_flux = tf.reduce_sum(img_weighted) + 1e-6
    
    # Weighted average coordinates
    cy = tf.reduce_sum(tf.reduce_sum(img_weighted, axis=1) * tf.cast(tf.range(h_int), tf.float32)) / total_flux
    cx = tf.reduce_sum(tf.reduce_sum(img_weighted, axis=0) * tf.cast(tf.range(w_int), tf.float32)) / total_flux
    
    # Clip centroid to keep the crop mostly within image bounds
    cy = tf.clip_by_value(cy, h * 0.1, h * 0.9)
    cx = tf.clip_by_value(cx, w * 0.1, w * 0.9)
    
    # Crop size
    ch = tf.cast(h * ratio, tf.int32)
    cw = tf.cast(w * ratio, tf.int32)
    
    offset_h = tf.cast(cy - tf.cast(ch, tf.float32) / 2.0, tf.int32)
    offset_w = tf.cast(cx - tf.cast(cw, tf.float32) / 2.0, tf.int32)
    
    # Final clamping to ensure valid crop
    offset_h = tf.clip_by_value(offset_h, 0, h_int - ch)
    offset_w = tf.clip_by_value(offset_w, 0, w_int - cw)
    
    return tf.image.crop_to_bounding_box(image, offset_h, offset_w, ch, cw)


def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor, image_size: int,
                               center_crop_ratio: float = 0.75):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = _smart_crop(img, ratio=center_crop_ratio)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img, label


# ══════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════

def apply_cutout(image: tf.Tensor, n_holes: int = 2, max_size_ratio: float = 0.20) -> tf.Tensor:
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    img_mean = tf.reduce_mean(image)

    mask = tf.ones_like(image)
    for _ in range(n_holes):
        hole_size_h = tf.random.uniform([], minval=tf.cast(tf.cast(h, tf.float32) * 0.05, tf.int32),
                                         maxval=tf.maximum(tf.cast(tf.cast(h, tf.float32) * max_size_ratio, tf.int32), 2),
                                         dtype=tf.int32)
        hole_size_w = tf.random.uniform([], minval=tf.cast(tf.cast(w, tf.float32) * 0.05, tf.int32),
                                         maxval=tf.maximum(tf.cast(tf.cast(w, tf.float32) * max_size_ratio, tf.int32), 2),
                                         dtype=tf.int32)
        cy = tf.random.uniform([], minval=hole_size_h // 2, maxval=h - hole_size_h // 2, dtype=tf.int32)
        cx = tf.random.uniform([], minval=hole_size_w // 2, maxval=w - hole_size_w // 2, dtype=tf.int32)

        y1 = tf.maximum(cy - hole_size_h // 2, 0)
        y2 = tf.minimum(cy + hole_size_h // 2, h)
        x1 = tf.maximum(cx - hole_size_w // 2, 0)
        x2 = tf.minimum(cx + hole_size_w // 2, w)

        top_pad = y1
        bottom_pad = h - y2
        left_pad = x1
        right_pad = w - x2
        hole_h = y2 - y1
        hole_w = x2 - x1

        hole = tf.zeros([hole_h, hole_w, 3])
        hole = tf.pad(hole, [[top_pad, bottom_pad], [left_pad, right_pad], [0, 0]], constant_values=1.0)
        mask = mask * hole

    image = image * mask + img_mean * (1.0 - mask)
    return image


def apply_poisson_noise(image: tf.Tensor, scale: float = 25.0) -> tf.Tensor:
    img_norm = image / 255.0
    img_scaled = img_norm * scale
    noisy = tf.random.poisson(shape=[], lam=tf.maximum(img_scaled, 1e-6))
    noisy = noisy / scale * 255.0
    return tf.clip_by_value(noisy, 0.0, 255.0)


def apply_gaussian_blur(image: tf.Tensor, sigma: float = 1.0) -> tf.Tensor:
    kernel_size = tf.cast(tf.math.ceil(sigma * 3.0) * 2 + 1, tf.int32)
    kernel_size = tf.maximum(kernel_size, 3)
    half = tf.cast(kernel_size // 2, tf.float32)
    x = tf.range(-half, half + 1, dtype=tf.float32)
    kernel_1d = tf.exp(-x ** 2 / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis] 
    kernel_2d = tf.tile(kernel_2d, [1, 1, 3, 1]) 
    
    image_4d = tf.expand_dims(image, 0)
    blurred = tf.nn.depthwise_conv2d(image_4d, kernel_2d, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(blurred, 0)


def apply_continuous_rotation(image: tf.Tensor) -> tf.Tensor:
    if tfa is not None:
        angle = tf.random.uniform([], 0.0, 2.0 * math.pi)
        return tfa.image.rotate(image, angle, interpolation='BILINEAR')
    else:
        angle = tf.random.uniform([], 0.0, 2.0 * math.pi)
        cos_a = tf.cos(angle)
        sin_a = tf.sin(angle)
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        cx, cy = w / 2.0, h / 2.0
        a2 = cx - cx * cos_a - cy * sin_a
        b2 = cy + cx * sin_a - cy * cos_a
        transform = [cos_a, sin_a, a2, -sin_a, cos_a, b2, 0.0, 0.0]
        transform = tf.cast(tf.stack(transform), tf.float32)
        transform = tf.reshape(transform, [1, 8])
        image_4d = tf.expand_dims(image, 0)
        rotated = tf.raw_ops.ImageProjectiveTransformV3(
            images=image_4d,
            transforms=transform,
            output_shape=tf.shape(image)[:2],
            fill_value=0.0,
            interpolation='BILINEAR',
            fill_mode='REFLECT',
        )
        return tf.squeeze(rotated, 0)


def augment_image(image: tf.Tensor, label: tf.Tensor):
    shape = tf.shape(image)
    orig_h, orig_w = shape[0], shape[1]

    image = apply_continuous_rotation(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if tf.random.uniform([]) < 0.3:
        image = apply_poisson_noise(image, scale=25.0)

    if tf.random.uniform([]) < 0.2:
        sigma = tf.random.uniform([], 0.5, 2.0)
        image = apply_gaussian_blur(image, sigma)

    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.15 * 255.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

    crop_h = tf.cast(tf.cast(orig_h, tf.float32) * 0.9, tf.int32)
    crop_w = tf.cast(tf.cast(orig_w, tf.float32) * 0.9, tf.int32)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, [orig_h, orig_w])

    if tf.random.uniform([]) > 0.7:
        image = apply_cutout(image, n_holes=1, max_size_ratio=0.15)

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


# ══════════════════════════════════════════════════════════════
# DATASET BUILDERS
# ══════════════════════════════════════════════════════════════

def build_dataset(
    image_paths: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    center_crop_ratio: float = 0.75,
    augment: bool = False,
    shuffle: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset mapping paths and 37-node target vectors."""
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(
        lambda p, y: load_and_preprocess_image(p, y, image_size, center_crop_ratio),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 10000), reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(lambda i, l: augment_image(i, l), num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ══════════════════════════════════════════════════════════════
# SPLITTING & ORCHESTRATION
# ══════════════════════════════════════════════════════════════

def build_datasets(labels_df: pd.DataFrame, config: Config):
    """Split data randomly for regression."""
    train_df, temp_df = train_test_split(
        labels_df,
        test_size=config.val_split + config.test_split,
        random_state=config.seed,
    )
    test_ratio_of_temp = config.test_split / (config.val_split + config.test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio_of_temp,
        random_state=config.seed,
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_labeled_images': len(labels_df),
    }

    return split_info, (train_df, val_df, test_df)
