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

# Binary label mappings for cascade stages
STAGE1_CLASSES = ['non_elliptical', 'elliptical']  # 0=non_elliptical, 1=elliptical
STAGE2_CLASSES = ['irregular', 'spiral']            # 0=irregular, 1=spiral


# ══════════════════════════════════════════════════════════════
# LABEL DETERMINATION (tightened for V25)
# ══════════════════════════════════════════════════════════════

def determine_class(row: pd.Series, config: Config) -> str:
    """Determine galaxy morphology class using tightened V25 thresholds."""
    is_elliptical = row['Class1.1'] >= config.elliptical_threshold
    is_spiral = (row['Class1.2'] >= config.spiral_disk_threshold) and (
        row['Class4.1'] >= config.spiral_arms_threshold
    )

    # V25: Tightened irregular label curation
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


def compute_sample_weights(df: pd.DataFrame, label_col: str = 'label_id', n_classes: int = 3) -> np.ndarray:
    """Computes weights inversely proportional to class frequency."""
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    weights_map = {cid: total / (n_classes * count) for cid, count in counts.items()}
    mean_w = np.mean(list(weights_map.values()))
    weights_map = {cid: w / mean_w for cid, w in weights_map.items()}
    return df[label_col].map(weights_map).to_numpy()


def generate_labels_df(solutions_csv: Path, image_dir: Path, config: Config) -> pd.DataFrame:
    """Generate 3-class labels DataFrame from Galaxy Zoo solutions CSV.
    
    V26: Also preserves raw voting fractions for soft label training.
    """
    df = pd.read_csv(solutions_csv)
    required_cols = {'GalaxyID', 'Class1.1', 'Class1.2', 'Class4.1', 'Class6.1', 'Class6.2'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {solutions_csv}: {sorted(missing)}')

    df['label'] = df.apply(lambda r: determine_class(r, config), axis=1)
    df = df[df['label'] != 'unknown'].copy()
    df['image_path'] = df['GalaxyID'].astype(str).apply(lambda gid: str(image_dir / f'{gid}.jpg'))
    df = df[df['image_path'].apply(lambda p: Path(p).exists())].copy()
    df['label_id'] = df['label'].map(CLASS_TO_ID)

    # V26: Preserve raw voting fractions for soft label training
    keep_cols = ['GalaxyID', 'image_path', 'label', 'label_id',
                 'Class1.1', 'Class1.2', 'Class4.1', 'Class6.1', 'Class6.2']
    return df[keep_cols]


def validate_class_counts(labels_df: pd.DataFrame, config: Config) -> None:
    """Validate minimum class counts after threshold tightening."""
    counts = labels_df['label'].value_counts()
    print('\n=== Class Distribution After V25 Threshold Tightening ===')
    for cls_name in CLASS_NAMES:
        cnt = counts.get(cls_name, 0)
        print(f'  {cls_name:>12s}: {cnt:>6d}')
    print(f'  {"TOTAL":>12s}: {len(labels_df):>6d}\n')

    min_counts = {
        'irregular': config.min_irregular_count,
        'spiral': config.min_spiral_count,
        'elliptical': config.min_elliptical_count,
    }
    for cls_name, min_count in min_counts.items():
        actual = counts.get(cls_name, 0)
        if actual < min_count:
            print(f'WARNING: {cls_name} count ({actual}) below minimum ({min_count}). '
                  f'Consider relaxing threshold by 0.02.')


def compute_alpha_from_counts(class_counts: dict[str, int]) -> list[float]:
    counts = np.array([class_counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float64)
    total = counts.sum()
    alpha = total / (len(CLASS_NAMES) * counts)
    alpha = alpha / alpha.sum()
    return alpha.tolist()


# ══════════════════════════════════════════════════════════════
# IMAGE LOADING & BASE PREPROCESSING
# ══════════════════════════════════════════════════════════════

def _center_crop(image: tf.Tensor, ratio: float = 0.75) -> tf.Tensor:
    """Center crop with configurable ratio (V25: 0.75 for tighter crop at 224px)."""
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    ch = tf.cast(tf.cast(h, tf.float32) * ratio, tf.int32)
    cw = tf.cast(tf.cast(w, tf.float32) * ratio, tf.int32)
    offset_h = (h - ch) // 2
    offset_w = (w - cw) // 2
    return tf.image.crop_to_bounding_box(image, offset_h, offset_w, ch, cw)


def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor, image_size: int,
                               center_crop_ratio: float = 0.75, weight: tf.Tensor = None):
    """Load JPEG, center crop, resize to target size."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = _center_crop(img, ratio=center_crop_ratio)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.clip_by_value(img, 0.0, 255.0)
    if weight is not None:
        return img, label, weight
    return img, label


# ══════════════════════════════════════════════════════════════
# AUGMENTATION — V25 CLASS-DIFFERENTIATED + MIXUP + CUTOUT
# ══════════════════════════════════════════════════════════════

def apply_cutout(image: tf.Tensor, n_holes: int = 2, max_size_ratio: float = 0.20) -> tf.Tensor:
    """Apply random square cutouts filled with image mean to force distributed feature learning."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    img_mean = tf.reduce_mean(image)

    mask = tf.ones_like(image)
    for _ in range(n_holes):
        # Random hole size between 5% and max_size_ratio of image
        hole_size_h = tf.random.uniform([], minval=tf.cast(tf.cast(h, tf.float32) * 0.05, tf.int32),
                                         maxval=tf.maximum(tf.cast(tf.cast(h, tf.float32) * max_size_ratio, tf.int32), 2),
                                         dtype=tf.int32)
        hole_size_w = tf.random.uniform([], minval=tf.cast(tf.cast(w, tf.float32) * 0.05, tf.int32),
                                         maxval=tf.maximum(tf.cast(tf.cast(w, tf.float32) * max_size_ratio, tf.int32), 2),
                                         dtype=tf.int32)
        # Random center position
        cy = tf.random.uniform([], minval=hole_size_h // 2, maxval=h - hole_size_h // 2, dtype=tf.int32)
        cx = tf.random.uniform([], minval=hole_size_w // 2, maxval=w - hole_size_w // 2, dtype=tf.int32)

        y1 = tf.maximum(cy - hole_size_h // 2, 0)
        y2 = tf.minimum(cy + hole_size_h // 2, h)
        x1 = tf.maximum(cx - hole_size_w // 2, 0)
        x2 = tf.minimum(cx + hole_size_w // 2, w)

        # Create hole mask using padding approach
        top_pad = y1
        bottom_pad = h - y2
        left_pad = x1
        right_pad = w - x2
        hole_h = y2 - y1
        hole_w = x2 - x1

        hole = tf.zeros([hole_h, hole_w, 3])
        hole = tf.pad(hole, [[top_pad, bottom_pad], [left_pad, right_pad], [0, 0]], constant_values=1.0)
        mask = mask * hole

    # Apply mask: replace cutout regions with image mean
    image = image * mask + img_mean * (1.0 - mask)
    return image


# ── V26: ASTRONOMY-SPECIFIC AUGMENTATIONS ──

def apply_poisson_noise(image: tf.Tensor, scale: float = 25.0) -> tf.Tensor:
    """Simulate CCD detector shot noise via Poisson process.
    
    Real telescope images have photon counting noise proportional to sqrt(signal).
    Scale controls SNR: higher = less noise (brighter source simulation).
    """
    # Normalize to [0, 1], apply Poisson, scale back
    img_norm = image / 255.0
    img_scaled = img_norm * scale
    # Poisson noise: output has same expected value but with shot noise
    noisy = tf.random.poisson(shape=[], lam=tf.maximum(img_scaled, 1e-6))
    noisy = noisy / scale * 255.0
    return tf.clip_by_value(noisy, 0.0, 255.0)


def apply_gaussian_blur(image: tf.Tensor, sigma: float = 1.0) -> tf.Tensor:
    """Simulate atmospheric PSF (Point Spread Function) via Gaussian blur.
    
    Real telescope observations are degraded by atmospheric turbulence ('seeing').
    Sigma controls the blur radius in pixels.
    """
    # Build 2D Gaussian kernel
    kernel_size = tf.cast(tf.math.ceil(sigma * 3.0) * 2 + 1, tf.int32)
    kernel_size = tf.maximum(kernel_size, 3)
    half = tf.cast(kernel_size // 2, tf.float32)
    x = tf.range(-half, half + 1, dtype=tf.float32)
    kernel_1d = tf.exp(-x ** 2 / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]  # [H, W, 1, 1]
    kernel_2d = tf.tile(kernel_2d, [1, 1, 3, 1])  # [H, W, 3, 1]
    
    # Apply depthwise convolution
    image_4d = tf.expand_dims(image, 0)  # [1, H, W, C]
    blurred = tf.nn.depthwise_conv2d(image_4d, kernel_2d, strides=[1, 1, 1, 1], padding='SAME')
    return tf.squeeze(blurred, 0)


def apply_continuous_rotation(image: tf.Tensor) -> tf.Tensor:
    """Apply continuous 0-360° rotation (pure TF, no tfa dependency).
    
    Galaxies have no preferred orientation axis — this is the most
    physically justified augmentation for astronomical images.
    """
    if tfa is not None:
        angle = tf.random.uniform([], 0.0, 2.0 * math.pi)
        return tfa.image.rotate(image, angle, interpolation='BILINEAR')
    else:
        # Pure TF continuous rotation via affine transform
        angle = tf.random.uniform([], 0.0, 2.0 * math.pi)
        cos_a = tf.cos(angle)
        sin_a = tf.sin(angle)
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        # Center-origin affine: translate, rotate, translate back
        cx, cy = w / 2.0, h / 2.0
        # Inverse transform matrix for tf.raw_ops.ImageProjectiveTransformV3
        # [a0, a1, a2, b0, b1, b2, c0, c1] where:
        # x_src = a0*x_dst + a1*y_dst + a2
        # y_src = b0*x_dst + b1*y_dst + b2
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


def augment_image(image: tf.Tensor, label: tf.Tensor, weight: tf.Tensor = None):
    """Standard augmentation with V26 astronomy-specific additions."""
    shape = tf.shape(image)
    orig_h, orig_w = shape[0], shape[1]

    # V26: Continuous rotation (0-360°) — galaxies have no preferred orientation
    image = apply_continuous_rotation(image)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # V26: Poisson CCD noise (30% probability)
    if tf.random.uniform([]) < 0.3:
        image = apply_poisson_noise(image, scale=25.0)

    # V26: PSF blur — atmospheric seeing simulation (20% probability)
    if tf.random.uniform([]) < 0.2:
        sigma = tf.random.uniform([], 0.5, 2.0)
        image = apply_gaussian_blur(image, sigma)

    # Brightness/contrast (50% probability)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.15 * 255.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

    # Random crop 90% and resize back
    crop_h = tf.cast(tf.cast(orig_h, tf.float32) * 0.9, tf.int32)
    crop_w = tf.cast(tf.cast(orig_w, tf.float32) * 0.9, tf.int32)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, [orig_h, orig_w])

    # Cutout (1 hole for standard augmentation)
    if tf.random.uniform([]) > 0.7:
        image = apply_cutout(image, n_holes=1, max_size_ratio=0.15)

    image = tf.clip_by_value(image, 0.0, 255.0)

    if weight is not None:
        return image, label, weight
    return image, label


def augment_image_irregular(image: tf.Tensor, label: tf.Tensor, weight: tf.Tensor = None):
    """Aggressive augmentation specifically for irregular galaxy class."""
    shape = tf.shape(image)
    orig_h, orig_w = shape[0], shape[1]

    if tfa is not None:
        angle = tf.random.uniform([], 0.0, 2.0 * math.pi)
        image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
    else:
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Stronger brightness/contrast for irregulars
    image = tf.image.random_brightness(image, max_delta=0.20 * 255.0)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)

    # Stronger zoom (±15%)
    crop_h = tf.cast(tf.cast(orig_h, tf.float32) * tf.random.uniform([], 0.85, 0.95), tf.int32)
    crop_w = tf.cast(tf.cast(orig_w, tf.float32) * tf.random.uniform([], 0.85, 0.95), tf.int32)
    crop_h = tf.maximum(crop_h, 1)
    crop_w = tf.maximum(crop_w, 1)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, [orig_h, orig_w])

    # Multi-hole cutout for irregulars (3 holes, always applied)
    image = apply_cutout(image, n_holes=3, max_size_ratio=0.20)

    # Random erasing (20% probability)
    if tf.random.uniform([]) > 0.8:
        erase_h = tf.random.uniform([], minval=2, maxval=tf.maximum(orig_h // 6, 3), dtype=tf.int32)
        erase_w = tf.random.uniform([], minval=2, maxval=tf.maximum(orig_w // 6, 3), dtype=tf.int32)
        ey = tf.random.uniform([], 0, tf.maximum(orig_h - erase_h, 1), dtype=tf.int32)
        ex = tf.random.uniform([], 0, tf.maximum(orig_w - erase_w, 1), dtype=tf.int32)
        noise = tf.random.uniform([erase_h, erase_w, 3], 0.0, 255.0)
        indices_h = tf.range(ey, ey + erase_h)
        indices_w = tf.range(ex, ex + erase_w)
        # Use scatter-nd approach for random erasing
        y_grid, x_grid = tf.meshgrid(indices_h, indices_w, indexing='ij')
        coords = tf.stack([tf.reshape(y_grid, [-1]), tf.reshape(x_grid, [-1])], axis=1)
        flat_noise = tf.reshape(noise, [-1, 3])
        # Create mask and apply
        mask = tf.ones([orig_h, orig_w, 1])
        mask_updates = tf.zeros([tf.shape(coords)[0], 1])
        mask = tf.tensor_scatter_nd_update(mask, coords, mask_updates)
        noise_full = tf.zeros_like(image)
        noise_full = tf.tensor_scatter_nd_update(noise_full, coords, flat_noise)
        image = image * mask + noise_full * (1.0 - mask)

    image = tf.clip_by_value(image, 0.0, 255.0)

    if weight is not None:
        return image, label, weight
    return image, label


# ══════════════════════════════════════════════════════════════
# MIXUP — WITHIN-CLASS FOR IRREGULARS
# ══════════════════════════════════════════════════════════════

def mixup_batch(images: tf.Tensor, labels: tf.Tensor, alpha: float = 0.4) -> tuple:
    """Apply within-batch Mixup: mix pairs of samples using Beta distribution.
    For within-class usage, all samples should be the same class.
    """
    batch_size = tf.shape(images)[0]
    # Sample lambda from Beta(alpha, alpha)
    lam = tf.random.uniform([], minval=0.3, maxval=0.7)  # Simplified Beta sampling

    # Shuffle indices for pairing
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    mixed_images = lam * images + (1.0 - lam) * shuffled_images
    mixed_labels = lam * labels + (1.0 - lam) * shuffled_labels

    return mixed_images, mixed_labels


# ══════════════════════════════════════════════════════════════
# DATASET BUILDERS — BINARY (CASCADE STAGES)
# ══════════════════════════════════════════════════════════════

def build_dataset(
    image_paths: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    center_crop_ratio: float = 0.75,
    weights: np.ndarray = None,
    augment: bool = False,
    augment_fn=None,
    shuffle: bool = False,
    cache: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset with optional augmentation and sample weighting.
    
    Args:
        labels: Either one-hot (N,C) for categorical or (N,) / (N,1) for binary.
        augment_fn: Custom augmentation function. If None and augment=True, uses default.
    """
    if weights is not None:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, weights))
        ds = ds.map(
            lambda p, y, w: load_and_preprocess_image(p, y, image_size, center_crop_ratio, w),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(
            lambda p, y: load_and_preprocess_image(p, y, image_size, center_crop_ratio),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 10000), reshuffle_each_iteration=True)

    if augment:
        aug_fn = augment_fn if augment_fn is not None else augment_image
        if weights is not None:
            ds = ds.map(lambda i, l, w: aug_fn(i, l, w), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda i, l: aug_fn(i, l), num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_binary_dataset_stage1(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    config: Config,
    augment: bool = False,
    shuffle: bool = False,
    weights: np.ndarray = None,
) -> tf.data.Dataset:
    """Build binary dataset for Stage 1: elliptical (1) vs non-elliptical (0).
    
    V26 Soft Labels: When config.use_soft_labels is True, the label is the
    raw Galaxy Zoo voting fraction Class1.1 (P(elliptical)) instead of a
    hard 0/1 binary. This preserves human uncertainty and reduces overfitting
    to noisy crowd-sourced boundaries.
    
    Note: Stage 1 uses class_weight in fit() for balancing, NOT sample_weight
    in the dataset. This avoids loss-scale instability during resolution jumps.
    """
    paths = df['image_path'].astype(str).to_numpy()

    if config.use_soft_labels and 'Class1.1' in df.columns:
        # Soft label: raw voting fraction P(elliptical) ∈ [0, 1]
        soft_labels = df['Class1.1'].astype(np.float32).to_numpy()
        soft_labels = np.clip(soft_labels, 0.0, 1.0).reshape(-1, 1)
    else:
        # Hard binary labels: 1 = elliptical, 0 = non-elliptical
        soft_labels = (df['label'] == 'elliptical').astype(np.float32).to_numpy()

    return build_dataset(
        paths, soft_labels, image_size, batch_size,
        center_crop_ratio=config.center_crop_ratio,
        weights=weights, augment=augment, shuffle=shuffle,
    )


def build_binary_dataset_stage2(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    config: Config,
    augment: bool = False,
    shuffle: bool = False,
    weights: np.ndarray = None,
    use_irregular_augment: bool = False,
) -> tf.data.Dataset:
    """Build binary dataset for Stage 2: spiral (1) vs irregular (0).
    Only include spiral and irregular samples (no ellipticals).
    
    V26 Soft Labels: When config.use_soft_labels is True, the label is a
    normalized spiral confidence derived from the voting fractions:
      soft_label = Class1.2 / (Class1.2 + Class6.1 + epsilon)
    This captures the degree of 'spiralness' vs 'oddness'.
    """
    # Filter to only spiral + irregular
    mask = df['label'].isin(['spiral', 'irregular'])
    df_filtered = df[mask].copy()

    paths = df_filtered['image_path'].astype(str).to_numpy()

    if config.use_soft_labels and 'Class1.2' in df_filtered.columns and 'Class6.1' in df_filtered.columns:
        # Soft label: normalized spiral confidence
        spiral_vote = df_filtered['Class1.2'].astype(np.float32).to_numpy()
        odd_vote = df_filtered['Class6.1'].astype(np.float32).to_numpy()
        soft_labels = (spiral_vote / (spiral_vote + odd_vote + 1e-7)).reshape(-1, 1)
        soft_labels = np.clip(soft_labels, 0.0, 1.0)
    else:
        # Hard binary labels: 1 = spiral, 0 = irregular
        soft_labels = (df_filtered['label'] == 'spiral').astype(np.float32).to_numpy().reshape(-1, 1)

    # Compute sample weights for the binary imbalanced problem
    # Use hard label assignment for weight computation (soft labels shouldn't affect balancing)
    hard_labels = (df_filtered['label'] == 'spiral').astype(np.float32).to_numpy()
    if weights is None and config.use_sample_weighting:
        n_spiral = (hard_labels == 1).sum()
        n_irregular = (hard_labels == 0).sum()
        weight_irregular = n_spiral / max(n_irregular, 1)
        weight_spiral = 1.0
        weights = np.where(hard_labels == 1, weight_spiral, weight_irregular).astype(np.float32)
        # Normalize to mean=1
        weights = weights / weights.mean()

    # Use stronger augmentation for irregular-heavy pipeline
    aug_fn = augment_image if not use_irregular_augment else None

    return build_dataset(
        paths, soft_labels, image_size, batch_size,
        center_crop_ratio=config.center_crop_ratio,
        weights=weights, augment=augment, augment_fn=aug_fn,
        shuffle=shuffle,
    )


def filter_stage2_training_data(
    train_df: pd.DataFrame,
    stage1_model_path: str,
    config: Config,
) -> pd.DataFrame:
    """Filter training data for Stage 2 using Stage 1 predictions.
    Includes samples predicted as non-elliptical AND borderline ellipticals
    (confidence < stage1_filter_confidence) to make Stage 2 robust.
    """
    from losses import BinaryFocalLoss
    
    model = tf.keras.models.load_model(
        stage1_model_path,
        custom_objects={'BinaryFocalLoss': BinaryFocalLoss},
    )

    paths = train_df['image_path'].astype(str).to_numpy()
    dummy_labels = np.zeros(len(paths), dtype=np.float32)
    ds = build_dataset(
        paths, dummy_labels, config.image_size_phase3, 32,
        center_crop_ratio=config.center_crop_ratio,
    )

    # Get Stage 1 predictions
    preds = model.predict(ds, verbose=0).flatten()

    # Keep samples where P(elliptical) < stage1_filter_confidence
    # This includes clear non-ellipticals + borderline cases
    keep_mask = preds < config.stage1_filter_confidence
    filtered_df = train_df[keep_mask].copy().reset_index(drop=True)

    # Also always include ALL spiral and irregular samples (even if Stage 1 is confident)
    spiral_irregular_mask = train_df['label'].isin(['spiral', 'irregular'])
    must_include = train_df[spiral_irregular_mask & ~keep_mask]
    if len(must_include) > 0:
        filtered_df = pd.concat([filtered_df, must_include], ignore_index=True)

    # Remove any ellipticals that passed through (they contaminate Stage 2)
    # But keep some for robustness (those with low Stage 1 confidence)
    final_mask = filtered_df['label'].isin(['spiral', 'irregular'])
    filtered_df = filtered_df[final_mask].copy().reset_index(drop=True)

    print(f'\nStage 2 Training Set (filtered from Stage 1):')
    print(f'  Total: {len(filtered_df)}')
    print(f'  Spiral: {(filtered_df["label"] == "spiral").sum()}')
    print(f'  Irregular: {(filtered_df["label"] == "irregular").sum()}')

    tf.keras.backend.clear_session()
    del model

    return filtered_df


# ══════════════════════════════════════════════════════════════
# SPLITTING & ORCHESTRATION
# ══════════════════════════════════════════════════════════════

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
    """Split data and return train/val/test DataFrames + class counts.
    Note: Actual tf.data.Dataset construction is deferred to training functions
    to support per-phase resolution curriculum.
    """
    train_df, val_df, test_df = _split_dataframe(labels_df, config)

    class_counts = labels_df['label'].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()

    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_labeled_images': len(labels_df),
        'class_distribution': class_counts,
        'config_snapshot': {
            'irregular_threshold': config.irregular_threshold,
            'irregular_margin': config.irregular_margin,
            'spiral_disk_threshold': config.spiral_disk_threshold,
            'spiral_arms_threshold': config.spiral_arms_threshold,
            'elliptical_threshold': config.elliptical_threshold,
        },
    }

    return class_counts, split_info, (train_df, val_df, test_df)
