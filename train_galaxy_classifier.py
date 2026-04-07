#!/usr/bin/env python3
"""CPU-efficient Galaxy Zoo classifier training pipeline."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from imblearn.over_sampling import RandomOverSampler

matplotlib.use("Agg")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_CLASSES = ("spiral", "elliptical", "irregular")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_images(data_dir: Path) -> Tuple[List[Path], List[str]]:
    image_paths: List[Path] = []
    labels: List[str] = []

    for class_name in TARGET_CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in VALID_EXTENSIONS and path.is_file():
                image_paths.append(path)
                labels.append(class_name)

    if not image_paths:
        raise ValueError(
            f"No images found in {data_dir}. Expected folders: {', '.join(TARGET_CLASSES)}"
        )
    return image_paths, labels


def ensure_unzipped(zip_path: Path, extract_to: Path) -> Path:
    """Unzips a file if the destination doesn't already exist. Returns extracted directory or file."""
    import zipfile
    
    # If the file already exists (unzipped), just return early
    if extract_to.exists() and (extract_to.is_dir() or extract_to.stat().st_size > 0):
        print(f"File already exists at {extract_to}")
        return extract_to
    
    if not zip_path.exists():
        print(f"Zip not found at {zip_path}, skipping extraction.")
        return extract_to

    print(f"Extracting {zip_path} to {extract_to.parent}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        names = zip_ref.namelist()
        print(f"Zip contents: {names[:5]}... (total {len(names)} files)")
        zip_ref.extractall(extract_to.parent)
    
    if extract_to.exists():
        print(f"Successful extraction to {extract_to}")
    else:
        print(f"Warning: {extract_to} not found after extraction. Actual contents of {extract_to.parent}:")
        print(list(extract_to.parent.iterdir()))

    return extract_to


def find_kaggle_file(pattern: str) -> Path | None:
    """Recursively search for a file in /kaggle/input."""
    root = Path("/kaggle/input")
    for path in root.rglob(pattern):
        return path
    return None


def determine_class(row: pd.Series) -> str:
    """Classification logic from the Galaxy Zoo challenge decision tree."""
    # If the crowd strongly voted that the galaxy has odd/irregular features
    if row['Class6.1'] > 0.5:
        return 'irregular'
    # If the crowd strongly voted that the galaxy is smooth and rounded
    elif row['Class1.1'] > 0.5:
        return 'elliptical'
    # If the crowd strongly voted it has a disk AND has spiral arms
    elif row['Class1.2'] > 0.5 and row['Class4.1'] > 0.5:
        return 'spiral'
    else:
        # If the galaxy is ambiguous or a star/artifact, we skip it
        return 'unknown'


def list_images_from_csv(csv_path: Path, image_dir: Path) -> Tuple[List[Path], List[str]]:
    """Lists images based on a CSV manifest. Handles both preprocessed and raw Kaggle CSVs."""
    import pandas as pd
    print(f"Loading image manifest from {csv_path}...")
    df = pd.read_csv(csv_path)

    # If it's a raw Kaggle CSV, we need to apply the classification logic first
    if 'filename' not in df.columns or 'label' not in df.columns:
        if 'GalaxyID' in df.columns:
            print("Raw Kaggle CSV detected. Applying classification logic...")
            df['label'] = df.apply(determine_class, axis=1)
            df['filename'] = df['GalaxyID'].astype(str) + '.jpg'
            # Filter out 'unknown'
            df = df[df['label'] != 'unknown'].copy()
            print(f"Processed raw CSV: {df['label'].value_counts().to_dict()}")
        else:
            raise ValueError("CSV must contain 'filename' and 'label' columns, or be a raw Kaggle CSV with 'GalaxyID'.")

    image_paths: List[Path] = []
    labels: List[str] = []

    for _, row in df.iterrows():
        img_path = image_dir / row['filename']
        if img_path.exists():
            image_paths.append(img_path)
            labels.append(row['label'])

    if not image_paths:
        raise ValueError(f"No valid images matching the CSV were found in {image_dir}")

    return image_paths, labels


def load_images(
    image_paths: Sequence[Path], labels: Sequence[str], image_size: int
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[str, int]]:
    class_to_idx = {name: i for i, name in enumerate(TARGET_CLASSES)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    x_data: List[np.ndarray] = []
    y_data: List[int] = []

    for path, label in zip(image_paths, labels):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        x_data.append(image)
        y_data.append(class_to_idx[label])

    if not x_data:
        raise ValueError("Images could not be decoded. Please verify your dataset.")

    x = np.array(x_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int64)
    return x, y, idx_to_class, class_to_idx


def split_dataset(
    x: np.ndarray, y: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Block 1 - 32 filters
    model.add(layers.Conv2D(32, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())

    # Block 2 - 64 filters
    model.add(layers.Conv2D(64, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())

    # Block 3 - 128 filters
    model.add(layers.Conv2D(128, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())

    # Block 4 - 256 filters
    model.add(layers.Conv2D(256, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(256, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D())

    # Global Head
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense Layer 1
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.4))
    
    # Dense Layer 2
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.3))
    
    # Output
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history: tf.keras.callbacks.History, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history.get("loss", []), label="Train Loss")
    axes[0].plot(history.history.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history.get("accuracy", []), label="Train Accuracy")
    axes[1].plot(history.history.get("val_accuracy", []), label="Val Accuracy")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_class_metrics(report_dict: dict, class_names: Sequence[str], out_path: Path) -> None:
    """Creates a grouped bar chart for precision, recall, and F1-score per class."""
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in metrics:
        values = [report_dict[cls][metric] for cls in class_names]
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=metric.capitalize())
        ax.bar_label(rects, padding=3, fmt="%.2f")
        multiplier += 1

    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics by Class")
    ax.set_xticks(x + width, class_names)
    ax.legend(loc="lower right", ncol=3)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def predict_with_tta(model: tf.keras.Model, x: np.ndarray, n_augments: int = 8) -> np.ndarray:
    """Performs inference with Test-Time Augmentation by averaging multiple augmented passes."""
    all_preds = []
    for _ in range(n_augments):
        # Apply random rigid transformations (flips + 90-deg rotations)
        x_aug = tf.image.random_flip_left_right(x)
        x_aug = tf.image.random_flip_up_down(x_aug)
        x_aug = tf.image.rot90(x_aug, k=random.randint(0, 3))
        
        preds = model.predict(x_aug, verbose=0)
        all_preds.append(preds)
    
    return np.mean(all_preds, axis=0)


def measure_inference_time(
    model: tf.keras.Model, sample: np.ndarray, runs: int = 20, use_tta: bool = False
) -> float:
    sample_batch = np.expand_dims(sample, axis=0)
    start = time.perf_counter()
    for _ in range(runs):
        if use_tta:
            _ = predict_with_tta(model, sample_batch, n_augments=8)
        else:
            _ = model.predict(sample_batch, verbose=0)
    end = time.perf_counter()
    return (end - start) / runs


def get_augmenter() -> tf.keras.Model:
    """Creates a Sequential model for real-time image augmentation."""
    return models.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.5), # ±180 degrees
            layers.RandomBrightness(0.15), # ±15%
            layers.RandomZoom(height_factor=0.2, width_factor=0.2), # 0-20% shift
        ]
    )


def prepare_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augmenter: tf.keras.Model | None = None,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Wraps NumPy arrays into a batched, prefetched tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    if augmenter is not None:
        # We use augmenter(x, training=True) to ensure layers like RandomFlip work correctly
        ds = ds.map(
            lambda x_img, y_lbl: (augmenter(x_img, training=True), y_lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU-efficient galaxy classifier.")

    # Detection logic for Kaggle environment
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    # Based on diagnostics, competition data is often in /kaggle/input/competitions/
    kaggle_input = Path("/kaggle/input/competitions/galaxy-zoo-the-galaxy-challenge")
    if not kaggle_input.exists():
        kaggle_input = Path("/kaggle/input/galaxy-zoo-the-galaxy-challenge")
    
    # Kaggle competition specific zips
    kaggle_csv_zip = kaggle_input / "training_solutions_rev1.zip"
    kaggle_img_zip = kaggle_input / "images_training_rev1.zip"
    
    # Determine defaults based on zip existence or unzipped paths
    if is_kaggle:
        default_csv = Path("/kaggle/temp/training_solutions_rev1.csv")
        default_img = Path("/kaggle/temp/images_training_rev1")
        default_out = Path("/kaggle/working/outputs")
    else:
        default_csv = None
        default_img = None
        default_out = Path("outputs")

    # Standard subfolder structure
    parser.add_argument("--data-dir", type=Path, help="Dataset root folder (contains class subfolders)")
    
    # Kaggle-style flat structure
    parser.add_argument("--labels-csv", type=Path, default=default_csv, help="CSV manifest for flat image structure")
    parser.add_argument("--image-dir", type=Path, default=default_img, help="Flat directory for images")
    
    # Kaggle zip paths (internal help)
    parser.add_argument("--labels-csv-zip", type=Path, default=kaggle_csv_zip, help=argparse.SUPPRESS)
    parser.add_argument("--image-dir-zip", type=Path, default=kaggle_img_zip, help=argparse.SUPPRESS)

    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--image-size", type=int, default=96, choices=[64, 96, 128])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0, help="0 means no cap")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    # Detection logic for Kaggle environment
    is_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    
    if not is_kaggle:
        # Locally, we stick to CPU to keep the local machine responsive
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], "GPU")
        print("Running on Local CPU mode.")
    else:
        print("Running on Kaggle - GPU acceleration enabled (if available).")

    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Automatic Kaggle zip handling
    if is_kaggle:
        print("--- Kaggle Environment Diagnostics ---")
        print(f"Current Directory: {os.getcwd()}")
        print("Listing /kaggle/input (recursively):")
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        
        print("Ensuring data is unzipped...")
        Path("/kaggle/temp").mkdir(exist_ok=True)
        
        # Robust discovery of zips
        csv_zip = args.labels_csv_zip or find_kaggle_file("training_solutions_rev1.zip")
        img_zip = args.image_dir_zip or find_kaggle_file("images_training_rev1.zip")
        
        print(f"Discovered CSV Zip: {csv_zip}")
        print(f"Discovered Image Zip: {img_zip}")

        if csv_zip and csv_zip.exists():
            ensure_unzipped(csv_zip, args.labels_csv)
        if img_zip and img_zip.exists():
            ensure_unzipped(img_zip, args.image_dir)

    if args.labels_csv and args.image_dir:
        image_paths, labels = list_images_from_csv(args.labels_csv, args.image_dir)
    elif args.data_dir:
        image_paths, labels = list_images(args.data_dir)
    else:
        raise ValueError("Must provide either --data-dir OR (--labels-csv AND --image-dir)")
    if args.max_images > 0:
        # Combine, shuffle, and unzip to ensure a random mix of classes
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)

        image_paths = list(image_paths)[: args.max_images]
        labels = list(labels)[: args.max_images]

    x, y, idx_to_class, _ = load_images(image_paths, labels, args.image_size)
    x_train_raw, x_val, x_test, y_train_int_raw, y_val_int, y_test_int = split_dataset(x, y, args.seed)

    # Balance the training set using RandomOverSampler
    print(f"Balancing training set (original size: {len(x_train_raw)})...")
    ros = RandomOverSampler(random_state=args.seed)
    # Flatten x_train for the resampler: (N, H, W, C) -> (N, H*W*C)
    x_train_flat = x_train_raw.reshape(x_train_raw.shape[0], -1)
    x_res, y_res = ros.fit_resample(x_train_flat, y_train_int_raw)
    # Reshape back to (N_new, H, W, C)
    x_train = x_res.reshape(-1, args.image_size, args.image_size, 3)
    y_train_int = y_res
    print(f"New training set size: {len(x_train)}")

    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=len(TARGET_CLASSES))
    y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=len(TARGET_CLASSES))
    y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=len(TARGET_CLASSES))

    class_weights_raw = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_int), y=y_train_int
    )
    class_weight = {int(i): float(w) for i, w in enumerate(class_weights_raw)}

    model = build_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=len(TARGET_CLASSES),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_dir / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    # Create optimized data pipelines
    print("Creating tf.data.Dataset pipelines...")
    augmenter = get_augmenter()
    train_ds = prepare_dataset(
        x_train, y_train, args.batch_size, augmenter=augmenter, shuffle=True
    )
    val_ds = prepare_dataset(x_val, y_val, args.batch_size, augmenter=None, shuffle=False)

    train_start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks,
    )
    train_seconds = time.perf_counter() - train_start

    print("Evaluating with Test-Time Augmentation (TTA)...")
    y_pred_probs = predict_with_tta(model, x_test, n_augments=8)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    cm = confusion_matrix(y_test_int, y_pred)
    report = classification_report(
        y_test_int,
        y_pred,
        target_names=[idx_to_class[i] for i in range(len(TARGET_CLASSES))],
        digits=4,
        output_dict=True,
    )

    model_path = args.output_dir / "galaxy_cnn.keras"
    curves_path = args.output_dir / "training_curves.png"
    cm_path = args.output_dir / "confusion_matrix.png"
    class_metrics_path = args.output_dir / "class_metrics.png"
    metrics_path = args.output_dir / "metrics.json"

    model.save(model_path)
    plot_training_curves(history, curves_path)
    plot_confusion_matrix(cm, [idx_to_class[i] for i in range(len(TARGET_CLASSES))], cm_path)
    plot_class_metrics(report, [idx_to_class[i] for i in range(len(TARGET_CLASSES))], class_metrics_path)

    # Standard inference timing
    inference_seconds = measure_inference_time(model, x_test[0], use_tta=False)
    # TTA inference timing
    tta_inference_seconds = measure_inference_time(model, x_test[0], use_tta=True)

    metrics = {
        "dataset": {
            "total_images": int(len(x)),
            "train_size": int(len(x_train)),
            "val_size": int(len(x_val)),
            "test_size": int(len(x_test)),
            "classes": idx_to_class,
            "image_size": args.image_size,
        },
        "training": {
            "epochs_requested": int(args.epochs),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "batch_size": int(args.batch_size),
            "train_time_seconds": train_seconds,
        },
        "evaluation": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "class_metrics_plot": str(class_metrics_path),
        },
        "inference": {
            "seconds_per_image": inference_seconds,
            "seconds_per_image_tta": tta_inference_seconds,
        },
        "model": {
            "parameter_count": int(model.count_params()),
            "model_path": str(model_path),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Training time: {train_seconds / 60:.2f} minutes")
    print(f"Inference latency: {inference_seconds * 1000:.2f} ms/image")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
