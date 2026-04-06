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
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
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


def measure_inference_time(model: tf.keras.Model, sample: np.ndarray, runs: int = 20) -> float:
    sample_batch = np.expand_dims(sample, axis=0)
    start = time.perf_counter()
    for _ in range(runs):
        _ = model.predict(sample_batch, verbose=0)
    end = time.perf_counter()
    return (end - start) / runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU-efficient galaxy classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root folder")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=64, choices=[64, 128])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=0, help="0 means no cap")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], "GPU")

    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = list_images(args.data_dir)
    if args.max_images > 0:
        # Combine, shuffle, and unzip to ensure a random mix of classes
        combined = list(zip(image_paths, labels))
        random.shuffle(combined)
        image_paths, labels = zip(*combined)

        image_paths = list(image_paths)[: args.max_images]
        labels = list(labels)[: args.max_images]

    x, y, idx_to_class, _ = load_images(image_paths, labels, args.image_size)
    x_train, x_val, x_test, y_train_int, y_val_int, y_test_int = split_dataset(x, y, args.seed)

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
            monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True
        )
    ]

    train_start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks,
    )
    train_seconds = time.perf_counter() - train_start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

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
    metrics_path = args.output_dir / "metrics.json"

    model.save(model_path)
    plot_training_curves(history, curves_path)
    plot_confusion_matrix(cm, [idx_to_class[i] for i in range(len(TARGET_CLASSES))], cm_path)

    inference_seconds = measure_inference_time(model, x_test[0])

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
        },
        "inference": {
            "seconds_per_image": inference_seconds,
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
