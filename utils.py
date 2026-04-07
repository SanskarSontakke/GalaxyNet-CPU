from __future__ import annotations

import json
import os
import random
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from config import Config


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
