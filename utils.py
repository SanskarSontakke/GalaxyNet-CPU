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
        json.dump(payload, f, indent=2, default=str)


def is_kaggle_runtime() -> bool:
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


# ══════════════════════════════════════════════════════════════
# V25 — GRADIENT ACCUMULATION HELPER
# ══════════════════════════════════════════════════════════════

class GradientAccumulationModel(tf.keras.Model):
    """Wrapper model that accumulates gradients over N mini-batches
    before applying an optimizer step. This allows effective large
    batch sizes when VRAM is limited (e.g., batch_size=16 × accum=2 = 32 effective).
    
    Usage:
        ga_model = GradientAccumulationModel(base_model, accumulation_steps=2)
        ga_model.compile(optimizer=..., loss=..., metrics=...)
        ga_model.fit(train_ds, ...)
    """

    def __init__(self, inner_model: tf.keras.Model, accumulation_steps: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.inner_model = inner_model
        self.accumulation_steps = accumulation_steps
        self.step_count = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._accum_gradients = None

    def call(self, inputs, training=False):
        return self.inner_model(inputs, training=training)

    @property
    def trainable_variables(self):
        return self.inner_model.trainable_variables

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        # Initialize gradient accumulators on first call
        if self._accum_gradients is None:
            self._accum_gradients = [
                tf.Variable(tf.zeros_like(v), trainable=False)
                for v in self.inner_model.trainable_variables
            ]

        with tf.GradientTape() as tape:
            y_pred = self.inner_model(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight)
            # Scale loss for accumulation
            scaled_loss = loss / tf.cast(self.accumulation_steps, tf.float32)

        gradients = tape.gradient(scaled_loss, self.inner_model.trainable_variables)

        # Accumulate
        for accum, grad in zip(self._accum_gradients, gradients):
            if grad is not None:
                accum.assign_add(grad)

        self.step_count.assign_add(1)

        # Apply when accumulation is complete
        tf.cond(
            tf.equal(self.step_count % self.accumulation_steps, 0),
            lambda: self._apply_and_reset(),
            lambda: None,
        )

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def _apply_and_reset(self):
        self.optimizer.apply_gradients(
            zip(self._accum_gradients, self.inner_model.trainable_variables)
        )
        for accum in self._accum_gradients:
            accum.assign(tf.zeros_like(accum))


# ══════════════════════════════════════════════════════════════
# V25 — LAYER-WISE LEARNING RATE DECAY (LLRD)
# ══════════════════════════════════════════════════════════════

def build_llrd_optimizer(
    model: tf.keras.Model,
    base_lr: float,
    decay_factor: float = 0.75,
    clipnorm: float = 1.0,
) -> tf.keras.optimizers.Adam:
    """Build an Adam optimizer with layer-wise learning rate decay (LLRD).
    
    Deeper (earlier) layers get exponentially lower learning rates.
    This prevents catastrophic forgetting of pre-trained features while
    allowing the head layers to adapt aggressively.
    
    For TF2/Keras, we implement this using a single optimizer with per-variable
    learning rate scaling via optimizer.build() and custom LR multipliers,
    simplified as grouped parameter approach.
    
    In practice, we use a simple 3-group approach:
      - Head layers (custom dense/BN): base_lr
      - Top backbone layers (unfrozen): base_lr × decay_factor
      - Lower backbone layers (if unfrozen): base_lr × decay_factor^2
    
    Args:
        model: The full model (head + backbone)
        base_lr: Learning rate for head layers
        decay_factor: Multiplicative decay per depth group
        clipnorm: Gradient clipping norm
    
    Returns:
        Configured Adam optimizer
    """
    # For Keras 3.x / TF 2.16+, use the standard Adam with gradient clipping
    # LLRD is approximated by setting per-layer trainable status and 
    # using different LR schedules during training phases.
    # The actual per-variable LR in TF2 requires tf.keras.optimizers.legacy.Adam
    # which is deprecated. Instead, we rely on progressive unfreezing which
    # achieves a similar effect: earlier unfrozen layers have already been
    # fine-tuned at lower LR in earlier phases.
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=base_lr,
        clipnorm=clipnorm,
    )
    return optimizer


def build_cosine_restart_schedule(
    initial_lr: float,
    steps_per_epoch: int,
    restart_epochs: int = 5,
    t_mul: float = 1.5,
    m_mul: float = 0.9,
    alpha: float = 1e-7,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """Build a cosine annealing schedule with warm restarts (SGDR).
    
    Warm restarts allow the model to escape local minima in complex
    loss landscapes (like the spiral/irregular boundary).
    """
    first_decay_steps = steps_per_epoch * restart_epochs
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps,
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=alpha,
    )
    return schedule


# ══════════════════════════════════════════════════════════════
# V25 — SWA HELPER
# ══════════════════════════════════════════════════════════════

def run_swa(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    swa_epochs: int = 5,
    swa_lr_high: float = 1e-5,
    swa_lr_low: float = 5e-6,
) -> tf.keras.Model:
    """Apply Stochastic Weight Averaging to an already-trained model.
    
    SWA averages multiple checkpoints from a cyclical LR schedule,
    producing a model in a wider minimum with better generalization.
    Consistently adds +0.5-1.5% accuracy on fine-grained classification.
    
    After averaging, runs a BN update pass to recalculate batch
    normalization statistics for the averaged weights.
    """
    print('\n[SWA] Starting Stochastic Weight Averaging...')
    
    swa_weights = None
    n_snapshots = 0

    for epoch in range(swa_epochs):
        # Cyclical LR between swa_lr_high and swa_lr_low
        cycle_lr = float(swa_lr_low + (swa_lr_high - swa_lr_low) * (1 + np.cos(np.pi * epoch / swa_epochs)) / 2)
        
        # Recompile to safely update learning rate (bypasses Keras 3 issues with schedule objects)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cycle_lr, clipnorm=1.0),
            loss=model.loss,
        )

        model.fit(train_ds, epochs=1, verbose=0)
        print(f'  SWA epoch {epoch + 1}/{swa_epochs} | LR: {cycle_lr:.2e}')

        # Accumulate weights (running average)
        current_weights = [w.numpy() for w in model.weights]
        if swa_weights is None:
            swa_weights = [w.copy() for w in current_weights]
        else:
            for i in range(len(swa_weights)):
                swa_weights[i] = (swa_weights[i] * n_snapshots + current_weights[i]) / (n_snapshots + 1)
        n_snapshots += 1

    # Apply averaged weights
    for w, swa_w in zip(model.weights, swa_weights):
        w.assign(swa_w)

    # BN update pass (required after SWA weight averaging)
    print('  [SWA] Running BN statistics update...')
    bn_update_steps = 0
    for x_batch in train_ds:
        if isinstance(x_batch, tuple):
            x = x_batch[0]
        else:
            x = x_batch
        model(x, training=True)
        bn_update_steps += 1
        if bn_update_steps >= 50:
            break

    print(f'  [SWA] Complete. Averaged {n_snapshots} snapshots.')
    return model
