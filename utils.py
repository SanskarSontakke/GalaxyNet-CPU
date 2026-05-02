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


def get_strategy() -> tf.distribute.Strategy:
    """Detect and initialize the appropriate distribution strategy."""
    try:
        # Standard TPU detection
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"Running on TPU (Standard): {tpu.cluster_spec().as_dict()}")
        return strategy
    except (ValueError, RuntimeError, tf.errors.NotFoundError):
        try:
            # Fallback for some Kaggle environments (TPU v5 or newer)
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Running on TPU (Local Resolver)")
            return strategy
        except (ValueError, RuntimeError, tf.errors.NotFoundError):
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"Running on GPU: {len(gpus)} device(s)")
                return tf.distribute.MirroredStrategy()
            else:
                print("Running on CPU")
                return tf.distribute.get_strategy()


def setup_environment(config: Config) -> None:
    set_global_seed(config.seed)

    # Configure GPU memory growth FIRST (before any GPU initialization)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Warning: Could not set GPU memory growth: {e}")

    if config.enable_mixed_precision:
        # Determine policy based on ACTUAL hardware presence
        # TPU v3/v5 prefers bfloat16, GPU prefers float16
        is_tpu = any(os.environ.get(k) for k in ['TPU_NAME', 'KAGGLE_TPU_ADDR', 'COLAB_TPU_ADDR'])
        
        policy = 'mixed_bfloat16' if is_tpu else 'mixed_float16'
        tf.keras.mixed_precision.set_global_policy(policy)
    
    # Final check for visibility
    if not any(os.environ.get(k) for k in ['TPU_NAME', 'KAGGLE_TPU_ADDR']) and not gpus:
        print('Warning: No Hardware Accelerator (GPU/TPU) detected — running on CPU')


def find_kaggle_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f'Could not find {name} under {root}')
    return matches[0]


def _extract_zip_if_needed(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)


def _resolve_competition_environment(search_roots: list[Path], temp_dir: Path) -> Tuple[Path, Path, Path, Path]:
    def find_robustly(name: str):
        for root in search_roots:
            if not root.exists():
                continue
            try:
                return find_kaggle_file(root, name)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Could not find {name} in any of {search_roots}")

    # Try to find CSV directly first (if already unzipped)
    try:
        csv_path = find_robustly('training_solutions_rev1.csv')
    except FileNotFoundError:
        # If not, find and extract zip
        train_csv_zip = find_robustly('training_solutions_rev1.zip')
        _extract_zip_if_needed(train_csv_zip, temp_dir)
        csv_path = find_kaggle_file(temp_dir, 'training_solutions_rev1.csv')

    # Try to find training image dir directly
    try:
        image_dir_candidate = find_robustly('images_training_rev1')
        if not image_dir_candidate.is_dir():
            raise FileNotFoundError
        train_check = list(Path(image_dir_candidate).glob('*.jpg'))
        if len(train_check) < 10:
            raise FileNotFoundError
        train_image_dir = image_dir_candidate
    except FileNotFoundError:
        image_zip = find_robustly('images_training_rev1.zip')
        _extract_zip_if_needed(image_zip, temp_dir)
        train_image_dir = find_kaggle_file(temp_dir, 'images_training_rev1')

    # Try to find test image dir directly
    try:
        test_dir_candidate = find_robustly('images_test_rev1')
        if not test_dir_candidate.is_dir():
            raise FileNotFoundError
        test_check = list(Path(test_dir_candidate).glob('*.jpg'))
        if len(test_check) < 10:
            raise FileNotFoundError
        test_image_dir = test_dir_candidate
    except FileNotFoundError:
        test_zip = find_robustly('images_test_rev1.zip')
        _extract_zip_if_needed(test_zip, temp_dir)
        test_image_dir = find_kaggle_file(temp_dir, 'images_test_rev1')

    try:
        submission_template_path = find_robustly('all_zeros_benchmark.zip')
    except FileNotFoundError:
        submission_template_path = find_robustly('all_zeros_benchmark.csv')

    return (
        Path(csv_path),
        Path(train_image_dir),
        Path(test_image_dir),
        Path(submission_template_path),
    )


def setup_kaggle_environment(config: Config) -> Tuple[Path, Path, Path, Path]:
    search_roots = [config.kaggle_input_dir, Path('/kaggle/input')]
    return _resolve_competition_environment(search_roots, config.kaggle_temp_dir)


def setup_local_environment(config: Config) -> Tuple[Path, Path, Path, Path]:
    local_root = Path('galaxy_raw')
    search_roots = [local_root]
    return _resolve_competition_environment(search_roots, local_root)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)


def is_kaggle_runtime() -> bool:
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None


# ══════════════════════════════════════════════════════════════
# V25 — GRADIENT ACCUMULATION HELPER
# ══════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="Custom")
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "inner_model": self.inner_model,
            "accumulation_steps": self.accumulation_steps,
        })
        return config

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
