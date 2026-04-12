from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Custom")
class BinaryFocalLoss(tf.keras.losses.Loss):
    """Binary focal loss for cascade binary classification stages.
    
    Focuses learning on hard examples via modulating factor (1-p)^gamma.
    Supports pos_weight for class imbalance and label smoothing.
    
    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
               Stage 1 (easy): 1.5, Stage 2 (hard): 2.5
        pos_weight: Weight for positive class. Set to n_neg/n_pos for imbalance.
        label_smoothing: Smooth labels to reduce overconfidence on noisy labels.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
        label_smoothing: float = 0.0,
        name: str = 'binary_focal_loss',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Ensure matching shapes to prevent unintended broadcasting [batch, batch]
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), tf.float32)

        # Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Clip predictions for numerical stability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Binary cross entropy components
        bce_pos = -y_true * tf.math.log(y_pred)
        bce_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)

        # Focal modulation weights
        focal_weight_pos = tf.pow(1.0 - y_pred, self.gamma)
        focal_weight_neg = tf.pow(y_pred, self.gamma)

        # Apply pos_weight for class imbalance
        loss = self.pos_weight * focal_weight_pos * bce_pos + focal_weight_neg * bce_neg

        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'pos_weight': self.pos_weight,
            'label_smoothing': self.label_smoothing,
        })
        return config


class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Categorical focal loss (kept for backwards compatibility and potential fallback).
    
    Multi-class focal loss with per-class alpha weighting and label smoothing.
    """

    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        name: str = 'categorical_focal_loss',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Ensure float32 for mixed precision
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        # Categorical cross entropy core
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Focal weight: (1 - p)^gamma
        weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * cross_entropy

        # Class balanced alpha scaling
        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=tf.float32)
            loss = alpha * loss

        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'label_smoothing': self.label_smoothing,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="Custom")
class OHEMBinaryLoss(tf.keras.losses.Loss):
    """Online Hard Example Mining loss for binary classification.
    
    Computes per-sample binary cross entropy, then only keeps the top-K%
    hardest examples (highest loss) for gradient computation. Easy examples
    that the model already classifies correctly are discarded.
    
    This forces the optimizer to spend 100% of its gradient budget on the
    confusing boundary cases (e.g., ambiguous spiral/irregular galaxies).
    
    Args:
        keep_ratio: Fraction of hardest examples to keep (0.7 = top 70%).
        label_smoothing: Optional label smoothing factor.
    """

    def __init__(
        self,
        keep_ratio: float = 0.70,
        label_smoothing: float = 0.0,
        name: str = 'ohem_binary_loss',
        **kwargs,
    ):
        super().__init__(name=name, reduction='none', **kwargs)
        self.keep_ratio = keep_ratio
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(tf.reshape(y_true, tf.shape(y_pred)), tf.float32)

        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Per-sample binary cross entropy
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        bce = tf.reduce_mean(bce, axis=-1)  # [batch_size]

        # Keep only the top-K% hardest examples
        batch_size = tf.shape(bce)[0]
        k = tf.maximum(tf.cast(tf.cast(batch_size, tf.float32) * self.keep_ratio, tf.int32), 1)

        # Get top-k losses
        top_k_losses, _ = tf.math.top_k(bce, k=k, sorted=False)

        return tf.reduce_mean(top_k_losses)

    def get_config(self):
        config = super().get_config()
        config.update({
            'keep_ratio': self.keep_ratio,
            'label_smoothing': self.label_smoothing,
        })
        return config


# Backwards compatibility alias
FocalLoss = CategoricalFocalLoss

# ══════════════════════════════════════════════════════════════
# CUSTOM METRICS FOR SOFT LABELS
# ══════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="Custom")
class SoftBinaryAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.cast(y_true >= 0.5, tf.float32)
        return super().update_state(y_true_bin, y_pred, sample_weight)

@tf.keras.utils.register_keras_serializable(package="Custom")
class SoftAUC(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.cast(y_true >= 0.5, tf.float32)
        return super().update_state(y_true_bin, y_pred, sample_weight)

@tf.keras.utils.register_keras_serializable(package="Custom")
class SoftPrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.cast(y_true >= 0.5, tf.float32)
        return super().update_state(y_true_bin, y_pred, sample_weight)

@tf.keras.utils.register_keras_serializable(package="Custom")
class SoftRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.cast(y_true >= 0.5, tf.float32)
        return super().update_state(y_true_bin, y_pred, sample_weight)
