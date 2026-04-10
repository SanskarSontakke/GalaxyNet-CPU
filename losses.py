from __future__ import annotations

import tensorflow as tf


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
        # Ensure float32 for numerical stability (critical for mixed precision)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

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


# Backwards compatibility alias
FocalLoss = CategoricalFocalLoss
