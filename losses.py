from __future__ import annotations

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: list[float] | None = None, gamma: float = 2.0, name: str = 'focal_loss', label_smoothing: float = 0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Elite Improvement: Label Smoothing
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Categorical cross entropy core
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Focal weight: (1 - p)^gamma
        weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * cross_entropy
        
        # Class balanced alpha scaling
        if self.alpha is not None:
            alpha = tf.constant(self.alpha, dtype=y_true.dtype)
            loss = alpha * loss
            
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha, 'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return config
