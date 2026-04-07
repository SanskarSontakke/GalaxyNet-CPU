from __future__ import annotations

import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None, name: str = 'focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = tf.constant(alpha if alpha is not None else [1.0, 1.0, 1.0], dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)

        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
        focal_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = -alpha_t * focal_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    def get_config(self):
        return {
            'gamma': self.gamma,
            'alpha': self.alpha.numpy().tolist(),
            'name': self.name,
        }
