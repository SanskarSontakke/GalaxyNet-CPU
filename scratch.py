import tensorflow as tf

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

y_true = tf.constant([[0.8], [0.2], [0.9]])
y_pred = tf.constant([[0.9], [0.3], [0.8]])

m = SoftBinaryAccuracy()
m.update_state(y_true, y_pred)
print(m.result().numpy())
