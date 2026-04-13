import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
def rmse_metric(y_true, y_pred):
    """Root Mean Squared Error for the 37 regression targets."""
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return tf.maximum(0.0, tf.sqrt(mse))

@tf.keras.utils.register_keras_serializable(package="Custom")
class RMSELoss(tf.keras.losses.Loss):
    """Direct RMSE loss function instead of standard MSE.
    
    Can be used directly to optimize against the Kaggle LB metric.
    Includes a small epsilon inside the square root for numerical stability.
    """
    def __init__(self, name="rmse_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        # Calculate MSE across the 37 features
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        # Apply sqrt with epsilon to prevent infinite gradient at exactly 0.0 variance
        return tf.sqrt(tf.maximum(mse, tf.keras.backend.epsilon()))

    def get_config(self):
        return super().get_config()
