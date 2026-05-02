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

@tf.keras.utils.register_keras_serializable(package="Custom")
class HierarchicalRMSELoss(tf.keras.losses.Loss):
    """Hierarchical RMSE loss that enforces Galaxy Zoo decision tree constraints.
    
    Ported from the legacy winning solution (benanne). This scales the errors of child 
    nodes by the probability of their parent nodes, ensuring the model focuses on 
    chemically/physically valid paths.
    """
    def __init__(self, name="hierarchical_rmse_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        # Scaling sequence: (child_slice_start, child_slice_end, parent_index)
        self.scaling_sequence = [
            (3, 5, 1),    # Q2 scaled by Q1.2 (Features)
            (5, 13, 4),   # Q3, Q4, Q5 scaled by Q2.2 (No edge-on)
            (15, 18, 0),  # Q7 scaled by Q1.1 (Smooth)
            (18, 25, 13), # Q8 scaled by Q6.1 (Odd features)
            (25, 28, 3),  # Q9 scaled by Q2.1 (Edge-on)
            (28, 37, 7)   # Q10, Q11 scaled by Q4.1 (Spirals)
        ]

    def call(self, y_true, y_pred):
        # We apply the scaling to the predictions to match the hierarchy
        # However, the ground truth targets are already weighted in the dataset.
        # So we just calculate RMSE on the raw targets vs our hierarchical predictions.
        y_pred_h = self._apply_hierarchy(y_pred)
        
        mse = tf.reduce_mean(tf.square(y_true - y_pred_h), axis=-1)
        return tf.sqrt(tf.maximum(mse, tf.keras.backend.epsilon()))

    def _apply_hierarchy(self, y_pred):
        """Applies the hierarchical constraints to the prediction vector."""
        # Start with clipped predictions [0, 1]
        y_h = tf.clip_by_value(y_pred, 0.0, 1.0)
        
        for start, end, parent_idx in self.scaling_sequence:
            parent_prob = y_h[:, parent_idx : parent_idx + 1]
            # Rescale the child slice
            child_slice = y_h[:, start:end] * parent_prob
            
            # Reconstruct y_h with the updated slice
            parts = []
            if start > 0:
                parts.append(y_h[:, :start])
            parts.append(child_slice)
            if end < 37:
                parts.append(y_h[:, end:])
            y_h = tf.concat(parts, axis=1)
            
        return y_h

    def get_config(self):
        return super().get_config()
