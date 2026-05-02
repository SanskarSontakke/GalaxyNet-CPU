from __future__ import annotations

import tensorflow as tf


def _get_backbone(architecture: str, input_tensor: tf.Tensor) -> tf.keras.Model:
    """Instantiate a backbone model by architecture name with ImageNet weights."""
    arch_map = {
        'EfficientNetV2B0': tf.keras.applications.EfficientNetV2B0,
        'EfficientNetV2B1': tf.keras.applications.EfficientNetV2B1,
        'EfficientNetV2B2': tf.keras.applications.EfficientNetV2B2,
        'ConvNeXtTiny': tf.keras.applications.ConvNeXtTiny,
    }
    if architecture not in arch_map:
        raise ValueError(
            f'Unsupported architecture: {architecture}. '
            f'Available: {list(arch_map.keys())}'
        )
    return arch_map[architecture](
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
    )


@tf.keras.utils.register_keras_serializable(package="Custom")
class MultiViewLayer(tf.keras.layers.Layer):
    """Generates 8 orientations (4 rotations + 4 flips) of the input batch."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: (Batch, H, W, 3)
        rot0 = inputs
        rot90 = tf.image.rot90(inputs, k=1)
        rot180 = tf.image.rot90(inputs, k=2)
        rot270 = tf.image.rot90(inputs, k=3)
        
        flip0 = tf.image.flip_left_right(rot0)
        flip90 = tf.image.flip_left_right(rot90)
        flip180 = tf.image.flip_left_right(rot180)
        flip270 = tf.image.flip_left_right(rot270)
        
        # Stack: (8, Batch, H, W, 3)
        views = tf.stack([rot0, rot90, rot180, rot270, flip0, flip90, flip180, flip270], axis=0)
        
        # Flatten: (8*Batch, H, W, 3)
        s = tf.shape(inputs)
        return tf.reshape(views, [8 * s[0], s[1], s[2], s[3]])

    def compute_output_shape(self, input_shape):
        batch = input_shape[0] * 8 if input_shape[0] is not None else None
        return (batch, input_shape[1], input_shape[2], input_shape[3])

@tf.keras.utils.register_keras_serializable(package="Custom")
class MultiViewAverage(tf.keras.layers.Layer):
    """Averages features or predictions across the 8 orientations."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: (8*Batch, D)
        s = tf.shape(inputs)
        d = inputs.shape[-1]
        # Reshape to (8, Batch, D)
        x = tf.reshape(inputs, [8, s[0] // 8, d])
        return tf.reduce_mean(x, axis=0)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0] // 8 if input_shape[0] is not None else None
        return (batch, input_shape[1])

def build_regression_model(
    architecture: str = 'EfficientNetV2B2',
    multi_view: bool = False,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build unified regression model: Predicts all 37 Galaxy Zoo probabilities.
    
    If multi_view is True, the model processes 8 rotations/flips and averages them.
    """
    inputs = tf.keras.Input(shape=(None, None, 3), name='image_input')
    
    x = inputs
    if multi_view:
        x = MultiViewLayer()(x)

    # Get backbone with ImageNet weights
    base_model = _get_backbone(architecture, x)
    
    # Process features
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(base_model.output)
    
    if multi_view:
        # Average features across the 8 views before the dense head
        x = MultiViewAverage()(x)

    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Dense(512, name='dense1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Activation('gelu', name='gelu1')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout1')(x)
    
    outputs = tf.keras.layers.Dense(
        37, activation='sigmoid', dtype='float32', name='regression_output'
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs,
        name=f'GalaxyNet_UnifiedRegression_{architecture}_MV' if multi_view else f'GalaxyNet_UnifiedRegression_{architecture}',
    )
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, last_n: int) -> int:
    """Gradually unfreezes the last N layers of the base model for fine-tuning.
    
    All layers before the last N are frozen (not trainable).
    Returns the number of unfrozen layers (those with trainable weights).
    """
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_up_to = max(0, total_layers - last_n)
    for layer in base_model.layers[:freeze_up_to]:
        layer.trainable = False
    # Unfreeze everything after the boundary
    for layer in base_model.layers[freeze_up_to:]:
        layer.trainable = True
    # Count layers that are actually trainable and have weights
    unfrozen = sum(1 for l in base_model.layers[freeze_up_to:] if len(l.weights) > 0)
    return unfrozen


def freeze_base(base_model: tf.keras.Model) -> None:
    """Freeze all layers of the base model (for warmup phase)."""
    base_model.trainable = False
