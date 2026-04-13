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


def build_regression_model(
    architecture: str = 'EfficientNetV2B2',
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build unified regression model: Predicts all 37 Galaxy Zoo probabilities.
    
    Architecture: backbone → GAP → BN → Dense(512) → BN → GELU → 
                  Dropout(0.4) → Dense(37, sigmoid)
    
    Using Sigmoid on the final layer logically bounds all voting fractions to [0.0, 1.0].
    
    Returns:
        (full_model, base_model) tuple for progressive unfreezing.
    """
    inputs = tf.keras.Input(shape=(None, None, 3), name='image_input')
    base_model = _get_backbone(architecture, inputs)

    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(base_model.output)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Dense(512, name='dense1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Activation('gelu', name='gelu1')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout1')(x)
    
    # 37 dimensional output bounded to [0, 1] interval natively using sigmoid
    outputs = tf.keras.layers.Dense(
        37, activation='sigmoid', dtype='float32', name='regression_output'
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs,
        name=f'GalaxyNet_UnifiedRegression_{architecture}',
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
