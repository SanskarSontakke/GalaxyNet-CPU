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


def build_stage1_model(
    architecture: str = 'EfficientNetV2B1',
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build Stage 1 binary classifier: Elliptical vs Non-Elliptical.
    
    Architecture: backbone → GAP → BN → Dense(256) → BN → ReLU → 
                  Dropout(0.35) → Dense(1, sigmoid)
    
    Returns:
        (full_model, base_model) tuple for progressive unfreezing.
    """
    # Dynamic input shape for resolution curriculum
    inputs = tf.keras.Input(shape=(None, None, 3), name='stage1_input')
    base_model = _get_backbone(architecture, inputs)

    x = tf.keras.layers.GlobalAveragePooling2D(name='stage1_gap')(base_model.output)
    x = tf.keras.layers.BatchNormalization(name='stage1_bn1')(x)
    x = tf.keras.layers.Dense(256, name='stage1_dense1')(x)
    x = tf.keras.layers.BatchNormalization(name='stage1_bn2')(x)
    x = tf.keras.layers.Activation('relu', name='stage1_relu')(x)
    x = tf.keras.layers.Dropout(0.35, name='stage1_dropout')(x)
    # Sigmoid output for binary classification, float32 for mixed precision
    outputs = tf.keras.layers.Dense(
        1, activation='sigmoid', dtype='float32', name='stage1_output'
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs,
        name=f'GalaxyNet_Stage1_{architecture}',
    )
    return model, base_model


def build_stage2_model(
    architecture: str = 'EfficientNetV2B2',
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build Stage 2 binary classifier: Spiral vs Irregular.
    
    Architecture: backbone → GAP → BN → Dense(512) → BN → GELU → 
                  Dropout(0.4) → Dense(256) → BN → GELU → 
                  Dropout(0.3) → Dense(1, sigmoid)
    
    Deeper head than Stage 1 because spiral/irregular boundary is harder.
    GELU activation for smoother gradients on the difficult boundary.
    
    Returns:
        (full_model, base_model) tuple for progressive unfreezing.
    """
    inputs = tf.keras.Input(shape=(None, None, 3), name='stage2_input')
    base_model = _get_backbone(architecture, inputs)

    x = tf.keras.layers.GlobalAveragePooling2D(name='stage2_gap')(base_model.output)
    x = tf.keras.layers.BatchNormalization(name='stage2_bn1')(x)
    x = tf.keras.layers.Dense(512, name='stage2_dense1')(x)
    x = tf.keras.layers.BatchNormalization(name='stage2_bn2')(x)
    x = tf.keras.layers.Activation('gelu', name='stage2_gelu1')(x)
    x = tf.keras.layers.Dropout(0.4, name='stage2_dropout1')(x)
    x = tf.keras.layers.Dense(256, name='stage2_dense2')(x)
    x = tf.keras.layers.BatchNormalization(name='stage2_bn3')(x)
    x = tf.keras.layers.Activation('gelu', name='stage2_gelu2')(x)
    x = tf.keras.layers.Dropout(0.3, name='stage2_dropout2')(x)
    # Sigmoid output for binary classification
    outputs = tf.keras.layers.Dense(
        1, activation='sigmoid', dtype='float32', name='stage2_output'
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs,
        name=f'GalaxyNet_Stage2_{architecture}',
    )
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, last_n: int) -> int:
    """Gradually unfreezes the last N layers of the base model for fine-tuning.
    
    All layers before the last N are frozen (not trainable).
    Returns the number of unfrozen layers.
    """
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_up_to = max(0, total_layers - last_n)
    for layer in base_model.layers[:freeze_up_to]:
        layer.trainable = False
    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    return unfrozen


def freeze_base(base_model: tf.keras.Model) -> None:
    """Freeze all layers of the base model (for warmup phase)."""
    base_model.trainable = False
