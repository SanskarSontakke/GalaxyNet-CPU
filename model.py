from __future__ import annotations

import tensorflow as tf


def build_model(image_size: int, num_classes: int, architecture: str = 'EfficientNetV2B0'):
    """Produces the base model and the modified classifier head."""
    # Elite Improvement: Multi-architecture support for Ensembling
    # Using (None, None, 3) enables the Resolution Curriculum to resize inputs dynamically
    inputs = tf.keras.Input(shape=(None, None, 3))
    if architecture == 'EfficientNetV2B0':
        base_model = tf.keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
    elif architecture == 'ConvNeXtTiny':
        # ConvNeXt architectures are excellent for spatial patterns
        base_model = tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=inputs)
    else:
        raise ValueError(f'Unsupported architecture: {architecture}')

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f'GalaxyNet_{architecture}')
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, last_n: int):
    """Gradually unfreezes layers for deep fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-last_n]:
        layer.trainable = False
    return last_n
