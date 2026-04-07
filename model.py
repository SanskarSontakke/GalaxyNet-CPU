from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_model(image_size: int = 128, num_classes: int = 3):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name='image')

    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='galaxynet_efficientnetv2b0')
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, last_n: int) -> int:
    base_model.trainable = True
    if last_n <= 0:
        for layer in base_model.layers:
            layer.trainable = False
        return 0

    for layer in base_model.layers[:-last_n]:
        layer.trainable = False
    for layer in base_model.layers[-last_n:]:
        layer.trainable = True
    return last_n
