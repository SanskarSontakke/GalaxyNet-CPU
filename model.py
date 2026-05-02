from __future__ import annotations

import tensorflow as tf


QUESTION_SLICES = (
    (0, 3),
    (3, 5),
    (5, 7),
    (7, 9),
    (9, 13),
    (13, 15),
    (15, 18),
    (18, 25),
    (25, 28),
    (28, 31),
    (31, 37),
)

SCALING_SEQUENCE = (
    (3, 5, 1),
    (5, 13, 4),
    (15, 18, 0),
    (18, 25, 13),
    (25, 28, 3),
    (28, 37, 7),
)


def _rotate_batch(images: tf.Tensor, angle_degrees: float) -> tf.Tensor:
    angle = tf.constant(angle_degrees * 3.141592653589793 / 180.0, dtype=tf.float32)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    image_shape = tf.shape(images)
    batch = image_shape[0]
    h = tf.cast(image_shape[1], tf.float32)
    w = tf.cast(image_shape[2], tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    a0 = cos_a
    a1 = sin_a
    a2 = cx - a0 * cx - a1 * cy
    b0 = -sin_a
    b1 = cos_a
    b2 = cy - b0 * cx - b1 * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    transforms = tf.tile(tf.reshape(transform, [1, 8]), [batch, 1])

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=image_shape[1:3],
        fill_value=0.0,
        interpolation='BILINEAR',
        fill_mode='REFLECT',
    )


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


@tf.keras.utils.register_keras_serializable(package="Custom")
class BenannePartExtractor(tf.keras.layers.Layer):
    """Create benanne-style views and 45x45 aligned parts."""
    def __init__(self, view_size: int = 69, part_size: int = 45, **kwargs):
        super().__init__(**kwargs)
        self.view_size = view_size
        self.part_size = part_size

    def call(self, inputs):
        regular_view = tf.image.resize(inputs, [self.view_size, self.view_size])
        rotated_view = tf.image.resize(_rotate_batch(inputs, 45.0), [self.view_size, self.view_size])

        views = [
            regular_view,
            tf.image.flip_left_right(regular_view),
            rotated_view,
            tf.image.flip_left_right(rotated_view),
        ]

        parts = []
        ps = self.part_size
        for view in views:
            parts.append(view[:, :ps, :ps, :])
            parts.append(tf.image.rot90(view[:, :ps, -ps:, :], k=1))
            parts.append(tf.image.rot90(view[:, -ps:, -ps:, :], k=2))
            parts.append(tf.image.rot90(view[:, -ps:, :ps, :], k=3))

        return tf.concat(parts, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"view_size": self.view_size, "part_size": self.part_size})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class PartFeatureMerge(tf.keras.layers.Layer):
    """Merge flattened part features back into per-example features."""
    def __init__(self, num_parts: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_parts = num_parts

    def call(self, inputs):
        shape = tf.shape(inputs)
        feature_dim = inputs.shape[-1]
        merged = tf.reshape(inputs, [self.num_parts, shape[0] // self.num_parts, feature_dim])
        merged = tf.transpose(merged, [1, 0, 2])
        return tf.reshape(merged, [shape[0] // self.num_parts, self.num_parts * feature_dim])

    def get_config(self):
        config = super().get_config()
        config["num_parts"] = self.num_parts
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class MaxoutDense(tf.keras.layers.Layer):
    """Dense layer with maxout pooling across linear pieces."""
    def __init__(self, units: int, pieces: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.pieces = pieces
        self.projection = tf.keras.layers.Dense(units * pieces, activation=None, dtype='float32')

    def call(self, inputs):
        outputs = self.projection(inputs)
        dynamic_shape = tf.shape(outputs)
        outputs = tf.reshape(outputs, [dynamic_shape[0], self.units, self.pieces])
        return tf.reduce_max(outputs, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "pieces": self.pieces})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class GalaxyOutputLayer(tf.keras.layers.Layer):
    """Normalizes per-question answers and applies Galaxy Zoo tree weights."""
    def __init__(self, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        positive = tf.nn.softplus(inputs) + self.epsilon
        question_probs = []
        for start, end in QUESTION_SLICES:
            question = positive[:, start:end]
            question_sum = tf.reduce_sum(question, axis=1, keepdims=True)
            question_probs.append(question / question_sum)

        outputs = tf.concat(question_probs, axis=1)
        for start, end, parent_idx in SCALING_SEQUENCE:
            parent_prob = outputs[:, parent_idx:parent_idx + 1]
            scaled_slice = outputs[:, start:end] * parent_prob
            outputs = tf.concat(
                [outputs[:, :start], scaled_slice, outputs[:, end:]],
                axis=1,
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config["epsilon"] = self.epsilon
        return config


def build_regression_model(
    architecture: str = 'EfficientNetV2B2',
    multi_view: bool = False,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build unified regression model: Predicts all 37 Galaxy Zoo probabilities.
    
    If multi_view is True, the model processes 8 rotations/flips and averages them.
    """
    inputs = tf.keras.Input(shape=(None, None, 3), name='image_input')

    if architecture == 'BenanneNetTF':
        x = BenannePartExtractor(name='benanne_parts')(inputs)
        x = tf.keras.layers.Conv2D(32, 6, activation='relu', padding='valid', name='conv1')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool1')(x)
        x = tf.keras.layers.Conv2D(64, 5, activation='relu', padding='valid', name='conv2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool2')(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='valid', name='conv3')(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='valid', name='conv4')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name='pool4')(x)
        x = tf.keras.layers.Flatten(name='flatten_parts')(x)
        x = PartFeatureMerge(name='merge_parts')(x)
        x = tf.keras.layers.Dropout(0.5, name='dropout1')(x)
        x = MaxoutDense(2048, pieces=2, name='maxout1')(x)
        x = tf.keras.layers.Dropout(0.5, name='dropout2')(x)
        x = MaxoutDense(2048, pieces=2, name='maxout2')(x)
        x = tf.keras.layers.Dropout(0.5, name='dropout3')(x)
        logits = tf.keras.layers.Dense(37, activation=None, dtype='float32', name='regression_logits')(x)
        outputs = GalaxyOutputLayer(name='regression_output')(logits)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='GalaxyNet_BenanneNetTF')
        return model, None

    x = inputs
    if multi_view:
        x = MultiViewLayer()(x)

    base_model = _get_backbone(architecture, x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(base_model.output)

    if multi_view:
        x = MultiViewAverage()(x)

    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.Dense(512, name='dense1')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.Activation('gelu', name='gelu1')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout1')(x)

    logits = tf.keras.layers.Dense(37, activation=None, dtype='float32', name='regression_logits')(x)
    outputs = GalaxyOutputLayer(name='regression_output')(logits)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f'GalaxyNet_UnifiedRegression_{architecture}_MV' if multi_view else f'GalaxyNet_UnifiedRegression_{architecture}',
    )
    return model, base_model


def unfreeze_top_layers(base_model: tf.keras.Model, last_n: int) -> int:
    """Gradually unfreezes the last N layers of the base model for fine-tuning.
    
    All layers before the last N are frozen (not trainable).
    Returns the number of unfrozen layers (those with trainable weights).
    """
    if base_model is None:
        return 0
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
    if base_model is not None:
        base_model.trainable = False
