"""Transform module for preprocessing data using TensorFlow Transform."""

from typing import Dict
import tensorflow as tf
import tensorflow_transform as tft

# Feature categorization
NUMERIC_FEATURE_KEYS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURE_KEYS = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

LABEL_KEY = "target"


def transformed_name(key: str) -> str:
    """Generates the transformed feature key name.

    Args:
        key: Original feature key.

    Returns:
        The name of the feature post transformation.
    """
    return f"{key}_xf"


def preprocessing_fn(
    inputs: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    """Preprocesses the input features.

    Args:
        inputs: Map of feature keys to raw tensors.

    Returns:
        Map of feature keys to transformed tensors.
    """
    outputs: Dict[str, tf.Tensor] = {}

    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = tf.cast(inputs[key], tf.float32)

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        inputs[LABEL_KEY], tf.int64
    )

    return outputs
