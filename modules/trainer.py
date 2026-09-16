"""Trainer module for training the Keras model in TFX pipeline."""

from typing import Dict, List
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

try:
    from modules.transform import (
        CATEGORICAL_FEATURE_KEYS,
        LABEL_KEY,
        NUMERIC_FEATURE_KEYS,
        transformed_name,
    )
except ImportError:
    from transform import (  # pylint: disable=import-error
        CATEGORICAL_FEATURE_KEYS,
        LABEL_KEY,
        NUMERIC_FEATURE_KEYS,
        transformed_name,
    )


def _gzip_reader_fn(filenames: List[str]) -> tf.data.TFRecordDataset:
    """Reads GZIP-compressed TFRecord files.

    Args:
        filenames: List of file paths.

    Returns:
        A TFRecordDataset with GZIP decompression.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(
    file_pattern: List[str],
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Generates features and label for training and evaluation.

    Args:
        file_pattern: Pattern of input data files.
        tf_transform_output: Output of transform component.
        batch_size: Batch size for training.

    Returns:
        A batched dataset yielding tuple of features and label.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=None,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def _build_keras_model() -> tf.keras.Model:
    """Builds a binary classification Keras model.

    Returns:
        A compiled Keras model.
    """
    input_layers: Dict[str, tf.keras.layers.Input] = {}

    for key in NUMERIC_FEATURE_KEYS:
        input_layers[transformed_name(key)] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name(key), dtype=tf.float32
        )

    for key in CATEGORICAL_FEATURE_KEYS:
        input_layers[transformed_name(key)] = tf.keras.layers.Input(
            shape=(1,), name=transformed_name(key), dtype=tf.float32
        )

    features = tf.keras.layers.concatenate(list(input_layers.values()))
    dense_1 = tf.keras.layers.Dense(64, activation="relu")(features)
    dropout_1 = tf.keras.layers.Dropout(0.2)(dense_1)
    dense_2 = tf.keras.layers.Dense(32, activation="relu")(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.1)(dense_2)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout_2)

    model = tf.keras.Model(inputs=input_layers, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def _get_serve_tf_examples_fn(
    model: tf.keras.Model,
    tf_transform_output: tft.TFTransformOutput,
):
    """Creates a serving function accepting serialized tf.train.Example.

    Args:
        model: Trained Keras model.
        tf_transform_output: Transform output artifact.

    Returns:
        TensorFlow function for inference serving.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Runs training and exports the trained model.

    Args:
        fn_args: Arguments passed by TFX Trainer component.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, batch_size=32
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, batch_size=32
    )

    model = _build_keras_model()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        patience=10,
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps or 10,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps or 5,
        epochs=15,
        callbacks=[early_stopping],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures,
    )
