"""Component initialization module for TFX Pipeline."""

# pylint: disable=too-many-arguments,too-many-locals,no-member,too-many-positional-arguments

import os
from typing import List, Union
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

PipelineComponent = Union[
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Resolver,
    Evaluator,
    Pusher,
]


def init_components(
    data_dir: str,
    transform_module_file: str,
    trainer_module_file: str,
    serving_model_dir: str,
    train_steps: int = 50,
    eval_steps: int = 20,
) -> List[PipelineComponent]:
    """Initializes all 9 TFX pipeline components.

    Args:
        data_dir: Directory containing the CSV dataset.
        transform_module_file: Path to transform module.
        trainer_module_file: Path to trainer module.
        serving_model_dir: Destination path for exported serving model.
        train_steps: Number of training steps.
        eval_steps: Number of evaluation steps.

    Returns:
        List of initialized TFX components.
    """
    # 1. ExampleGen
    example_gen = CsvExampleGen(input_base=data_dir)

    # 2. StatisticsGen
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    # 3. SchemaGen
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    # 4. ExampleValidator
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # 5. Transform
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module_file,
    )

    # 6. Trainer
    trainer = Trainer(
        module_file=trainer_module_file,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps),
    )

    # 7. Resolver (Latest Blessed Model)
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    # 8. Evaluator
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="target")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.5}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="AUC"),
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    # 9. Pusher
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=os.path.abspath(serving_model_dir)
            )
        ),
    )

    return [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
