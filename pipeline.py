"""Main pipeline execution script using TensorFlow Extended (TFX) and Apache Beam."""

import os
import sys
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from modules.components import init_components

# Pipeline Configurations
PIPELINE_NAME = "rizalbasri-pipeline"
PIPELINE_ROOT = "rizalbasri-pipeline"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, PIPELINE_ROOT)
METADATA_DIR = os.path.join(BASE_DIR, "tfx_metadata")
SERVING_DIR = os.path.join(BASE_DIR, "serving_model_dir")

TRANSFORM_MODULE = os.path.join(BASE_DIR, "modules", "transform.py")
TRAINER_MODULE = os.path.join(BASE_DIR, "modules", "trainer.py")


def create_pipeline() -> pipeline.Pipeline:
    """Creates and configures the TFX pipeline with Apache Beam.

    Returns:
        Configured TFX Pipeline instance.
    """
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(SERVING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    components = init_components(
        data_dir=DATA_DIR,
        transform_module_file=TRANSFORM_MODULE,
        trainer_module_file=TRAINER_MODULE,
        serving_model_dir=SERVING_DIR,
        train_steps=50,
        eval_steps=20,
    )

    metadata_connection = metadata.sqlite_metadata_connection_config(
        os.path.join(METADATA_DIR, "metadata.db")
    )

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=OUTPUT_DIR,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_connection,
    )


def run() -> None:
    """Runs the pipeline via BeamDagRunner."""
    print(f"Starting pipeline execution: {PIPELINE_NAME}")
    pipeline_obj = create_pipeline()
    BeamDagRunner().run(pipeline_obj)
    print("Pipeline execution completed successfully!")


if __name__ == "__main__":
    run()
