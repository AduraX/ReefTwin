"""Kubeflow Pipeline definition for ReefTwin.

Compile: python -m pipelines.kfp.pipeline
Submit:  kfp run create -f pipelines/kfp/reeftwin_pipeline.yaml
"""

from kfp import dsl, compiler

from pipelines.kfp.components import (
    generate_iot_data,
    generate_noaa_data,
    build_features_component,
    train_model_component,
)


@dsl.pipeline(
    name="reeftwin-fti",
    description="ReefTwin Feature-Training-Inference pipeline",
)
def reeftwin_pipeline(iot_rows: int = 5000, noaa_days: int = 60):
    iot_task = generate_iot_data(rows=iot_rows)
    noaa_task = generate_noaa_data(days=noaa_days)

    features_task = build_features_component(
        iot_dataset=iot_task.outputs["iot_dataset"],
        noaa_dataset=noaa_task.outputs["noaa_dataset"],
    )

    train_model_component(
        features_dataset=features_task.outputs["features_dataset"],
    )


def compile_pipeline(output: str = "pipelines/kfp/reeftwin_pipeline.yaml") -> str:
    compiler.Compiler().compile(pipeline_func=reeftwin_pipeline, package_path=output)
    print(f"Pipeline compiled: {output}")
    return output


if __name__ == "__main__":
    compile_pipeline()
