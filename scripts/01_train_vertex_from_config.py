
from google.cloud import aiplatform

# Basic parameters
project='ksolvik-misc'
location='us-east1'
bucket='aic-fire-amazon'
config_json='gs://aic-fire-amazon/configs/embeddings_oneyear_config.json'

aiplatform.init(project=project, location=location, staging_bucket=bucket)

# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomTrainingJob
job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=f"fire-risk-model",
    python_package_gcs_uri="gs://aic-fire-amazon/python_packages/aic_risk_modeling-0.0.1.tar.gz",
    python_module_name="aic_risk_modeling.train.train_model",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-16.py310:latest",
)

job.run(
    machine_type="n1-highmem-4",
    scheduling_strategy=aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT,
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        f"--config_path={config_json}",
    ],
)
