
from google.cloud import aiplatform
import argparse
import google


parser = argparse.ArgumentParser()
parser.add_argument('config_json',
                    help='Path to config json on GCS'
                    )
parser.add_argument('display_name',
                    help='Job display name on Vertex AI'
                    )
args = parser.parse_args()

# Basic parameters
project = google.auth.default()[1]
location='us-east1'
bucket='res-id'
config_json=args.config_json
display_name=args.display_name

print(config_json)
print(display_name)

aiplatform.init(project=project, location=location, staging_bucket=bucket)

# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomTrainingJob
job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=args.display_name,
    python_package_gcs_uri="gs://res-id/fire-amazon/python_packages/aic_risk_modeling-0.1.0.tar.gz",
    python_module_name="aic_risk_modeling.train.trainer",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-16.py310:latest",
)
job.run(
    machine_type="n1-highmem-8",
    scheduling_strategy=aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT,
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=[
        f"--config_path={config_json}",
    ],
)
