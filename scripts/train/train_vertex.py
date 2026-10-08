"""Submit a training config to Vertex AI.

Usage: train_vertex.py gs://.../config.json DISPLAY_NAME"""
from google.cloud import aiplatform
import argparse
import google
import os


parser = argparse.ArgumentParser()
parser.add_argument('config_json',
                    help='Path to config json on GCS'
                    )
parser.add_argument('display_name',
                    help='Job display name on Vertex AI'
                    )
args = parser.parse_args()

project = google.auth.default()[1]
location='us-east1'
bucket='aic-amazon'
config_json=args.config_json
display_name=args.display_name

print(config_json)
print(display_name)

aiplatform.init(project=project, location=location, staging_bucket=bucket)

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=args.display_name,
    python_package_gcs_uri="gs://aic-amazon/python_packages/aic_risk_modeling-0.3.7.tar.gz",
    python_module_name="aic_risk_modeling.train.trainer",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
)
job.run(
    machine_type="n1-highmem-16",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    boot_disk_size_gb=200,
    args=[
        f"--config_path={config_json}",
    ],
    sync=False,
)
job.wait_for_resource_creation()
print(job.resource_name)
# sync=False leaves a non-daemon monitor thread; os._exit avoids waiting for the job.
os._exit(0)
