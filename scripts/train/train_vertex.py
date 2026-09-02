
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
bucket='aic-amazon'
config_json=args.config_json
display_name=args.display_name

print(config_json)
print(display_name)

aiplatform.init(project=project, location=location, staging_bucket=bucket)

# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomTrainingJob
# Container provides torch + CUDA; the package's core deps (tensorflow-cpu for
# data loading, tensorflow-metadata, google-cloud-storage) are pip-installed
# from the sdist. Available containers:
# https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=args.display_name,
    python_package_gcs_uri="gs://aic-amazon/python_packages/aic_risk_modeling-0.2.7.tar.gz",
    python_module_name="aic_risk_modeling.train.trainer",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
)
job.run(
    machine_type="n1-highmem-8",
#    scheduling_strategy=aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT,
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    boot_disk_size_gb=100, # 100 is default
    args=[
        f"--config_path={config_json}",
    ],
)
