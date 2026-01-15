
from google.cloud import aiplatform

# Training parameters
epochs = 10
model_type='mlp'
gcs_data_dir='gs://aic-fire-amazon/results_2024_5k/'
tfrecord_pattern='*.tfrecord.gz'
output_band='BurnDate'
patch_size=128
batch_size=4

# Basic parameters
project='ksolvik-misc'
location='us-east1'
bucket='aic-fire-amazon'
 
aiplatform.init(project=project, location=location, staging_bucket=bucket)

# https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomTrainingJob
job = aiplatform.CustomTrainingJob(
    display_name=f"fire-risk-model",
    script_path="../src/aic_risk_modeling/train/train_model.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-16.py310:latest",
)

job.run(
    machine_type="n1-highmem-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=0,
    args=[
        f"--model_type={model_type}",
        f"--gcs_data_dir={gcs_data_dir}",
        f"--tfrecord_pattern={tfrecord_pattern}",
        f"--output_band={output_band}",
        f"--patch_size={patch_size}",
        f"--batch_size={batch_size}",
        f"--epochs={epochs}",
    ],
)