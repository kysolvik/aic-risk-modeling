# Usage: DATAFLOW_SA=<service-account-email> scripts/preprocessing/run_df.sh <target_year> <random_seed> [extra args]
#export IMAGE_URL=us-east1-docker.pkg.dev/macedo-lab-general-9051/geebeam/kysolvik/geebeam:latest
export IMAGE_URL=kysolvik/geebeam:0.5.3
target_year=$1
random_seed=$2
other_args=$3

python scripts/preprocessing/geebeam_ali_inputs.py \
    --region us-east1 \
    --worker_zone us-east1-b \
    --runner DataflowRunner \
    --max_num_workers=4 \
    --num_workers=1 \
    --experiments=use_runner_v2 \
    --machine_type=n2-highmem-2 \
    --sdk_container_image="${IMAGE_URL}" \
    --random_seed=$random_seed \
    --target_year=$target_year \
    --service_account_email="${DATAFLOW_SA:?set DATAFLOW_SA to the Dataflow service account email}" \
    $other_args \
    --use_public_ips
#    --no_use_public_ips # For columbia project

