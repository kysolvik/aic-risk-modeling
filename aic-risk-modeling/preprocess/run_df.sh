python geebeam_main.py \
    --config ./example_config.json \
    --region_of_interest ../data/Limites_RAISG_2025/Lim_Raisg.shp \
    --output_path gs://aic-fire-amazon/results_v3/ \
    --runner DataflowRunner \
    --max_num_workers=16 \
    --num_workers=8 \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-east1-docker.pkg.dev/ksolvik-misc/columbia-aic-risk-modeling/fire-risk-preprocess/beam_python_prebuilt_sdk:9472ab08-b36c-4096-a6a2-bedcd892ac0c


# Uncomment to build container image for faster deployment
# Pushes to Google Archive Registry
#    --prebuild_sdk_container_engine=local_docker \
#    --docker_registry_push_url=us-east1-docker.pkg.dev/ksolvik-misc/columbia-aic-risk-modeling/fire-risk-preprocess
#    --requirements_file ./pipeline_requirements.txt \
