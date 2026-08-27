REGION='us-east1'
PROJECT='macedo-lab-general-9051'

IMG=$REGION-docker.pkg.dev/$PROJECT/aic-containers/aic-predict:$(git rev-parse --short HEAD)

gcloud run jobs create aic-predict \
  --project=$PROJECT --region=$REGION \
  --image="$IMG" \
  --cpu=8 --memory=16Gi \
  --task-timeout=2h --max-retries=0 --tasks=1 \
  --service-account=$SA \
  --set-env-vars=\
CONFIG_PATH=gs://aic-amazon/configs/mtsvit_test_v39.json,\
CHECKPOINT=gs://aic-amazon/models/mtsvit_test_v39.pt,\
DATA_DIR=gs://aic-amazon/data/fullgrid_v2/allpreds_2024/,\
OUTPUT_URI=gs://aic-amazon/preds/mtsvit_v39_2024/,\
EDGE_CROP=0,INVERT_YRES=1,MOSAIC=1,BATCH_SIZE=4,OMP_NUM_THREADS=8
