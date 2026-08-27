for m in v37 v38 v40;
do
    for y in 2023 2024 2025;
    do
         gcloud run jobs execute aic-predict --region=$REGION --project=$PROJECT --wait \
            --update-env-vars=\
DATA_DIR=gs://aic-amazon/data/fullgrid_v2/allpreds_${y}/,\
OUTPUT_URI=gs://aic-amazon/preds/mtsvit_${m}_${y}/,\
CONFIG_PATH=gs://aic-amazon/configs/mtsvit_test_${m}.json,\
CHECKPOINT=gs://aic-amazon/models/mtsvit_test_${m}.pt
    done
done
                
