# Containerized prediction

This is a CPU-only image. It runs one data dir (usually a year of chips) through
`predict.py` or `attribute.py`, mosaics the output, and uploads to GCS. It runs as two
Cloud Run jobs: `aic-predict` for a year (~20 min) and `aic-attribute` for exact Shapley
(~27 s/chip, so `run_cv_attribute.sh` splits each year by shard).

## Build

The image installs `src/` at build time (`uv sync --locked`), so rebuild it after any
change to `src/` or `scripts/predict/`:

```bash
uv lock
gcloud builds submit --config=cloudbuild.yaml --region=us-east1 \
  --project=macedo-lab-general-9051 --substitutions=_TAG=$(git rev-parse --short HEAD) .
```

`.gcloudignore` includes `.dockerignore`, an allowlist. Without it, `gcloud` falls back
to `.gitignore`, whose `*.tif` rule drops `assets/example_v3.tif`. The image then builds
but fails at runtime.

## Jobs (create once)

```bash
REGION=us-east1; PROJECT=macedo-lab-general-9051
SA=<project-number>-compute@developer.gserviceaccount.com
IMG=$REGION-docker.pkg.dev/$PROJECT/aic-containers/aic-predict:$(git rev-parse --short HEAD)

gcloud run jobs create aic-predict --project=$PROJECT --region=$REGION --image="$IMG" \
  --cpu=8 --memory=16Gi --task-timeout=2h --max-retries=0 --tasks=1 --service-account=$SA
gcloud run jobs create aic-attribute --project=$PROJECT --region=$REGION --image="$IMG" \
  --cpu=8 --memory=32Gi --task-timeout=24h --max-retries=0 --tasks=1 --service-account=$SA
```

- `--task-timeout`: the default is 10 minutes. It kills runs partway, and the exit trap
  still uploads the partial chips, so check the `[predict] wrote N chips` count.
- `--memory`: Cloud Run's filesystem is in RAM, so it must also hold the scratch chips
  and the mosaics.
- A job pins its image digest. The CV scripts re-point the job (`gcloud run jobs update
  --image`) and refuse to run if that update didn't take.

## Running

The CV scripts read rows from `out/cv/protocol.csv` and submit one async execution per
(fold, year). Each script's header documents its env overrides.

```bash
docker/run_cv_predict.sh factored_v3p_union4_monthlyattn_wide_yeargain folds
docker/download_cv_preds.sh factored_v3p_union4_monthlyattn_wide_yeargain folds
YEARS="2023 2024 2025" docker/run_cv_attribute.sh factored_v3p_union4_monthlyattn_wide_yeargain final
YEARS="2023 2024 2025" docker/download_cv_attr.sh factored_v3p_union4_monthlyattn_wide_yeargain final
```

Single run with explicit env (`--update-env-vars` overrides the job's values):

```bash
gcloud run jobs execute aic-predict --region=$REGION --project=$PROJECT --async --update-env-vars=\
CONFIG_PATH=gs://aic-amazon/configs/cv/<arch>/final_all.json,\
CHECKPOINT=gs://aic-amazon/models/cv/<arch>/final_all.pt,\
STATS_PATH=gs://aic-amazon/data/fullgrid_v3_patched/stats_cv/final_all.json,\
DATA_DIR=gs://aic-amazon/data/fullgrid_v3_patched/allpreds_2024/,\
OUTPUT_URI=gs://aic-amazon/preds/cv/<arch>/final_all_2024/,UPLOAD_TILES=1
```

Always pass the training `STATS_PATH`: normalizing with per-year stats erases the
year-to-year signal. Check for the `[predict] normalizing with stats: ...` log line.

## Environment variables

- **Required:** `CONFIG_PATH`, `CHECKPOINT`, `DATA_DIR`, `OUTPUT_URI`.
- **Optional:** `MODE` (`predict`/`attribute`), `STATS_PATH`, `TFRECORD_PATTERN`,
  `MAX_CHIPS`, `BATCH_SIZE`, `SEED`, `EDGE_CROP` (0), `MOSAIC` (1), `MOSAIC_NAME`
  (`preds`), `UPLOAD_TILES` (0), `PROFILE_TEMPLATE` (default `assets/example_v3.tif`,
  MODIS sinusoidal 463 m).
- **Attribute only:** `DRIVERS` (a `gs://` spec, since `configs/` is not in the image),
  `SHAPLEY`, `SHAPLEY_SAMPLES`, `POS_WEIGHT`, `WRITE_MASK`.

## Local test

```bash
docker build -f docker/Dockerfile -t aic-predict:dev .
docker run --rm --user 0:0 -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" -e HOME=/root \
  -e CONFIG_PATH=... -e CHECKPOINT=... -e STATS_PATH=... -e DATA_DIR=... \
  -e TFRECORD_PATTERN='full-00005-of-*.tfrecord.gz' -e MAX_CHIPS=3 -e BATCH_SIZE=1 \
  -e OUTPUT_URI=gs://aic-amazon/preds/_test/ aic-predict:dev
```

`--user 0:0` is only needed under rootless Docker, where it lets the container read the
mounted credentials.
