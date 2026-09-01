# Containerized prediction

CPU-only image that runs a full year of prediction as a **Cloud Run job**.

The container does the whole job: predict → mosaic → upload.

Cloud Run rather than a Compute Engine VM because [the container startup agent
that backed `gcloud compute instances create-with-container` was deprecated on
2025-07-21][dep], and Cloud Run is Google's recommended replacement for
"stateless container applications and small to medium jobs". It also removes
the VPC/Private-Google-Access and VM-lifecycle problems entirely: a Cloud Run
job reaches GCS with no networking configuration, and it stops on its own.

[dep]: https://cloud.google.com/compute/docs/deprecations/container-startup-agent-on-compute

Measured: a full year is 1,813 chips at ~2.2 s/batch of 4 on 8 vCPU, so
**~18–20 minutes** end to end, producing ~150 MB of chips plus the mosaics.

## One-time project setup

```bash
PROJECT=macedo-lab-general-9051
REGION=us-east1
SA=211266473926-compute@developer.gserviceaccount.com

# Neither Cloud Build nor Cloud Run is enabled on this project yet.
gcloud services enable cloudbuild.googleapis.com run.googleapis.com --project=$PROJECT

# The existing `geebeam` repo is a REMOTE_REPOSITORY (a Docker Hub pull-through
# cache) and cannot be pushed to. A standard repo is required.
gcloud artifacts repositories create aic-containers \
  --repository-format=docker --location=$REGION \
  --description="First-party container images" --project=$PROJECT
```

No IAM grants should be needed: `$SA` already holds `roles/editor`, and
`gs://aic-amazon` grants object access via the legacy `projectEditor` ACL.

## Build

```bash
gcloud builds submit --config=cloudbuild.yaml --region=$REGION --project=$PROJECT \
  --substitutions=_TAG=$(git rev-parse --short HEAD) .
```

If this fails on a missing default Cloud Build service account or logs bucket
(common right after enabling the API), add:

```bash
  --service-account=projects/$PROJECT/serviceAccounts/$SA \
  --default-buckets-behavior=REGIONAL_USER_OWNED_BUCKET
```

Expect ~8–12 min cold, ~2 min for a source-only change (the dependency layer is
cached). The `.gcloudignore` in the repo root is load-bearing: without it
`gcloud` derives one from `.gitignore`, whose `*.tif` rule would silently drop
`assets/example.tif` and produce an image that builds fine and fails at runtime.

## Create the job (once)

```bash
IMG=$REGION-docker.pkg.dev/$PROJECT/aic-containers/aic-predict:$(git rev-parse --short HEAD)

gcloud run jobs create aic-predict \
  --project=$PROJECT --region=$REGION \
  --image="$IMG" \
  --cpu=8 --memory=16Gi \
  --task-timeout=2h --max-retries=0 --tasks=1 \
  --service-account=$SA \
  --set-env-vars=\
CONFIG_PATH=gs://aic-amazon/configs/mtsvit_test_v40.json,\
CHECKPOINT=gs://aic-amazon/models/mtsvit_test_v40.pt,\
DATA_DIR=gs://aic-amazon/data/fullgrid_v2/allpreds_2025/,\
OUTPUT_URI=gs://aic-amazon/preds/mtsvit_v40_2025/,\
EDGE_CROP=0,INVERT_YRES=1,MOSAIC=1,BATCH_SIZE=4,OMP_NUM_THREADS=8
```

Load-bearing flags:

- **`--task-timeout=2h` is mandatory.** The default is **10 minutes**, which is
  shorter than the ~18-minute run — without this the task is killed partway.
  (The trap in `entrypoint.sh` still uploads whatever finished, so a timeout
  looks like a suspiciously short run rather than an obvious failure. Watch the
  `[predict] wrote N chips` count.) The ceiling is 168 h.
- **`--cpu=8` is the maximum** a Cloud Run job allows, so this is half the
  16 vCPU a VM could give. That is already reflected in the ~18 min estimate.
  Valid memory for 8 vCPU is 4–32 GiB.
- **`--memory=16Gi` also has to cover scratch files.** Cloud Run's filesystem is
  in-memory, so everything `entrypoint.sh` writes to `/scratch` counts against
  this. A year is ~150 MB of chips plus mosaics, so 16 GiB is generous; do not
  drop to 4 GiB.
- `--max-retries=0` — the default is 3, and a failure here is not usually
  transient, so retrying just burns ~20 min three more times.
- `--tasks=1` — no sharding needed at this runtime. See "Going parallel" below
  if that changes.
- No `--vpc-connector` / `--subnet`. Cloud Run reaches GCS over Google's
  network with no VPC configuration; the Private Google Access and Cloud NAT
  concerns that constrained the VM design do not apply.

## Run it

```bash
gcloud run jobs execute aic-predict --region=$REGION --project=$PROJECT --wait
```

For another year or model, override at execution time instead of editing the
job:

```bash
gcloud run jobs execute aic-predict --region=$REGION --project=$PROJECT --wait \
  --update-env-vars=\
DATA_DIR=gs://aic-amazon/data/fullgrid_v2/allpreds_2023/,\
OUTPUT_URI=gs://aic-amazon/preds/mtsvit_v39_2023/
```

Watch it:

```bash
gcloud logging read \
  'resource.type="cloud_run_job" AND resource.labels.job_name="aic-predict"' \
  --project=$PROJECT --limit=100 --format='value(textPayload)'
```

The line to look for is `[predict] normalizing with stats: ...` — it is the
proof that the pooled training stats are in use rather than per-year stats.
Then check `[predict] wrote N chips` against the ~1,813 a full grid should
produce, and `[sync] uploaded ...` for the mosaics.

## Going parallel

If a year ever outgrows the timeout, the data is already 40 shards named
`full-000NN-of-00040.tfrecord.gz`, so one task per shard needs no code change —
`entrypoint.sh` only has to derive `TFRECORD_PATTERN` from
`CLOUD_RUN_TASK_INDEX`, and the job runs with `--tasks=40 --parallelism=10`.
Two consequences: `UPLOAD_TILES=1` becomes required (each task must ship its own
chips), and `MOSAIC` has to move to a separate final pass over the uploaded
chips, because no single task sees the whole grid.

## Attribution runs

`MODE=attribute` runs `attribute.py` instead of `predict.py` through the same
entrypoint (predict → mosaic → upload becomes attribute → mosaic → upload). It
writes one multi-band per-driver raster per chip — Shapley values with
`SHAPLEY=1`, one-at-a-time occlusion otherwise — and mosaics them exactly like a
predict run. The driver spec is read from `gs://` (`DRIVERS`), because `configs/`
is not copied into the image.

**Cost.** Exact Shapley is **`2^N` model forwards per chip** for `N` driver
groups, versus 1 for predict and `N+2` for one-at-a-time.

Two ways to reduce, both in `attribution.shapley_bands`: fewer driver groups (cost is
exponential in group *count*, not membership size), and `SHAPLEY_SAMPLES=S`
(Monte-Carlo, ~`N*S` forwards; the residual-band efficiency check still holds).

### The `aic-attribute` job (create once)

A **separate** job from `aic-predict`: `--memory` and `--task-timeout` are
job-level, not per-execution, and both differ. Memory is **32 Gi** (the max at
8 vCPU) because attribution writes ~10 float32 bands/chip (≈10× predict) into
Cloud Run's in-memory filesystem alongside the mosaic. Set `--task-timeout` from
the measured `s/chip` (ceiling 168 h); the 24 h below is a placeholder.

```bash
IMG=$REGION-docker.pkg.dev/$PROJECT/aic-containers/aic-predict:$(git rev-parse --short HEAD)

gcloud run jobs create aic-attribute \
  --project=$PROJECT --region=$REGION --image="$IMG" \
  --cpu=8 --memory=32Gi --task-timeout=24h --max-retries=0 --tasks=1 \
  --service-account=$SA \
  --set-env-vars=\
MODE=attribute,SHAPLEY=1,\
DRIVERS=gs://aic-amazon/configs/attribution_drivers_v44.json,\
CONFIG_PATH=gs://aic-amazon/configs/mtsvit_test_v44.json,\
CHECKPOINT=gs://aic-amazon/models/mtsvit_test_v44.pt,\
DATA_DIR=gs://aic-amazon/data/fullgrid_v2/allpreds_2024/,\
OUTPUT_URI=gs://aic-amazon/attr/mtsvit_v44_2024/,\
EDGE_CROP=0,INVERT_YRES=1,MOSAIC=1,BATCH_SIZE=4,OMP_NUM_THREADS=8
```

Then run a year (or loop years) with `docker/run_attribute.sh`, or override at
execution time as with predict. Remember Cloud Run pins the tag to a digest at
create time, so after a rebuild the job needs `gcloud run jobs update
aic-attribute --image="$IMG"` before `execute` sees the new code.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MODE` | `predict` | `predict` → `predict.py`; `attribute` → `attribute.py` |
| `CONFIG_PATH` | *required* | training config JSON, local or `gs://` |
| `CHECKPOINT` | *required* | `.pt` checkpoint, local or `gs://` |
| `DATA_DIR` | *required* | directory of `*.tfrecord.gz`, local or `gs://` |
| `OUTPUT_URI` | *required* | `gs://` prefix for the mosaics |
| `STATS_PATH` | `config['stats_path']` | normalization stats override |
| `TFRECORD_PATTERN` | `*.tfrecord.gz` | narrow to one shard for test runs |
| `MAX_CHIPS` | all | stop after N chips |
| `BATCH_SIZE` | 4 | |
| `SEED` | unset | makes shard/chip order reproducible |
| `EDGE_CROP` | 0 | |
| `INVERT_YRES` | 1 | pass `--invert_yres` |
| `MOSAIC` | 1 | run `gdalbuildvrt`/`gdal_translate` |
| `MOSAIC_NAME` | `preds` | output basename |
| `UPLOAD_TILES` | 0 | also upload the ~3,600 per-chip tiles |
| `DRIVERS` | built-in default | *(attribute)* `gs://` driver-spec JSON |
| `SHAPLEY` | 0 | *(attribute)* `1` → Shapley values, else OAT occlusion |
| `SHAPLEY_SAMPLES` | unset | *(attribute)* Monte-Carlo permutations (~`N*S` forwards) |
| `POS_WEIGHT` | `config['pos_weight']` | *(attribute)* deflation weight |
| `WRITE_MASK` | 0 | *(attribute)* also write the ground-truth `mask_*.tif` |

## Local test

```bash
docker build -f docker/Dockerfile -t aic-predict:dev .

docker run --rm --user 0:0 \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" -e HOME=/root \
  -e CONFIG_PATH=gs://aic-amazon/configs/mtsvit_test_v40.json \
  -e CHECKPOINT=gs://aic-amazon/models/mtsvit_test_v40.pt \
  -e DATA_DIR=gs://aic-amazon/data/fullgrid_v2/allpreds_2025/ \
  -e TFRECORD_PATTERN='full-00005-of-00040.tfrecord.gz' \
  -e MAX_CHIPS=3 -e SEED=0 -e BATCH_SIZE=1 \
  -e OUTPUT_URI=gs://aic-amazon/preds/_test/ \
  aic-predict:dev
```

`--user 0:0` is only needed under **rootless** Docker, where container uid 1000
maps to a subuid that cannot read the bind-mounted ADC file. On Cloud Run the
container runs as `appuser` and takes credentials from the metadata server, so
no mount and no key file are involved.
