# aic-risk-modeling

Annual fire-risk forecasts for the Amazon basin. Each January, the model predicts the
probability that every ~463 m pixel burns that year, using drivers known by the end of
the previous year: climate indices, weather, vegetation, land use, fire history and
terrain.

The paper model is a factored network. Its logit is a sum of separable terms:

    logit = gamma(t) * (1 + g_res(location)) + m (coarse intensity) + s (pixel susceptibility) + c (local context)

- `gamma(t)` is a frozen year offset, fit offline from SOI and the previous year's
  basin burn.
- `s` and `c` are disjoint by construction, so they separate ignition from spread.

## Installation

Python >= 3.10 (developed on 3.11), managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra torch_cpu --group dev     # package + CPU torch + pytest
```

Add `--extra vertex` (google-cloud-aiplatform) to launch training on Vertex AI.

`torch` is optional on purpose: Vertex AI training gets it from the prebuilt PyTorch
container. Data loading uses tf.data, so `tensorflow-cpu` is a core dependency.

## Layout

- `src/aic_risk_modeling/`
  - `preprocess/`: climate-index download.
  - `train/`: tf.data TFRecord loaders, normalization, the PyTorch models
    (`models.py`, factored model in `factored.py`) and the training loop (`trainer.py`).
  - `eval/`: metrics, calibration, year offset, chip I/O and driver attribution.
  - `predict/`: the prediction pipeline shared by `scripts/predict/`.
- `scripts/`: command-line steps of the pipeline, described below. Each script has a
  `Usage:` line in its module docstring.
- `docker/`: the Cloud Run prediction/attribution image and the CV submit/download
  scripts (see [docker/README.md](docker/README.md)).
- `configs/example_config.json`: the paper model's `final_all` training config. Other
  configs are local-only and live on GCS.

## Pipeline

1. **Inputs** (`scripts/preprocessing/`)
   - `geebeam_ali_inputs.py` exports one TFRecord dir per target year from Earth Engine
     via Dataflow (`run_df.sh`), giving 128x128 chips on a MODIS sinusoidal grid.
   - `patch_features.py` replaces bands in existing exports.
   - `pool_stats_pbtxt.py` pools per-year stats into the training `stats_path`.
   - `preprocess_fire_detections.py` (`run_*_prep.sh`) rasterizes VIIRS/MODIS hotspots.
2. **Train**
   - Locally: `python -m aic_risk_modeling.train.trainer --config_path CONFIG`.
   - On Vertex AI: `python scripts/train/train_vertex.py gs://.../config.json JOB_NAME`.
     This needs the package sdist uploaded first, as
     `gs://aic-amazon/python_packages/aic_risk_modeling-<version>.tar.gz`.
   - Checkpoints bundle their config, so `aic_risk_modeling.train.load_model(path)`
     rebuilds the model.
3. **Temporal CV** (`scripts/cross_validation/`)
   - `cv_make_folds.py --arch ARCH` writes the protocol: forward-pair folds trained on
     2013..t-1 and evaluated on {t, t+1} for t = 2018..2022, plus `final_all`. 2024/25
     are a write-once test set that never enters a fold.
   - Before launching, check with `config_check.py` and `cv_preflight.py`.
   - Score with `cv_collect_results.py` (or `cv_score_parallel.py`), then run the
     post-hoc ablations with `cv_year_sensitivity.py`.
4. **Predict and attribute** (Cloud Run, see `docker/`)
   - Submit with `run_cv_predict.sh` or `run_cv_attribute.sh`, and fetch with
     `download_cv_preds.sh` or `download_cv_attr.sh`.
   - To run locally instead, use `scripts/predict/predict.py` and `attribute.py`, then
     `mosaic.sh`.
5. **Analysis** (`scripts/analysis/`)
   - The gamma fit: `fit_year_offset.py`, using the panels from `build_target_panel.py`
     and `extract_chip_panel.py`.
   - Label mosaics and climatology: `build_label_mosaics.py` and `build_climatology.py`.
   - The frozen Platt calibrator and expected-vs-actual totals: `calibrated_year_totals.py`.
6. **Figures** (`scripts/figures/`, written to `out/figures/` as PNG @ 300 dpi + PDF)

   | Figure | Script |
   |---|---|
   | 1, model landscape | `make_risk_landscape_figure.py` |
   | 2, architecture + timeline | `make_architecture_figure.py` |
   | 3, spatial pyramid | `make_pyramid_figure.py` |
   | 4, risk map and chips | `make_risk_figure_2024.py` |
   | 5, expected vs actual | `plot_expected_actual.py` |
   | 6, forest vs non-forest | `make_forest_figure.py` |
   | 7, Shapley drivers | `make_shapley_figure.py`, `make_shapley_maps.py` |

   Also: `make_forecast_figure.py` and `make_soi_burn_figure.py`.

## Tests

```bash
uv sync --extra torch_cpu --group dev
uv run pytest
```

CI (`.github/workflows/tests.yml`) runs the same on pushes to `main` and on pull
requests. Tests that need local-only files (checkpoints in `out/`, configs, the chip
panel) are skipped.

## License

MIT, see [LICENSE](LICENSE).
