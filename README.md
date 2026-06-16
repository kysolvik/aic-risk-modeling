# aic-risk-modeling

Amazon basin fire risk modeling utilities.

The package is organized into three subpackages:

- **`preprocess`** — preprocessing utilities (e.g. downloading climate indices)
- **`train`** — PyTorch models and training loop, plus tf.data-based TFRecord loaders
  (PyTorch itself comes from the optional `train` dependency group)
- **`eval`** — evaluation utilities for comparing model predictions against ground truth

Models and the training loop are PyTorch. Data loading still uses tf.data TFRecord
pipelines, so `tensorflow-cpu` is a core dependency (it stays off the GPU); `torch` is
not — on Vertex AI it comes from the prebuilt PyTorch training containers.

Subpackages are imported lazily (PEP 562), so `import aic_risk_modeling` does not import
heavy dependencies until you actually access `aic_risk_modeling.train`.

## Installation

Requires Python >= 3.10 (developed against 3.11).

### With uv (recommended)

```bash
uv sync
```

To include PyTorch for training:

```bash
uv sync --group train
```

### With pip

```bash
pip install .
```

For training, also install PyTorch (any build >= 2.3):

```bash
pip install torch
```

## Usage

### `preprocess`

Download non-spatial climate indices from NOAA. Supported indices: `amo`, `soi`, `oni`, `mei`, `tna`.

```python
from aic_risk_modeling.preprocess import download_clim_indices

# Returns a monthly pandas DataFrame indexed by date, with a single 'metric' column
df = download_clim_indices("oni", year_start=2000, year_end=2020)
print(df.head())
```

Arguments:

- `index_name` — one of `amo`, `soi`, `oni`, `mei`, `tna`
- `year_start` — first year to include (samples are monthly)
- `year_end` — last year to include

### `train`

Training is config-driven (see `configs/example_config.json`): the config declares the
TFRecord data dirs, input feature groups (each with its own branch model, e.g.
`convlstm_bottleneck`, `unet_lite`, `mlp_for_fusion`, `transformer`, `identity`), the
decoder (`fusion`, or the TSViT/MTSViT-inspired `mtsvit` — see
`configs/mtsvit_test_v1.json`), loss, and optimizer settings.

```bash
python -m aic_risk_modeling.train.trainer --config_path configs/example_config.json
```

The trainer checkpoints on best `val_pr_auc` with early stopping, logs per-epoch metrics
to `training.csv`, and saves the best model to `model_output_path` (local or `gs://`).
Models are saved as a `torch.save` payload containing the config and the `state_dict`,
so they can be rebuilt without the original config file:

```python
from aic_risk_modeling.train import load_model

model = load_model("gs://bucket/models/fusion_test.pt")  # returns an eval-mode nn.Module
```

To launch on Vertex AI (uses the prebuilt PyTorch GPU container; the package sdist on
GCS provides the data-loading deps):

```bash
python scripts/train_vertex_new.py gs://bucket/path/to/config.json my-job-name
```

### `eval`

Compute classification metrics (accuracy, precision, recall, F1, Cohen's kappa, PR AUC) for
fire model predictions against ground truth.

#### As a library

```python
from aic_risk_modeling.eval import load_preprocess_inputs, calc_stats

# Inputs may be paired CSVs (predictions need a 'pred' column, ground truth a 'label' column)
# or paired GeoTIFFs (the first band of each is read).
predictions, ground_truth = load_preprocess_inputs(
    "predictions.csv",
    "ground_truth.csv",
)

stats = calc_stats(predictions, ground_truth, threshold=0.5)
# stats == (accuracy, precision, recall, f1, kappa, pr_auc, n_truth, n_pred)
```

Pass `grouped=True` to additionally break out metrics per unique non-zero ground-truth value.

#### As a command-line tool

```bash
python -m aic_risk_modeling.eval \
    --predictions predictions.csv \
    --ground_truth ground_truth.csv \
    --threshold 0.5
```

Options:

- `--predictions` (required) — path to predictions (`.tif`, or `.csv` with a `pred` column)
- `--ground_truth` (required) — path to ground truth (`.tif`, or `.csv` with a `label` column)
- `--threshold` — threshold for converting predictions to binary (default `0.5`)
- `--grouped` — also report metrics grouped by each unique ground-truth value

> Note: predictions and ground truth must use the same format — both CSV or both GeoTIFF.

## License

MIT See [LICENSE](LICENSE).
