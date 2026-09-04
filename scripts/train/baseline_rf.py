"""Standalone tabular Random-Forest / gradient-boosting baseline.

The neural baselines (MLP/LSTM/U-Net/ConvLSTM) and the MTSViT champion are
segmentation models trained through the tf.data + PyTorch pipeline. A tree model
does not fit that pipeline, so this trains one directly on a PER-PIXEL tabular
view of the SAME variable pool and label as the reference config
(`configs/mtsvit_test_v44.json`), then scores every val pixel so the result drops
straight into `scripts/analysis/compare_baselines.py` alongside the neural nets.

"Matched info": each pixel's feature row is every (image band x timestep) value
at that pixel, plus the per-chip climate-index sequence and coordinates broadcast
across the chip -- i.e. exactly the variables v44 sees, flattened to a table.
Trees are scale-invariant, so no normalization is applied (unlike the NN path);
only finite nodata sentinels are cleaned to NaN. HistGradientBoosting consumes
NaN natively and is the default (fast on this volume); RandomForest is available
with median imputation.

Usage:
    # 1. Fit on train years (subsampled, class-balanced)
    .venv/bin/python scripts/train/baseline_rf.py fit \
        --config configs/mtsvit_test_v44.json \
        --data_dirs gs://aic-amazon/data/fullgrid_v2/allpreds_20{13,14,15,16,17,18,19,20,21,22}/ \
        --model_out out/baselines/rf/model.joblib --model histgb

    # 2. Predict every val pixel -> (scores, labels) npz for the eval harness
    .venv/bin/python scripts/train/baseline_rf.py predict \
        --config configs/mtsvit_test_v44.json \
        --data_dirs gs://aic-amazon/data/fullgrid_v2/allpreds_20{23,24}/ \
        --model_out out/baselines/rf/model.joblib \
        --pred_out out/baselines/rf/val_preds.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import tensorflow as tf  # noqa: E402

from aic_risk_modeling.train import data_loader  # noqa: E402
from aic_risk_modeling.train.trainer import load_config  # noqa: E402

PATCH = 128
PATCH_PIXELS = PATCH * PATCH

# Finite nodata sentinels that would otherwise become ordinary split points.
NODATA = {"im_Elevation": -32767.0, "im_accessibility": -9999.0}
# gt0 / gt0_bool both boolean-ize; that is the only transform kind in the configs.
_GT0 = {"gt0", "gt0_bool", "gt1_bool", "gt2_bool"}


def feature_plan(config):
    """Derive the per-pixel tabular layout from an `input_features` config.

    Returns a dict describing the columns to build for each chip:
      image_cols : list of (schema_key, transform_or_None) -- one per pixel
      vec_cols   : list of (schema_key, length) climate-index sequences,
                   broadcast across the chip (constant per pixel)
      scalar_cols: list of schema_key -- per-chip scalars (coords), broadcast
      label_keys : the output bands to union into the binary target
      names      : flat ordered column names (for feature-importance readouts)
    """
    image_cols, vec_cols, scalar_cols = [], [], []
    for br in config["input_features"].values():
        transforms = br.get("transforms", {}) or {}
        is_image = len(br["shape"]) == 2  # [128,128] spatial rasters
        ts = br["timesteps"]
        if is_image:
            bases = br["feature_names"]
            keys = ([f"{b}_{t}" for b in bases for t in ts] if ts else list(bases))
            for k in keys:
                base = k.rsplit("_", 1)[0] if ts else k
                tf_name = transforms.get(k) or transforms.get(base)
                image_cols.append((k, tf_name))
        elif br["shape"] == [1]:
            scalar_cols.extend(br["feature_names"])
        else:
            length = br["shape"][0]
            vec_cols.extend((k, length) for k in br["feature_names"])
    out = config["output_features"]
    ots = out.get("timesteps") or []
    label_keys = ([f"{b}_{t}" for b in out["feature_names"] for t in ots]
                  if ots else list(out["feature_names"]))
    # Column names in the exact order chip_matrix stacks them: image, vec, scalar.
    names = ([k for k, _ in image_cols]
             + [f"{k}_m{i}" for k, length in vec_cols for i in range(length)]
             + list(scalar_cols))
    return {"image_cols": image_cols, "vec_cols": vec_cols,
            "scalar_cols": scalar_cols, "label_keys": label_keys, "names": names}


def wanted_keys(plan):
    keys = {k for k, _ in plan["image_cols"]}
    keys |= {k for k, _ in plan["vec_cols"]}
    keys |= set(plan["scalar_cols"]) | set(plan["label_keys"])
    return keys


def _apply_transform(arr, tf_name):
    if tf_name in _GT0:
        return (arr > 0).astype(np.float32)
    return arr  # normalize_* transforms are for the NN path; trees skip them


def chip_matrix(rec, plan):
    """One parsed chip -> (X (PATCH_PIXELS, F) float32, y (PATCH_PIXELS,) bool).

    `rec` maps schema key -> numpy array: image bands are (128,128), climate
    indices are (length,), coordinates are scalars/(1,). Pure numpy so it is
    unit-testable without any TFRecords.
    """
    cols = []
    for key, tf_name in plan["image_cols"]:
        arr = np.asarray(rec[key], dtype=np.float32).reshape(-1)
        nod = NODATA.get(key.rsplit("_", 1)[0], NODATA.get(key))
        if nod is not None:
            arr = np.where(arr == nod, np.nan, arr)
        cols.append(_apply_transform(arr, tf_name))
    for key, length in plan["vec_cols"]:
        vec = np.asarray(rec[key], dtype=np.float32).reshape(-1)[:length]
        cols.extend(np.full(PATCH_PIXELS, v, np.float32) for v in vec)
    for key in plan["scalar_cols"]:
        v = float(np.asarray(rec[key]).reshape(-1)[0])
        cols.append(np.full(PATCH_PIXELS, v, np.float32))
    X = np.stack(cols, axis=1)

    y = np.zeros(PATCH_PIXELS, dtype=bool)
    for key in plan["label_keys"]:
        y |= np.asarray(rec[key], dtype=np.float32).reshape(-1) > 0
    return X, y


def _write_out(path, writer):
    """Write via `writer(local_path)`, copying to GCS when `path` is gs://.

    Cloud Run's filesystem is ephemeral (tmpfs), so outputs must land in a
    bucket; joblib/np.savez only speak local paths, hence the temp+copy.
    """
    if path.startswith("gs://"):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, os.path.basename(path))
            writer(local)
            tf.io.gfile.copy(local, path, overwrite=True)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        writer(path)


def _load_joblib(path):
    from joblib import load
    if path.startswith("gs://"):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, os.path.basename(path))
            tf.io.gfile.copy(path, local)
            return load(local)
    return load(path)


def _iter_chips(data_dirs, plan, pattern, max_chips=None):
    spec = None
    seen = 0
    for d in data_dirs:
        schema = data_loader.load_schema_from_gcs(d)
        full = data_loader.schema_to_feature_spec(schema)
        want = wanted_keys(plan)
        missing = [k for k in want if k not in full]
        if missing:
            raise KeyError(f"{len(missing)} bands absent from {d}: {missing[:6]}")
        spec = {k: full[k] for k in want}
        shards = sorted(tf.io.gfile.glob(os.path.join(d, pattern)))
        ds = tf.data.TFRecordDataset(shards, compression_type="GZIP")
        ds = ds.map(lambda x: tf.io.parse_single_example(x, spec))
        for rec in ds.as_numpy_iterator():
            yield chip_matrix(rec, plan)
            seen += 1
            if max_chips and seen >= max_chips:
                return


def cmd_fit(args):
    from joblib import dump
    config = load_config(args.config)
    plan = feature_plan(config)
    rng = np.random.default_rng(args.seed)
    Xs, ys, n_rows = [], [], 0
    for X, y in _iter_chips(args.data_dirs, plan, args.tfrecord_pattern,
                            args.max_chips):
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        take_neg = min(neg.size, max(1, args.neg_per_pos * pos.size))
        keep = np.concatenate([pos, rng.choice(neg, take_neg, replace=False)])
        Xs.append(X[keep])
        ys.append(y[keep])
        n_rows += keep.size
        if n_rows >= args.max_train_rows:
            break
    X = np.concatenate(Xs)[: args.max_train_rows]
    y = np.concatenate(ys)[: args.max_train_rows]
    print(f"[fit] {X.shape[0]} rows x {X.shape[1]} features; "
          f"positives {int(y.sum())} ({y.mean():.3%})", flush=True)

    clf = _build_estimator(args)
    clf.fit(X, y)
    payload = {"model": clf, "names": plan["names"], "config": args.config,
               "kind": args.model}
    _write_out(args.model_out, lambda p: dump(payload, p))
    print(f"[fit] wrote {args.model_out}", flush=True)


def _build_estimator(args):
    if args.model == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=args.n_estimators, learning_rate=0.05,
            max_leaf_nodes=63, l2_regularization=1.0,
            class_weight="balanced", random_state=args.seed)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators, n_jobs=-1, random_state=args.seed,
        class_weight="balanced_subsample", max_depth=args.max_depth)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("rf", rf)])


def cmd_predict(args):
    bundle = _load_joblib(args.model_out)
    clf = bundle["model"]
    config = load_config(args.config)
    plan = feature_plan(config)
    scores, labels = [], []
    n = 0
    for X, y in _iter_chips(args.data_dirs, plan, args.tfrecord_pattern,
                            args.max_chips):
        p = clf.predict_proba(X)[:, 1].astype(np.float32)
        scores.append(p)
        labels.append(y)
        n += 1
        if n % 50 == 0:
            print(f"[predict] {n} chips", flush=True)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    _write_out(args.pred_out,
               lambda p: np.savez_compressed(p, scores=scores, labels=labels))
    print(f"[predict] {scores.size} pixels ({labels.mean():.3%} positive) "
          f"-> {args.pred_out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True,
                        help="reference config for the variable pool + label")
    common.add_argument("--data_dirs", nargs="+", required=True)
    common.add_argument("--tfrecord_pattern", default="*.tfrecord.gz")
    common.add_argument("--model_out", required=True)
    common.add_argument("--max_chips", type=int, default=None)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--model", choices=("histgb", "rf"), default="histgb")
    common.add_argument("--n_estimators", type=int, default=300)
    common.add_argument("--max_depth", type=int, default=None)

    f = sub.add_parser("fit", parents=[common])
    f.add_argument("--max_train_rows", type=int, default=1_000_000)
    f.add_argument("--neg_per_pos", type=int, default=10)
    f.set_defaults(func=cmd_fit)

    pr = sub.add_parser("predict", parents=[common])
    pr.add_argument("--pred_out", required=True)
    pr.set_defaults(func=cmd_predict)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
