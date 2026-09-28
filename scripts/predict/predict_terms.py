"""Write the factored model's additive log-odds terms as separate raster bands.

`predict.py` writes only the combined sigmoid. FactoredFireModel is additive in
log-odds --

    logit(x, t) = gamma(t) + m(x, t) + s(x, t) + c(x, t)
                  when       where       ignition   neighbourhood
                             (coarse)    (pointwise) (spread analogy)

-- and exposes `forward_terms` (src/aic_risk_modeling/train/factored.py). This
script runs that instead of `forward` and emits one band per term so the
ignition (`s`) and neighbourhood/spread (`c`) contributions can be mapped
separately.

Caveat: the bands are LOG-ODDS contributions, not probabilities. `s` and `c` are
fit against the same single burn label (one BCE), so `c` is a neighbourhood
association, not a mechanistic spread probability; see the plan/notes.

Everything except the forward pass and the band assembly is reused verbatim from
predict.py (same directory, so `import predict` resolves).
"""

import argparse
import os

import numpy as np
import rasterio as rio
import torch
from tqdm import tqdm

import aic_risk_modeling as arm

# predict.py lives next to this file; sys.path[0] is this dir when run as a script.
from predict import (
    DEFAULT_PROFILE_TEMPLATE,
    TFRECORD_PATTERN,
    add_md_sidecar,
    resolve_stats_path,
    set_raw_x_y,
    write_batch,
)

# Band layout of the written terms_*.tif. `prob` is the only non-log-odds band;
# it must reproduce predict.py's out_*.tif exactly (identity check).
TERM_BANDS = ["ignition_s", "spread_c", "coarse_m", "gamma", "logit_total", "prob"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--edge_crop", type=int, default=0)
    parser.add_argument("--invert_yres", action="store_true")
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--profile_template", type=str, default=DEFAULT_PROFILE_TEMPLATE)
    parser.add_argument("--tfrecord_pattern", type=str, default=TFRECORD_PATTERN)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_chips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--check_identity",
        action="store_true",
        help="also run model(inputs) and assert the assembled prob band matches "
        "it to <1e-4 (confirms the terms sum to the real logit)",
    )
    return parser.parse_args()


def assemble_terms(model, inputs):
    """Return an (B, H, W, K) tensor of the TERM_BANDS, in log-odds (+ prob)."""
    terms = model.forward_terms(inputs)
    s = terms["s"]                       # (B, 1, H, W)
    c = terms["c"]                       # (B, 1, H, W)
    m = terms["m"]                       # (B, 1, H, W)
    gamma = terms["gamma"]               # (B, 1, 1, 1) -- global per-year offset
    # year = gamma(t)*(1 + g_res(x)) = gamma + gamma*g_res. The optional per-location gain
    # is folded into the effective per-chip year contribution so logit_total stays exact and
    # the band schema is unchanged; it is zero for models without year_gain.
    year_gain = terms.get("year_gain", torch.zeros_like(gamma))   # (B, 1, 1, 1) -- per-chip
    year = gamma + year_gain
    logit = year + m + s + c             # broadcast the per-chip year term over H, W
    prob = torch.sigmoid(logit.float())
    year_full = year.expand_as(s)        # materialize the scalar as a plane
    stack = torch.cat([s, c, m, year_full, logit, prob], dim=1)  # (B, K, H, W)
    return stack.permute(0, 2, 3, 1)     # (B, H, W, K)


def main():
    args = parse_args()
    config = arm.train.trainer.load_config(args.config_path)
    stats_path = resolve_stats_path(args.stats_path, config, args.data_dir)
    print(f"[predict_terms] normalizing with stats: {stats_path}", flush=True)
    config = add_md_sidecar(config)

    ds = arm.train.build_merged_dataset(
        [args.data_dir], args.tfrecord_pattern, batch_size=args.batch_size,
        cache=False, axis="examples", shuffle=False, seed=args.seed)
    ds = ds.map(set_raw_x_y)
    normalize_list = arm.train.get_normalize_list(config)
    robust_features = arm.train.get_robust_normalize_list(config)
    norm_func = arm.train.create_normalizer(
        stats_path, normalize_list, robust_features=robust_features)
    ds = ds.map(norm_func)
    ds = arm.train.select_bands_transform(
        ds, input_feature_config=config["input_features"],
        output_feature_config=config["output_features"])

    model = arm.train.trainer.load_model(args.checkpoint)
    if not hasattr(model, "forward_terms"):
        raise TypeError(
            f"checkpoint {args.checkpoint} loaded a {type(model).__name__}, which "
            "has no forward_terms; predict_terms only applies to FactoredFireModel. "
            "A stale/mismatched checkpoint can load cleanly into the wrong model "
            "(see memory: predict-image-staleness-trap).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    with rio.open(args.profile_template) as src:
        profile = src.profile
    profile.update(dtype=rio.float32, count=1, compress="lzw")
    base_transform = profile["transform"]

    n_chips = 0
    max_id_err = 0.0
    with torch.no_grad():
        # No autocast: the additive head is read in fp32 (matches attribute.py),
        # so the term bands are the exact log-odds the model produced.
        for inputs, labels, weights, *_ in tqdm(
                arm.train.trainer._torch_batches(ds, device),
                desc="Predicting terms", unit="batch"):
            stack = assemble_terms(model, inputs)  # (B, H, W, K)
            if args.check_identity:
                ref = model(inputs)                # (B, H, W) sigmoid
                prob = stack[..., TERM_BANDS.index("prob")]
                max_id_err = max(max_id_err, float((prob - ref).abs().max()))

            md_sidecar = inputs["md_sidecar"]
            md_x_raw = md_sidecar[:, 0, 0].cpu().numpy()
            md_y_raw = md_sidecar[:, 0, 1].cpu().numpy()

            write_batch(
                stack.float().cpu().numpy(), labels.cpu().numpy(),
                md_x_raw, md_y_raw, base_transform, profile,
                args.output_dir, args.edge_crop, args.invert_yres,
                band_names=TERM_BANDS, out_prefix="terms")

            n_chips += int(labels.shape[0])
            if args.max_chips and n_chips >= args.max_chips:
                break

    if args.check_identity:
        print(f"[predict_terms] max |prob_band - model(inputs)| = {max_id_err:.2e}",
              flush=True)
        assert max_id_err < 1e-4, "assembled prob band does not match model forward"
    print(f"[predict_terms] wrote {n_chips} chips to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
