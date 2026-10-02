#!/usr/bin/env python
"""GCS-free CPU build check for training configs. Read-only, no Vertex.

Builds a config's real model (build_all_models + build_decoder) and runs one
forward + backward pass on synthetic batches shaped exactly as build_all_models
derives them from the config. Validates decoder wiring -- band counts, timestep
folding, md_monthly length, coord/transformer branches, and the output head --
without reading any gs:// data, so it is the cheap gate before a Vertex launch.

Complements cv_preflight.py, which covers only the factored/gamma line: this
handles any decoder (fusion, vit, factored, ...) and needs no gamma. The 3-way
union target is a label-side combine in the data pipeline and is not exercised
here (see tests/test_union_label.py); val_cache is train-time only
(tests/test_val_cache.py).

Usage:
    .venv/bin/python scripts/cross_validation/config_check.py            # the non-gamma v3p archs
    .venv/bin/python scripts/cross_validation/config_check.py configs/unet_v3p_union4.json ...
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from aic_risk_modeling.train import trainer  # noqa: E402

B = 2
# The factored arch is left out: its gamma coeffs_path is on GCS.
DEFAULT_CONFIGS = ["configs/lstm_v3p_union4.json", "configs/mlp_v3p_union4_flat.json",
                   "configs/unet_v3p_union4.json", "configs/vit_test_v3p_union4.json"]


def synth_shape(spec):
    """Batch tensor shape for one input group, mirroring build_all_models."""
    T = len(spec["timesteps"])
    C = len(spec["feature_names"])
    shape = spec["shape"]
    if T > 0 and spec.get("stack_timesteps"):
        return [B, T] + shape + [C]
    if T > 0:
        return [B] + shape + [C * T]
    return [B] + shape + [C]


def check(config_path):
    cfg = json.load(open(config_path))
    name = os.path.basename(config_path)
    print(f"\n=== {name}  decoder={cfg['decoder']} ===")
    branches = trainer.build_all_models(cfg["input_features"])
    model = trainer.build_decoder(cfg["decoder"], branches, cfg.get("decoder_config"))
    inputs = {g: torch.randn(*synth_shape(spec))
              for g, spec in cfg["input_features"].items()}
    # A year_offset (gamma) arch looks md_year up in its table and refuses anything
    # outside it, so feed real in-range years there instead of noise.
    year_group = (cfg.get("decoder_config") or {}).get("year_group")
    if year_group in inputs:
        years = torch.tensor([2018.0, 2019.0][:B] + [2019.0] * max(0, B - 2))
        inputs[year_group] = years.reshape([B] + [1] * (inputs[year_group].dim() - 1)).expand_as(
            inputs[year_group]).clone()
    for g, t in inputs.items():
        print(f"  in  {g:16s} {tuple(t.shape)}")
    out = model(inputs)
    print(f"  out {tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")
    assert out.shape[0] == B and tuple(out.shape[-2:]) == tuple(cfg["output_features"]["shape"]), out.shape
    assert torch.isfinite(out).all()
    # One backward step must reach every branch's params.
    loss = torch.nn.functional.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6),
                                                     torch.zeros_like(out))
    loss.backward()
    grads = [torch.isfinite(p.grad).all() for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  backward OK: {len(grads)}/{n_params} trainable params got finite grads")
    assert grads and all(grads)
    print(f"  PASS {name}")


def main():
    paths = sys.argv[1:] or DEFAULT_CONFIGS
    for p in paths:
        check(os.path.join(REPO, p) if not os.path.isabs(p) else p)
    print("\nALL CONFIG CHECKS PASSED")


if __name__ == "__main__":
    main()
