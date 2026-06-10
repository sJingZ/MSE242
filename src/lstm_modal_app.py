#!/usr/bin/env python3
"""Modal entrypoint for running the project's models on the cloud.

This wraps the local training scripts (currently ``cnn_lstm.py``) so they can be
launched on Modal's GPU workers straight from the CLI, with every result pulled
back to the local ``results/`` directory.

Quick start
-----------
1. One-time auth (opens a browser tab)::

       pip install -r requirements.txt
       modal setup

2. Run a model on a remote GPU and save results locally::

       # fast smoke test on a T4
       modal run src/modal_app.py --model cnn_lstm --quick

       # full run, pick the GPU and hyper-params
       modal run src/modal_app.py --model cnn_lstm --gpu A10G \
           --max-epochs 50 --batch-size 256 --lr 1e-3 --markets all

       # also pull the trained checkpoint (.pt) back
       modal run src/modal_app.py --model cnn_lstm --quick --save-model

3. List the models you can run::

       modal run src/modal_app.py --list-models

Results land exactly where local runs put them:
    results/experiments.jsonl              (one appended line per run)
    results/runs/<model>_<timestamp>.json  (full per-run record)
    results/runs/<model>_<timestamp>.pt    (only with --save-model)
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Paths (resolved on the *local* machine, at definition time)                  #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "of"
RESULTS_DIR = PROJECT_ROOT / "results"

# Remote layout inside the container.
REMOTE_CODE_DIR = "/root/models"
REMOTE_DATA_DIR = "/root/data/processed/of"

# --------------------------------------------------------------------------- #
# Model registry: CLI name -> (module file, importable module name)            #
# Add new models here and they become selectable via `--model <name>`.         #
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "cnn_lstm": {
        "file": SRC_DIR / "cnn_lstm.py",
        "module": "cnn_lstm",
        "label": "CNN-LSTM (OF order-flow mid-price-return forecaster)",
    },
    "lstm": {
        "file": SRC_DIR / "lstm.py",
        "module": "lstm",
        "label": "LSTM (OF order-flow mid-price-return forecaster)",
    },
    "lstm_mlp": {
        "file": SRC_DIR / "lstm_mlp.py",
        "module": "lstm_mlp",
        "label": "LSTM-MLP (OF order-flow mid-price-return forecaster)",
    },
}


def _build_image() -> modal.Image:
    """Container image: deps + every registered model file + the OF dataset.

    Runs at import time on BOTH the local machine and the remote worker, so it
    must not touch / validate the local filesystem (those paths don't exist on
    the worker). Existence checks live in the local entrypoint instead.
    """
    img = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("numpy", "scikit-learn", "torch")
    )
    # Attach local files only on the local side. On the worker the image is
    # loaded by its built id, so re-scanning local paths (which don't exist
    # there) is both unnecessary and unsafe.
    if modal.is_local():
        # Ship the dataset so the worker loads it the same way local runs do.
        img = img.add_local_dir(str(DATA_DIR), REMOTE_DATA_DIR)
        # Ship every registered model source file.
        for spec in MODEL_REGISTRY.values():
            f = spec["file"]
            img = img.add_local_file(str(f), f"{REMOTE_CODE_DIR}/{f.name}")
    return img


def _validate_local_files() -> None:
    """Local-only sanity checks (run from the entrypoint before dispatching)."""
    if not DATA_DIR.exists():
        raise SystemExit(
            f"Dataset dir not found: {DATA_DIR}\n"
            "Run the OF pipeline notebook first to produce data/processed/of/."
        )
    for name, spec in MODEL_REGISTRY.items():
        if not spec["file"].exists():
            raise SystemExit(f"Model '{name}' source not found: {spec['file']}")


app = modal.App("mse242-models")
image = _build_image()


# --------------------------------------------------------------------------- #
# Remote worker                                                                #
# --------------------------------------------------------------------------- #
# GPUs the --gpu flag can pick from.
GPU_CHOICES = ("T4", "L4", "A10G", "A100", "H100")


def _train_impl(model: str, params: dict, return_model: bool) -> dict:
    """Import the selected model on the worker and run one experiment.

    Returns the model's result payload: ``{"record": ..., "model_state": ...}``.
    We write nothing remotely (the container fs is ephemeral); the local
    entrypoint persists everything to the repo's ``results/`` dir.
    """
    import importlib

    sys.path.insert(0, REMOTE_CODE_DIR)
    mod = importlib.import_module(MODEL_REGISTRY[model]["module"])

    cfg_fields = {f.name for f in dataclasses.fields(mod.Config)}
    cfg = mod.Config(**{k: v for k, v in params.items() if k in cfg_fields})
    cfg.data_dir = REMOTE_DATA_DIR          # read the dataset shipped in the image
    cfg.results_dir = "/root/results"       # unused (write=False) but keep it valid

    out = mod.run_experiment(cfg, write=False, return_model=return_model)
    return out


# Register one Modal Function per GPU type. This lets the CLI --gpu flag select
# the accelerator on any Modal version, without relying on Function.with_options
# (only available in Modal >= 1.4).
TRAIN_FNS = {
    gpu: app.function(
        image=image,
        gpu=gpu,
        timeout=4 * 60 * 60,
        name=f"train_remote_{gpu.lower()}",
    )(_train_impl)
    for gpu in GPU_CHOICES
}


# --------------------------------------------------------------------------- #
# Local helpers                                                                #
# --------------------------------------------------------------------------- #
def _save_results(model: str, payload: dict, save_model: bool,
                  runs_subdir: str = "") -> None:
    """Persist a remote run's payload into the local results/ directory.

    Self-contained (stdlib only) so the Modal CLI's Python env does not need
    torch/numpy installed just to write the JSON results back. The format mirrors
    cnn_lstm.record_experiment exactly: a per-run JSON in results/runs/ plus an
    appended line in results/experiments.jsonl.

    ``runs_subdir`` redirects the per-run detail files (JSON + .pt + _pnl.npz)
    into results/runs/<runs_subdir>/ to keep e.g. a tuning sweep organized. The
    append-only experiments.jsonl log stays global so ranking tools see every
    run regardless of subdir.
    """
    import json
    import re

    record = payload["record"]
    runs_dir = RESULTS_DIR / "runs"
    if runs_subdir:
        runs_dir = runs_dir / runs_subdir
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = record["timestamp"].replace(":", "").replace("-", "").replace(".", "_")
    safe_ts = re.sub(r"[^0-9A-Za-z_]+", "", ts)
    detail_path = runs_dir / f"{model}_{safe_ts}.json"
    detail_path.write_text(json.dumps(record, indent=2, default=str))

    log_path = RESULTS_DIR / "experiments.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    print(f"\nSaved run detail -> {detail_path}")
    print(f"Appended to experiment log -> {log_path}")

    if save_model and payload.get("model_state"):
        pt_path = detail_path.with_suffix(".pt")
        pt_path.write_bytes(payload["model_state"])
        print(f"Saved model checkpoint -> {pt_path}")

    # Models that return an out-of-sample PnL series (e.g. lstm) ship it back as
    # .npz bytes; persist it next to the run JSON. cnn_lstm omits the key today,
    # so this is a no-op there.
    if payload.get("pnl"):
        pnl_path = detail_path.with_name(detail_path.stem + "_pnl.npz")
        pnl_path.write_bytes(payload["pnl"])
        print(f"Saved OOS PnL series -> {pnl_path}")


def _apply_quick(p: dict) -> dict:
    """Mirror cnn_lstm's --quick flag for tiny/fast smoke runs."""
    p["max_epochs"] = min(p.get("max_epochs", 50), 2)
    p["train_stride"] = max(p.get("train_stride", 25), 200)
    p["eval_stride"] = max(p.get("eval_stride", 10), 200)
    p["max_train_windows"] = p.get("max_train_windows") or 4000
    p["linear_fit_windows"] = min(p.get("linear_fit_windows", 40000), 4000)
    if p.get("tag", "") in ("", "modal"):
        p["tag"] = "modal-quick"
    return p


# --------------------------------------------------------------------------- #
# CLI entrypoint                                                               #
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(
    model: str = "cnn_lstm",
    gpu: str = "T4",
    list_models: bool = False,
    save_model: bool = False,
    quick: bool = False,
    # --- training knobs (forwarded to the model's Config) ---
    markets: str = "all",
    window: int = 100,
    train_stride: int = 25,
    eval_stride: int = 10,
    max_train_windows: int = 0,
    hidden: int = 64,
    cnn_filters: int = 32,
    inception_filters: int = 64,
    # --- lstm-only knobs (ignored by cnn_lstm; see _train_impl field filter) ---
    num_layers: int = 1,
    dropout: float = 0.0,
    forget_bias_init: float = 1.0,
    # --- lstm_mlp-only head knobs (ignored by other models) ---
    mlp_hidden: int = 64,
    mlp_layers: int = 1,
    mlp_dropout: float = 0.0,
    mlp_activation: str = "relu",
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_epochs: int = 50,
    patience: int = 5,
    grad_clip: float = 1.0,
    linear_benchmark: bool = True,
    linear_fit_windows: int = 40000,
    seed: int = 42,
    tag: str = "modal",
    runs_subdir: str = "",  # results/runs/<subdir>/ for per-run JSON/.pt/.npz
):
    """Run a registered model on Modal and pull its results back locally.

    Use ``--model <name>`` to pick which model to run and ``--gpu <type>``
    (T4 | L4 | A10G | A100 | H100) to pick the accelerator.
    """
    if list_models:
        print("Available models (use --model <name>):")
        for name, spec in MODEL_REGISTRY.items():
            print(f"  {name:<12} {spec['label']}")
        return

    if model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown model '{model}'. Available: {sorted(MODEL_REGISTRY)} "
            "(or run with --list-models)."
        )

    gpu = gpu.upper()
    if gpu not in TRAIN_FNS:
        raise SystemExit(
            f"Unknown gpu '{gpu}'. Available: {list(GPU_CHOICES)}."
        )
    _validate_local_files()

    # Only forward fields cnn_lstm.Config actually accepts; device stays "auto"
    # so the worker picks up the GPU automatically.
    params = dict(
        markets=markets, window=window, train_stride=train_stride,
        eval_stride=eval_stride, max_train_windows=max_train_windows,
        hidden=hidden, cnn_filters=cnn_filters,
        inception_filters=inception_filters, num_layers=num_layers,
        dropout=dropout, forget_bias_init=forget_bias_init,
        mlp_hidden=mlp_hidden, mlp_layers=mlp_layers,
        mlp_dropout=mlp_dropout, mlp_activation=mlp_activation,
        batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, max_epochs=max_epochs, patience=patience,
        grad_clip=grad_clip, linear_benchmark=linear_benchmark,
        linear_fit_windows=linear_fit_windows, seed=seed, tag=tag,
    )
    if quick:
        params = _apply_quick(params)

    print(f"Running model '{model}' on Modal (gpu={gpu}) ...")
    payload = TRAIN_FNS[gpu].remote(model, params, save_model)

    _save_results(model, payload, save_model, runs_subdir)
    print("Done.")
