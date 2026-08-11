"""Push gate for RegressionModel (reference signature) engine changes.

Companion to ``engine_validation.py``, which gates the spatial model only. The
reference model is step 1 of every cell2location workflow and has its own
performance characteristics: unlike the spatial model it minibatches by default
(``batch_size=2500``), and every one of its latents is global -- its
``list_obs_plate_vars()["sites"]`` is empty -- so a minibatch step subsamples
data and scales the likelihood, rather than indexing per-cell latents.

Gates, same philosophy as the spatial harness:

  train   ms/epoch <= baseline * 0.95   (an engine change must actually pay)
  elbo    |final - baseline| / |baseline| <= 0.005
  guard   checks > 0 and not diverged and max relative difference finite

An engine change touching shared flat-engine code must pass BOTH harnesses
before it merges.

Exit code 0 only if every gate passes. Anything else: do not push.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASELINE = Path(__file__).parent / "reference_baseline.json"

# 10,000 cells keeps four 2,500-cell minibatches per epoch -- enough that
# per-step overhead is measured in the regime real references run in, while the
# run still finishes in seconds. Genes match the spatial harness.
N_OBS, N_GENES, EPOCHS, BATCH = 10000, 10000, 30, 2500


def build():
    import logging

    for name in ("lightning.pytorch", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    from scvi.data import synthetic_iid

    import cell2location.accel as accel
    from cell2location.models import RegressionModel

    torch.manual_seed(0)
    np.random.seed(0)
    adata = synthetic_iid(n_labels=5, batch_size=N_OBS // 2, n_genes=N_GENES)
    accel.prepare_anndata(adata)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    return RegressionModel(adata), adata


def main():
    quiet = {"enable_progress_bar": False, "enable_model_summary": False}
    model, adata = build()
    model.mps_numerical_guard_every_n_steps = 10

    t0 = time.perf_counter()
    model.train(max_epochs=EPOCHS, batch_size=BATCH, accelerator="mps", **quiet)
    train_ms = (time.perf_counter() - t0) / EPOCHS * 1000
    elbo = float(np.asarray(model.history_["elbo_train"]).ravel()[-1])
    guard = model.numerical_guard_.summary() if model.numerical_guard_ else {"checks": 0, "diverged": True}

    metrics = {
        "train_ms_per_epoch": train_ms,
        "final_elbo": elbo,
        "guard": guard,
        "flat_engine_used": bool(getattr(model, "flat_engine_used_", False)),
        "config": {"n_obs": N_OBS, "n_genes": N_GENES, "epochs": EPOCHS, "batch_size": BATCH},
    }
    print(json.dumps(metrics, indent=2, default=str))

    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(metrics, indent=2, default=str))
        print("No baseline existed; wrote this run AS the baseline (run on master!). "
              "Gates not evaluated.")
        return 2

    base = json.loads(BASELINE.read_text())
    checks = {
        "train_faster": train_ms <= base["train_ms_per_epoch"] * 0.95,
        "elbo_parity": abs(elbo - base["final_elbo"]) / abs(base["final_elbo"]) <= 0.005,
        "guard_clean": guard.get("checks", 0) > 0 and not guard.get("diverged", True)
                       and np.isfinite(guard.get("max_relative_difference", np.inf)),
    }
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if all(checks.values()):
        print("ALL GATES PASS -- pushable.")
        return 0
    print("GATES FAILED -- do not push.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
