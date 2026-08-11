"""Push gate for RegressionModel (reference signature) engine changes.

Companion to ``engine_validation.py``, which gates the spatial model only. The
reference model is step 1 of every cell2location workflow and has its own
performance characteristics: unlike the spatial model it minibatches by default
(``batch_size=2500``), and every one of its latents is global -- its
``list_obs_plate_vars()["sites"]`` is empty -- so a minibatch step subsamples
data and scales the likelihood, rather than indexing per-cell latents.

Gates, same philosophy as the spatial harness:

  train   ms/epoch <= baseline * 1.05    (no regression)
  paid    ms/epoch <= baseline * 0.95    (dropped by --no-regression)
  elbo    |final - baseline| / |baseline| <= 0.005
  guard   checks > 0 and not diverged and max relative difference finite

An engine change touching shared flat-engine code must pass BOTH harnesses
before it merges -- the targeted arm normally, the other with
``--no-regression``.

Thresholds are ratios against master, so **recapture the baseline on master
after every merged engine change**. A stale baseline passes everything: this
one held its pre-flat-engine capture (569.6 ms/epoch) while master ran at
182.2, so a change could regress and still clear the gate by 3x.

Exit code 0 only if every gate passes. Anything else: do not push.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

BASELINE = Path(__file__).parent / "reference_baseline.json"

# 10,000 cells keeps four 2,500-cell minibatches per epoch -- enough that
# per-step overhead is measured in the regime real references run in, while the
# run still finishes in seconds. Genes match the spatial harness.
N_OBS, N_GENES, EPOCHS, BATCH = 10000, 10000, 30, 2500

# Reported timing is the MINIMUM of the measured runs, not the median; see the
# note in engine_validation.py for why. This arm shows the same asymmetry: its
# own baseline capture ran 86.8 / 49.8 / 49.3 ms/epoch, where the minimum is the
# uncontended cost and the 86.8 is one run that met something else on the
# machine. All runs are recorded so the spread stays visible.
from _gate import (  # noqa: E402
    QUIET, REPEATS, quiet_libraries, repeat_runs, verdict,
)
import json  # noqa: E402


def build():
    import torch
    from scvi.data import synthetic_iid

    quiet_libraries()

    import cell2location.accel as accel
    from cell2location.models import RegressionModel

    torch.manual_seed(0)
    np.random.seed(0)
    adata = synthetic_iid(n_labels=5, batch_size=N_OBS // 2, n_genes=N_GENES)
    accel.prepare_anndata(adata)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    return RegressionModel(adata), adata


def _one_run():
    """A timed training run with the guard OFF.

    The guard cross-checks the loss on the CPU, and at this shape those CPU
    forward passes both dominate the wall clock and carry most of its variance:
    timing with the guard on left an ~18% spread across repeats, enough for the
    gate to certify a no-op change as 5% faster (observed). Timing now measures
    training, and the guard is verified in its own run below.
    """
    model, _adata = build()

    t0 = time.perf_counter()
    model.train(max_epochs=EPOCHS, batch_size=BATCH, accelerator="mps", **QUIET)
    train_ms = (time.perf_counter() - t0) / EPOCHS * 1000
    elbo = float(np.asarray(model.history_["elbo_train"]).ravel()[-1])
    return {
        "train_ms_per_epoch": train_ms,
        "final_elbo": elbo,
        "flat_engine_used": bool(getattr(model, "flat_engine_used_", False)),
    }


def _guard_run():
    """One guarded run, for the numerical gate only -- never for timing."""
    model, _adata = build()
    model.mps_numerical_guard_every_n_steps = 10
    model.train(max_epochs=EPOCHS, batch_size=BATCH, accelerator="mps", **QUIET)
    if model.numerical_guard_ is None:
        return {"checks": 0, "diverged": True}
    return model.numerical_guard_.summary()


def main():
    runs = repeat_runs(_one_run)
    metrics = dict(min(runs, key=lambda r: r["train_ms_per_epoch"]))
    metrics["guard"] = _guard_run()
    metrics["train_ms_all_runs"] = [round(r["train_ms_per_epoch"], 1) for r in runs]
    metrics["config"] = {
        "n_obs": N_OBS, "n_genes": N_GENES, "epochs": EPOCHS,
        "batch_size": BATCH, "repeats": REPEATS,
    }

    base_exists = BASELINE.exists()
    base = json.loads(BASELINE.read_text()) if base_exists else {}
    guard = metrics["guard"]
    extra = {} if not base_exists else {
        "elbo_parity": abs(metrics["final_elbo"] - base["final_elbo"]) / abs(base["final_elbo"]) <= 0.005,
        "guard_clean": guard.get("checks", 0) > 0 and not guard.get("diverged", True)
                       and np.isfinite(guard.get("max_relative_difference", np.inf)),
    }
    return verdict(metrics, BASELINE, extra, {"train": "train_ms_per_epoch"},
                   "--no-regression" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
