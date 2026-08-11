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

# Timing is the MEDIAN of REPEATS runs after WARMUP_RUNS discarded ones. One run
# is not enough: a single-run measurement of this workload was observed at
# 364 ms/epoch against a 130 ms median on the same code, far more than the 5%
# the gate is trying to resolve. The first run additionally pays warmup, and
# without freeing between runs the harness measures its own accumulated
# footprint -- both were visible as monotone trends in the per-run timings, which
# is why those are reported alongside the median.
WARMUP_RUNS, REPEATS = 1, 3


def build():
    import logging

    for name in ("lightning.pytorch", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    import pyro
    from scvi.data import synthetic_iid

    # pyro's parameter store is process-global; repeated runs in one process
    # must not inherit the previous run's guide parameters (which are also on
    # the previous run's device). See the note in engine_validation.build.
    pyro.clear_param_store()

    import cell2location.accel as accel
    from cell2location.models import RegressionModel

    torch.manual_seed(0)
    np.random.seed(0)
    adata = synthetic_iid(n_labels=5, batch_size=N_OBS // 2, n_genes=N_GENES)
    accel.prepare_anndata(adata)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    return RegressionModel(adata), adata


QUIET = {"enable_progress_bar": False, "enable_model_summary": False}


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
    import gc

    runs = []
    for i in range(WARMUP_RUNS + REPEATS):
        gc.collect()
        torch.mps.empty_cache()
        result = _one_run()
        if i >= WARMUP_RUNS:
            runs.append(result)
    order = sorted(runs, key=lambda r: r["train_ms_per_epoch"])
    metrics = dict(order[len(order) // 2])  # the median run, reported whole
    metrics["guard"] = _guard_run()
    metrics["train_ms_all_runs"] = [round(r["train_ms_per_epoch"], 1) for r in runs]
    metrics["config"] = {
        "n_obs": N_OBS, "n_genes": N_GENES, "epochs": EPOCHS,
        "batch_size": BATCH, "repeats": REPEATS,
    }
    print(json.dumps(metrics, indent=2, default=str))

    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(metrics, indent=2, default=str))
        print("No baseline existed; wrote this run AS the baseline (run on master!). "
              "Gates not evaluated.")
        return 2

    base = json.loads(BASELINE.read_text())
    train_ms = metrics["train_ms_per_epoch"]
    elbo = metrics["final_elbo"]
    guard = metrics["guard"]
    no_regression_only = "--no-regression" in sys.argv

    train_ratio = train_ms / base["train_ms_per_epoch"]
    checks = {
        "train_no_regression": train_ratio <= 1.05,
        "elbo_parity": abs(elbo - base["final_elbo"]) / abs(base["final_elbo"]) <= 0.005,
        "guard_clean": guard.get("checks", 0) > 0 and not guard.get("diverged", True)
                       and np.isfinite(guard.get("max_relative_difference", np.inf)),
    }
    if not no_regression_only:
        checks["something_got_faster"] = train_ratio <= 0.95
    print(f"  train {train_ratio:.3f}x baseline"
          f"{'  [no-regression mode]' if no_regression_only else ''}")
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if all(checks.values()):
        print("ALL GATES PASS -- pushable.")
        return 0
    print("GATES FAILED -- do not push.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
