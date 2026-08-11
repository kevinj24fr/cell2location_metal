"""Push gate for spatial-model engine changes: better performance AND no
precision regression.

Runs the current checkout at Visium scale on MPS and verdicts against
``benchmarks/engine_baseline.json`` (captured on master). Gates:

  train   ms/epoch <= baseline * 1.05    (no regression)
  export  seconds  <= baseline * 1.05    (no regression)
  paid    at least one of the two <= baseline * 0.95
  elbo    |final - baseline| / |baseline| <= 0.005
  guard   checks > 0 and not diverged and max relative difference finite
  export  fast-vs-looped summary parity within Monte-Carlo error
          (medians: means <= 2%, quantiles <= 3%)

Run with ``--no-regression`` to drop the "paid" requirement. That is the mode
for a change aimed at the OTHER arm (the reference model, gated by
``reference_validation.py``) that touches shared flat-engine code: it still
must not regress this arm, but there is no reason for it to speed this arm up.

The thresholds are ratios against master, so **the baseline must be recaptured
on master after every merged engine change**. A stale baseline does not fail
loudly -- it passes everything. This one held its pre-flat-engine capture long
enough that changes were clearing it by 3x while regressing against master.

Exit code 0 only if every gate passes. Anything else: do not push.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _gate import (  # noqa: E402
    QUIET, REPEATS, quiet_libraries, repeat_runs, verdict,
)

# ``--minibatch`` gates the spatial model's MINIBATCH configuration, which is a
# different engine path with different performance characteristics, so it keeps
# its own baseline. Full batch is the model's default and the arm everything
# else here describes; a minibatch caller is someone whose data does not fit.
MINIBATCH = "--minibatch" in sys.argv
BATCH_SIZE = 1250

BASELINE = Path(__file__).parent / (
    "engine_minibatch_baseline.json" if MINIBATCH else "engine_baseline.json"
)
N_OBS, N_GENES, EPOCHS = 5000, 10000, 30




def build():
    import torch
    from scvi.data import synthetic_iid

    quiet_libraries()

    import cell2location.accel as accel
    from cell2location.models import Cell2location, RegressionModel

    torch.manual_seed(0)
    np.random.seed(0)
    adata = synthetic_iid(n_labels=5, batch_size=N_OBS // 2, n_genes=N_GENES)
    accel.prepare_anndata(adata)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    ref = RegressionModel(adata)
    ref.train(max_epochs=3, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    cell_state_df = ref._compute_cluster_averages(key="labels")
    Cell2location.setup_anndata(adata, batch_key="batch")
    model = Cell2location(adata, cell_state_df=cell_state_df, N_cells_per_location=8, detection_alpha=20)
    return model, adata


def _train_kwargs():
    return {"batch_size": BATCH_SIZE} if MINIBATCH else {}


def _guard_run():
    """One guarded run, for the numerical gate only -- never for timing.

    The guard cross-checks the loss on the CPU. Those CPU forward passes carry
    most of the wall-clock variance, and the more of them a configuration runs
    the worse it gets: the minibatch arm fires twelve per run and timing it with
    the guard on gave a 91% spread across repeats (868.7 / 1480.1 / 774.0).
    """
    model, _adata = build()
    model.mps_numerical_guard_every_n_steps = 10
    model.train(max_epochs=EPOCHS, accelerator="mps", **_train_kwargs(), **QUIET)
    if model.numerical_guard_ is None:
        return {"checks": 0, "diverged": True}
    return model.numerical_guard_.summary()


def _timed_run():
    """One build + train + fast export, guard OFF. Returns its metrics and its
    model/adata, so parity can be run against the reported run specifically."""
    model, adata = build()

    t0 = time.perf_counter()
    model.train(max_epochs=EPOCHS, accelerator="mps", **_train_kwargs(), **QUIET)
    train_ms = (time.perf_counter() - t0) / EPOCHS * 1000
    elbo = float(np.asarray(model.history_["elbo_train"]).ravel()[-1])

    t0 = time.perf_counter()
    out_fast = model.export_posterior(adata, sample_kwargs={"num_samples": 1000, "batch_size": adata.n_obs})
    export_s = time.perf_counter() - t0
    fast = {k: np.asarray(out_fast.obsm[k]) for k in out_fast.obsm}

    return {
        "train_ms_per_epoch": train_ms,
        "final_elbo": elbo,
        "export_seconds": export_s,
        "flat_engine_used": bool(getattr(model, "flat_engine_used_", False)),
    }, model, adata, fast


def _parity(model, adata, fast):
    """Fast-vs-looped summary parity, computed once on the reported run.

    Deterministic given the trained guide, so unlike the timings it does not
    need repeating -- and the looped sampler is the most expensive thing here.
    Only exists on checkouts that HAVE the fast path; on a baseline checkout the
    export above already ran the looped sampler, so parity is empty.
    """
    try:
        from cell2location.accel import _sampling
    except ImportError:
        return {}

    orig = _sampling.vectorized_posterior_samples

    def _refuse(*a, **k):
        raise _sampling.NotVectorizable("forced looped baseline")

    _sampling.vectorized_posterior_samples = _refuse
    try:
        out_slow = model.export_posterior(
            adata, sample_kwargs={"num_samples": 1000, "batch_size": adata.n_obs}
        )
    finally:
        _sampling.vectorized_posterior_samples = orig

    parity = {}
    for k in fast:
        if k in out_slow.obsm:
            s = np.asarray(out_slow.obsm[k])
            parity[k] = float(np.median(np.abs(fast[k] - s) / (np.abs(s) + 1e-6)))
    return parity


def main():
    # Runs are seeded identically and produce bit-identical ELBO, guard and
    # parity values, so the kept run's numbers describe them all; only timing
    # varies. See _gate.repeat_runs for why earlier runs are freed as they go.
    timings, kept = repeat_runs(_timed_run, keep_last=True)
    metrics, model, adata, fast = kept
    metrics = dict(metrics)
    metrics["train_ms_per_epoch"] = min(t["train_ms_per_epoch"] for t in timings)
    metrics["export_seconds"] = min(t["export_seconds"] for t in timings)
    key = next(k for k in fast if k.startswith("means"))

    # Parity first: it exports from the kept model, whose guide parameters live
    # in pyro's global store. _guard_run builds another model and clears that
    # store, which would pull the kept model's parameters out from under it.
    metrics["export_parity_median_rel"] = _parity(model, adata, fast)
    metrics["guard"] = _guard_run()
    metrics["train_ms_all_runs"] = [round(t["train_ms_per_epoch"], 1) for t in timings]
    metrics["export_seconds_all_runs"] = [round(t["export_seconds"], 2) for t in timings]
    metrics["repeats"] = REPEATS
    metrics["minibatch"] = BATCH_SIZE if MINIBATCH else None

    base_exists = BASELINE.exists()
    base = json.loads(BASELINE.read_text()) if base_exists else {}
    guard, parity = metrics["guard"], metrics["export_parity_median_rel"]
    extra = {} if not base_exists else {
        "elbo_parity": abs(metrics["final_elbo"] - base["final_elbo"]) / abs(base["final_elbo"]) <= 0.005,
        "guard_clean": guard.get("checks", 0) > 0 and not guard.get("diverged", True)
                       and np.isfinite(guard.get("max_relative_difference", np.inf)),
        "export_means_parity": parity.get(key, 1.0) <= 0.02,
        "export_quantile_parity": all(v <= 0.03 for kk, v in parity.items() if kk.startswith("q")),
    }
    return verdict(metrics, BASELINE, extra,
                   {"train": "train_ms_per_epoch", "export": "export_seconds"},
                   "--no-regression" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
