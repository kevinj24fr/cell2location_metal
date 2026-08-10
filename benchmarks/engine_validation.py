"""Push gate for engine changes: better performance AND no precision regression.

Runs the current checkout at Visium scale on MPS and verdicts against
``benchmarks/engine_baseline.json`` (captured on master). Gates:

  train   ms/epoch <= baseline * 0.95   (an engine change must actually pay)
  elbo    |final - baseline| / |baseline| <= 0.005
  guard   checks > 0 and not diverged and max relative difference finite
  export  seconds <= baseline * 0.5, and fast-vs-looped summary parity within
          Monte-Carlo error (medians: means <= 2%, quantiles <= 3%)

Exit code 0 only if every gate passes. Anything else: do not push.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASELINE = Path(__file__).parent / "engine_baseline.json"
N_OBS, N_GENES, EPOCHS = 5000, 10000, 30


def build():
    import logging

    for name in ("lightning.pytorch", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    from scvi.data import synthetic_iid

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


def main():
    quiet = {"enable_progress_bar": False, "enable_model_summary": False}
    model, adata = build()
    model.mps_numerical_guard_every_n_steps = 10

    t0 = time.perf_counter()
    model.train(max_epochs=EPOCHS, accelerator="mps", **quiet)
    train_ms = (time.perf_counter() - t0) / EPOCHS * 1000
    elbo = float(np.asarray(model.history_["elbo_train"]).ravel()[-1])
    guard = model.numerical_guard_.summary() if model.numerical_guard_ else {"checks": 0, "diverged": True}

    t0 = time.perf_counter()
    out_fast = model.export_posterior(adata, sample_kwargs={"num_samples": 1000, "batch_size": adata.n_obs})
    export_s = time.perf_counter() - t0
    key = next(k for k in out_fast.obsm if k.startswith("means"))
    fast = {k: np.asarray(out_fast.obsm[k]) for k in out_fast.obsm}

    from cell2location.accel import _sampling

    orig = _sampling.vectorized_posterior_samples

    def _refuse(*a, **k):
        raise _sampling.NotVectorizable("forced looped baseline")

    _sampling.vectorized_posterior_samples = _refuse
    out_slow = model.export_posterior(adata, sample_kwargs={"num_samples": 1000, "batch_size": adata.n_obs})
    _sampling.vectorized_posterior_samples = orig

    parity = {}
    for k in fast:
        if k in out_slow.obsm:
            s = np.asarray(out_slow.obsm[k])
            parity[k] = float(np.median(np.abs(fast[k] - s) / (np.abs(s) + 1e-6)))

    metrics = {"train_ms_per_epoch": train_ms, "final_elbo": elbo, "guard": guard,
               "export_seconds": export_s, "export_parity_median_rel": parity}
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
        "export_faster": export_s <= base["export_seconds"] * 0.5,
        "export_means_parity": parity.get(key, 1.0) <= 0.02,
        "export_quantile_parity": all(v <= 0.03 for kk, v in parity.items() if kk.startswith("q")),
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
