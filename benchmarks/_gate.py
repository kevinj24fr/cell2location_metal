"""Shared machinery for the push gates.

``engine_validation.py`` (spatial model, full batch and ``--minibatch``) and
``reference_validation.py`` (reference signature model) gate different
workloads against different baselines, but they answer the same question in the
same way, and the verdict logic drifting between them would be a silent
problem. It lives here once.

Why the statistic is a minimum
------------------------------
Contention can only ADD time to a run, so under load the minimum estimates the
uncontended cost while the median tracks whatever else the machine is doing. On
a loaded machine the spatial workload's runs spread 32.0 / 67.4 / 68.6 ms/epoch
while its minimum stayed within 1 ms of a quiet-machine capture. Individual runs
are always reported alongside so the spread stays visible.

Why the no-regression tolerance is loose
----------------------------------------
A no-regression check has to be looser than the instrument can resolve or it
fires on noise. Measured across independent captures of IDENTICAL master code,
export minima differ by up to 5.4% and within-capture spreads run 1.7-3.8%. A
5% tolerance sat inside that and failed a change that did not touch export at
all. This is calibrated to the measurement, not to any particular change: a real
regression is a factor, not a few percent, and the improvement gate is what
demands actual evidence.
"""

import json

#: Discard this many runs before measuring (the first pays warmup), then report
#: the minimum of this many measured runs.
WARMUP_RUNS, REPEATS = 1, 3

#: Ratio-to-baseline above which a metric counts as regressed.
NO_REGRESSION_TOLERANCE = 1.15

#: Ratio-to-baseline at or below which a metric counts as a real improvement.
IMPROVEMENT_THRESHOLD = 0.95

QUIET = {"enable_progress_bar": False, "enable_model_summary": False}


def quiet_libraries():
    """Silence Lightning's per-run chatter and clear pyro's global param store.

    That store is process-global, so repeated runs in one process otherwise
    inherit the previous run's guide parameters -- which are also on the previous
    run's device, surfacing as a mid-forward "found at least two devices" error
    rather than as anything resembling its cause.
    """
    import logging

    import pyro

    for name in ("lightning.pytorch", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    pyro.clear_param_store()


def repeat_runs(one_run, keep_last=False):
    """Run ``one_run`` WARMUP_RUNS + REPEATS times, returning the measured ones.

    Frees each result before the next unless it is the one being kept: holding
    every repeat's model alive put several full-size datasets plus their
    posterior samples in memory and made each run slower than the last, which is
    the harness measuring its own footprint.
    """
    import gc

    import torch

    measured, kept = [], None
    total = WARMUP_RUNS + REPEATS
    for i in range(total):
        if kept is not None:
            del kept
            kept = None
        gc.collect()
        torch.mps.empty_cache()
        result = one_run()
        if i >= WARMUP_RUNS:
            measured.append(result[0] if keep_last else result)
        if keep_last and i == total - 1:
            kept = result
    return (measured, kept) if keep_last else measured


def verdict(metrics, baseline_path, extra_checks, ratio_keys, no_regression_only):
    """Evaluate the gates and print them. Returns the process exit code.

    ``ratio_keys`` maps a display label to the metric key it reads (the same key
    is read from the baseline); each is checked for no-regression, and at least
    one must show a real improvement unless ``no_regression_only``.
    ``extra_checks`` carries the correctness gates (ELBO parity, guard, export
    parity), which are pass/fail, not ratios.
    """
    print(json.dumps(metrics, indent=2, default=str))

    if not baseline_path.exists():
        baseline_path.write_text(json.dumps(metrics, indent=2, default=str))
        print("No baseline existed; wrote this run AS the baseline (run on master!). "
              "Gates not evaluated.")
        return 2

    base = json.loads(baseline_path.read_text())
    ratios = {label: metrics[key] / base[key] for label, key in ratio_keys.items()}
    checks = {f"{name}_no_regression": r <= NO_REGRESSION_TOLERANCE for name, r in ratios.items()}
    checks.update(extra_checks)
    if not no_regression_only:
        # A change aimed at this arm must actually pay somewhere.
        checks["something_got_faster"] = min(ratios.values()) <= IMPROVEMENT_THRESHOLD

    print("  " + ", ".join(f"{n} {r:.3f}x baseline" for n, r in ratios.items())
          + ("  [no-regression mode]" if no_regression_only else ""))
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if all(checks.values()):
        print("ALL GATES PASS -- pushable.")
        return 0
    print("GATES FAILED -- do not push.")
    return 1
