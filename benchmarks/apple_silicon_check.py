#!/usr/bin/env python3
"""Validate and benchmark cell2location on Apple silicon.

Run this natively on macOS (not in a container -- Metal is not available inside
Docker or a Linux VM)::

    python benchmarks/apple_silicon_check.py                 # everything
    python benchmarks/apple_silicon_check.py --skip-train    # ops + parity only
    python benchmarks/apple_silicon_check.py --json out.json # machine-readable

What it checks, in order:

1. **Environment** -- torch build, Metal availability, CoreML presence.
2. **Op coverage** -- which kernels exist on MPS and which need a CPU fallback.
3. **Numerical parity** -- MPS vs CPU for ``lgamma`` (all four dispatch modes) and
   for the full negative-binomial log-likelihood. This is the important section:
   a wrong ``lgamma`` produces a plausible-looking but incorrect ELBO, and nothing
   downstream will tell you.
4. **Kernel benchmarks** -- matmul and NB likelihood at cell2location-like shapes.
5. **End-to-end training** -- a short SVI run on synthetic data, CPU vs MPS,
   comparing both wall-clock and final ELBO.

Exit code is non-zero if any parity check fails, so it can gate CI on a Mac runner.
"""

import argparse
import json
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

try:
    from cell2location.accel import _ops, report, supports_op
    from cell2location.accel._ops import LGAMMA_MODES
except ImportError:  # pragma: no cover - allows running from a source checkout
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from cell2location.accel import _ops, report, supports_op
    from cell2location.accel._ops import LGAMMA_MODES


# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

#: Device under test. Overridable with ``--device`` purely so the script itself can be
#: smoke-tested on a machine without Metal -- comparing CPU against CPU proves nothing
#: numerically, but it does prove every code path below runs.
TARGET_DEVICE = "mps"


def _supports_colour() -> bool:
    return sys.stdout.isatty()


def mark(ok: Optional[bool]) -> str:
    if ok is None:
        symbol, colour = "~", YELLOW
    else:
        symbol, colour = ("ok", GREEN) if ok else ("FAIL", RED)
    return f"{colour}{symbol}{RESET}" if _supports_colour() else symbol


def info(value: bool) -> str:
    """Informational yes/no. Distinct from :func:`mark`, which reports pass/fail --
    "not an Apple silicon machine" is a fact, not a test failure."""
    return "yes" if value else "no"


def heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# --------------------------------------------------------------------------------------
# 1. environment
# --------------------------------------------------------------------------------------


def check_environment() -> Dict[str, Any]:
    heading("1. Environment")
    status = report()

    print(f"  platform        : {status['platform']}")
    print(f"  torch           : {status['torch_version']}")
    print(f"  apple silicon   : {info(status['apple_silicon'])}")
    print(f"  MPS built       : {info(status['mps_built'])}")
    print(f"  MPS available   : {info(status['mps_available'])}")
    print(f"  default device  : {status['default_device']}")
    print(f"  lgamma mode     : {status['lgamma_mode']}")
    print(f"  coremltools     : {info(status['coreml_available'])} (needed only for ANE export)")

    if status["mps_available"]:
        print(f"  memory          : {status['memory']}")
    else:
        print(f"\n{YELLOW if _supports_colour() else ''}  Metal is unavailable. Requires macOS 12.3+ on Apple")
        print(
            f"  silicon, running natively -- containers and VMs cannot see the GPU.{RESET if _supports_colour() else ''}"
        )

    return status


# --------------------------------------------------------------------------------------
# 2. op coverage
# --------------------------------------------------------------------------------------


def check_op_support() -> Dict[str, bool]:
    heading("2. MPS operator coverage")
    ops = [
        ("lgamma", "negative-binomial likelihood"),
        ("digamma", "gradients of Gamma/Beta sites"),
        ("poisson", "posterior predictive sampling"),
        ("standard_gamma", "Gamma sampling in NB.sample()"),
        ("erfinv", "Normal quantiles in export_posterior"),
        ("cumsum", "misc"),
        ("logsumexp", "TraceEnum_ELBO with discrete sites"),
        ("index_add", "minibatch scatter"),
        ("sort", "quantile computation"),
        ("multinomial", "categorical sampling"),
        ("linalg_cholesky", "multivariate guides"),
    ]
    results = {}
    for name, why in ops:
        ok = supports_op(name, device=TARGET_DEVICE)
        results[name] = ok
        suffix = (
            "" if ok else f"  {DIM if _supports_colour() else ''}-> CPU fallback{RESET if _supports_colour() else ''}"
        )
        print(
            f"  {mark(ok):>12}  {name:<16} {DIM if _supports_colour() else ''}{why}{RESET if _supports_colour() else ''}{suffix}"
        )
    return results


# --------------------------------------------------------------------------------------
# 3. numerical parity
# --------------------------------------------------------------------------------------


def _error(result: torch.Tensor, reference: torch.Tensor, tol: float) -> Dict[str, Any]:
    """Combined absolute/relative comparison, matching ``torch.allclose`` semantics.

    A pure relative metric is misleading for ``lgamma``: the function passes through
    zero at x=1 and x=2, so any implementation shows unbounded relative error there
    while being perfectly usable. What propagates into a summed log-likelihood is the
    absolute error, so both are reported and the verdict uses the combined budget.
    """
    a64, b64 = result.cpu().double(), reference.cpu().double()
    diff = (a64 - b64).abs()
    # ``tol`` serves as both atol and rtol. The relative term matters at large
    # arguments (lgamma(1e4) is ~8e4, where float32 rounding alone is ~1e-2); the
    # absolute term matters at the roots. A genuinely broken kernel misses by order 1
    # and trips either way.
    budget = tol + tol * b64.abs()
    return {
        "max_abs_error": float(diff.max()),
        "max_rel_error": float((diff / b64.abs().clamp_min(1e-6)).max()),
        "passed": bool((diff <= budget).all()),
    }


def check_lgamma_parity(tolerance: float = 1e-4) -> List[Dict[str, Any]]:
    heading("3a. lgamma parity (MPS vs CPU, float32)")
    print(
        f"  {DIM if _supports_colour() else ''}Includes a deliberately broadcast (stride-0) input, which is the shape\n"
        f"  that has historically returned wrong results on MPS "
        f"(pytorch/pytorch#132605).{RESET if _supports_colour() else ''}\n"
    )

    torch.manual_seed(0)
    cases = {
        "contiguous 1D": torch.rand(4096) * 100 + 1e-3,
        "contiguous 2D": torch.rand(512, 2000) * 50 + 1e-3,
        "broadcast view": (torch.rand(1, 2000) * 50 + 1e-3).expand(512, 2000),
        "large values": torch.rand(4096) * 1e4 + 1.0,
        "small values": torch.rand(4096) * 1e-2 + 1e-4,
    }

    rows = []
    for mode in [m for m in LGAMMA_MODES if m != "auto"]:
        for case_name, x_cpu in cases.items():
            reference = torch.lgamma(x_cpu.double().contiguous())
            try:
                result = _ops.lgamma(x_cpu.to(TARGET_DEVICE), mode=mode)
                stats = _error(result, reference, tolerance)
            except Exception as exc:  # noqa: BLE001
                print(f"  {mark(False):>12}  {mode:<11} {case_name:<16} raised {type(exc).__name__}: {exc}")
                rows.append({"mode": mode, "case": case_name, "passed": False, "error": str(exc)})
                continue

            print(
                f"  {mark(stats['passed']):>12}  {mode:<11} {case_name:<16} "
                f"abs {stats['max_abs_error']:.2e}  rel {stats['max_rel_error']:.2e}"
            )
            rows.append({"mode": mode, "case": case_name, **stats})

    return rows


def check_nb_parity(tolerance: float = 1e-4) -> List[Dict[str, Any]]:
    heading("3b. Negative-binomial log-likelihood parity")
    torch.manual_seed(0)

    shapes = [(256, 2000), (2048, 5000)]
    rows = []
    for n_obs, n_genes in shapes:
        value = torch.poisson(torch.full((n_obs, n_genes), 5.0))
        mu = torch.rand(n_obs, n_genes) * 20 + 0.1
        theta = torch.rand(1, n_genes) * 10 + 0.1  # broadcast against value -- the risky case

        reference = _ops.log_nb_positive(value.double(), mu.double(), theta.double())

        try:
            result = _ops.log_nb_positive(value.to(TARGET_DEVICE), mu.to(TARGET_DEVICE), theta.to(TARGET_DEVICE))
            stats = _error(result, reference, tolerance)
        except Exception as exc:  # noqa: BLE001
            print(f"  {mark(False):>12}  {n_obs}x{n_genes}  raised {type(exc).__name__}: {exc}")
            rows.append({"shape": [n_obs, n_genes], "passed": False, "error": str(exc)})
            continue

        # The summed term is what training actually optimises, so report it too:
        # elementwise noise can cancel, or accumulate, and only this tells you which.
        sum_err = abs(float(result.cpu().double().sum()) - float(reference.sum())) / abs(float(reference.sum()))
        print(
            f"  {mark(stats['passed']):>12}  {n_obs}x{n_genes:<8} "
            f"elementwise abs {stats['max_abs_error']:.2e}   summed rel {sum_err:.2e}"
        )
        rows.append({"shape": [n_obs, n_genes], "sum_rel_error": sum_err, **stats})

    return rows


# --------------------------------------------------------------------------------------
# 4. kernel benchmarks
# --------------------------------------------------------------------------------------


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _time(fn: Callable, device: str, n_warmup: int = 3, n_iters: int = 10) -> float:
    for _ in range(n_warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / n_iters


def benchmark_kernels(devices: List[str]) -> List[Dict[str, Any]]:
    heading("4. Kernel benchmarks")
    print(
        f"  {DIM if _supports_colour() else ''}Shapes chosen to mirror a Visium slide against a reference signature\n"
        f"  matrix: (n_locations x n_cell_types) @ (n_cell_types x n_genes).{RESET if _supports_colour() else ''}\n"
    )

    configs = [
        ("abundance matmul", (4992, 50, 12000)),
        ("abundance matmul (large)", (20000, 80, 20000)),
    ]

    rows = []
    for label, (n_obs, n_factors, n_genes) in configs:
        timings = {}
        for device in devices:
            try:
                w = torch.rand(n_obs, n_factors, device=device)
                g = torch.rand(n_factors, n_genes, device=device)
                timings[device] = _time(lambda: w @ g, device)
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<26} {device:<5} failed: {exc}")
                timings[device] = None

        _print_timing_row(label, timings)
        rows.append({"benchmark": label, "shape": [n_obs, n_factors, n_genes], "seconds": timings})

    for label, (n_obs, n_genes) in [("NB log-likelihood", (4992, 12000))]:
        timings = {}
        for device in devices:
            try:
                value = torch.poisson(torch.full((n_obs, n_genes), 5.0)).to(device)
                mu = (torch.rand(n_obs, n_genes) * 20 + 0.1).to(device)
                theta = (torch.rand(1, n_genes) * 10 + 0.1).to(device)
                timings[device] = _time(lambda: _ops.log_nb_positive(value, mu, theta), device)
            except Exception as exc:  # noqa: BLE001
                print(f"  {label:<26} {device:<5} failed: {exc}")
                timings[device] = None

        _print_timing_row(label, timings)
        rows.append({"benchmark": label, "shape": [n_obs, n_genes], "seconds": timings})

    return rows


def _print_timing_row(label: str, timings: Dict[str, Optional[float]]) -> None:
    parts = [f"{d}={t * 1e3:.1f}ms" for d, t in timings.items() if t is not None]
    speedup = ""
    if timings.get("cpu") and timings.get(TARGET_DEVICE) and TARGET_DEVICE != "cpu":
        factor = timings["cpu"] / timings[TARGET_DEVICE]
        colour = GREEN if factor > 1 else RED
        arrow = f"{factor:.2f}x"
        speedup = f"   {colour}{arrow}{RESET}" if _supports_colour() else f"   {arrow}"
    print(f"  {label:<26} {'  '.join(parts)}{speedup}")


# --------------------------------------------------------------------------------------
# 5. end-to-end training
# --------------------------------------------------------------------------------------


def _quiet_trainer_kwargs() -> Dict[str, Any]:
    """Silence Lightning's progress bar and logger.

    Timings here are the measurement, and a per-step progress bar both adds overhead
    and buries the results under thousands of lines of carriage returns.

    Note that ``logger`` is deliberately left enabled: scvi-tools populates
    ``model.history_`` from it, and the ELBO comparison below is the whole point of
    this section.
    """
    import logging as _logging

    _logging.getLogger("lightning.pytorch").setLevel(_logging.ERROR)
    _logging.getLogger("pytorch_lightning").setLevel(_logging.ERROR)
    return {"enable_progress_bar": False, "enable_model_summary": False}


def benchmark_training(devices: List[str], max_epochs: int = 200, n_obs: int = 500, n_genes: int = 400):
    heading("5. End-to-end training (synthetic data)")
    try:
        from scvi.data import synthetic_iid

        from cell2location.models import Cell2location, RegressionModel
    except ImportError as exc:
        print(f"  skipped: {exc}")
        return []

    import cell2location.accel as accel

    rows = []
    for device in devices:
        accelerator = "cpu" if device == "cpu" else device
        try:
            torch.manual_seed(0)
            np.random.seed(0)

            adata = synthetic_iid(n_labels=5, batch_size=n_obs // 2, n_genes=n_genes)
            accel.prepare_anndata(adata)
            adata.obsm["X_spatial"] = np.random.normal(0, 1, [adata.n_obs, 2])

            RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
            sc_model = RegressionModel(adata)

            start = time.perf_counter()
            sc_model.train(max_epochs=max_epochs, accelerator=accelerator, **_quiet_trainer_kwargs())
            reference_seconds = time.perf_counter() - start

            signatures = sc_model.samples if hasattr(sc_model, "samples") else None
            del signatures

            cell_state_df = sc_model._compute_cluster_averages(key="labels")

            Cell2location.setup_anndata(adata, batch_key="batch")
            st_model = Cell2location(
                adata,
                cell_state_df=cell_state_df,
                N_cells_per_location=8,
                detection_alpha=20,
            )

            start = time.perf_counter()
            st_model.train(max_epochs=max_epochs, accelerator=accelerator, **_quiet_trainer_kwargs())
            spatial_seconds = time.perf_counter() - start

            final_elbo = float(np.asarray(st_model.history_["elbo_train"]).ravel()[-1])

            print(
                f"  {mark(True):>12}  {device:<5} reference {reference_seconds:6.1f}s   "
                f"spatial {spatial_seconds:6.1f}s   final -ELBO {final_elbo:.4e}"
            )
            rows.append(
                {
                    "device": device,
                    "reference_seconds": reference_seconds,
                    "spatial_seconds": spatial_seconds,
                    "final_elbo": final_elbo,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {mark(False):>12}  {device:<5} failed: {type(exc).__name__}: {exc}")
            rows.append({"device": device, "error": f"{type(exc).__name__}: {exc}"})

    _compare_training(rows)
    return rows


def _compare_training(rows: List[Dict[str, Any]]) -> None:
    by_device = {r["device"]: r for r in rows if "error" not in r}
    if "cpu" not in by_device or TARGET_DEVICE not in by_device or TARGET_DEVICE == "cpu":
        return

    cpu, mps = by_device["cpu"], by_device[TARGET_DEVICE]
    speedup = cpu["spatial_seconds"] / mps["spatial_seconds"]
    print(f"\n  spatial-model speedup (MPS vs CPU): {speedup:.2f}x")

    elbo_gap = abs(cpu["final_elbo"] - mps["final_elbo"]) / abs(cpu["final_elbo"])
    verdict = "consistent" if elbo_gap < 0.05 else "DIVERGED -- investigate before trusting results"
    print(f"  final -ELBO relative difference:    {elbo_gap:.3%}  ({verdict})")
    print(
        f"  {DIM if _supports_colour() else ''}Some difference is expected: SVI is stochastic and RNG streams differ\n"
        f"  across devices. A gap above ~5% on a fixed seed points at a numerical bug,\n"
        f"  not at sampling noise.{RESET if _supports_colour() else ''}"
    )


# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-train", action="store_true", help="skip the end-to-end training benchmark")
    parser.add_argument("--skip-bench", action="store_true", help="skip kernel benchmarks")
    parser.add_argument("--max-epochs", type=int, default=200, help="epochs for the training benchmark")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max relative error for parity checks")
    parser.add_argument("--json", type=str, default=None, help="write full results to this path")
    parser.add_argument("--html", type=str, default=None, help="write a self-contained HTML report to this path")
    parser.add_argument(
        "--device",
        default="mps",
        help="device under test. Only useful as --device cpu, to smoke-test this script " "on a machine without Metal.",
    )
    args = parser.parse_args()

    global TARGET_DEVICE
    TARGET_DEVICE = args.device

    results: Dict[str, Any] = {"environment": check_environment()}

    if TARGET_DEVICE != "cpu" and not results["environment"]["mps_available"]:
        print("\nMetal unavailable -- skipping every MPS-dependent section.")
        print("(Pass --device cpu to smoke-test this script's own code paths.)")
        _write_outputs(args, results)
        return 0

    if TARGET_DEVICE == "cpu":
        print(
            "\nRunning against CPU. This validates that the script works; it cannot\n"
            "tell you anything about Metal correctness or performance."
        )

    results["op_support"] = check_op_support()
    results["lgamma_parity"] = check_lgamma_parity(args.tolerance)
    results["nb_parity"] = check_nb_parity(args.tolerance)

    devices = ["cpu"] if TARGET_DEVICE == "cpu" else ["cpu", TARGET_DEVICE]
    if not args.skip_bench:
        results["kernel_benchmarks"] = benchmark_kernels(devices)
    if not args.skip_train:
        results["training_benchmarks"] = benchmark_training(devices, max_epochs=args.max_epochs)

    failed = _summarise(results)
    _write_outputs(args, results)

    return 1 if failed else 0


def _summarise(results: Dict[str, Any]) -> bool:
    heading("Summary")

    auto_modes = {"contiguous", "native"}
    parity = results.get("lgamma_parity", []) + results.get("nb_parity", [])
    default_path = [r for r in parity if r.get("mode", "contiguous") == "contiguous"]
    failures = [r for r in default_path if not r["passed"]]

    if not failures:
        print(f"  {mark(True)}  Default configuration is numerically sound on this machine.")
    else:
        print(f"  {mark(False)}  {len(failures)} parity check(s) failed on the default 'contiguous' lgamma path.")
        working = _first_working_mode(results.get("lgamma_parity", []), auto_modes)
        if working:
            print(f"      Workaround: export CELL2LOCATION_MPS_LGAMMA={working}")
        else:
            print("      No lgamma mode passed. Train on CPU and open an issue with this output.")

    missing = [name for name, ok in results.get("op_support", {}).items() if not ok]
    if missing:
        print(f"  {mark(None)}  CPU fallback in use for: {', '.join(missing)} (correct, but slower).")

    return bool(failures)


def _first_working_mode(rows: List[Dict[str, Any]], exclude: set) -> Optional[str]:
    by_mode: Dict[str, List[bool]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row["passed"])
    for mode, passes in by_mode.items():
        if mode not in exclude and all(passes):
            return mode
    return None


def _write_outputs(args, results: Dict[str, Any]) -> None:
    if args.json:
        _write_json(args.json, results)
    if args.html:
        _write_html(args.html, results)


def _write_html(path: str, results: Dict[str, Any]) -> None:
    """Render the HTML report, degrading to a clear message if the module is missing.

    A failed report must never mask the validation result -- that is the actual
    output, and it has already been printed by the time this runs.
    """
    try:
        from report_html import render_report
    except ImportError:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
        try:
            from report_html import render_report
        except ImportError as exc:  # pragma: no cover
            print(f"\nCould not write the HTML report: {exc}")
            return

    import datetime

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w") as handle:
        handle.write(render_report(results, generated_at=stamp))
    print(f"Wrote {path}")


def _write_json(path: str, results: Dict[str, Any]) -> None:
    payload = dict(results)
    payload["_meta"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    sys.exit(main())
