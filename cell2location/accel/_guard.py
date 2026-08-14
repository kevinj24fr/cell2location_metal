"""Runtime detection of silent numerical divergence between Metal and CPU.

The failure mode that matters on this backend is not a crash. It is a kernel that
returns plausible numbers: the loss curve descends, training completes, the figures
look reasonable, and the cell abundances are wrong. `benchmarks/apple_silicon_check.py`
tests for this before a run, but it tests synthetic data at shapes I chose. Your
dataset has its own dynamic range, its own sparsity, its own dispersion values.

This callback closes that gap by checking during the actual run: every
``every_n_steps``, it recomputes the current minibatch's loss on the CPU and compares.
Same parameters, same batch, same seed -- so the two numbers should agree to float32
rounding, and a systematic disagreement means one of the two is wrong.

Cost is one CPU forward pass every N steps. At the default N=1000 over a 30k-step run
that is 30 extra evaluations, which is noise against the total.

This is cheap insurance against the one thing you cannot otherwise detect.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = ["NumericalGuard", "compare_loss_across_devices"]


def compare_loss_across_devices(
    module,
    args,
    kwargs,
    reference_device: str = "cpu",
) -> Optional[Dict[str, float]]:
    """Evaluate the same ELBO term on the module's device and on ``reference_device``.

    The guide is traced once on the training device, and the *same* sampled latents
    are then replayed through the model on both devices; the quantity compared is
    the model log-joint (data likelihood plus priors) under identical latents. That
    is deterministic -- any disagreement is arithmetic, never sampling -- and it is
    precisely the arithmetic the accelerated kernels compute. The guide's own
    log-density is deliberately excluded: autoguides record constrained sites as
    ``Delta`` distributions whose log-prob is an exact equality test, so recomputing
    the transform on another device fails on one-ulp differences by construction.

    The module itself is moved for the reference evaluation and moved back whatever
    happens. Moving, not copying, is load-bearing: PyroModule parameters resolve
    through the global param store at trace time, so any copy -- however deep --
    still reads tensors living on the original device.

    Returns ``None`` when the comparison cannot be made (module already on the
    reference device, or an evaluation failed). Otherwise returns both values and
    their relative difference.

    Scope: this validates FORWARD arithmetic only. Gradients of the fused kernel
    are validated by its own self-test (forward and backward against eager);
    a wrong backward in any other custom op would not be caught here.
    """
    import pyro

    from ._device import device_of

    original_device = device_of(module)
    if original_device.type == reference_device:
        return None

    # Dropout (module.dropout_p != 0) draws fresh masks per forward; two devices
    # would disagree on healthy runs. eval() disables it for both evaluations.
    was_training = getattr(module, "training", False)
    if was_training:
        module.eval()
    try:
        return _compare_in_eval(module, args, kwargs, reference_device, original_device)
    finally:
        if was_training:
            module.train()


def _compare_in_eval(module, args, kwargs, reference_device, original_device):
    import pyro

    try:
        with torch.no_grad():
            guide_trace = pyro.poutine.trace(module.guide).get_trace(*args, **kwargs)
            model_trace = pyro.poutine.trace(pyro.poutine.replay(module.model, trace=guide_trace)).get_trace(
                *args, **kwargs
            )
            device_loss = float(model_trace.log_prob_sum())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Numerical guard: could not evaluate on %s (%s).", original_device, exc)
        return None

    try:
        reference_trace = _sample_values_to(guide_trace, reference_device)
        ref_args = [a.to(reference_device) if isinstance(a, torch.Tensor) else a for a in args]
        ref_kwargs = {k: (v.to(reference_device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}

        module.to(reference_device)
        try:
            with torch.no_grad():
                ref_model_trace = pyro.poutine.trace(pyro.poutine.replay(module.model, trace=reference_trace)).get_trace(
                    *ref_args, **ref_kwargs
                )
                reference_loss = float(ref_model_trace.log_prob_sum())
        finally:
            module.to(original_device)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Numerical guard: could not evaluate on %s (%s).", reference_device, exc)
        return None

    scale = max(abs(reference_loss), 1e-12)
    relative_difference = abs(device_loss - reference_loss) / scale
    if not math.isfinite(relative_difference):
        # A NaN would sail through every ``>`` comparison and read as agreement --
        # the one thing a guard must never do. Any non-finite loss is a divergence.
        relative_difference = float("inf")
    return {
        "device_loss": device_loss,
        "reference_loss": reference_loss,
        "relative_difference": relative_difference,
    }


def _sample_values_to(trace, device):
    """A minimal trace holding only the sample sites' values, moved to ``device``.

    Replay reads nothing but ``nodes[name]["value"]`` for sample sites; carrying the
    original distribution objects over would drag their device tensors along.
    """
    from pyro.poutine.trace_struct import Trace

    moved = Trace()
    for name, node in trace.nodes.items():
        if node.get("type") != "sample":
            continue
        new_node = dict(node)
        value = node.get("value")
        if isinstance(value, torch.Tensor):
            new_node["value"] = value.detach().to(device)
        moved.add_node(name, **new_node)
    return moved


class NumericalGuard:
    """Lightning callback comparing Metal against CPU during training.

    Parameters
    ----------
    every_n_steps
        Comparison interval. 0 disables.
    tolerance
        Relative difference above which a warning is raised. The default of 1e-3 is
        loose enough to ignore float32 rounding and reordered reductions, and tight
        enough to catch a wrong kernel, which misses by percent or more.
    max_warnings
        Stop warning after this many, so a genuinely diverged run does not produce
        thirty identical messages.
    """

    def __init__(self, every_n_steps: int = 1000, tolerance: float = 1e-3, max_warnings: int = 3):
        self.every_n_steps = every_n_steps
        self.tolerance = tolerance
        self.max_warnings = max_warnings
        self.history: List[Dict[str, Any]] = []
        self._warnings_issued = 0

    #: A heavy-tail SVI draw makes at most a check or two of many exceed
    #: tolerance (the handoff measured ~1 outlier per 700 epochs; the guard
    #: samples far less often than that). A rate above this is not a tail, it is
    #: a consistent bias -- e.g. a deterministically wrong kernel, which is
    #: reproducible and so escapes the per-check benign flag. Set well above the
    #: heavy-tail rate and well below any systematic error's ~100%.
    benign_fraction: float = 0.05

    @property
    def diverged(self) -> bool:
        """Whether the run really diverged, tolerating only benign heavy-tail draws.

        A check is exonerated only if the flat guard verified it benign: on an
        over-tolerance check it re-evaluates the device log-joint at the same
        latents and sets ``reproducible_benign`` when the device agrees with
        itself -- a genuine fp32 gap on an extreme single draw, which does not
        corrupt device-only training. Real divergence is caught two ways, so a
        deterministically-wrong kernel (reproducible on every check, thus flagged
        benign individually) cannot hide:

        * any over-tolerance check that is NOT benign -- non-finite, device
          non-reproducible (the fused-kernel race), or a pyro-path record with no
          flag at all (unchanged behaviour); or
        * more than ``benign_fraction`` of all checks over tolerance, however
          reproducible -- a rate that high is bias, not a tail.
        """
        over = [r for r in self.history if r["relative_difference"] > self.tolerance]
        if any(not r.get("reproducible_benign", False) for r in over):
            return True
        return bool(self.history) and len(over) / len(self.history) > self.benign_fraction

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"checks": 0, "diverged": False}
        differences = [record["relative_difference"] for record in self.history]
        return {
            "checks": len(differences),
            "diverged": self.diverged,
            "max_relative_difference": max(differences),
            "mean_relative_difference": sum(differences) / len(differences),
            "tolerance": self.tolerance,
        }

    def check(self, module, args, kwargs, step: int) -> Optional[Dict[str, float]]:
        """Run one comparison and record it. Returns the record, or None if skipped."""
        result = compare_loss_across_devices(module, args, kwargs)
        if result is None:
            return None

        record = {"step": step, **result}
        self.history.append(record)

        if result["relative_difference"] > self.tolerance and self._warnings_issued < self.max_warnings:
            self._warnings_issued += 1
            logger.warning(
                "Numerical guard: step %d loss differs by %.3g between %s and CPU "
                "(%.6g vs %.6g, tolerance %.3g). This is larger than float32 rounding explains. "
                "Re-run benchmarks/apple_silicon_check.py, and consider "
                "CELL2LOCATION_MPS_LGAMMA=stirling or accelerator='cpu' until it is resolved.",
                step,
                result["relative_difference"],
                "the GPU",
                result["device_loss"],
                result["reference_loss"],
                self.tolerance,
            )
        return record

    def as_callback(self):
        """Wrap as a Lightning ``Callback``. Imported lazily to keep this module light."""
        from lightning.pytorch.callbacks import Callback

        guard = self

        class _NumericalGuardCallback(Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if guard.every_n_steps <= 0 or trainer.global_step == 0:
                    return
                if trainer.global_step % guard.every_n_steps != 0:
                    return

                module = getattr(pl_module, "module", None)
                if module is None:
                    return

                try:
                    args, kwargs = module._get_fn_args_from_batch(batch)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Numerical guard: could not extract batch args (%s).", exc)
                    return

                guard.check(module, args, kwargs, trainer.global_step)

            def on_train_end(self, trainer, pl_module):
                summary = guard.summary()
                if summary["checks"] == 0:
                    logger.warning(
                        "Numerical guard was armed but completed ZERO checks -- the run is "
                        "unverified, not verified. Every comparison failed or was skipped; "
                        "enable debug logging to see why."
                    )
                    return
                if summary["diverged"]:
                    logger.warning(
                        "Numerical guard: %d/%d checks exceeded tolerance (worst %.3g). "
                        "Treat these results as unverified.",
                        sum(1 for r in guard.history if r["relative_difference"] > guard.tolerance),
                        summary["checks"],
                        summary["max_relative_difference"],
                    )
                else:
                    logger.info(
                        "Numerical guard: %d checks, worst relative difference %.3g -- " "GPU and CPU agree.",
                        summary["checks"],
                        summary["max_relative_difference"],
                    )

        return _NumericalGuardCallback()
