"""The flat training engine (task #13): pyro-free full-batch SVI on two flat tensors.

Every guide parameter lives in two flat leaves -- unconstrained loc and
softplus-unconstrained rho, matching AutoNormal's SoftplusPositive scale
parameterization (verified per site at build time, not assumed). Each step draws
one eps vector, reconstructs per-site tensors by slicing, and optimizes
``-(flat_log_joint - flat_log_q)`` with plain (unclipped) Adam -- deliberately.
scvi's pyro path documents ClippedAdam but actually builds unclipped
``pyro.optim.Adam``; an elementwise +-10 clamp we once added "for fidelity"
caused late-horizon instability (loss bottoming at ~3600 epochs then spiking
~1e6 nats with systematic ~2-posterior-sd abundance drift at 5000x10000 scale)
because clamping the skewed, large-magnitude per-draw gradients of a summed
log-likelihood biases their mean. Removing it restored parity: 100% of
abundances within 1 posterior sd of the pyro path, r 0.992 (2026-08-11).
Do not reintroduce clipping without a trajectory-validation artifact. The
correctness chain to pyro is tests/test_flat_train.py -> test_flat_elbo.py ->
test_flat_joint.py.

Scope matches the flat log-joint: full batch, no dropout, no initial-value
branches, single particle, unscaled ELBO. The caller falls back to the pyro
path whenever any of that fails to hold; the guide is only written to after a
finite run, so a diverged flat run leaves the model exactly as the pyro path
would find it.
"""

import logging
import math
import os
from types import SimpleNamespace

import torch

from ._convergence import EARLY_STOP_ENV_VAR, RelativeEarlyStopping
from ._flat_joint import flat_log_joint
from ._sampling import NotVectorizable, _autonormal_site_params

logger = logging.getLogger(__name__)

__all__ = ["FLAT_ENGINE_ENV_VAR", "FlatGuideState", "flat_training_loss", "flat_log_q_from_state", "run_flat_training"]

#: Set to 0 to disable the flat engine and train through the pyro path unchanged.
FLAT_ENGINE_ENV_VAR = "CELL2LOCATION_MPS_FLAT_ENGINE"

_LOG_2PI = math.log(2.0 * math.pi)


def _softplus_inv(x):
    return x + torch.log(-torch.expm1(-x))


class FlatGuideState:
    """Two flat leaf tensors plus the site -> slice map to rebuild guide tensors."""

    def __init__(self, names, shapes, slices, transforms, loc, rho):
        self.names = names
        self.shapes = shapes
        self.slices = slices
        self.transforms = transforms
        self.loc = loc
        self.rho = rho

    @classmethod
    def from_guide(cls, guide):
        from pyro.infer.autoguide.utils import deep_getattr

        names, shapes, slices, transforms = [], [], [], []
        locs, rhos = [], []
        offset = 0
        for name, loc, scale, transform in _autonormal_site_params(guide):
            unconstrained = deep_getattr(guide.scales, name + "_unconstrained")
            if not torch.allclose(torch.nn.functional.softplus(unconstrained.detach()), scale.detach(), rtol=1e-5):
                raise NotVectorizable(f"site {name}: guide scale is not softplus-parameterized")
            n = loc.numel()
            names.append(name)
            shapes.append(tuple(loc.shape))
            slices.append(slice(offset, offset + n))
            transforms.append(transform)
            locs.append(loc.detach().reshape(-1))
            rhos.append(unconstrained.detach().reshape(-1))
            offset += n
        loc_flat = torch.cat(locs).clone().requires_grad_(True)
        rho_flat = torch.cat(rhos).clone().requires_grad_(True)
        return cls(names, shapes, slices, transforms, loc_flat, rho_flat)

    def unpack(self, flat):
        return {
            name: flat[sl].view(shape) for name, shape, sl in zip(self.names, self.shapes, self.slices)
        }

    def constrain(self, u_flat):
        return {
            name: transform(u_flat[sl].view(shape))
            for name, shape, sl, transform in zip(self.names, self.shapes, self.slices, self.transforms)
        }

    def write_back(self, guide):
        """Copy the trained parameters into the guide so export reads them."""
        from pyro.infer.autoguide.utils import deep_getattr

        with torch.no_grad():
            for name, shape, sl in zip(self.names, self.shapes, self.slices):
                loc_param = deep_getattr(guide.locs, name)
                loc_param.data.copy_(self.loc[sl].view(shape))
                rho_param = deep_getattr(guide.scales, name + "_unconstrained")
                rho_param.data.copy_(self.rho[sl].view(shape))


def flat_log_q_from_state(loc_flat, scale_flat, u_flat, state):
    """log q of the draw from the flat tensors: one Normal over the concatenated
    vector (sites are independent under mean-field) minus per-site jacobians."""
    normal_lp = (
        -0.5 * ((u_flat - loc_flat) / scale_flat).pow(2) - torch.log(scale_flat) - 0.5 * _LOG_2PI
    ).sum()
    ladj = u_flat.new_zeros(())
    for shape, sl, transform in zip(state.shapes, state.slices, state.transforms):
        u = u_flat[sl].view(shape)
        ladj = ladj + transform.log_abs_det_jacobian(u, transform(u)).sum()
    return normal_lp - ladj


def flat_training_loss(module, state, args, kwargs, eps):
    """-(single-particle ELBO estimate) at u = loc + softplus(rho) * eps."""
    scale_flat = torch.nn.functional.softplus(state.rho)
    u_flat = state.loc + scale_flat * eps
    log_joint = flat_log_joint(module, args, kwargs, state.constrain(u_flat))
    return -(log_joint - flat_log_q_from_state(state.loc, scale_flat, u_flat, state))


def _flat_guard_check(guard, module, state, args, kwargs, epoch):
    """One guard comparison: flat_log_joint at the same fresh draw on the training
    device and on CPU. Identical latents make it deterministic -- any disagreement
    is device arithmetic, and it is exactly the arithmetic the flat engine trains
    on (the pyro-path guard compares the replayed model log-joint instead)."""
    with torch.no_grad():
        u_flat = state.loc + torch.nn.functional.softplus(state.rho) * torch.randn_like(state.loc)
        latents = state.constrain(u_flat)
        try:
            device_loss = float(flat_log_joint(module, args, kwargs, latents))
        except Exception as exc:  # noqa: BLE001 - a failed check must not kill training
            logger.debug("Flat guard: device evaluation failed (%s).", exc)
            return None
        original_device = state.loc.device
        try:
            cpu_latents = {name: value.detach().cpu() for name, value in latents.items()}
            cpu_args = tuple(a.cpu() if torch.is_tensor(a) else a for a in args)
            module.to("cpu")
            try:
                reference_loss = float(flat_log_joint(module, cpu_args, kwargs, cpu_latents))
            finally:
                module.to(original_device)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Flat guard: CPU evaluation failed (%s).", exc)
            return None

    scale = max(abs(reference_loss), 1e-12)
    relative_difference = abs(device_loss - reference_loss) / scale
    if not math.isfinite(relative_difference):
        # Non-finite must read as divergence, never as agreement.
        relative_difference = float("inf")
    record = {
        "step": epoch,
        "device_loss": device_loss,
        "reference_loss": reference_loss,
        "relative_difference": relative_difference,
    }
    guard.history.append(record)
    if relative_difference > guard.tolerance:
        logger.warning(
            "Numerical guard (flat engine): epoch %d log-joint differs by %.3g between "
            "the GPU and CPU (%.6g vs %.6g, tolerance %.3g).",
            epoch, relative_difference, device_loss, reference_loss, guard.tolerance,
        )
    return record


def _full_batch_args(model, device):
    from scvi.dataloaders import AnnDataLoader

    dl = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=model.adata.n_obs)
    args, kwargs = model.module._get_fn_args_from_batch(next(iter(dl)))
    args = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
    return args, kwargs


def _make_optimizer(state, lr):
    try:
        return torch.optim.Adam([state.loc, state.rho], lr=lr, fused=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.Adam([state.loc, state.rho], lr=lr)


def run_flat_training(model, kwargs) -> bool:
    """Train the model's guide with the flat engine. Returns False (guide untouched)
    if the engine diverges; raises NotVectorizable if the guide is out of scope."""
    module = model.module
    device = torch.device("mps")
    module.to(device)
    args, batch_kwargs = _full_batch_args(model, device)

    guide = module.guide
    if getattr(guide, "prototype_trace", None) is None:
        with torch.no_grad():
            guide(*args, **batch_kwargs)

    state = FlatGuideState.from_guide(guide)
    optimizer = _make_optimizer(state, kwargs.get("lr", 0.002))
    max_epochs = kwargs.get("max_epochs", 30000)
    logger.info(
        "Flat engine: training %s up to %d epochs (%d parameters, lr=%g).",
        type(getattr(module, "model", module)).__name__, max_epochs, state.loc.numel(), kwargs.get("lr", 0.002),
    )

    guard = None
    if any(type(cb).__name__ == "_NumericalGuardCallback" for cb in kwargs.get("callbacks") or []):
        guard = getattr(model, "numerical_guard_", None)

    stopper = None
    stop_cfg = getattr(model, "mps_early_stopping", None)
    env_off = os.environ.get(EARLY_STOP_ENV_VAR, "1").lower() in ("0", "false", "no")
    if stop_cfg and not env_off:
        stopper = RelativeEarlyStopping(**stop_cfg)
        model.early_stopping_ = stopper

    loss_fn = flat_training_loss
    if getattr(model, "mps_flat_compile", True):
        try:
            loss_fn = torch.compile(flat_training_loss)
        except Exception as exc:  # noqa: BLE001 - compile support varies by torch build
            logger.info("torch.compile unavailable for the flat step (%s); running eager.", exc)

    losses = []
    for epoch in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        eps = torch.randn_like(state.loc)
        try:
            loss = loss_fn(module, state, args, batch_kwargs, eps)
        except Exception as exc:  # noqa: BLE001 - a compiled step may fail at runtime
            if loss_fn is flat_training_loss:
                raise
            logger.warning("Compiled flat step failed (%s); retrying eager.", exc)
            loss_fn = flat_training_loss
            loss = loss_fn(module, state, args, batch_kwargs, eps)
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            logger.warning(
                "Flat engine diverged at epoch %d (loss=%r); guide left untouched, "
                "falling back to the pyro path.", epoch, loss_value,
            )
            return False
        loss.backward()
        optimizer.step()
        losses.append(loss_value)

        if guard is not None and guard.every_n_steps and epoch % guard.every_n_steps == 0:
            _flat_guard_check(guard, module, state, args, batch_kwargs, epoch)

        if stopper is not None:
            shim = SimpleNamespace(
                callback_metrics={stopper.monitor: loss_value}, current_epoch=epoch, should_stop=False
            )
            stopper.on_train_epoch_end(shim, None)
            if shim.should_stop:
                break

    state.write_back(guide)
    module.eval()
    logger.info("Flat engine: finished after %d epochs (final loss %.6g).", len(losses), losses[-1])

    import pandas as pd

    history = pd.DataFrame({"elbo_train": losses})
    history.index.name = "epoch"
    model.history_ = {"elbo_train": history}
    model.is_trained_ = True
    return True
