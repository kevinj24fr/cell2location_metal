"""The flat training engine (task #13): pyro-free full-batch SVI on two flat tensors.

Every guide parameter lives in two flat leaves -- unconstrained loc and
softplus-unconstrained rho, matching AutoNormal's SoftplusPositive scale
parameterization (verified per site at build time, not assumed). Each step draws
one eps vector, reconstructs per-site tensors by slicing, and optimizes
``-(flat_log_joint - flat_log_q)`` with Adam; gradients are clamped elementwise
at +-10 to mirror the pyro path's ClippedAdam. The correctness chain to pyro is
tests/test_flat_train.py -> test_flat_elbo.py -> test_flat_joint.py.

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

#: Elementwise gradient clamp, mirroring pyro ClippedAdam's default clip_norm.
_GRAD_CLAMP = 10.0


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
        state.loc.grad.clamp_(-_GRAD_CLAMP, _GRAD_CLAMP)
        state.rho.grad.clamp_(-_GRAD_CLAMP, _GRAD_CLAMP)
        optimizer.step()
        losses.append(loss_value)

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
