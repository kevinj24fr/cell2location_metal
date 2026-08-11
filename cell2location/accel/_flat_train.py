"""The flat training engine: pyro-free SVI on two flat tensors.

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

Both full-batch and minibatch training run here. Which sites a minibatch step
has to subsample and scale comes from each model's own ``list_obs_plate_vars``:
the reference model declares none, so only its likelihood scales, while the
spatial model's five per-location latents are subsampled with the data and
their priors and log q scale too.

Scope otherwise matches the flat log-joint: no dropout, single particle,
unscaled ELBO, and a model whose transcription covers it (``log_joint_for``).
The caller falls back to the pyro path whenever any of that fails to hold; the
guide is only written to after a finite run, so a diverged flat run leaves the
model exactly as the pyro path would find it.
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

__all__ = [
    "FLAT_ENGINE_ENV_VAR",
    "FlatGuideState",
    "flat_training_loss",
    "flat_log_q_from_state",
    "pack_module",
    "run_flat_training",
    "run_flat_minibatch_training",
    "flat_minibatch_loss",
    "flat_log_q_minibatch",
    "local_plate_sites",
]

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

    def constrain_rows(self, u_flat, rows, local_sites):
        """Constrained latents with per-observation sites cut to ``rows``.

        The guide's local parameters stay full size and are drawn whole -- they
        are tiny next to the data (locations x factors, against locations x
        genes) -- and only the rows this batch needs enter the graph. Rows not in
        the batch get no gradient, which is what a subsampled plate means.
        """
        out = {}
        for name, shape, sl, transform in zip(self.names, self.shapes, self.slices, self.transforms):
            u = u_flat[sl].view(shape)
            if name in local_sites:
                u = u[rows]
            out[name] = transform(u)
        return out

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


def local_plate_sites(module):
    """Names of latent sites inside the observation plate, or None if unreadable.

    Read from the model's own ``list_obs_plate_vars`` so a model that grows or
    loses a per-observation site changes behaviour without an edit here.

    None and the empty set mean different things and must not be conflated: an
    empty set says "every latent is global, subsample the data alone", while
    None says the declaration could not be read at all, which is not a licence
    to subsample. Callers route None to pyro.
    """
    mod = getattr(module, "model", None)
    mod = getattr(mod, "_orig_mod", mod)
    lister = getattr(mod, "list_obs_plate_vars", None)
    if lister is None:
        return None
    try:
        return frozenset(lister()["sites"])
    except Exception:  # noqa: BLE001 - an unreadable declaration is not a green light
        return None


def flat_log_q_minibatch(state, loc_flat, scale_flat, u_flat, rows, local_sites, plate_scale):
    """log q for a subsampled plate: global sites whole, local sites cut to the
    batch's rows and scaled, matching how pyro scales guide sites inside a plate.

    Per-site rather than one fused Normal over the flat vector, because the local
    and global blocks are interleaved in that vector and carry different scales.
    """
    if not local_sites:
        # Every site global: log q is unscaled and the sites are contiguous in the
        # flat vector, so the fused whole-vector form applies unchanged. Taking
        # the per-site path here instead costs real time -- it replaces one large
        # Normal evaluation with one per site, and measured 1.26x slower on the
        # reference model, which has nine of them.
        return flat_log_q_from_state(loc_flat, scale_flat, u_flat, state)

    glob = u_flat.new_zeros(())
    local = u_flat.new_zeros(())
    for name, shape, sl, transform in zip(state.names, state.shapes, state.slices, state.transforms):
        u = u_flat[sl].view(shape)
        loc = loc_flat[sl].view(shape)
        scale = scale_flat[sl].view(shape)
        if name in local_sites:
            u, loc, scale = u[rows], loc[rows], scale[rows]
        normal_lp = (
            -0.5 * ((u - loc) / scale).pow(2) - torch.log(scale) - 0.5 * _LOG_2PI
        ).sum()
        term = normal_lp - transform.log_abs_det_jacobian(u, transform(u)).sum()
        if name in local_sites:
            local = local + term
        else:
            glob = glob + term
    return glob + plate_scale * local


def flat_minibatch_loss(module, state, args, kwargs, eps, rows, local_sites,
                        plate_scale, log_joint_fn):
    """-(single-particle ELBO estimate) for one minibatch.

    Serves both models. The reference model's ``local_sites`` is empty, so every
    site is global and only its transcription's likelihood carries the scale;
    the spatial model's five per-location sites are subsampled here and their
    priors and log q scale along with the likelihood.
    """
    scale_flat = torch.nn.functional.softplus(state.rho)
    u_flat = state.loc + scale_flat * eps
    latents = state.constrain_rows(u_flat, rows, local_sites)
    log_joint = log_joint_fn(module, args, kwargs, latents, plate_scale)
    log_q = flat_log_q_minibatch(
        state, state.loc, scale_flat, u_flat, rows, local_sites, plate_scale
    )
    return -(log_joint - log_q)


def flat_training_loss(module, state, args, kwargs, eps, log_joint_fn=None):
    """-(single-particle ELBO estimate) at u = loc + softplus(rho) * eps.

    ``log_joint_fn`` selects the model's transcription; it defaults to the spatial
    one so existing callers and contracts are unchanged. The reference model
    passes its own, and for a minibatch that transcription carries the plate's
    n_obs/batch scale on the likelihood -- the guide term is unscaled either way,
    because that model's latents are all global.
    """
    if log_joint_fn is None:
        log_joint_fn = flat_log_joint
    scale_flat = torch.nn.functional.softplus(state.rho)
    u_flat = state.loc + scale_flat * eps
    log_joint = log_joint_fn(module, args, kwargs, state.constrain(u_flat))
    return -(log_joint - flat_log_q_from_state(state.loc, scale_flat, u_flat, state))


#: Small model buffers the flat loss reads; packed into one tensor for the hot
#: loop. `cell_state` (large) and `n_batch` (int) stay as-is.
_PACKED_MODEL_ATTRS = (
    "m_g_mu_mean_var_ratio_hyp", "m_g_mu_hyp", "m_g_alpha_hyp_mean",
    "N_cells_per_location", "N_cells_mean_var_ratio", "B_groups_per_location",
    "ones_1_n_groups", "n_groups_tensor", "factors_per_groups", "n_factors_tensor",
    "w_sf_mean_var_ratio_tensor", "detection_mean_hyp_prior_alpha",
    "detection_mean_hyp_prior_beta", "detection_hyp_prior_alpha", "ones_n_batch_1",
    "gene_add_alpha_hyp_prior_alpha", "gene_add_alpha_hyp_prior_beta",
    "gene_add_mean_hyp_prior_alpha", "gene_add_mean_hyp_prior_beta",
    "alpha_g_phi_hyp_prior_alpha", "alpha_g_phi_hyp_prior_beta", "ones",
)


#: The reference model's small buffers. Its hyperparameters differ from the
#: spatial model's and it has no ``cell_state``; ``n_obs`` rides along as a plain
#: int because the plate scale needs it.
_REFERENCE_PACKED_ATTRS = (
    "detection_mean_hyp_prior_alpha", "detection_mean_hyp_prior_beta",
    "gene_add_alpha_hyp_prior_alpha", "gene_add_alpha_hyp_prior_beta",
    "gene_add_mean_hyp_prior_alpha", "gene_add_mean_hyp_prior_beta",
    "alpha_g_phi_hyp_prior_alpha", "alpha_g_phi_hyp_prior_beta", "ones",
)

#: model class name -> (buffers to pack, attributes to carry through untouched).
_PACK_SPEC = {
    "LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel": (
        _PACKED_MODEL_ATTRS, ("cell_state", "n_batch"),
    ),
    "RegressionBackgroundDetectionTechPyroModel": (
        _REFERENCE_PACKED_ATTRS, ("n_batch", "n_factors", "n_obs"),
    ),
}


class _PackedModel:
    """The model's small hyperparameter buffers as lazy slices of ONE tensor.

    Attribute access slices inside the caller's graph, so a torch.compile'd flat
    loss sees a single input buffer where the module would contribute 21 -- the
    difference between fitting and not fitting Metal's hard 31-constant-buffer
    kernel limit. Values are copied once at construction; the model's buffers are
    training-constant, and test_packed_model_proxy_matches_module pins equality.
    """

    def __init__(self, pyro_model):
        packed_attrs, passthrough = _PACK_SPEC[type(pyro_model).__name__]
        # One stray float64 buffer (detection_mean_hyp_prior_beta upstream) would
        # promote the whole pack via torch.cat; pin to the model's working dtype.
        dtype = getattr(pyro_model, "cell_state", pyro_model.ones).dtype
        chunks, layout, offset = [], {}, 0
        for name in packed_attrs:
            tensor = getattr(pyro_model, name).detach()
            layout[name] = (offset, offset + tensor.numel(), tuple(tensor.shape))
            chunks.append(tensor.reshape(-1).to(dtype))
            offset += tensor.numel()
        self._packed = torch.cat(chunks)
        self._layout = layout
        for name in passthrough:
            setattr(self, name, getattr(pyro_model, name))

    def __getattr__(self, name):
        layout = object.__getattribute__(self, "_layout")
        if name not in layout:
            raise AttributeError(name)
        start, stop, shape = layout[name]
        return object.__getattribute__(self, "_packed")[start:stop].view(shape)


def pack_module(module):
    """A module stand-in for the hot training loop: ``.model`` is the packed view."""
    mod = module.model
    return SimpleNamespace(model=_PackedModel(getattr(mod, "_orig_mod", mod)))


def _flat_guard_check(guard, module, state, args, kwargs, epoch, log_joint_fn=None,
                      rows=None, local_sites=frozenset(), plate_scale=None):
    """One guard comparison: the flat log-joint at the same fresh draw on the
    training device and on CPU. Identical latents make it deterministic -- any
    disagreement is device arithmetic, and it is exactly the arithmetic the flat
    engine trains on (the pyro-path guard compares the replayed model log-joint).

    On a minibatch the draw must be cut the same way the training step cuts it,
    ``rows`` and all, or the comparison is not of the arithmetic being trained.
    A guard that cannot evaluate is reported at warning level: silence there
    reads as agreement, and "checks: 0" is a verifier that never ran."""
    if log_joint_fn is None:
        log_joint_fn = flat_log_joint
    extra = () if plate_scale is None else (plate_scale,)
    with torch.no_grad():
        u_flat = state.loc + torch.nn.functional.softplus(state.rho) * torch.randn_like(state.loc)
        latents = (
            state.constrain(u_flat) if rows is None
            else state.constrain_rows(u_flat, rows, local_sites)
        )
        try:
            device_loss = float(log_joint_fn(module, args, kwargs, latents, *extra))
        except Exception as exc:  # noqa: BLE001 - a failed check must not kill training
            logger.warning("Flat guard: device evaluation failed (%s); check skipped.", exc)
            return None
        original_device = state.loc.device
        try:
            cpu_latents = {name: value.detach().cpu() for name, value in latents.items()}
            cpu_args = tuple(a.cpu() if torch.is_tensor(a) else a for a in args)
            module.to("cpu")
            try:
                reference_loss = float(log_joint_fn(module, cpu_args, kwargs, cpu_latents, *extra))
            finally:
                module.to(original_device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Flat guard: CPU evaluation failed (%s); check skipped.", exc)
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


def _minibatch_loader(model, batch_size):
    """scvi's own loader, so batching and the registry match the pyro path exactly."""
    from scvi.dataloaders import AnnDataLoader

    return AnnDataLoader(model.adata_manager, shuffle=True, batch_size=batch_size)


#: Fraction of the MPS recommended working set the resident copy may occupy.
#: The copy is only part of the step's footprint -- activations of the same order
#: are allocated per step -- so this stays well clear of the whole budget.
_RESIDENT_MEMORY_FRACTION = 0.25


def _resident_fits(model, device) -> bool:
    """True when the whole training matrix can live on the device with headroom.

    A caller passing ``batch_size`` may be doing it because the data does not
    fit, so residency is a measured decision, not an assumption: estimate the
    copy against the driver's recommended working set and decline if it is a
    meaningful fraction of it. Declining costs performance; guessing wrong costs
    the run.
    """
    if device.type != "mps":
        return False
    adata = getattr(model, "adata", None)
    if adata is None:
        return False
    try:
        budget = torch.mps.recommended_max_memory()
    except (AttributeError, RuntimeError):
        return False
    if not budget:
        return False
    # float32 counts plus the per-observation index/batch/label columns.
    estimate = adata.n_obs * (adata.n_vars + 3) * 4
    return estimate <= _RESIDENT_MEMORY_FRACTION * budget


class _ResidentBatches:
    """Minibatches cut from one device-resident copy of the data.

    Profiling the streaming path at 10,000x10,000 showed the flat step itself at
    11.4 ms while scvi's loader collation plus the host-to-device copy cost
    16.7 ms -- 59% of the step spent moving data that never changes. Staging it
    once and gathering rows on the device removes that per-step CPU work.

    Batch composition is equivalent to the loader's: a fresh permutation each
    epoch, same batch size, trailing partial batch kept (the plate scale reads
    the actual batch length, so a short last batch stays correct).
    """

    def __init__(self, model, batch_size, device):
        args, self.kwargs = _full_batch_args(model, device)
        self.args = args
        self.n_obs = args[0].shape[0]
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        perm = torch.randperm(self.n_obs, device=self.device)
        for start in range(0, self.n_obs, self.batch_size):
            rows = perm[start : start + self.batch_size]
            yield tuple(
                a[rows] if torch.is_tensor(a) and a.shape[:1] == (self.n_obs,) else a
                for a in self.args
            ), self.kwargs

    def __len__(self):
        return (self.n_obs + self.batch_size - 1) // self.batch_size


def _to_device(args, device):
    return tuple(a.to(device) if torch.is_tensor(a) else a for a in args)


def run_flat_minibatch_training(model, kwargs, log_joint_fn) -> bool:
    """Minibatch flat training. Serves both models through one path.

    What a minibatch step has to scale depends on the model. The reference
    model's ``list_obs_plate_vars()["sites"]`` is empty -- every latent global --
    so only the likelihood scales. The spatial model has five per-location
    latents, so those latents are subsampled with the data and their priors and
    log q scale too. Both cases fall out of reading the model's own declaration.

    Each step's loss estimates the FULL-data negative ELBO, and history records
    the epoch mean of them -- what scvi's ``elbo_train`` is -- so the two paths'
    histories and early-stopping signals mean the same thing.

    Batches come from scvi's AnnDataLoader unless the data comfortably fits on
    the device: a caller who passed ``batch_size`` may have done so because it
    does not, and silently materialising all of it would defeat the reason they
    asked.

    Returns False (guide untouched) if the engine diverges.
    """
    module = model.module
    device = torch.device("mps")
    module.to(device)

    batch_size = kwargs["batch_size"]
    resident = _resident_fits(model, device)
    if resident:
        batches = _ResidentBatches(model, batch_size, device)

        def epoch_batches():
            return iter(batches)
    else:
        loader = _minibatch_loader(model, batch_size)

        def epoch_batches():
            for batch in loader:
                args, batch_kwargs = module._get_fn_args_from_batch(batch)
                yield _to_device(args, device), batch_kwargs

    probe_args, batch_kwargs = next(epoch_batches())
    local_sites = local_plate_sites(module) or frozenset()
    n_obs = model.adata.n_obs

    guide = module.guide
    if getattr(guide, "prototype_trace", None) is None:
        with torch.no_grad():
            guide(*probe_args, **batch_kwargs)

    state = FlatGuideState.from_guide(guide)
    packed = pack_module(module)
    lr = kwargs.get("lr", 0.002)
    optimizer = _make_optimizer(state, lr)
    max_epochs = kwargs.get("max_epochs", 30000)
    logger.info(
        "Flat engine (minibatch, %s data): training %s up to %d epochs "
        "(%d parameters, batch_size=%d, lr=%g).",
        "device-resident" if resident else "streamed",
        type(getattr(module, "model", module)).__name__, max_epochs,
        state.loc.numel(), batch_size, lr,
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

    losses = []
    step = 0
    for epoch in range(max_epochs):
        epoch_losses = []
        for args, batch_kwargs in epoch_batches():
            # args[1] is ind_x: the batch's ORIGINAL observation indices, which
            # are exactly the guide rows a subsampled plate touches.
            rows = args[1].reshape(-1).long()
            plate_scale = float(n_obs) / float(args[0].shape[0])

            optimizer.zero_grad(set_to_none=True)
            eps = torch.randn_like(state.loc)
            loss = flat_minibatch_loss(packed, state, args, batch_kwargs, eps, rows,
                                       local_sites, plate_scale, log_joint_fn)
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                logger.warning(
                    "Flat engine diverged at epoch %d (loss=%r); guide left untouched, "
                    "falling back to the pyro path.", epoch, loss_value,
                )
                return False
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_value)

            if guard is not None and guard.every_n_steps and step % guard.every_n_steps == 0:
                _flat_guard_check(guard, module, state, args, batch_kwargs, step,
                                  log_joint_fn, rows, local_sites, plate_scale)
            step += 1

        # scvi logs elbo_train as the epoch MEAN of per-step losses; match it so
        # history_ and the early-stopping signal mean the same thing on both paths.
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)

        if stopper is not None:
            shim = SimpleNamespace(
                callback_metrics={stopper.monitor: epoch_loss}, current_epoch=epoch, should_stop=False
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


def _make_optimizer(state, lr):
    try:
        return torch.optim.Adam([state.loc, state.rho], lr=lr, fused=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.Adam([state.loc, state.rho], lr=lr)


def run_flat_training(model, kwargs, log_joint_fn=None) -> bool:
    """Train the model's guide with the flat engine, full batch. Returns False
    (guide untouched) if the engine diverges; raises NotVectorizable if the guide
    is out of scope. ``log_joint_fn`` defaults to the spatial transcription."""
    module = model.module
    device = torch.device("mps")
    module.to(device)
    args, batch_kwargs = _full_batch_args(model, device)

    guide = module.guide
    if getattr(guide, "prototype_trace", None) is None:
        with torch.no_grad():
            guide(*args, **batch_kwargs)

    state = FlatGuideState.from_guide(guide)
    packed = pack_module(module)
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
            loss = loss_fn(packed, state, args, batch_kwargs, eps, log_joint_fn)
        except Exception as exc:  # noqa: BLE001 - a compiled step may fail at runtime
            if loss_fn is flat_training_loss:
                raise
            logger.warning("Compiled flat step failed (%s); retrying eager.", exc)
            loss_fn = flat_training_loss
            loss = loss_fn(packed, state, args, batch_kwargs, eps, log_joint_fn)
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
            _flat_guard_check(guard, module, state, args, batch_kwargs, epoch, log_joint_fn)

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
