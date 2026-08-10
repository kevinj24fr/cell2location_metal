"""Training-time glue shared by the user-facing model classes."""

import logging
import os

from ._compat import prepare_module_for_device
from ._convergence import EARLY_STOP_ENV_VAR, RelativeEarlyStopping
from ._device import resolve_accelerator
from ._dtype import check_anndata_dtype, prepare_anndata
from ._guard import NumericalGuard
from ._memory import MPSCacheCallback

logger = logging.getLogger(__name__)

__all__ = ["AppleSiliconTrainMixin", "GUARD_ENV_VAR", "device_cached_plan"]

#: Set to a step interval (or ``1`` for the default interval) to enable the runtime
#: divergence guard without changing any code.
GUARD_ENV_VAR = "CELL2LOCATION_MPS_GUARD"

_DEFAULT_GUARD_INTERVAL = 1000

#: Train kwargs the flat engine implements (or checks and matches exactly). Anything
#: else present routes training to the pyro path.
_FLAT_HANDLED_TRAIN_KWARGS = frozenset(
    {
        "max_epochs", "batch_size", "train_size", "lr", "accelerator", "device",
        "plan_kwargs", "callbacks", "enable_progress_bar", "enable_model_summary",
    }
)

#: Callbacks whose behavior the flat engine reproduces itself (early stopping) or
#: that a full-batch flat run does not need (periodic MPS cache release).
_FLAT_EQUIVALENT_CALLBACKS = frozenset({"_MPSCacheCallback", "_RelativeEarlyStoppingCallback"})


def _guard_interval_from_env() -> int:
    raw = os.environ.get(GUARD_ENV_VAR, "0").strip().lower()
    if raw in ("", "0", "false", "no"):
        return 0
    if raw in ("1", "true", "yes"):
        return _DEFAULT_GUARD_INTERVAL
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring.", GUARD_ENV_VAR, raw)
        return 0


def device_cached_plan(plan_cls):
    """Subclass a Lightning training plan so a single-batch epoch is moved to the
    GPU once and reused.

    Full-batch training re-collates and re-copies the identical dataset every epoch
    -- 24 ms of a 149 ms epoch at Visium scale, for zero information. Caching is
    gated at runtime on ``trainer.num_training_batches == 1`` during the training
    phase: a single-batch epoch has the same content every time (order within the
    batch is carried by its own index tensor, and the ELBO sums over locations, so
    even shuffling changes nothing), while genuine minibatches and validation
    batches always transfer normally.
    """
    if getattr(plan_cls, "_c2l_device_cached", False):
        return plan_cls

    class _DeviceCachedFullBatchPlan(plan_cls):
        _c2l_device_cached = True
        _c2l_device_batch = None

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            trainer = getattr(self, "trainer", None)
            cacheable = (
                trainer is not None
                and getattr(trainer, "training", False)
                and getattr(trainer, "num_training_batches", 0) == 1
            )
            if not cacheable:
                return super().transfer_batch_to_device(batch, device, dataloader_idx)
            if self._c2l_device_batch is None:
                self._c2l_device_batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
            return self._c2l_device_batch

    _DeviceCachedFullBatchPlan.__name__ = f"DeviceCached{plan_cls.__name__}"
    _DeviceCachedFullBatchPlan.__qualname__ = _DeviceCachedFullBatchPlan.__name__
    return _DeviceCachedFullBatchPlan


class AppleSiliconTrainMixin:
    """Prepares a training run for the Metal backend.

    Lightning's ``accelerator="auto"`` already picks MPS on an Apple silicon Mac, so
    this runs on the default path -- there is no opt-in for a user to forget, and
    equally no behaviour change on Linux/CUDA where every branch here is skipped.
    """

    #: How often to release cached MPS blocks during training. 0 disables.
    mps_empty_cache_every_n_steps: int = 500

    #: How often to cross-check the loss against the CPU. 0 disables, which is the
    #: default because the check costs a CPU forward pass. Set this attribute, or
    #: ``CELL2LOCATION_MPS_GUARD``, when a run's results matter enough to verify --
    #: which, for anything headed into a paper, is all of them.
    mps_numerical_guard_every_n_steps: int = 0

    #: Populated during training when the guard is active, so the outcome can be
    #: inspected afterwards: ``model.numerical_guard_.summary()``.
    numerical_guard_ = None

    #: Convergence-based early stopping for Metal runs: stop when the best ELBO has
    #: not improved by rel_tol (relative) within patience epochs, never before
    #: min_epochs. Defaults are the validated "tight" operating point (3.1x over
    #: upstream's fixed 30k epochs; every abundance within 1 posterior sd of the
    #: full run, r=0.990). A looser point ({"rel_tol": 1e-4, "patience": 500,
    #: "min_epochs": 1000}) reaches ~7x but drifts beyond the posterior's own
    #: resolution and is deliberately not the default. Set to None (or
    #: CELL2LOCATION_MPS_EARLY_STOP=0) to reproduce upstream exactly.
    mps_early_stopping: dict = {"rel_tol": 1e-5, "patience": 2000, "min_epochs": 2000}

    #: Populated when early stopping is active; ``model.early_stopping_.stopped_epoch``
    #: records where (None if the run reached max_epochs).
    early_stopping_ = None

    #: torch.compile the flat engine's step function (eager fallback on failure).
    #: Off by default: at torch 2.12 inductor fuses the flat loss into a single
    #: Metal kernel whose input count exceeds Metal's 31-constant-buffer limit, so
    #: compilation always fails and the attempt costs seconds per train() call.
    #: Revisit alongside the fused mu/alpha kernel (packing the module's ~30 scalar
    #: hyperparameter buffers into one tensor removes the cause).
    mps_flat_compile: bool = False

    #: True when the last ``train()`` ran on the flat engine rather than pyro.
    flat_engine_used_ = False

    def _flat_train_if_applicable(self, kwargs: dict) -> bool:
        """Train with the flat engine when its verified scope holds; False otherwise.

        Scope: MPS device, full batch, single-particle unscaled Trace_ELBO, default
        optimizer, an AutoNormal guide, no dropout, no initial-value branches. Any
        miss -- including a divergence mid-run -- leaves training to the pyro path.
        """
        from ._flat_train import FLAT_ENGINE_ENV_VAR, run_flat_training
        from ._sampling import NotVectorizable

        self.flat_engine_used_ = False
        if os.environ.get(FLAT_ENGINE_ENV_VAR, "1").strip().lower() in ("0", "false", "no"):
            return False
        # Any train kwarg the flat engine does not implement routes to pyro rather
        # than being silently ignored. scvi's load() warmup depends on this: it
        # calls train(max_steps=1), and an engine that ignores max_steps would
        # retrain a freshly loaded model to convergence.
        unknown = set(kwargs) - _FLAT_HANDLED_TRAIN_KWARGS
        if unknown:
            logger.info("Flat engine: unhandled train kwargs %s; training through pyro.", sorted(unknown))
            return False
        _, device = resolve_accelerator(kwargs.get("accelerator", "auto"), kwargs.get("device", "auto"))
        if device.type != "mps":
            return False
        if kwargs.get("batch_size") is not None or kwargs.get("train_size", 1) != 1:
            return False
        callbacks = kwargs.get("callbacks") or []
        if any(type(cb).__name__ not in _FLAT_EQUIVALENT_CALLBACKS for cb in callbacks):
            # Unrecognized callbacks (including the replay-based numerical guard,
            # which has no flat equivalent yet) need the Lightning loop.
            return False
        plan = kwargs.get("plan_kwargs") or {}
        loss_fn = plan.get("loss_fn")
        if loss_fn is not None and (
            type(loss_fn).__name__ != "Trace_ELBO" or getattr(loss_fn, "num_particles", 1) != 1
        ):
            return False
        if plan.get("scale_elbo", 1.0) != 1.0 or "optim" in plan or "optim_kwargs" in plan:
            return False
        pyro_model = getattr(self.module, "model", None)
        if getattr(pyro_model, "np_init_vals", None) is not None:
            return False
        if getattr(pyro_model, "dropout_p", 0.0):
            return False
        if getattr(pyro_model, "training_wo_observed", False):
            return False
        try:
            self.flat_engine_used_ = run_flat_training(self, kwargs)
        except NotVectorizable as exc:
            logger.info("Flat engine unavailable (%s); training through pyro.", exc)
            return False
        return self.flat_engine_used_

    def _get_posterior_samples(self, args, kwargs, **sample_kwargs):
        """Vectorized posterior sampling when it is exactly equivalent; loop otherwise.

        The fast path requires a mean-field AutoNormal guide (site marginals ARE the
        joint) and a batch covering every observation (a minibatched caller stitches
        per-batch results, which a full-size draw would break)."""
        from ._sampling import NotVectorizable, vectorized_posterior_samples

        module = sample_kwargs.get("model") or self.module
        plain = not sample_kwargs.get("return_observed") and not sample_kwargs.get("exclude_vars")
        n_obs = getattr(getattr(self, "adata", None), "n_obs", None)
        full_batch = bool(args) and hasattr(args[0], "shape") and args[0].shape[0] == n_obs
        if plain and full_batch:
            try:
                return vectorized_posterior_samples(
                    module, args, kwargs,
                    num_samples=sample_kwargs.get("num_samples", 1000),
                    return_sites=sample_kwargs.get("return_sites"),
                )
            except NotVectorizable as exc:
                logger.info("Vectorized posterior sampling unavailable (%s); using the looped sampler.", exc)
        return super()._get_posterior_samples(args, kwargs, **sample_kwargs)

    def _prepare_apple_silicon(self, kwargs: dict) -> None:
        _, device = resolve_accelerator(kwargs.get("accelerator", "auto"), kwargs.get("device", "auto"))
        if device.type != "mps":
            return

        # Lightning moves the training plan with ``_apply``, which recurses into
        # child modules without ever calling their ``to()`` -- so the CompatMixin
        # interception never fires on this path and float64 buffers reach the MPS
        # move intact. Downcast here, while everything is still on the CPU.
        module = getattr(self, "module", None)
        if module is not None:
            prepare_module_for_device(module, device)

        manager = getattr(self, "adata_manager", None)
        adata = getattr(manager, "adata", None)
        if adata is not None:
            layer = None
            try:
                registry = manager.data_registry["X"]
                if registry.attr_name == "layers":
                    layer = registry.attr_key
            except Exception:  # noqa: BLE001 - registry layouts vary across scvi versions
                layer = None
            if not check_anndata_dtype(adata, layer=layer):
                # Not a crash risk (scvi casts per batch), but that cast re-runs over
                # the full matrix every epoch; converting once here is free speed.
                logger.info("Metal backend: converting the count matrix to float32 in place.")
                prepare_anndata(adata, layer=layer)

        plan_cls = getattr(self, "_training_plan_cls", None)
        if plan_cls is not None:
            self._training_plan_cls = device_cached_plan(plan_cls)

        callbacks = kwargs.setdefault("callbacks", [])

        if self.mps_empty_cache_every_n_steps:
            if not any(type(cb).__name__ == "_MPSCacheCallback" for cb in callbacks):
                callbacks.append(MPSCacheCallback(every_n_steps=self.mps_empty_cache_every_n_steps).as_callback())

        stop_cfg = self.mps_early_stopping
        env_off = os.environ.get(EARLY_STOP_ENV_VAR, "1").lower() in ("0", "false", "no")
        if stop_cfg and not env_off:
            if not any(type(cb).__name__ == "_RelativeEarlyStoppingCallback" for cb in callbacks):
                self.early_stopping_ = RelativeEarlyStopping(**stop_cfg)
                callbacks.append(self.early_stopping_.as_callback())

        interval = self.mps_numerical_guard_every_n_steps or _guard_interval_from_env()
        if interval and not any(type(cb).__name__ == "_NumericalGuardCallback" for cb in callbacks):
            self.numerical_guard_ = NumericalGuard(every_n_steps=interval)
            callbacks.append(self.numerical_guard_.as_callback())
            logger.info(
                "Numerical guard active: the loss will be cross-checked against the CPU every %d steps. "
                "Inspect the outcome afterwards with model.numerical_guard_.summary().",
                interval,
            )
