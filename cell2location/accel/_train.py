"""Training-time glue shared by the user-facing model classes."""

import logging
import os

from ._device import resolve_accelerator
from ._dtype import check_anndata_dtype
from ._guard import NumericalGuard
from ._memory import MPSCacheCallback

logger = logging.getLogger(__name__)

__all__ = ["AppleSiliconTrainMixin", "GUARD_ENV_VAR"]

#: Set to a step interval (or ``1`` for the default interval) to enable the runtime
#: divergence guard without changing any code.
GUARD_ENV_VAR = "CELL2LOCATION_MPS_GUARD"

_DEFAULT_GUARD_INTERVAL = 1000


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

    def _prepare_apple_silicon(self, kwargs: dict) -> None:
        _, device = resolve_accelerator(kwargs.get("accelerator", "auto"), kwargs.get("device", "auto"))
        if device.type != "mps":
            return

        adata = getattr(self, "adata_manager", None)
        adata = getattr(adata, "adata", None)
        if adata is not None and not check_anndata_dtype(adata):
            logger.warning(
                "The count matrix is not float32. The Metal backend has no float64 kernels, so "
                "moving minibatches to the GPU will raise. Fix it once, before setup_anndata(), with "
                "`cell2location.accel.prepare_anndata(adata)`."
            )

        callbacks = kwargs.setdefault("callbacks", [])

        if self.mps_empty_cache_every_n_steps:
            if not any(type(cb).__name__ == "_MPSCacheCallback" for cb in callbacks):
                callbacks.append(MPSCacheCallback(every_n_steps=self.mps_empty_cache_every_n_steps).as_callback())

        interval = self.mps_numerical_guard_every_n_steps or _guard_interval_from_env()
        if interval and not any(type(cb).__name__ == "_NumericalGuardCallback" for cb in callbacks):
            self.numerical_guard_ = NumericalGuard(every_n_steps=interval)
            callbacks.append(self.numerical_guard_.as_callback())
            logger.info(
                "Numerical guard active: the loss will be cross-checked against the CPU every %d steps. "
                "Inspect the outcome afterwards with model.numerical_guard_.summary().",
                interval,
            )
