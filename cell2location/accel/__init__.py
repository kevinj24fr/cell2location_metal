"""Apple silicon acceleration for cell2location.

Two distinct pieces of Apple hardware, with very different roles:

* **Metal / MPS (the GPU).** This is where training happens. The work in this
  subpackage is mostly about making the existing Pyro model *correct* and *movable*
  on Metal -- float64 buffers, a broken broadcast ``lgamma`` kernel, and missing
  random-number generators are each enough to stop it dead.
* **The Neural Engine (ANE).** Inference-only, float16, CoreML-only. It cannot run
  variational inference. The one component it can run is the amortised guide's
  encoder network, exported via :mod:`cell2location.accel.coreml`.

Quick start on a Mac::

    import cell2location
    from cell2location.accel import configure, report

    configure()          # sets allocator watermarks, checks the backend
    print(report())      # what is available, what will fall back

    model.train(max_epochs=30000, accelerator="mps")

Everything degrades gracefully: on Linux/CUDA every helper here is a no-op or
delegates to the original scvi-tools code path.
"""

import logging
from typing import Any, Dict, Optional

from ._compat import AppleSiliconCompatMixin, compile_is_safe, prepare_module_for_device
from ._device import (
    DISABLE_MPS_ENV_VAR,
    device_of,
    is_apple_silicon,
    mps_is_available,
    mps_is_disabled,
    parse_device_args_safe,
    resolve_accelerator,
)
from ._dtype import (
    UNSUPPORTED_MPS_DTYPES,
    check_anndata_dtype,
    default_float_dtype,
    downcast_unsupported_,
    prepare_anndata,
    sanitize_args,
    sanitize_tensor,
)
from ._fallback import (
    clear_support_cache,
    cpu_fallback,
    enable_global_torch_fallback,
    run_on_cpu,
    supports_op,
)
from ._fused_nb import (
    FUSED_NB_ENV_VAR,
    disable_fused_nb,
    enable_fused_nb,
    fused_nb_enabled,
    fused_nb_status,
    verify_fused_kernel,
)
from ._guard import NumericalGuard, compare_loss_across_devices
from ._memory import (
    MPSCacheCallback,
    configure_memory,
    current_allocated_memory,
    empty_cache,
    memory_summary,
)
from ._ops import (
    LGAMMA_MODE_ENV_VAR,
    LGAMMA_MODES,
    current_lgamma_mode,
    digamma,
    lgamma,
    lgamma_stirling,
    log_nb_positive,
)
from ._planner import (
    MemoryPlan,
    plan_memory,
    recommended_max_memory,
    total_unified_memory,
)
from ._train import GUARD_ENV_VAR, AppleSiliconTrainMixin

logger = logging.getLogger(__name__)

__all__ = [
    "AppleSiliconCompatMixin",
    "FUSED_NB_ENV_VAR",
    "GUARD_ENV_VAR",
    "MemoryPlan",
    "NumericalGuard",
    "compare_loss_across_devices",
    "disable_fused_nb",
    "enable_fused_nb",
    "fused_nb_enabled",
    "fused_nb_status",
    "plan_memory",
    "recommended_max_memory",
    "total_unified_memory",
    "verify_fused_kernel",
    "AppleSiliconTrainMixin",
    "DISABLE_MPS_ENV_VAR",
    "LGAMMA_MODES",
    "LGAMMA_MODE_ENV_VAR",
    "MPSCacheCallback",
    "UNSUPPORTED_MPS_DTYPES",
    "check_anndata_dtype",
    "clear_support_cache",
    "compile_is_safe",
    "configure",
    "configure_memory",
    "cpu_fallback",
    "current_allocated_memory",
    "current_lgamma_mode",
    "default_float_dtype",
    "device_of",
    "digamma",
    "downcast_unsupported_",
    "empty_cache",
    "enable_global_torch_fallback",
    "is_apple_silicon",
    "lgamma",
    "lgamma_stirling",
    "log_nb_positive",
    "memory_summary",
    "mps_is_available",
    "mps_is_disabled",
    "parse_device_args_safe",
    "prepare_anndata",
    "prepare_module_for_device",
    "report",
    "resolve_accelerator",
    "run_on_cpu",
    "sanitize_args",
    "sanitize_tensor",
    "supports_op",
]

#: Ops cell2location depends on, probed by :func:`report`.
PROBED_OPS = (
    "lgamma",
    "digamma",
    "poisson",
    "standard_gamma",
    "erfinv",
    "cumsum",
    "logsumexp",
    "index_add",
    "sort",
    "multinomial",
    "linalg_cholesky",
)


def configure(
    high_watermark_ratio: Optional[float] = None,
    low_watermark_ratio: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Prepare the process for Metal execution and return a status dictionary.

    Call this before building the model. Setting allocator watermarks afterwards has
    no effect, because PyTorch reads them at the first MPS allocation.
    """
    configure_memory(high_watermark_ratio=high_watermark_ratio, low_watermark_ratio=low_watermark_ratio)
    status = report()

    if verbose:
        if not status["mps_available"]:
            reason = "explicitly disabled" if mps_is_disabled() else "unavailable"
            logger.info("Metal backend %s; cell2location will use %s.", reason, status["default_accelerator"])
        else:
            missing = [name for name, ok in status["op_support"].items() if not ok]
            logger.info(
                "Metal backend ready (%s). %s",
                status["memory"],
                f"CPU fallback for: {', '.join(missing)}." if missing else "All probed ops native.",
            )

    return status


def report() -> Dict[str, Any]:
    """Snapshot of what this machine can and cannot do, for logs and bug reports."""
    import platform

    import torch

    available = mps_is_available()
    accelerator, device = resolve_accelerator("auto", "auto")

    return {
        "platform": f"{platform.system()} {platform.machine()} (macOS {platform.mac_ver()[0] or 'n/a'})",
        "torch_version": torch.__version__,
        "apple_silicon": is_apple_silicon(),
        "mps_built": bool(getattr(getattr(torch.backends, "mps", None), "is_built", lambda: False)()),
        "mps_available": available,
        "mps_disabled_by_env": mps_is_disabled(),
        "default_accelerator": accelerator,
        "default_device": str(device),
        "lgamma_mode": current_lgamma_mode(),
        "coreml_available": _coreml_available(),
        "fused_nb": fused_nb_status(),
        "unified_memory_gb": (round(total_unified_memory() / 1e9, 1) if total_unified_memory() else None),
        "op_support": {name: supports_op(name) for name in PROBED_OPS} if available else {},
        "memory": memory_summary() if available else "n/a",
    }


def _coreml_available() -> bool:
    try:
        from .coreml import coremltools_available

        return coremltools_available()
    except Exception:  # pragma: no cover
        return False
