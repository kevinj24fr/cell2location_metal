"""Unified-memory management for the Metal backend.

Apple silicon shares one physical memory pool between CPU and GPU. That removes
host/device copies -- a real win for cell2location, whose training loop otherwise
ships every minibatch across PCIe -- but it also means a runaway allocator competes
with the rest of the machine instead of hitting a separate VRAM ceiling.

PyTorch exposes two watermark ratios for this. Both are read once, at the first MPS
allocation, so they must be set before the model touches the GPU.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "HIGH_WATERMARK_ENV_VAR",
    "LOW_WATERMARK_ENV_VAR",
    "configure_memory",
    "empty_cache",
    "current_allocated_memory",
    "memory_summary",
    "MPSCacheCallback",
]

HIGH_WATERMARK_ENV_VAR = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
LOW_WATERMARK_ENV_VAR = "PYTORCH_MPS_LOW_WATERMARK_RATIO"


def configure_memory(
    high_watermark_ratio: Optional[float] = None,
    low_watermark_ratio: Optional[float] = None,
) -> None:
    """Set MPS allocator watermarks as a fraction of total system memory.

    Parameters
    ----------
    high_watermark_ratio
        Hard cap. Allocations beyond this fail. ``0.0`` disables the limit entirely,
        which is what you want for a large spatial dataset on a 128 GB Mac Studio but
        risks swapping the whole machine if the model does not fit.
    low_watermark_ratio
        Soft cap at which the allocator starts returning cached blocks to the system.

    Has no effect if called after the first MPS allocation.
    """
    if high_watermark_ratio is not None:
        os.environ[HIGH_WATERMARK_ENV_VAR] = str(high_watermark_ratio)
    if low_watermark_ratio is not None:
        os.environ[LOW_WATERMARK_ENV_VAR] = str(low_watermark_ratio)

    if current_allocated_memory() > 0:
        logger.warning(
            "MPS watermark ratios were set after the allocator was initialised; "
            "they will not take effect until a fresh process."
        )


def empty_cache() -> None:
    """Release cached MPS blocks back to the system."""
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception as exc:  # pragma: no cover
            logger.debug("torch.mps.empty_cache() failed: %s", exc)


def current_allocated_memory() -> int:
    """Bytes currently allocated on the MPS device (0 when unavailable)."""
    mps = getattr(torch, "mps", None)
    if mps is None or not hasattr(mps, "current_allocated_memory"):
        return 0
    try:
        return int(mps.current_allocated_memory())
    except Exception:  # pragma: no cover
        return 0


def memory_summary() -> str:
    """Human-readable snapshot of MPS allocator state."""
    mps = getattr(torch, "mps", None)
    if mps is None:
        return "MPS unavailable"
    parts = [f"allocated={current_allocated_memory() / 1e9:.2f} GB"]
    if hasattr(mps, "driver_allocated_memory"):
        try:
            parts.append(f"driver={mps.driver_allocated_memory() / 1e9:.2f} GB")
        except Exception:  # pragma: no cover
            pass
    parts.append(f"high_watermark={os.environ.get(HIGH_WATERMARK_ENV_VAR, 'default')}")
    return ", ".join(parts)


class MPSCacheCallback:
    """Lightning callback that trims the MPS cache periodically.

    Long cell2location runs (tens of thousands of SVI steps) accumulate cached blocks
    that unified memory makes everyone else's problem. Dropping the cache every
    ``every_n_steps`` keeps the rest of the machine responsive at negligible cost.

    Implemented lazily as a plain class so importing this module does not pull in
    lightning; :meth:`as_callback` returns the real callback instance.
    """

    def __init__(self, every_n_steps: int = 500):
        self.every_n_steps = every_n_steps

    def as_callback(self):
        from lightning.pytorch.callbacks import Callback

        every_n_steps = self.every_n_steps

        class _MPSCacheCallback(Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if every_n_steps > 0 and trainer.global_step > 0 and trainer.global_step % every_n_steps == 0:
                    empty_cache()

        return _MPSCacheCallback()
