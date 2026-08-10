"""Module-level glue that makes an existing Pyro module safe to move onto Metal."""

import logging
from typing import Any, Optional

import torch

from ._device import mps_is_available
from ._dtype import downcast_unsupported_

logger = logging.getLogger(__name__)

__all__ = ["AppleSiliconCompatMixin", "prepare_module_for_device", "compile_is_safe"]


def _extract_device(args, kwargs) -> Optional[torch.device]:
    """Pull the target device out of ``nn.Module.to`` style arguments."""
    device = kwargs.get("device")
    if device is None:
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
                break
            if isinstance(arg, str):
                try:
                    device = torch.device(arg)
                except (RuntimeError, ValueError):
                    continue
                break
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
    return torch.device(device) if device is not None else None


def prepare_module_for_device(module: torch.nn.Module, device: Any) -> torch.nn.Module:
    """Make ``module`` movable to ``device``, in place.

    For MPS this means downcasting float64 buffers/parameters *before* the move,
    since the cast is impossible once the tensor is on the device.
    """
    device = torch.device(device)
    if device.type == "mps":
        downcast_unsupported_(module, device)
    return module


class AppleSiliconCompatMixin:
    """Mix into a ``PyroBaseModuleClass`` to make ``.to("mps")`` work.

    cell2location builds most of its hyperparameter buffers directly from NumPy
    arrays, which default to float64. MPS has no float64 kernels, so the plain
    ``.to("mps")`` raises before any model code runs. Intercepting the move lets us
    normalise dtypes while the tensors are still on the CPU.
    """

    def to(self, *args, **kwargs):
        device = _extract_device(args, kwargs)
        if device is not None and device.type == "mps":
            prepare_module_for_device(self, device)
        return super().to(*args, **kwargs)


def compile_is_safe(device: Any = None) -> bool:
    """Whether ``torch.compile`` should be used on ``device``.

    Allowed on Metal by default: on torch >= 2.12, TorchInductor's Metal backend
    compiles the Pyro graphs cell2location produces and matches the fused kernel's
    arithmetic (verified against CPU by the numerical guard, which compiled Metal
    runs arm automatically). ``CELL2LOCATION_ALLOW_MPS_COMPILE=0`` is the kill
    switch that makes ``train_compiled`` degrade to eager rather than compile.
    """
    device = torch.device(device) if device is not None else None
    on_mps = (device is not None and device.type == "mps") or (device is None and mps_is_available())
    if not on_mps:
        return True

    import os

    override = os.environ.get("CELL2LOCATION_ALLOW_MPS_COMPILE", "1").lower()
    return override in ("1", "true", "yes")
