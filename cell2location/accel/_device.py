"""Device resolution for Apple silicon (Metal / MPS) alongside CUDA and CPU.

The upstream code path relies on :func:`scvi.model._utils.parse_device_args`, which is
written around CUDA semantics (integer device indices, ``validate_single_device``).
The MPS backend exposes exactly one logical device and has no index, so passing an
integer through that path either raises or silently resolves to CPU.

This module provides a drop-in replacement that understands ``"mps"`` and keeps the
existing CUDA/CPU behaviour untouched.
"""

import logging
import os
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "DISABLE_MPS_ENV_VAR",
    "is_apple_silicon",
    "mps_is_available",
    "mps_is_disabled",
    "resolve_accelerator",
    "parse_device_args_safe",
    "device_of",
]

#: Set this environment variable to ``1`` to make ``accelerator="auto"`` ignore MPS.
DISABLE_MPS_ENV_VAR = "CELL2LOCATION_DISABLE_MPS"


def is_apple_silicon() -> bool:
    """True when running on an arm64 macOS host."""
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mps_is_disabled() -> bool:
    """True when the user has explicitly opted out of the Metal backend."""
    return os.environ.get(DISABLE_MPS_ENV_VAR, "0").lower() in ("1", "true", "yes")


def mps_is_available() -> bool:
    """True when PyTorch was built with MPS and a Metal device is usable."""
    if mps_is_disabled():
        return False
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_built() and backend.is_available())
    except Exception:  # pragma: no cover - defensive, older torch builds
        return False


def _auto_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if mps_is_available():
        return "mps"
    return "cpu"


def resolve_accelerator(
    accelerator: Union[str, None] = "auto",
    device: Union[int, str, None] = "auto",
) -> Tuple[str, torch.device]:
    """Resolve ``(accelerator, torch.device)`` with Metal support.

    Parameters
    ----------
    accelerator
        One of ``"auto"``, ``"cpu"``, ``"gpu"``, ``"cuda"``, ``"mps"``.
        ``"gpu"`` resolves to CUDA when present and to MPS on Apple silicon,
        which matches what a user on a Mac means by "use the GPU".
    device
        Device index for CUDA. Ignored for MPS, which has no index.

    Returns
    -------
    Tuple of the Lightning accelerator string and the concrete ``torch.device``.
    """
    accelerator = (accelerator or "auto").lower()

    if accelerator == "auto":
        accelerator = _auto_accelerator()
    elif accelerator == "gpu":
        if torch.cuda.is_available():
            accelerator = "cuda"
        elif mps_is_available():
            accelerator = "mps"
        else:
            raise RuntimeError("accelerator='gpu' requested but neither CUDA nor MPS is available.")

    if accelerator == "mps":
        if not mps_is_available():
            raise RuntimeError(
                "accelerator='mps' requested but the Metal backend is unavailable. "
                "Requires macOS 12.3+, an Apple silicon Mac and a PyTorch build with MPS enabled."
            )
        # MPS has a single logical device and no index.
        return "mps", torch.device("mps")

    if accelerator in ("cuda", "gpu"):
        index = 0 if device in ("auto", None) else int(device)
        return "gpu", torch.device(f"cuda:{index}")

    return "cpu", torch.device("cpu")


def parse_device_args_safe(
    accelerator: str = "auto",
    devices: Union[int, str] = "auto",
    return_device: Optional[str] = "torch",
    validate_single_device: bool = True,
):
    """Drop-in replacement for :func:`scvi.model._utils.parse_device_args`.

    Delegates to scvi-tools for CUDA/CPU so behaviour is unchanged there, and short
    circuits for MPS which scvi-tools does not model.
    """
    resolved_accelerator, torch_device = resolve_accelerator(accelerator, devices)

    if torch_device.type == "mps":
        if return_device == "torch":
            return "mps", "auto", torch_device
        return "mps", "auto", "mps"

    from scvi.model._utils import parse_device_args as _scvi_parse

    return _scvi_parse(
        accelerator=resolved_accelerator,
        devices=devices,
        return_device=return_device,
        validate_single_device=validate_single_device,
    )


def device_of(obj) -> torch.device:
    """Best-effort device lookup for a module or tensor."""
    if isinstance(obj, torch.Tensor):
        return obj.device
    for param in getattr(obj, "parameters", lambda: iter(()))():
        return param.device
    for buffer in getattr(obj, "buffers", lambda: iter(()))():
        return buffer.device
    return torch.device("cpu")
