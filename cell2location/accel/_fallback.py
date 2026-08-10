"""CPU fallbacks for operations the Metal backend does not implement.

PyTorch offers a blanket ``PYTORCH_ENABLE_MPS_FALLBACK=1`` switch, but it has two
drawbacks for a library: it must be set before ``import torch`` (so a library cannot
turn it on for its users), and it silently routes *every* missing op through the CPU,
which turns a one-line gap into an invisible performance cliff.

Instead we fall back explicitly at the few call sites that need it -- currently the
random-number generators behind ``NegativeBinomial.sample`` -- and probe support at
runtime so a future PyTorch release that adds the kernel is picked up automatically.
"""

import functools
import logging
import os
from typing import Callable, Dict

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "MPS_FALLBACK_ENV_VAR",
    "run_on_cpu",
    "cpu_fallback",
    "supports_op",
    "clear_support_cache",
    "enable_global_torch_fallback",
]

MPS_FALLBACK_ENV_VAR = "PYTORCH_ENABLE_MPS_FALLBACK"

_SUPPORT_CACHE: Dict[str, bool] = {}


def enable_global_torch_fallback() -> bool:
    """Set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` if torch has not been imported yet.

    Returns True if the flag is now in effect. Only useful from a launcher script;
    once ``torch`` is loaded the variable is no longer read.
    """
    if os.environ.get(MPS_FALLBACK_ENV_VAR) == "1":
        return True
    import sys

    if "torch" in sys.modules:
        logger.warning(
            "%s must be set before torch is imported; cell2location uses targeted "
            "per-op fallbacks instead, so this is usually unnecessary.",
            MPS_FALLBACK_ENV_VAR,
        )
        return False
    os.environ[MPS_FALLBACK_ENV_VAR] = "1"
    return True


def run_on_cpu(fn: Callable, *args, _target_device=None, **kwargs):
    """Execute ``fn`` with all tensor arguments moved to CPU, restoring the device.

    ``_target_device`` overrides where results are sent; by default the device of the
    first tensor argument is used.
    """
    device = _target_device

    def _to_cpu(x):
        nonlocal device
        if isinstance(x, torch.Tensor):
            if device is None:
                device = x.device
            return x.cpu()
        return x

    cpu_args = [_to_cpu(a) for a in args]
    cpu_kwargs = {k: _to_cpu(v) for k, v in kwargs.items()}

    result = fn(*cpu_args, **cpu_kwargs)

    if device is None or torch.device(device).type == "cpu":
        return result
    return _move_back(result, device)


def _move_back(result, device):
    if isinstance(result, torch.Tensor):
        return result.to(device)
    if isinstance(result, dict):
        return {k: _move_back(v, device) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        moved = [_move_back(v, device) for v in result]
        return type(result)(moved) if not isinstance(result, tuple) else tuple(moved)
    return result


def cpu_fallback(op_name: str):
    """Decorator: run the wrapped function on CPU when its inputs live on MPS.

    ``op_name`` is only used for logging and for the support probe cache.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            on_mps = any(isinstance(a, torch.Tensor) and a.device.type == "mps" for a in args) or any(
                isinstance(v, torch.Tensor) and v.device.type == "mps" for v in kwargs.values()
            )
            if not on_mps:
                return fn(*args, **kwargs)
            logger.debug("Metal backend: running %s on CPU (no MPS kernel).", op_name)
            return run_on_cpu(fn, *args, **kwargs)

        return wrapper

    return decorator


_PROBES: Dict[str, Callable[[torch.device], torch.Tensor]] = {
    "poisson": lambda d: torch.poisson(torch.ones(4, device=d)),
    "standard_gamma": lambda d: torch._standard_gamma(torch.ones(4, device=d)),
    "lgamma": lambda d: torch.lgamma(torch.ones(4, device=d)),
    "digamma": lambda d: torch.digamma(torch.ones(4, device=d)),
    "erfinv": lambda d: torch.erfinv(torch.zeros(4, device=d)),
    "cumsum": lambda d: torch.cumsum(torch.ones(4, device=d), dim=0),
    "logsumexp": lambda d: torch.logsumexp(torch.ones(4, device=d), dim=0),
    "index_add": lambda d: torch.zeros(4, device=d).index_add(
        0, torch.zeros(4, dtype=torch.long, device=d), torch.ones(4, device=d)
    ),
    "sort": lambda d: torch.sort(torch.rand(8, device=d)).values,
    "multinomial": lambda d: torch.multinomial(torch.ones(4, device=d), 2, replacement=True),
    "linalg_cholesky": lambda d: torch.linalg.cholesky(torch.eye(3, device=d)),
}


def supports_op(op_name: str, device: str = "mps", use_cache: bool = True) -> bool:
    """Probe whether ``op_name`` actually runs on ``device``.

    Unknown op names return ``False``; add entries to ``_PROBES`` to extend coverage.
    """
    key = f"{device}:{op_name}"
    if use_cache and key in _SUPPORT_CACHE:
        return _SUPPORT_CACHE[key]

    probe = _PROBES.get(op_name)
    if probe is None:
        _SUPPORT_CACHE[key] = False
        return False

    try:
        result = probe(torch.device(device))
        if isinstance(result, torch.Tensor):
            result.cpu()  # force synchronisation so lazy failures surface here
        supported = True
    except Exception as exc:  # noqa: BLE001 - any failure means "unsupported"
        logger.debug("Metal backend: op %s unavailable on %s (%s).", op_name, device, exc)
        supported = False

    _SUPPORT_CACHE[key] = supported
    return supported


def clear_support_cache() -> None:
    _SUPPORT_CACHE.clear()
