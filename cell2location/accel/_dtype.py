"""dtype normalisation for the Metal backend.

The MPS backend has no float64 kernels at all -- moving a ``torch.float64`` tensor to
an MPS device raises ``TypeError``. cell2location creates a large number of buffers
straight from user-supplied NumPy arrays (``N_cells_per_location``, ``cell_state``,
``init_vals``, ...). NumPy defaults to float64, so those buffers are float64 whenever
the caller did not explicitly downcast, and the model dies the moment it is moved to
the GPU on a Mac.

Rather than rewriting several dozen ``register_buffer`` calls, we normalise dtypes at
the boundary: just before anything is moved onto an MPS device.
"""

import logging
import warnings
from typing import Any, Iterable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "UNSUPPORTED_MPS_DTYPES",
    "default_float_dtype",
    "downcast_unsupported_",
    "sanitize_tensor",
    "sanitize_args",
    "prepare_anndata",
    "check_anndata_dtype",
]

#: dtypes with no MPS kernel coverage, mapped to their closest supported dtype.
UNSUPPORTED_MPS_DTYPES = {
    torch.float64: torch.float32,
    torch.complex128: torch.complex64,
}


def default_float_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    """Widest floating dtype usable on ``device``."""
    if device is not None and torch.device(device).type == "mps":
        return torch.float32
    return torch.get_default_dtype()


def _needs_downcast(dtype: torch.dtype, device: torch.device) -> bool:
    return device.type == "mps" and dtype in UNSUPPORTED_MPS_DTYPES


def downcast_unsupported_(module: torch.nn.Module, device: Any = "mps") -> torch.nn.Module:
    """Cast float64 parameters and buffers of ``module`` in place, recursively.

    Must be called while the module still lives on a device that supports float64
    (i.e. CPU), because the cast itself is impossible once on MPS.

    Returns the same module, for chaining.
    """
    device = torch.device(device)
    if device.type != "mps":
        return module

    converted = []

    for name, buf in list(module.named_buffers(recurse=True)):
        if buf is not None and _needs_downcast(buf.dtype, device):
            target = UNSUPPORTED_MPS_DTYPES[buf.dtype]
            _set_by_path(module, name, buf.to(target), is_buffer=True)
            converted.append(f"{name} ({buf.dtype} -> {target})")

    for name, param in list(module.named_parameters(recurse=True)):
        if param is not None and _needs_downcast(param.dtype, device):
            target = UNSUPPORTED_MPS_DTYPES[param.dtype]
            with torch.no_grad():
                param.data = param.data.to(target)
            converted.append(f"{name} ({param.dtype} -> {target})")

    if converted:
        logger.info(
            "Metal backend: downcast %d float64 tensors to float32 (MPS has no float64 kernels). " "First few: %s",
            len(converted),
            ", ".join(converted[:5]),
        )
    return module


def _set_by_path(root: torch.nn.Module, dotted: str, value: torch.Tensor, is_buffer: bool = True) -> None:
    parts = dotted.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    leaf = parts[-1]
    if is_buffer:
        # bypass ``register_buffer`` re-validation, the tensor is already registered
        obj._buffers[leaf] = value
    else:
        setattr(obj, leaf, value)


def sanitize_tensor(tensor: Any, device: Any) -> Any:
    """Move a tensor to ``device``, downcasting first if the dtype is unsupported."""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    device = torch.device(device)
    if _needs_downcast(tensor.dtype, device):
        tensor = tensor.to(UNSUPPORTED_MPS_DTYPES[tensor.dtype])
    return tensor.to(device)


def sanitize_args(args: Iterable, kwargs: dict, device: Any):
    """Apply :func:`sanitize_tensor` across a positional/keyword argument pair."""
    args = [sanitize_tensor(a, device) for a in args]
    kwargs = {k: sanitize_tensor(v, device) for k, v in kwargs.items()}
    return args, kwargs


def check_anndata_dtype(adata, layer: Optional[str] = None) -> bool:
    """Return True when the count matrix is already in an MPS-friendly dtype."""
    matrix = adata.X if layer is None else adata.layers[layer]
    dtype = getattr(matrix, "dtype", None)
    return dtype is not None and np.dtype(dtype) == np.float32


def prepare_anndata(adata, layer: Optional[str] = None, inplace: bool = True):
    """Downcast an AnnData count matrix to float32.

    float64 count matrices are the second most common way training fails on Metal:
    the model itself is clean but every minibatch arrives as float64 and the transfer
    to the GPU raises. Counts are integers well inside float32's exact range, so this
    is lossless for real data.
    """
    matrix = adata.X if layer is None else adata.layers[layer]
    dtype = np.dtype(getattr(matrix, "dtype", np.float32))

    if dtype == np.float32:
        return adata

    if not inplace:
        adata = adata.copy()

    if dtype == np.float64:
        max_exact = 2**24
        try:
            observed_max = matrix.max()
        except Exception:  # pragma: no cover - exotic backed matrices
            observed_max = 0
        if observed_max > max_exact:
            warnings.warn(
                f"Count matrix contains values above {max_exact}, which float32 cannot represent "
                "exactly. Values will be rounded. Consider training on CPU if this matters.",
                UserWarning,
                stacklevel=2,
            )

    converted = matrix.astype(np.float32)
    if layer is None:
        adata.X = converted
    else:
        adata.layers[layer] = converted

    logger.info("Metal backend: converted count matrix from %s to float32.", dtype)
    return adata
