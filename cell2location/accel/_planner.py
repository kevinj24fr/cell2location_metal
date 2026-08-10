"""Memory planning for training on unified memory.

Apple silicon shares one physical pool between CPU and GPU. Most of the time that is
described as a convenience -- no host-to-device copies -- but the more interesting
consequence is capacity. A discrete GPU gives you 24 GB and no way to exceed it; a
Mac Studio can be configured with 128 or 512 GB, all of it addressable by the GPU.

For cell2location that changes what is possible rather than just what is fast.
Minibatching a spatial model is not free: ``N_cells_per_location`` and the detection
efficiency priors couple locations, and the amortised guide has to approximate what
full-batch inference computes exactly. Full-batch training is the better estimator,
and on a large-memory Mac it fits at scales where a workstation GPU simply cannot run
it.

This module answers the concrete question -- will this dataset train full-batch on
this machine, and if not what batch size should I use -- with an estimate rather than
a shrug.

Estimates are deliberately conservative. Being told 8192 and getting an allocation
failure at step 3000 is much worse than being told 4096.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["MemoryPlan", "plan_memory", "total_unified_memory", "recommended_max_memory"]

_BYTES_PER_FLOAT32 = 4

#: Peak live intermediates per element of the (n_obs x n_genes) likelihood, in units
#: of full-size float32 tensors. The eager negative-binomial expression allocates
#: roughly a dozen; autograd retains a subset for the backward pass. Measured against
#: the expression in ``_ops.log_nb_positive``, rounded up.
_LIKELIHOOD_INTERMEDIATES = 14

#: With the fused Metal kernel, only the inputs and the saved tensors survive.
_LIKELIHOOD_INTERMEDIATES_FUSED = 5

#: Optimiser state: ClippedAdam keeps two moments per parameter, plus the gradient.
_OPTIMISER_MULTIPLIER = 3

#: Headroom left for the OS, the allocator's own fragmentation, and everything else
#: the user has open. Unified memory means overshooting does not just fail -- it
#: swaps, and takes the whole machine down with it.
_SAFETY_FRACTION = 0.7


@dataclass
class MemoryPlan:
    """The outcome of :func:`plan_memory`."""

    n_obs: int
    n_genes: int
    n_factors: int
    available_bytes: int
    full_batch_bytes: int
    fits_full_batch: bool
    recommended_batch_size: Optional[int]
    fused_kernel_assumed: bool
    notes: list = field(default_factory=list)

    @property
    def full_batch_gb(self) -> float:
        return self.full_batch_bytes / 1e9

    @property
    def available_gb(self) -> float:
        return self.available_bytes / 1e9

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "n_genes": self.n_genes,
            "n_factors": self.n_factors,
            "available_gb": round(self.available_gb, 2),
            "full_batch_gb": round(self.full_batch_gb, 2),
            "fits_full_batch": self.fits_full_batch,
            "recommended_batch_size": self.recommended_batch_size,
            "fused_kernel_assumed": self.fused_kernel_assumed,
            "notes": self.notes,
        }

    def __str__(self) -> str:
        lines = [
            f"Dataset      : {self.n_obs:,} locations x {self.n_genes:,} genes x {self.n_factors} cell types",
            f"Available    : {self.available_gb:.1f} GB usable of unified memory",
            f"Full batch   : {self.full_batch_gb:.1f} GB estimated peak",
        ]
        if self.fits_full_batch:
            lines.append("Verdict      : fits -- train with batch_size=None for exact full-batch inference")
        else:
            lines.append(f"Verdict      : does not fit -- use batch_size={self.recommended_batch_size:,}")
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def total_unified_memory() -> Optional[int]:
    """Total physical memory in bytes, or None if it cannot be determined."""
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
        return None


def recommended_max_memory() -> Optional[int]:
    """PyTorch's own view of the allocation ceiling on this device, if exposed."""
    import torch

    fn = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
    if fn is None:
        return None
    try:
        return int(fn())
    except Exception:  # pragma: no cover
        return None


def _available_bytes(memory_budget_gb: Optional[float]) -> tuple:
    notes = []

    if memory_budget_gb is not None:
        return int(memory_budget_gb * 1e9), notes

    ceiling = recommended_max_memory()
    if ceiling:
        return int(ceiling * _SAFETY_FRACTION), notes

    total = total_unified_memory()
    if total:
        notes.append(
            "torch.mps.recommended_max_memory() unavailable; estimating from total system memory "
            f"({total / 1e9:.0f} GB) at {_SAFETY_FRACTION:.0%}"
        )
        return int(total * _SAFETY_FRACTION), notes

    notes.append("could not determine system memory; assuming 16 GB, pass memory_budget_gb to override")
    return int(16e9 * _SAFETY_FRACTION), notes


def plan_memory(
    adata=None,
    n_obs: Optional[int] = None,
    n_genes: Optional[int] = None,
    n_factors: int = 50,
    memory_budget_gb: Optional[float] = None,
    fused_kernel: bool = False,
) -> MemoryPlan:
    """Estimate whether cell2location will train full-batch on this machine.

    Parameters
    ----------
    adata
        Spatial AnnData. ``n_obs``/``n_genes`` are read from it when given.
    n_factors
        Number of reference cell types (columns of ``cell_state_df``).
    memory_budget_gb
        Override the automatic budget. Useful for planning for a machine you are not
        currently sitting at.
    fused_kernel
        Assume the fused Metal likelihood kernel is active, which removes most of the
        full-size intermediates. See :mod:`cell2location.accel._fused_nb`.

    Examples
    --------
    >>> from cell2location.accel import plan_memory
    >>> print(plan_memory(n_obs=200_000, n_genes=18_000, memory_budget_gb=340))
    """
    if adata is not None:
        n_obs = n_obs or int(adata.n_obs)
        n_genes = n_genes or int(adata.n_vars)
    if n_obs is None or n_genes is None:
        raise ValueError("Provide either adata or both n_obs and n_genes.")

    available, notes = _available_bytes(memory_budget_gb)

    per_element = _LIKELIHOOD_INTERMEDIATES_FUSED if fused_kernel else _LIKELIHOOD_INTERMEDIATES
    bytes_per_obs = n_genes * _BYTES_PER_FLOAT32 * per_element

    # Parameters that exist regardless of batch size: the location-by-factor abundance
    # matrix dominates, and it is full-size even when the likelihood is minibatched.
    parameter_bytes = (n_obs * n_factors + n_genes * n_factors + n_obs + n_genes) * _BYTES_PER_FLOAT32
    parameter_bytes *= _OPTIMISER_MULTIPLIER

    # The count matrix itself, resident. On unified memory this is shared rather than
    # duplicated across host and device -- which is exactly the saving being exploited.
    data_bytes = n_obs * n_genes * _BYTES_PER_FLOAT32

    full_batch_bytes = parameter_bytes + data_bytes + n_obs * bytes_per_obs
    fits = full_batch_bytes <= available

    if fits:
        recommended = None
        notes.append(
            "full-batch avoids the approximation minibatching introduces for "
            "location-coupled priors -- prefer it when it fits"
        )
    else:
        spare = available - parameter_bytes - data_bytes
        if spare <= 0:
            recommended = 256
            notes.append(
                "parameters and data alone exceed the budget; the suggested batch size is a floor, "
                "not an estimate. Consider gene filtering, or a machine with more memory."
            )
        else:
            raw = spare / bytes_per_obs
            recommended = max(256, 1 << int(math.floor(math.log2(max(raw, 256)))))
            notes.append(f"largest batch that fits is ~{int(raw):,}; rounded down to a power of two")

    if not fused_kernel and not fits:
        fused_full = (
            parameter_bytes + data_bytes + n_obs * n_genes * _BYTES_PER_FLOAT32 * (_LIKELIHOOD_INTERMEDIATES_FUSED)
        )
        if fused_full <= available:
            notes.append(
                f"the fused Metal kernel would bring the full-batch estimate to "
                f"{fused_full / 1e9:.1f} GB, which does fit -- see CELL2LOCATION_MPS_FUSED_NB"
            )

    return MemoryPlan(
        n_obs=n_obs,
        n_genes=n_genes,
        n_factors=n_factors,
        available_bytes=available,
        full_batch_bytes=full_batch_bytes,
        fits_full_batch=fits,
        recommended_batch_size=recommended,
        fused_kernel_assumed=fused_kernel,
        notes=notes,
    )
