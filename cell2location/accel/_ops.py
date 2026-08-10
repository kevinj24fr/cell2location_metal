"""Numerically-guarded elementwise ops for the Metal backend.

Two separate problems are handled here.

1. **Correctness.** ``torch.lgamma`` on MPS has returned wrong values for
   non-contiguous / broadcast inputs (pytorch/pytorch#132605). cell2location calls
   ``lgamma`` on exactly such tensors inside the negative-binomial log-likelihood --
   ``theta`` is routinely a ``(1, n_genes)`` view broadcast against a
   ``(n_obs, n_genes)`` batch. A silently wrong ELBO is far worse than a crash, so
   inputs are materialised before the kernel runs.

2. **Escape hatches.** If a given macOS / PyTorch combination still disagrees with
   CPU, ``lgamma`` can be switched to a pure-composition Stirling implementation
   (built only from ops with solid MPS coverage) or forced onto the CPU, without
   touching model code. Controlled by ``CELL2LOCATION_MPS_LGAMMA``.
"""

import math
import os
from typing import Optional

import torch

__all__ = [
    "LGAMMA_MODE_ENV_VAR",
    "LGAMMA_MODES",
    "lgamma",
    "digamma",
    "lgamma_stirling",
    "log_nb_positive",
    "eager_log_nb_positive",
    "current_lgamma_mode",
]

LGAMMA_MODE_ENV_VAR = "CELL2LOCATION_MPS_LGAMMA"

#: ``auto``       - native kernel off-MPS, contiguous-guarded native kernel on MPS
#: ``native``     - always call ``torch.lgamma`` unchanged
#: ``contiguous`` - materialise broadcast views, then call ``torch.lgamma``
#: ``stirling``   - shifted Stirling series composed from basic ops (no lgamma kernel)
#: ``cpu``        - evaluate on CPU and copy the result back
LGAMMA_MODES = ("auto", "native", "contiguous", "stirling", "cpu")

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# Number of recurrence steps applied before the asymptotic expansion. The series
# converges rapidly for z >= 8, and shifting by a constant keeps the op count fixed
# (no data-dependent control flow, which matters for graph capture).
_STIRLING_SHIFT = 8


def current_lgamma_mode() -> str:
    mode = os.environ.get(LGAMMA_MODE_ENV_VAR, "auto").lower()
    if mode not in LGAMMA_MODES:
        raise ValueError(f"{LGAMMA_MODE_ENV_VAR} must be one of {LGAMMA_MODES}, got {mode!r}")
    return mode


def lgamma_stirling(x: torch.Tensor) -> torch.Tensor:
    r"""``log |Gamma(x)|`` for ``x > 0``, built only from log/reciprocal/arithmetic.

    Uses the recurrence :math:`\log\Gamma(x) = \log\Gamma(x + n) - \sum_{k<n}\log(x+k)`
    to push the argument into the regime where the asymptotic expansion

    .. math::
        \log\Gamma(z) \approx (z - \tfrac12)\log z - z + \tfrac12\log 2\pi
                              + \frac{1}{12z} - \frac{1}{360z^3}
                              + \frac{1}{1260z^5} - \frac{1}{1680z^7}

    is accurate to far better than float32 resolution (truncation error below 1e-11
    at ``z >= 8``). Fully differentiable, so autograd recovers ``digamma`` for free.

    **Accuracy is absolute, not relative.** The recurrence subtracts two quantities of
    similar magnitude, so in float32 the result carries roughly 1e-6 of absolute error.
    Around the roots of ``lgamma`` (``x = 1`` and ``x = 2``, where the true value is
    zero) that is an unbounded *relative* error. This is harmless here -- the value
    feeds a summed log-likelihood, where absolute error is what propagates -- but it
    means "agrees to N significant figures" is the wrong way to test this function.
    In float64 the same code is accurate to ~1e-9 relative across the useful range.
    """
    shift_terms = torch.zeros((), dtype=x.dtype, device=x.device)
    for k in range(_STIRLING_SHIFT):
        shift_terms = shift_terms + torch.log(x + k)

    z = x + _STIRLING_SHIFT
    inv = torch.reciprocal(z)
    inv2 = inv * inv

    series = inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 * (1.0 / 1260.0 + inv2 * (-1.0 / 1680.0))))

    log_gamma_shifted = (z - 0.5) * torch.log(z) - z + _LOG_SQRT_2PI + series
    return log_gamma_shifted - shift_terms


def _materialise(x: torch.Tensor) -> torch.Tensor:
    """Force a dense, contiguous layout so broadcast views cannot trip MPS kernels."""
    if x.is_contiguous():
        return x
    return x.contiguous()


def lgamma(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
    """MPS-guarded ``torch.lgamma``."""
    mode = mode or current_lgamma_mode()
    on_mps = x.device.type == "mps"

    if mode == "auto":
        mode = "contiguous" if on_mps else "native"

    if mode == "native":
        return torch.lgamma(x)
    if mode == "contiguous":
        return torch.lgamma(_materialise(x))
    if mode == "stirling":
        return lgamma_stirling(x)
    if mode == "cpu":
        return torch.lgamma(x.cpu()).to(x.device)
    raise ValueError(f"Unknown lgamma mode {mode!r}")


def digamma(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
    """MPS-guarded ``torch.digamma``, same dispatch rules as :func:`lgamma`."""
    mode = mode or current_lgamma_mode()
    on_mps = x.device.type == "mps"

    if mode == "auto":
        mode = "contiguous" if on_mps else "native"

    if mode == "cpu":
        return torch.digamma(x.cpu()).to(x.device)
    if mode == "stirling":
        # derivative of the Stirling composition; cheaper to let autograd do it
        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(lgamma_stirling(x_).sum(), x_)
        return grad
    if mode == "contiguous":
        return torch.digamma(_materialise(x))
    return torch.digamma(x)


def log_nb_positive(
    value: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative-binomial log-likelihood in the ``(mu, theta)`` parameterisation.

    Numerically identical to the upstream implementation; the difference is that no
    ``lgamma`` argument is ever a stride-0 broadcast view, which is what makes the
    result trustworthy on Metal.

    Note the deliberate asymmetry: ``lgamma(theta)`` is evaluated on the small
    unexpanded tensor and broadcast afterwards. Expanding it first would be correct
    too, but would allocate a full ``(n_obs, n_genes)`` intermediate per call for no
    reason -- and on unified memory that bandwidth is the thing you are trying to save.

    When the fused Metal kernel is enabled *and* has verified itself against this
    function on the current machine, it is used instead. It computes the same thing in
    a single pass; see :mod:`cell2location.accel._fused_nb`.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    if value.device.type == "mps":
        # Imported lazily to keep this module importable without _fused_nb and back.
        from ._fused_nb import fused_log_nb_positive

        fused = fused_log_nb_positive(value, mu, theta, eps)
        if fused is not None:
            return fused

    return eager_log_nb_positive(value, mu, theta, eps)


def eager_log_nb_positive(
    value: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """The NB log-likelihood with no kernel dispatch, ever.

    This is the reference the fused Metal kernel verifies itself against, so it must
    not be able to reach the fused path -- verification calling back into the kernel
    under verification recurses forever.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)

    # ``value + theta`` materialises a fresh contiguous tensor, so it is already safe.
    # ``theta`` may be a broadcast view -- ``lgamma`` guards it via ``_materialise``.
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + value * (torch.log(mu + eps) - log_theta_mu_eps)
        + lgamma(value + theta)
        - lgamma(theta)
        - lgamma(value + 1)
    )

    return res
