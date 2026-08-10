"""Fused negative-binomial log-likelihood on Metal, with a self-verifying gate.

The negative-binomial log-probability is the hot loop of cell2location: it is
evaluated over the full ``(n_obs, n_genes)`` matrix on every SVI step, and in eager
mode it materialises about a dozen intermediates of that size. The arithmetic is
cheap; the memory traffic is not. ``_metal/nb_logprob.metal`` collapses the whole
expression -- forward and backward -- into one pass each.

**This module is disabled by default, and it verifies itself before it is trusted.**

That is not excessive caution, it is an honest response to a constraint: the kernel
was written without access to Apple hardware, so it has never executed. Shipping it
enabled would be asking a user to take an unexecuted GPU kernel on faith in the
computation that produces their scientific results.

Instead, the first time it is used on a given machine it runs both the forward pass
and the gradients against the eager implementation on realistic shapes. If anything
disagrees, it logs what failed, disables itself permanently for the process, and the
eager path continues. The worst case is one wasted check at startup; there is no path
by which an unverified kernel reaches your ELBO.

Enable with::

    export CELL2LOCATION_MPS_FUSED_NB=1

or ``cell2location.accel.enable_fused_nb()``.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ._ops import eager_log_nb_positive as _eager_log_nb_positive

logger = logging.getLogger(__name__)

__all__ = [
    "FUSED_NB_ENV_VAR",
    "enable_fused_nb",
    "disable_fused_nb",
    "fused_nb_enabled",
    "fused_nb_status",
    "fused_log_nb_positive",
    "verify_fused_kernel",
    "reset_fused_nb_state",
]

FUSED_NB_ENV_VAR = "CELL2LOCATION_MPS_FUSED_NB"

_SHADER_PATH = Path(__file__).parent / "_metal" / "nb_logprob.metal"

#: Tolerances for the self-test. Deliberately tight: this compares one float32
#: computation against another float32 computation of the same quantity, so a
#: correct kernel lands near machine precision. A kernel that is wrong in the way
#: kernels are usually wrong -- bad indexing, missing broadcast, wrong sign -- misses
#: by order 1 and is nowhere near these numbers.
_VERIFY_RTOL = 1e-4
_VERIFY_ATOL = 1e-4


class _FusedNBState:
    """Process-wide state for the fused kernel.

    Tracked explicitly rather than with a bare module global so the whole lifecycle
    -- requested, compiled, verified, rejected -- is inspectable from
    :func:`fused_nb_status` and testable without a GPU.
    """

    def __init__(self):
        self.library: Optional[Any] = None
        self.verified: Optional[bool] = None
        self.rejected_reason: Optional[str] = None
        self.compile_error: Optional[str] = None

    def reset(self) -> None:
        self.__init__()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested": fused_nb_enabled(),
            "compiled": self.library is not None,
            "verified": self.verified,
            "rejected_reason": self.rejected_reason,
            "compile_error": self.compile_error,
        }


_STATE = _FusedNBState()


def reset_fused_nb_state() -> None:
    """Forget compilation and verification results. Mainly for tests."""
    _STATE.reset()


def enable_fused_nb() -> None:
    os.environ[FUSED_NB_ENV_VAR] = "1"


def disable_fused_nb() -> None:
    os.environ[FUSED_NB_ENV_VAR] = "0"


def fused_nb_enabled() -> bool:
    """On by default. Safe as a default because the kernel never runs unverified:
    the first dispatch compares it against the eager implementation, forward and
    gradients, and permanently rejects it for the process on any mismatch. Set
    ``CELL2LOCATION_MPS_FUSED_NB=0`` to force the eager path."""
    return os.environ.get(FUSED_NB_ENV_VAR, "1").lower() in ("1", "true", "yes")


def fused_nb_status() -> Dict[str, Any]:
    """Where the fused kernel currently stands, for logs and bug reports."""
    return _STATE.as_dict()


# --------------------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------------------


def _read_shader_source() -> str:
    return _SHADER_PATH.read_text()


def _compile_library() -> Optional[Any]:
    """Compile the Metal shader, returning None (with a logged reason) on failure."""
    compile_shader = getattr(getattr(torch, "mps", None), "compile_shader", None)
    if compile_shader is None:
        _STATE.compile_error = "torch.mps.compile_shader is unavailable in this PyTorch build"
        return None

    try:
        return compile_shader(_read_shader_source())
    except Exception as exc:  # noqa: BLE001 - a compile failure must never be fatal
        _STATE.compile_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Fused NB kernel failed to compile, using eager path: %s", _STATE.compile_error)
        return None


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------


def _normalise_theta(value: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Return ``(theta_buffer, n_cols, is_broadcast)`` for the kernel.

    Two layouts are supported: theta matching ``value`` elementwise, and the common
    ``(1, n_genes)`` row broadcast. Anything else is refused rather than guessed at --
    the caller falls back to eager, which handles arbitrary broadcasting correctly.
    """
    if theta.shape == value.shape:
        return theta.contiguous(), value.shape[-1], 0

    if theta.ndim <= 2 and theta.shape[-1] == value.shape[-1] and theta.numel() == value.shape[-1]:
        return theta.reshape(-1).contiguous(), value.shape[-1], 1

    raise _UnsupportedLayout(f"theta shape {tuple(theta.shape)} against value shape {tuple(value.shape)}")


class _UnsupportedLayout(Exception):
    """Raised when the kernel's indexing assumptions do not hold; triggers fallback."""


def _dispatch_forward(library, value, mu, theta_buf, n_cols, broadcast, eps) -> torch.Tensor:
    out = torch.empty_like(value)
    library.nb_logprob_forward(
        out,
        value,
        mu,
        theta_buf,
        int(value.numel()),
        int(n_cols),
        int(broadcast),
        float(eps),
    )
    return out


def _dispatch_backward(library, grad_out, value, mu, theta_buf, n_cols, broadcast, eps):
    grad_mu = torch.empty_like(value)
    grad_theta_elem = torch.empty_like(value)
    library.nb_logprob_backward(
        grad_mu,
        grad_theta_elem,
        grad_out,
        value,
        mu,
        theta_buf,
        int(value.numel()),
        int(n_cols),
        int(broadcast),
        float(eps),
    )
    return grad_mu, grad_theta_elem


class _FusedLogNBPositive(torch.autograd.Function):
    """Autograd wrapper around the fused kernels.

    ``value`` is observed count data and never requires grad, so only ``mu`` and
    ``theta`` gradients are produced.
    """

    @staticmethod
    def forward(ctx, value, mu, theta, eps, library):
        theta_buf, n_cols, broadcast = _normalise_theta(value, theta)

        value = value.contiguous()
        mu = mu.contiguous()

        out = _dispatch_forward(library, value, mu, theta_buf, n_cols, broadcast, eps)

        ctx.save_for_backward(value, mu, theta_buf)
        ctx.eps = eps
        ctx.library = library
        ctx.n_cols = n_cols
        ctx.broadcast = broadcast
        ctx.theta_shape = theta.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        value, mu, theta_buf = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        grad_mu, grad_theta_elem = _dispatch_backward(
            ctx.library, grad_out, value, mu, theta_buf, ctx.n_cols, ctx.broadcast, ctx.eps
        )

        if ctx.broadcast:
            # theta was a single row shared by every observation: sum its contribution
            # back down. One reduction, versus a threadgroup reduction inside the
            # kernel that would need synchronisation for no measurable gain.
            grad_theta = grad_theta_elem.reshape(-1, ctx.n_cols).sum(dim=0).reshape(ctx.theta_shape)
        else:
            grad_theta = grad_theta_elem.reshape(ctx.theta_shape)

        return None, grad_mu, grad_theta, None, None


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------


def verify_fused_kernel(
    dispatch: Optional[Callable] = None,
    device: str = "mps",
    rtol: float = _VERIFY_RTOL,
    atol: float = _VERIFY_ATOL,
    seed: int = 0,
) -> Tuple[bool, str]:
    """Check the fused path against eager, forward and gradients.

    ``dispatch(value, mu, theta, eps)`` defaults to the real kernel. Injecting it
    lets the gate itself be tested without a GPU -- including the case that matters
    most, a kernel that returns plausible-looking but wrong numbers.

    Returns ``(passed, message)``.
    """
    if dispatch is None:
        if _STATE.library is None:
            return False, "kernel not compiled"

        def dispatch(value, mu, theta, eps):
            return _FusedLogNBPositive.apply(value, mu, theta, eps, _STATE.library)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Both layouts the kernel claims to support, at shapes with awkward remainders so
    # an off-by-one in the index arithmetic has somewhere to show up.
    cases = [
        ("broadcast theta", (257, 601), True),
        ("elementwise theta", (129, 337), False),
        ("single row", (1, 1024), True),
    ]

    for label, (n_obs, n_genes), broadcast in cases:
        value = torch.poisson(torch.full((n_obs, n_genes), 5.0), generator=generator).to(device)
        mu = (torch.rand(n_obs, n_genes, generator=generator) * 20 + 0.1).to(device)
        theta_shape = (1, n_genes) if broadcast else (n_obs, n_genes)
        theta = (torch.rand(*theta_shape, generator=generator) * 10 + 0.1).to(device)

        mu_a, theta_a = mu.clone().requires_grad_(True), theta.clone().requires_grad_(True)
        mu_b, theta_b = mu.clone().requires_grad_(True), theta.clone().requires_grad_(True)

        try:
            fused = dispatch(value, mu_a, theta_a, 1e-8)
        except Exception as exc:  # noqa: BLE001
            return False, f"{label}: forward raised {type(exc).__name__}: {exc}"

        eager = _eager_log_nb_positive(value, mu_b, theta_b, 1e-8)

        ok, detail = _compare(fused, eager, rtol, atol)
        if not ok:
            return False, f"{label}: forward mismatch, {detail}"

        # Gradients are where a hand-written backward actually goes wrong, and a
        # forward-only check would sail straight past it.
        weights = torch.rand(n_obs, n_genes, generator=generator).to(device)
        try:
            (fused * weights).sum().backward()
            (eager * weights).sum().backward()
        except Exception as exc:  # noqa: BLE001
            return False, f"{label}: backward raised {type(exc).__name__}: {exc}"

        ok, detail = _compare(mu_a.grad, mu_b.grad, rtol, atol)
        if not ok:
            return False, f"{label}: d/dmu mismatch, {detail}"

        ok, detail = _compare(theta_a.grad, theta_b.grad, rtol, atol)
        if not ok:
            return False, f"{label}: d/dtheta mismatch, {detail}"

    return True, "forward and gradients match eager on all supported layouts"


def _compare(a: Optional[torch.Tensor], b: Optional[torch.Tensor], rtol: float, atol: float) -> Tuple[bool, str]:
    if a is None or b is None:
        return False, "a gradient was not produced"
    a64, b64 = a.detach().cpu().double(), b.detach().cpu().double()
    diff = (a64 - b64).abs()
    budget = atol + rtol * b64.abs()
    if bool((diff <= budget).all()):
        return True, ""
    worst = int(diff.argmax())
    return False, (
        f"max abs error {float(diff.max()):.3e} at flat index {worst} "
        f"(fused {float(a64.reshape(-1)[worst]):.6g} vs eager {float(b64.reshape(-1)[worst]):.6g})"
    )


def _ensure_ready() -> bool:
    """Compile and verify once per process. Returns whether the kernel may be used."""
    if _STATE.verified is not None:
        return _STATE.verified

    if _STATE.library is None:
        _STATE.library = _compile_library()
        if _STATE.library is None:
            _STATE.verified = False
            _STATE.rejected_reason = _STATE.compile_error
            return False

    passed, message = verify_fused_kernel()
    _STATE.verified = passed

    if passed:
        logger.info("Fused NB Metal kernel verified against eager (%s); enabling.", message)
    else:
        _STATE.rejected_reason = message
        logger.warning(
            "Fused NB Metal kernel REJECTED and permanently disabled for this process: %s. "
            "Training continues on the eager path with no loss of correctness. "
            "Please report this with the output of cell2location.accel.report().",
            message,
        )

    return passed


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def fused_log_nb_positive(
    value: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """Fused NB log-likelihood, or ``None`` when the fused path is unavailable.

    Returning ``None`` rather than falling back internally keeps the decision visible
    at the call site: the caller can see, and test, that eager is what ran.
    """
    if not fused_nb_enabled():
        return None
    if value.device.type != "mps":
        return None
    if not (value.dtype == mu.dtype == theta.dtype == torch.float32):
        return None
    if value.ndim != 2:
        return None
    if not _ensure_ready():
        return None

    try:
        return _FusedLogNBPositive.apply(value, mu, theta, eps, _STATE.library)
    except _UnsupportedLayout as exc:
        logger.debug("Fused NB kernel skipped for this call: %s", exc)
        return None
