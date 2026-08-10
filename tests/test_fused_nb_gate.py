"""Tests for the fused-kernel gate.

The Metal kernel itself cannot be executed without Apple hardware. What *can* be
tested anywhere -- and what actually protects the user -- is the gate around it: does
verification catch a wrong kernel, does rejection stick, and does the eager path
survive every failure mode.

So these tests inject fake dispatch functions with specific defects and assert the
gate rejects each one. A gate that only ever sees a correct kernel has not been
tested.
"""

import numpy as np
import pytest
import torch

from cell2location.accel import _fused_nb
from cell2location.accel._fused_nb import (
    _normalise_theta,
    _UnsupportedLayout,
    fused_log_nb_positive,
    fused_nb_status,
    reset_fused_nb_state,
    verify_fused_kernel,
)
from cell2location.accel._ops import eager_log_nb_positive, log_nb_positive


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    reset_fused_nb_state()
    monkeypatch.delenv(_fused_nb.FUSED_NB_ENV_VAR, raising=False)
    yield
    reset_fused_nb_state()


# --------------------------------------------------------------------------------------
# fake kernels, each broken in a way a real kernel plausibly would be
# --------------------------------------------------------------------------------------


def _correct(value, mu, theta, eps):
    return eager_log_nb_positive(value, mu, theta, eps)


def _wrong_by_a_little(value, mu, theta, eps):
    """The dangerous case: right shape, right magnitude, quietly wrong.

    A 0.5% error is invisible in a loss curve and fatal to the results.
    """
    return eager_log_nb_positive(value, mu, theta, eps) * 1.005


def _forgets_the_broadcast(value, mu, theta, eps):
    """Indexes theta by row instead of by column -- a classic kernel indexing bug."""
    if theta.shape != value.shape:
        theta = theta.reshape(-1)[0].expand_as(value)
    return eager_log_nb_positive(value, mu, theta, eps)


def _correct_forward_broken_gradient(value, mu, theta, eps):
    """Forward is exact, backward has a sign error on d/dtheta.

    This is the defect a forward-only check misses, and it is also the most likely
    one: the backward pass is hand-derived, the forward is transcribed.
    """
    return _SignFlippedThetaGrad.apply(value, mu, theta, eps)


class _SignFlippedThetaGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, mu, theta, eps):
        with torch.enable_grad():
            mu_ = mu.detach().requires_grad_(True)
            theta_ = theta.detach().requires_grad_(True)
            out = eager_log_nb_positive(value, mu_, theta_, eps)
        ctx.save_for_backward(value, mu_, theta_)
        ctx.out = out
        ctx.eps = eps
        return out.detach()

    @staticmethod
    def backward(ctx, grad_out):
        value, mu_, theta_ = ctx.saved_tensors
        grad_mu, grad_theta = torch.autograd.grad(ctx.out, (mu_, theta_), grad_out, retain_graph=True)
        return None, grad_mu, -grad_theta, None


def _raises(value, mu, theta, eps):
    raise RuntimeError("Metal shader compilation failed")


# --------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------


def test_gate_accepts_a_correct_kernel():
    passed, message = verify_fused_kernel(dispatch=_correct, device="cpu")
    assert passed, message


def test_gate_rejects_a_subtly_wrong_kernel():
    """0.5% off. Loud enough to matter scientifically, quiet enough to miss by eye."""
    passed, message = verify_fused_kernel(dispatch=_wrong_by_a_little, device="cpu")
    assert not passed
    assert "forward mismatch" in message


def test_gate_rejects_a_broadcast_indexing_bug():
    passed, message = verify_fused_kernel(dispatch=_forgets_the_broadcast, device="cpu")
    assert not passed
    assert "broadcast theta" in message


def test_gate_rejects_a_wrong_gradient_with_a_correct_forward():
    passed, message = verify_fused_kernel(dispatch=_correct_forward_broken_gradient, device="cpu")
    assert not passed
    assert "d/dtheta" in message


def test_gate_survives_a_kernel_that_raises():
    passed, message = verify_fused_kernel(dispatch=_raises, device="cpu")
    assert not passed
    assert "RuntimeError" in message


def test_gate_reports_where_the_error_is():
    """A rejection message has to be actionable, not just 'it failed'."""
    _, message = verify_fused_kernel(dispatch=_wrong_by_a_little, device="cpu")
    assert "flat index" in message
    assert "fused" in message and "eager" in message


# --------------------------------------------------------------------------------------
# fallback behaviour
# --------------------------------------------------------------------------------------


def test_disabled_by_default():
    assert fused_log_nb_positive(torch.ones(2, 2), torch.ones(2, 2), torch.ones(1, 2)) is None
    assert fused_nb_status()["requested"] is False


def test_enabled_but_not_on_mps_returns_none(monkeypatch):
    monkeypatch.setenv(_fused_nb.FUSED_NB_ENV_VAR, "1")
    result = fused_log_nb_positive(torch.ones(2, 2), torch.ones(2, 2), torch.ones(1, 2))
    assert result is None, "CPU tensors must never take the Metal path"


def test_float64_input_is_refused(monkeypatch):
    monkeypatch.setenv(_fused_nb.FUSED_NB_ENV_VAR, "1")
    value = torch.ones(2, 2, dtype=torch.float64)
    assert fused_log_nb_positive(value, value, value) is None


def test_eager_path_is_used_when_the_kernel_is_unavailable():
    """The property that actually matters: the model still trains, and correctly."""
    torch.manual_seed(0)
    value = torch.poisson(torch.full((16, 32), 4.0))
    mu = torch.rand(16, 32) * 10 + 0.1
    theta = torch.rand(1, 32) * 5 + 0.1

    result = eager_log_nb_positive(value, mu, theta)
    assert torch.isfinite(result).all()
    assert result.shape == (16, 32)


# --------------------------------------------------------------------------------------
# theta layout handling
# --------------------------------------------------------------------------------------


def test_normalise_theta_detects_row_broadcast():
    value = torch.ones(8, 5)
    theta = torch.ones(1, 5)
    buf, n_cols, broadcast = _normalise_theta(value, theta)
    assert broadcast == 1
    assert n_cols == 5
    assert buf.shape == (5,)


def test_normalise_theta_detects_elementwise():
    value = torch.ones(8, 5)
    buf, n_cols, broadcast = _normalise_theta(value, torch.ones(8, 5))
    assert broadcast == 0
    assert n_cols == 5
    assert buf.shape == (8, 5)


def test_normalise_theta_refuses_layouts_it_cannot_index():
    """Refusing beats guessing: the caller falls back to eager, which is general."""
    value = torch.ones(8, 5)
    with pytest.raises(_UnsupportedLayout):
        _normalise_theta(value, torch.ones(8, 1))


def test_normalise_theta_makes_buffers_contiguous():
    value = torch.ones(8, 5)
    theta = torch.ones(5, 8).t()  # non-contiguous, same shape as value
    buf, _, _ = _normalise_theta(value, theta)
    assert buf.is_contiguous(), "the kernel indexes buffers linearly and assumes contiguity"


# --------------------------------------------------------------------------------------
# shader source sanity (checkable without a GPU)
# --------------------------------------------------------------------------------------


def test_shader_source_declares_both_kernels():
    source = _fused_nb._read_shader_source()
    assert "kernel void nb_logprob_forward" in source
    assert "kernel void nb_logprob_backward" in source


def test_shader_bounds_checks_every_kernel():
    """Grid sizes get rounded up; without the guard that is an out-of-bounds write."""
    source = _fused_nb._read_shader_source()
    assert source.count("if (idx >= n_total)") == 2


def test_shader_constants_match_the_python_reference():
    """The kernel and :func:`lgamma_stirling` must use the same shift, or forward
    results will differ from the reference the gate compares them against."""
    from cell2location.accel._ops import _STIRLING_SHIFT

    source = _fused_nb._read_shader_source()
    assert f"constant int SHIFT = {_STIRLING_SHIFT};" in source


def test_shader_digamma_series_matches_a_numeric_reference():
    """Transcribe the MSL digamma series back into Python and check it against
    torch.digamma. Catches a mistyped coefficient, which is otherwise invisible until
    gradients are subtly wrong on hardware I cannot reach."""
    shift = 8
    x = torch.linspace(0.5, 50.0, 1000, dtype=torch.float64)

    accumulated = torch.zeros_like(x)
    for k in range(shift):
        accumulated = accumulated + 1.0 / (x + k)

    z = x + shift
    inv, inv2 = 1.0 / z, 1.0 / (z * z)
    series = (
        torch.log(z)
        - 0.5 * inv
        + inv2 * (-1.0 / 12.0 + inv2 * (1.0 / 120.0 + inv2 * (-1.0 / 252.0 + inv2 * (1.0 / 240.0))))
    )

    assert torch.allclose(series - accumulated, torch.digamma(x), rtol=1e-10, atol=1e-10)


def test_shader_lgamma_series_matches_a_numeric_reference():
    shift = 8
    x = torch.linspace(0.5, 50.0, 1000, dtype=torch.float64)

    accumulated = torch.zeros_like(x)
    for k in range(shift):
        accumulated = accumulated + torch.log(x + k)

    z = x + shift
    inv, inv2 = 1.0 / z, 1.0 / (z * z)
    series = inv * (1.0 / 12.0 + inv2 * (-1.0 / 360.0 + inv2 * (1.0 / 1260.0 + inv2 * (-1.0 / 1680.0))))
    result = (z - 0.5) * torch.log(z) - z + 0.5 * float(np.log(2 * np.pi)) + series - accumulated

    assert torch.allclose(result, torch.lgamma(x), rtol=1e-10, atol=1e-10)


def test_shader_gradient_formulae_match_autograd():
    """The backward kernel's formulae, transcribed from the .metal source and checked
    against autograd on the eager forward. If the derivation is wrong, it is wrong
    here too -- and on hardware I cannot test."""
    torch.manual_seed(0)
    eps = 1e-8
    value = torch.poisson(torch.full((64, 32), 5.0)).double()
    mu = (torch.rand(64, 32) * 20 + 0.1).double().requires_grad_(True)
    theta = (torch.rand(64, 32) * 10 + 0.1).double().requires_grad_(True)

    grad_out = torch.rand(64, 32).double()
    (eager_log_nb_positive(value, mu, theta, eps) * grad_out).sum().backward()

    v, m, t = value, mu.detach(), theta.detach()
    denom = t + m + eps
    ratio = (t + v) / denom

    expected_grad_mu = grad_out * (v / (m + eps) - ratio)
    expected_grad_theta = grad_out * (
        torch.log(t + eps) + t / (t + eps) - torch.log(denom) - ratio + torch.digamma(v + t) - torch.digamma(t)
    )

    assert torch.allclose(mu.grad, expected_grad_mu, rtol=1e-8, atol=1e-10)
    assert torch.allclose(theta.grad, expected_grad_theta, rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------------------
# the real kernel, end to end
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Metal backend")
@pytest.mark.skipif(not hasattr(getattr(torch, "mps", object()), "compile_shader"), reason="needs compile_shader")
def test_real_kernel_compiles_verifies_and_engages_on_mps(monkeypatch):
    """enable -> first dispatch -> compile -> verify -> use, on the actual GPU.

    Two ways this has broken before: the verification's eager reference dispatched
    back into the kernel being verified (infinite recursion), and the comparison
    widened MPS tensors to float64, which MPS refuses -- either way the kernel
    never engages on the hardware it was written for.
    """
    monkeypatch.setenv(_fused_nb.FUSED_NB_ENV_VAR, "1")

    torch.manual_seed(0)
    value = torch.poisson(torch.full((64, 128), 5.0))
    mu = torch.rand(64, 128) * 20 + 0.1
    theta = torch.rand(1, 128) * 10 + 0.1

    result = log_nb_positive(value.to("mps"), mu.to("mps"), theta.to("mps"))
    reference = log_nb_positive(value, mu, theta)
    assert torch.allclose(result.cpu(), reference, rtol=1e-4, atol=1e-4)

    status = _fused_nb.fused_nb_status()
    assert status["verified"] is True, status
