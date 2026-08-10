"""Tests for the Apple silicon acceleration layer.

Everything here runs on any platform. The MPS-specific assertions are skipped when
Metal is unavailable, so CI on Linux still exercises the dtype logic, the device
resolution rules and the numerical correctness of the guarded ``lgamma``.
"""

import numpy as np
import pytest
import torch

from cell2location import accel
from cell2location.accel import _ops
from cell2location.accel._dtype import downcast_unsupported_

mps_available = accel.mps_is_available()
requires_mps = pytest.mark.skipif(not mps_available, reason="Metal backend unavailable")


# --------------------------------------------------------------------------------------
# device resolution
# --------------------------------------------------------------------------------------


def test_resolve_accelerator_cpu_is_explicit():
    accelerator, device = accel.resolve_accelerator("cpu", "auto")
    assert accelerator == "cpu"
    assert device.type == "cpu"


def test_resolve_accelerator_auto_never_raises():
    accelerator, device = accel.resolve_accelerator("auto", "auto")
    assert accelerator in ("cpu", "gpu", "mps")
    assert isinstance(device, torch.device)


def test_resolve_accelerator_rejects_unavailable_mps():
    if mps_available:
        pytest.skip("MPS is available here, nothing to reject")
    with pytest.raises(RuntimeError, match="Metal backend is unavailable"):
        accel.resolve_accelerator("mps", "auto")


@requires_mps
def test_mps_device_has_no_index():
    """MPS is a single logical device; an index would be meaningless."""
    _, device = accel.resolve_accelerator("mps", 0)
    assert device.type == "mps"
    assert device.index is None


def test_disable_env_var_is_respected(monkeypatch):
    monkeypatch.setenv(accel.DISABLE_MPS_ENV_VAR, "1")
    assert accel.mps_is_disabled()
    assert not accel.mps_is_available()


# --------------------------------------------------------------------------------------
# dtype normalisation
# --------------------------------------------------------------------------------------


class _Float64Module(torch.nn.Module):
    """Mimics how cell2location registers hyperparameters straight from NumPy."""

    def __init__(self):
        super().__init__()
        self.register_buffer("from_numpy", torch.tensor(np.array([1.0, 2.0])))  # float64
        self.register_buffer("already_float32", torch.ones(3, dtype=torch.float32))
        self.register_buffer("an_integer", torch.tensor(7))
        self.inner = torch.nn.Linear(3, 2).double()


def test_numpy_buffers_really_are_float64():
    """Guards the premise: if this ever stops being true, the fix is unnecessary."""
    module = _Float64Module()
    assert module.from_numpy.dtype == torch.float64


def test_downcast_converts_float64_recursively():
    module = _Float64Module()
    downcast_unsupported_(module, "mps")

    assert module.from_numpy.dtype == torch.float32
    assert module.inner.weight.dtype == torch.float32
    assert module.already_float32.dtype == torch.float32


def test_downcast_leaves_integers_alone():
    module = _Float64Module()
    downcast_unsupported_(module, "mps")
    assert module.an_integer.dtype == torch.int64


def test_downcast_is_a_noop_for_cpu_and_cuda():
    module = _Float64Module()
    downcast_unsupported_(module, "cpu")
    assert module.from_numpy.dtype == torch.float64


def test_sanitize_tensor_preserves_values():
    x = torch.tensor([1.5, 2.5], dtype=torch.float64)
    out = accel.sanitize_tensor(x, "cpu")
    assert torch.equal(out, x)


def test_prepare_anndata_downcasts_counts():
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(np.random.poisson(3, size=(20, 10)).astype(np.float64))

    assert not accel.check_anndata_dtype(adata)
    accel.prepare_anndata(adata)
    assert accel.check_anndata_dtype(adata)
    assert adata.X.dtype == np.float32


def test_prepare_anndata_warns_on_unrepresentable_values():
    anndata = pytest.importorskip("anndata")
    counts = np.full((4, 4), 2.0**25, dtype=np.float64)
    adata = anndata.AnnData(counts)

    with pytest.warns(UserWarning, match="float32 cannot represent"):
        accel.prepare_anndata(adata)


# --------------------------------------------------------------------------------------
# lgamma
# --------------------------------------------------------------------------------------


def _max_error(result: torch.Tensor, reference: torch.Tensor, rtol: float, atol: float):
    """Combined absolute/relative error, matching ``torch.allclose`` semantics.

    A pure relative metric is meaningless near ``lgamma``'s roots at x=1 and x=2,
    where the true value passes through zero.
    """
    diff = (result.double() - reference.double()).abs()
    budget = atol + rtol * reference.double().abs()
    return float(diff.max()), bool((diff <= budget).all())


@pytest.mark.parametrize("mode", ["native", "contiguous", "stirling", "cpu"])
def test_lgamma_modes_agree_with_reference_on_cpu(mode):
    x = torch.linspace(1e-3, 500.0, 5000, dtype=torch.float32)
    reference = torch.lgamma(x.double())
    result = _ops.lgamma(x, mode=mode)

    max_abs, ok = _max_error(result, reference, rtol=1e-5, atol=1e-5)
    assert ok, f"mode={mode} max abs error {max_abs:.3e}"


def test_stirling_float32_error_near_the_roots_is_absolute():
    """Characterises the cancellation regime described in the docstring.

    Around x=1 and x=2 the true value is ~0, so the recurrence subtracts two similar
    magnitudes and the surviving error is absolute. Pinning it here means a change to
    the shift constant or the series shows up as a test failure rather than as
    slightly-off ELBOs six months later.
    """
    x = torch.linspace(0.5, 3.0, 2000, dtype=torch.float32)
    error = (_ops.lgamma_stirling(x).double() - torch.lgamma(x.double())).abs().max()
    assert error < 1e-5, f"absolute error near the roots is {error:.3e}"


def test_stirling_float32_error_at_large_arguments_is_relative():
    """Away from the roots the error is ordinary float32 rounding on a large value.

    lgamma(500) is ~2605, so ~1e-4 of absolute error there is float32 working exactly
    as specified -- which is why the parity checks use a combined abs/rel budget
    rather than an absolute one.
    """
    x = torch.linspace(50.0, 500.0, 2000, dtype=torch.float32)
    reference = torch.lgamma(x.double())
    rel_error = ((_ops.lgamma_stirling(x).double() - reference).abs() / reference.abs()).max()
    assert rel_error < 1e-6, f"relative error at large x is {rel_error:.3e}"


def test_stirling_lgamma_handles_the_full_useful_range():
    """Counts and dispersions in real data span roughly 1e-4 to 1e5."""
    x = torch.logspace(-4, 5, 2000, dtype=torch.float64)
    reference = torch.lgamma(x)
    result = _ops.lgamma_stirling(x)

    rel_error = ((result - reference).abs() / reference.abs().clamp_min(1e-6)).max()
    assert rel_error < 1e-9


def test_stirling_lgamma_is_differentiable():
    x = torch.rand(64, dtype=torch.float64, requires_grad=True) * 10 + 0.5
    (grad,) = torch.autograd.grad(_ops.lgamma_stirling(x).sum(), x)
    assert torch.allclose(grad, torch.digamma(x), rtol=1e-7)


def test_unknown_lgamma_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown lgamma mode"):
        _ops.lgamma(torch.ones(3), mode="definitely-not-a-mode")


def test_lgamma_mode_env_var_is_validated(monkeypatch):
    monkeypatch.setenv(_ops.LGAMMA_MODE_ENV_VAR, "nonsense")
    with pytest.raises(ValueError, match="must be one of"):
        _ops.current_lgamma_mode()


@requires_mps
@pytest.mark.parametrize("mode", ["contiguous", "stirling", "cpu"])
def test_lgamma_on_broadcast_view_matches_cpu(mode):
    """The exact shape that has historically been miscomputed on MPS."""
    generator = torch.Generator().manual_seed(0)
    base = torch.rand(1, 2000, generator=generator, dtype=torch.float32) * 50 + 1e-3
    broadcast = base.expand(512, 2000)

    reference = torch.lgamma(broadcast.double().contiguous())
    result = _ops.lgamma(broadcast.to("mps"), mode=mode).cpu()

    max_abs, ok = _max_error(result, reference, rtol=1e-5, atol=1e-5)
    assert ok, f"mode={mode} max abs error {max_abs:.3e}"


# --------------------------------------------------------------------------------------
# negative binomial
# --------------------------------------------------------------------------------------


def test_guarded_nb_matches_the_upstream_formula():
    """The guarded implementation must not change results on CPU."""
    torch.manual_seed(0)
    value = torch.poisson(torch.full((128, 300), 5.0))
    mu = torch.rand(128, 300) * 20 + 0.1
    theta = torch.rand(1, 300) * 10 + 0.1

    upstream = (
        theta * (torch.log(theta + 1e-8) - torch.log(theta + mu + 1e-8))
        + value * (torch.log(mu + 1e-8) - torch.log(theta + mu + 1e-8))
        + torch.lgamma(value + theta)
        - torch.lgamma(theta)
        - torch.lgamma(value + 1)
    )
    guarded = _ops.log_nb_positive(value, mu, theta)

    assert torch.allclose(guarded, upstream, rtol=1e-6, atol=1e-6)


def test_nb_distribution_log_prob_uses_the_guarded_path():
    from cell2location.distributions.NegativeBinomial import NegativeBinomial

    torch.manual_seed(0)
    value = torch.poisson(torch.full((32, 50), 4.0))
    mu = torch.rand(32, 50) * 10 + 0.1
    theta = torch.rand(1, 50) * 5 + 0.1

    dist = NegativeBinomial(mu=mu, theta=theta, validate_args=False)
    assert torch.allclose(dist.log_prob(value), _ops.log_nb_positive(value, mu, theta), rtol=1e-6)


@requires_mps
def test_nb_sampling_works_on_mps():
    """Falls back to CPU when the Gamma/Poisson kernels are missing; either way it
    must return an MPS tensor of the right shape."""
    from cell2location.distributions.NegativeBinomial import NegativeBinomial

    mu = (torch.rand(16, 32) * 10 + 0.1).to("mps")
    theta = (torch.rand(16, 32) * 5 + 0.1).to("mps")

    samples = NegativeBinomial(mu=mu, theta=theta, validate_args=False).sample()
    assert samples.shape == (16, 32)
    assert samples.device.type == "mps"
    assert torch.isfinite(samples).all()


# --------------------------------------------------------------------------------------
# fallback plumbing
# --------------------------------------------------------------------------------------


def test_run_on_cpu_restores_the_original_device():
    x = torch.ones(4)
    result = accel.run_on_cpu(lambda t: t * 2, x)
    assert result.device == x.device
    assert torch.equal(result, torch.full((4,), 2.0))


def test_supports_op_returns_false_for_unknown_ops():
    assert accel.supports_op("an_op_that_does_not_exist") is False


def test_report_is_serialisable():
    import json

    json.dumps(accel.report(), default=str)


# --------------------------------------------------------------------------------------
# module integration
# --------------------------------------------------------------------------------------


@requires_mps
def test_module_with_float64_buffers_moves_to_mps():
    from cell2location.accel._compat import AppleSiliconCompatMixin

    class _Module(AppleSiliconCompatMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("hyperparameter", torch.tensor(np.array([1.0, 2.0])))

    module = _Module()
    assert module.hyperparameter.dtype == torch.float64

    module.to("mps")  # would raise TypeError without the mixin
    assert module.hyperparameter.device.type == "mps"
    assert module.hyperparameter.dtype == torch.float32


def test_compile_is_allowed_on_mps_by_default(monkeypatch):
    """train_compiled() is already an explicit opt-in; requiring an env var on top
    of it was hostile. Measured on M2 Ultra / torch 2.12: inductor's Metal backend
    matches the hand-written fused kernel on the NB likelihood."""
    monkeypatch.delenv("CELL2LOCATION_ALLOW_MPS_COMPILE", raising=False)
    assert accel.compile_is_safe("cpu") is True
    assert accel.compile_is_safe("mps") is True


def test_compile_can_be_disabled_on_mps(monkeypatch):
    monkeypatch.setenv("CELL2LOCATION_ALLOW_MPS_COMPILE", "0")
    assert accel.compile_is_safe("mps") is False
    assert accel.compile_is_safe("cpu") is True, "the kill switch is Metal-specific"


# --------------------------------------------------------------------------------------
# GammaPoisson routing
# --------------------------------------------------------------------------------------


def test_gamma_poisson_is_bit_identical_to_pyro_on_cpu():
    """Off Metal the subclass must not change training arithmetic at all."""
    import pyro.distributions as dist

    from cell2location.accel import GammaPoisson

    torch.manual_seed(0)
    value = torch.poisson(torch.full((64, 128), 5.0))
    alpha = torch.rand(1, 128) * 10 + 0.1
    mu = torch.rand(64, 128) * 20 + 0.1

    ours = GammaPoisson(concentration=alpha, rate=alpha / mu).log_prob(value)
    pyros = dist.GammaPoisson(concentration=alpha, rate=alpha / mu).log_prob(value)
    assert torch.equal(ours, pyros)


@requires_mps
def test_gamma_poisson_on_mps_matches_cpu_reference():
    import pyro.distributions as dist

    from cell2location.accel import GammaPoisson

    torch.manual_seed(0)
    value = torch.poisson(torch.full((64, 128), 5.0))
    alpha = torch.rand(1, 128) * 10 + 0.1
    mu = torch.rand(64, 128) * 20 + 0.1

    reference = dist.GammaPoisson(concentration=alpha, rate=alpha / mu).log_prob(value)
    result = GammaPoisson(
        concentration=alpha.to("mps"), rate=(alpha / mu).to("mps")
    ).log_prob(value.to("mps"))

    max_abs, ok = _max_error(result.cpu(), reference, rtol=1e-4, atol=1e-4)
    assert ok, f"max abs error {max_abs:.3e}"


@requires_mps
def test_gamma_poisson_survives_plate_expansion():
    """Pyro plates call expand(), whose base implementation hardcodes the parent
    class -- losing the subclass would silently lose the Metal routing."""
    import pyro.distributions as dist

    from cell2location.accel import GammaPoisson

    torch.manual_seed(0)
    value = torch.poisson(torch.full((64, 128), 5.0))
    alpha = torch.rand(1, 128) * 10 + 0.1
    mu = torch.rand(64, 128) * 20 + 0.1

    expanded = GammaPoisson(
        concentration=alpha.to("mps"), rate=(alpha / mu).to("mps")
    ).expand(torch.Size([64, 128]))
    assert type(expanded) is GammaPoisson

    reference = dist.GammaPoisson(concentration=alpha, rate=alpha / mu).log_prob(value)
    max_abs, ok = _max_error(expanded.log_prob(value.to("mps")).cpu(), reference, rtol=1e-4, atol=1e-4)
    assert ok, f"max abs error {max_abs:.3e}"


def test_spatial_modules_use_the_routed_gamma_poisson():
    """The whole point: the training likelihood must go through accel dispatch."""
    import inspect

    from cell2location.models import _cell2location_module

    source = inspect.getsource(_cell2location_module)
    assert "dist.GammaPoisson(" not in source
    assert "GammaPoisson(" in source
