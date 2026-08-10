"""Tests for the runtime divergence guard and the unified-memory planner.

The guard's whole purpose is to fire when a device quietly disagrees with the CPU.
Testing it against a device that agrees would prove nothing, so the divergence is
injected directly.
"""

import pytest
import torch

from cell2location.accel import _planner, _train
from cell2location.accel._guard import NumericalGuard
from cell2location.accel._planner import plan_memory

# --------------------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------------------


def _record(step: int, device_loss: float, reference_loss: float) -> dict:
    return {
        "step": step,
        "device_loss": device_loss,
        "reference_loss": reference_loss,
        "relative_difference": abs(device_loss - reference_loss) / abs(reference_loss),
    }


def test_guard_is_quiet_when_devices_agree():
    guard = NumericalGuard(tolerance=1e-3)
    guard.history = [_record(i, 1000.0, 1000.0001) for i in range(5)]

    assert not guard.diverged
    assert guard.summary()["checks"] == 5
    assert guard.summary()["max_relative_difference"] < 1e-3


def test_guard_flags_a_diverged_device():
    """1% off: exactly the magnitude a wrong kernel produces and a loss curve hides."""
    guard = NumericalGuard(tolerance=1e-3)
    guard.history = [_record(0, 1000.0, 1000.0), _record(1000, 1010.0, 1000.0)]

    assert guard.diverged
    assert guard.summary()["max_relative_difference"] == pytest.approx(0.01)


def test_guard_summary_is_safe_before_any_check():
    guard = NumericalGuard()
    assert guard.summary() == {"checks": 0, "diverged": False}
    assert not guard.diverged


def test_guard_warning_is_rate_limited(caplog):
    """A diverged 30k-step run must not emit thirty identical warnings."""
    guard = NumericalGuard(tolerance=1e-6, max_warnings=2)

    class _Diverging:
        pass

    def fake_compare(module, args, kwargs, reference_device="cpu"):
        return {"device_loss": 1.1, "reference_loss": 1.0, "relative_difference": 0.1}

    import cell2location.accel._guard as guard_module

    original = guard_module.compare_loss_across_devices
    guard_module.compare_loss_across_devices = fake_compare
    try:
        with caplog.at_level("WARNING"):
            for step in range(5):
                guard.check(_Diverging(), [], {}, step)
    finally:
        guard_module.compare_loss_across_devices = original

    assert len(guard.history) == 5, "every check is still recorded"
    warnings = [r for r in caplog.records if "Numerical guard" in r.message]
    assert len(warnings) == 2, "but only max_warnings are surfaced"


def test_guard_records_every_check_even_when_silent():
    guard = NumericalGuard(tolerance=1.0)

    def fake_compare(module, args, kwargs, reference_device="cpu"):
        return {"device_loss": 1.0, "reference_loss": 1.0, "relative_difference": 0.0}

    import cell2location.accel._guard as guard_module

    original = guard_module.compare_loss_across_devices
    guard_module.compare_loss_across_devices = fake_compare
    try:
        for step in range(3):
            guard.check(object(), [], {}, step)
    finally:
        guard_module.compare_loss_across_devices = original

    assert [r["step"] for r in guard.history] == [0, 1, 2]


def test_compare_returns_none_when_already_on_the_reference_device():
    """No point comparing CPU against CPU, and it must not pretend otherwise."""
    from cell2location.accel._guard import compare_loss_across_devices

    module = torch.nn.Linear(2, 2)
    assert compare_loss_across_devices(module, [], {}) is None


# --------------------------------------------------------------------------------------
# guard wiring
# --------------------------------------------------------------------------------------


def test_guard_env_var_parsing(monkeypatch):
    monkeypatch.setenv(_train.GUARD_ENV_VAR, "1")
    assert _train._guard_interval_from_env() == 1000

    monkeypatch.setenv(_train.GUARD_ENV_VAR, "250")
    assert _train._guard_interval_from_env() == 250

    monkeypatch.setenv(_train.GUARD_ENV_VAR, "0")
    assert _train._guard_interval_from_env() == 0

    monkeypatch.delenv(_train.GUARD_ENV_VAR, raising=False)
    assert _train._guard_interval_from_env() == 0


def test_guard_env_var_ignores_nonsense(monkeypatch, caplog):
    monkeypatch.setenv(_train.GUARD_ENV_VAR, "sometimes")
    with caplog.at_level("WARNING"):
        assert _train._guard_interval_from_env() == 0
    assert any("not an integer" in r.message for r in caplog.records)


def test_prepare_apple_silicon_is_a_noop_off_metal():
    """The entire mixin must be invisible on Linux and CUDA."""

    class _Model(_train.AppleSiliconTrainMixin):
        adata_manager = None

    kwargs = {"accelerator": "cpu"}
    _Model()._prepare_apple_silicon(kwargs)
    assert "callbacks" not in kwargs


# --------------------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------------------


def test_planner_says_yes_when_memory_is_plentiful():
    plan = plan_memory(n_obs=50_000, n_genes=12_000, memory_budget_gb=400)
    assert plan.fits_full_batch
    assert plan.recommended_batch_size is None


def test_planner_says_no_and_suggests_a_batch_size_when_tight():
    plan = plan_memory(n_obs=200_000, n_genes=18_000, memory_budget_gb=16)
    assert not plan.fits_full_batch
    assert plan.recommended_batch_size >= 256


def test_planner_batch_size_is_a_power_of_two():
    plan = plan_memory(n_obs=200_000, n_genes=18_000, memory_budget_gb=64)
    size = plan.recommended_batch_size
    assert size & (size - 1) == 0, f"{size} is not a power of two"


def test_planner_is_monotonic_in_budget():
    """More memory must never yield a smaller recommendation."""
    sizes = []
    for budget in (16, 32, 64, 128):
        plan = plan_memory(n_obs=500_000, n_genes=20_000, memory_budget_gb=budget)
        sizes.append(plan.recommended_batch_size or float("inf"))
    assert sizes == sorted(sizes), sizes


def test_planner_never_recommends_below_the_floor():
    """A hopeless case must still return something runnable, flagged in the notes."""
    plan = plan_memory(n_obs=2_000_000, n_genes=30_000, memory_budget_gb=8)
    assert plan.recommended_batch_size == 256
    assert any("floor" in note for note in plan.notes)


def test_planner_credits_the_fused_kernel_with_lower_peak():
    eager = plan_memory(n_obs=100_000, n_genes=15_000, memory_budget_gb=100, fused_kernel=False)
    fused = plan_memory(n_obs=100_000, n_genes=15_000, memory_budget_gb=100, fused_kernel=True)
    assert fused.full_batch_bytes < eager.full_batch_bytes


def test_planner_mentions_the_fused_kernel_when_it_would_change_the_answer():
    """The suggestion has to be conditional on it actually helping, not boilerplate."""
    plans = [
        plan_memory(n_obs=n, n_genes=15_000, memory_budget_gb=budget)
        for n, budget in [(60_000, 60), (80_000, 70), (100_000, 90), (120_000, 100)]
    ]
    mentions = [any("fused Metal kernel would" in note for note in p.notes) for p in plans]
    assert any(mentions), "expected at least one borderline case to surface the suggestion"

    for plan, mentioned in zip(plans, mentions):
        if mentioned:
            assert not plan.fits_full_batch, "only suggest it when eager does not already fit"


def test_planner_reads_dimensions_from_anndata():
    anndata = pytest.importorskip("anndata")
    import numpy as np

    adata = anndata.AnnData(np.zeros((120, 34), dtype=np.float32))
    plan = plan_memory(adata, memory_budget_gb=8)
    assert plan.n_obs == 120
    assert plan.n_genes == 34


def test_planner_requires_dimensions():
    with pytest.raises(ValueError, match="Provide either adata"):
        plan_memory()


def test_planner_output_is_readable_and_serialisable():
    import json

    plan = plan_memory(n_obs=4992, n_genes=12_000, memory_budget_gb=64)
    text = str(plan)
    assert "Verdict" in text and "4,992" in text
    json.dumps(plan.as_dict())


def test_planner_safety_fraction_leaves_real_headroom():
    """Unified memory means overshooting swaps the whole machine, not just the job."""
    assert 0.5 <= _planner._SAFETY_FRACTION <= 0.85
