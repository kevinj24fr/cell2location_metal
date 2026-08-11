"""Contract for the flat trainer (engine task #13, final step).

The flat trainer owns two flat tensors -- unconstrained loc and
softplus-unconstrained rho (matching AutoNormal's SoftplusPositive scale
parameterization) -- reconstructs per-site tensors by slicing, and optimizes
-(flat_log_joint - flat_log_q). Its correctness chain is: one training step's
loss must equal -flat_elbo at the same draw (and flat_elbo is already pinned
against pyro replay by test_flat_elbo.py), state must round-trip losslessly
guide->flat->guide so posterior export sees the trained parameters, and the
model.train() wiring must use the engine only when its scope holds (MPS,
full batch, single particle, unscaled Trace_ELBO) with the kill switch and
divergence fallback restoring the pyro path. Never weaken these to pass.
"""

import os

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")

flat_train = pytest.importorskip(
    "cell2location.accel._flat_train",
    reason="flat trainer not built yet (task #13); this file is its contract",
)
from cell2location.accel import _flat_joint as flat  # noqa: E402

mps_available = torch.backends.mps.is_available()


def _make_model(seed=0):
    import pandas as pd

    from cell2location.models import Cell2location

    torch.manual_seed(seed)
    np.random.seed(seed)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=100, n_genes=60)
    Cell2location.setup_anndata(adata, batch_key="batch")
    sig = pd.DataFrame(np.random.rand(60, 3) + 0.1, index=adata.var_names, columns=list("abc"))
    return Cell2location(adata, cell_state_df=sig, N_cells_per_location=8, detection_alpha=20)


@pytest.fixture(scope="module")
def warmed_model():
    """Trained 2 epochs on CPU so the guide has a prototype trace and real params."""
    model = _make_model()
    model.train(max_epochs=2, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    return model


def _batch(model):
    from scvi.dataloaders import AnnDataLoader

    dl = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=model.adata.n_obs)
    return model.module._get_fn_args_from_batch(next(iter(dl)))


# --- flat state <-> guide round trip ---


def test_flat_state_matches_guide_params(warmed_model):
    from cell2location.accel._sampling import _autonormal_site_params

    state = flat_train.FlatGuideState.from_guide(warmed_model.module.guide)
    locs = state.unpack(state.loc)
    scales = state.unpack(torch.nn.functional.softplus(state.rho))
    for name, loc, scale, _transform in _autonormal_site_params(warmed_model.module.guide):
        assert torch.allclose(locs[name], loc.detach()), name
        assert torch.allclose(scales[name], scale.detach(), rtol=1e-5), name


def test_write_back_round_trips_into_guide(warmed_model):
    from cell2location.accel._sampling import _autonormal_site_params

    state = flat_train.FlatGuideState.from_guide(warmed_model.module.guide)
    with torch.no_grad():
        state.loc += 0.125
        state.rho -= 0.25
    state.write_back(warmed_model.module.guide)
    locs = state.unpack(state.loc)
    scales = state.unpack(torch.nn.functional.softplus(state.rho))
    for name, loc, scale, _transform in _autonormal_site_params(warmed_model.module.guide):
        assert torch.allclose(loc.detach(), locs[name], rtol=1e-5, atol=1e-7), name
        assert torch.allclose(scale.detach(), scales[name], rtol=1e-5, atol=1e-7), name


# --- one training step == -flat_elbo at the same draw ---


def test_step_loss_and_grads_match_flat_elbo(warmed_model):
    args, kwargs = _batch(warmed_model)
    state = flat_train.FlatGuideState.from_guide(warmed_model.module.guide)
    torch.manual_seed(7)
    eps = torch.randn_like(state.loc)

    loss = flat_train.flat_training_loss(warmed_model.module, state, args, kwargs, eps)
    loss_grads = torch.autograd.grad(loss, [state.loc, state.rho], retain_graph=False)

    loc = state.loc.detach().clone().requires_grad_(True)
    rho = state.rho.detach().clone().requires_grad_(True)
    u_flat = loc + torch.nn.functional.softplus(rho) * eps
    unconstrained = state.unpack(u_flat)
    reference = -(
        flat.flat_log_joint(warmed_model.module, args, kwargs, flat.constrain_latents(warmed_model.module, unconstrained))
        - flat_train.flat_log_q_from_state(loc, torch.nn.functional.softplus(rho), u_flat, state)
    )
    ref_grads = torch.autograd.grad(reference, [loc, rho])

    assert abs(float(loss) - float(reference)) <= 1e-4 * abs(float(reference))
    for got, want in zip(loss_grads, ref_grads):
        scale = want.abs().max().clamp_min(1e-6)
        assert (got - want).abs().max() <= 1e-3 * scale


def test_flat_log_q_from_state_matches_per_site(warmed_model):
    """The flat-tensor log q (one Normal over the concatenated vector plus per-site
    jacobians) must equal the per-site flat_log_q already pinned against pyro."""
    args, kwargs = _batch(warmed_model)
    del args, kwargs
    state = flat_train.FlatGuideState.from_guide(warmed_model.module.guide)
    torch.manual_seed(3)
    eps = torch.randn_like(state.loc)
    scale_flat = torch.nn.functional.softplus(state.rho)
    u_flat = state.loc + scale_flat * eps

    result = float(flat_train.flat_log_q_from_state(state.loc, scale_flat, u_flat, state))
    reference = float(flat.flat_log_q(warmed_model.module, state.unpack(u_flat)))
    assert abs(result - reference) <= 1e-4 * abs(reference)


def test_packed_model_proxy_matches_module(warmed_model):
    """Buffer packing (one tensor for all small hyperparameter buffers, lazily
    sliced inside the loss so a compiled graph reads one buffer, not 21 -- Metal
    caps kernels at 31 constant buffers) must be loss- and gradient-identical to
    reading the module's buffers directly."""
    args, kwargs = _batch(warmed_model)
    state = flat_train.FlatGuideState.from_guide(warmed_model.module.guide)
    torch.manual_seed(11)
    eps = torch.randn_like(state.loc)

    packed = flat_train.pack_module(warmed_model.module)
    loss_p = flat_train.flat_training_loss(packed, state, args, kwargs, eps)
    grads_p = torch.autograd.grad(loss_p, [state.loc, state.rho])

    loss_m = flat_train.flat_training_loss(warmed_model.module, state, args, kwargs, eps)
    grads_m = torch.autograd.grad(loss_m, [state.loc, state.rho])

    assert abs(float(loss_p) - float(loss_m)) <= 1e-6 * abs(float(loss_m))
    for gp, gm in zip(grads_p, grads_m):
        assert (gp - gm).abs().max() <= 1e-6 * gm.abs().max().clamp_min(1e-6)
    # the packing invariant itself: every packed attribute views one storage
    ptrs = {
        packed.model.m_g_mu_hyp.untyped_storage().data_ptr(),
        packed.model.ones.untyped_storage().data_ptr(),
        packed.model.alpha_g_phi_hyp_prior_beta.untyped_storage().data_ptr(),
        packed.model.ones_1_n_groups.untyped_storage().data_ptr(),
    }
    assert len(ptrs) == 1


# --- model.train() wiring ---


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_flat_engine_trains_end_to_end(monkeypatch):
    model = _make_model(seed=1)
    model.mps_flat_compile = False
    model.train(max_epochs=40, enable_progress_bar=False, enable_model_summary=False)
    assert model.flat_engine_used_ is True
    assert model.is_trained_ is True
    losses = model.history["elbo_train"]
    assert len(losses) == 40
    assert np.isfinite(np.asarray(losses, dtype=float)).all()
    # Training must have moved the guide parameters that posterior export reads.
    # A twin model built with the same seed has identical init (init_to_mean), so
    # warming its guide without training reproduces the starting locs.
    twin = _make_model(seed=1)
    args, kwargs = _batch(twin)
    with torch.no_grad():
        twin.module.guide(*args, **kwargs)
    trained = flat_train.FlatGuideState.from_guide(model.module.guide)
    init = flat_train.FlatGuideState.from_guide(twin.module.guide)
    assert trained.loc.shape == init.loc.shape
    assert not torch.allclose(trained.loc.cpu(), init.loc.cpu(), rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_kill_switch_falls_back_to_pyro(monkeypatch):
    monkeypatch.setenv(flat_train.FLAT_ENGINE_ENV_VAR, "0")
    model = _make_model(seed=2)
    model.train(max_epochs=2, enable_progress_bar=False, enable_model_summary=False)
    assert model.flat_engine_used_ is False
    assert model.is_trained_ is True


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_minibatch_runs_on_the_flat_engine():
    """Was a fallback contract while spatial minibatch was unimplemented. The
    engine now subsamples the observation plate -- local latents included -- and
    the arithmetic is pinned against pyro replay in test_flat_joint_minibatch.py.
    A spatial minibatch caller therefore stays on the flat engine."""
    model = _make_model(seed=3)
    model.train(max_epochs=2, batch_size=50, enable_progress_bar=False, enable_model_summary=False)
    assert model.flat_engine_used_ is True
    assert model.is_trained_ is True


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_minibatch_falls_back_when_a_local_site_is_not_row_indexed():
    """Subsampling local latents means indexing their guide parameters by
    observation. A site declared per-observation whose parameter is not shaped
    that way would be silently mis-indexed, so it must route to pyro instead."""
    model = _make_model(seed=3)
    pyro_model = getattr(model.module.model, "_orig_mod", model.module.model)
    original = pyro_model.list_obs_plate_vars
    try:
        pyro_model.list_obs_plate_vars = lambda: {
            "name": "obs_plate", "input": [], "sites": {"m_g": 1},  # m_g is per-GENE
        }
        model.train(max_epochs=2, batch_size=50, enable_progress_bar=False,
                    enable_model_summary=False)
        assert model.flat_engine_used_ is False
        assert model.is_trained_ is True
    finally:
        pyro_model.list_obs_plate_vars = original


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_unknown_train_kwargs_fall_back_to_pyro(tmp_path):
    """scvi's load() warms up pyro state via model.train(max_steps=1, ...). The flat
    engine does not implement max_steps -- ignoring it retrains a loaded model for
    thousands of epochs. Any train kwarg outside the engine's implemented set must
    route to the pyro path."""
    from cell2location.models import Cell2location

    model = _make_model(seed=5)
    model.train(max_epochs=2, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    model.save(str(tmp_path / "m"), overwrite=True, save_anndata=True)
    loaded = Cell2location.load(str(tmp_path / "m"))
    assert loaded.flat_engine_used_ is False


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_numerical_guard_runs_on_flat_engine():
    """A guarded run must stay guarded on the flat engine: the harness's guard_clean
    gate requires checks > 0, not diverged, finite max relative difference. The flat
    guard compares flat_log_joint at the current draw on MPS vs CPU -- the actual
    training arithmetic, same latents, so any disagreement is device arithmetic."""
    model = _make_model(seed=6)
    model.mps_flat_compile = False
    model.mps_numerical_guard_every_n_steps = 2
    model.train(max_epochs=5, enable_progress_bar=False, enable_model_summary=False)
    assert model.flat_engine_used_ is True
    summary = model.numerical_guard_.summary()
    assert summary["checks"] >= 2
    assert summary["diverged"] is False
    assert np.isfinite(summary["max_relative_difference"])


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_divergence_restores_params_and_falls_back(monkeypatch):
    model = _make_model(seed=4)
    model.mps_flat_compile = False

    real_loss = flat_train.flat_training_loss

    def poisoned(module, state, args, kwargs, eps, log_joint_fn=None):
        loss = real_loss(module, state, args, kwargs, eps, log_joint_fn)
        return loss * torch.nan

    monkeypatch.setattr(flat_train, "flat_training_loss", poisoned)
    model.train(max_epochs=2, enable_progress_bar=False, enable_model_summary=False)
    # non-finite loss => flat engine aborts, pyro path completes the run
    assert model.flat_engine_used_ is False
    assert model.is_trained_ is True
