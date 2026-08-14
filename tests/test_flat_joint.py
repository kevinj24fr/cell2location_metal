"""Equivalence contract for the flat log-joint engine (task #13).

The flat engine hand-transcribes the Cell2location model's joint log-density out of
pyro's effect-handler machinery. Hand transcription of a 12-site hierarchical model
is exactly where silent math errors live, so the contract is machine-checked: for
MANY random draws of all latents, the flat log-joint must equal pyro's replayed
``model_trace.log_prob_sum()`` to float32 tolerance. Until `flat_log_joint` exists
these tests are the executable specification; they must never be weakened to pass.
"""

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")

flat = pytest.importorskip(
    "cell2location.accel._flat_joint",
    reason="flat engine not built yet (task #13); this file is its contract",
)


@pytest.fixture(scope="module")
def spatial_model():
    import pandas as pd

    from cell2location.models import Cell2location

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=100, n_genes=60)
    Cell2location.setup_anndata(adata, batch_key="batch")
    sig = pd.DataFrame(np.random.rand(60, 3) + 0.1, index=adata.var_names, columns=list("abc"))
    model = Cell2location(adata, cell_state_df=sig, N_cells_per_location=8, detection_alpha=20)
    model.train(max_epochs=2, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    return model


def _batch(model):
    from scvi.dataloaders import AnnDataLoader

    dl = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=model.adata.n_obs)
    return model.module._get_fn_args_from_batch(next(iter(dl)))


def _pyro_log_joint(module, args, kwargs, latents):
    import pyro

    from cell2location.accel._guard import _sample_values_to

    trace = _sample_values_to_from_dict(latents)
    replayed = pyro.poutine.trace(pyro.poutine.replay(module.model, trace=trace)).get_trace(*args, **kwargs)
    return float(replayed.log_prob_sum())


def _sample_values_to_from_dict(latents):
    from pyro.poutine.trace_struct import Trace

    trace = Trace()
    for name, value in latents.items():
        trace.add_node(name, type="sample", value=value, is_observed=False, infer={})
    return trace


@pytest.mark.parametrize("seed", range(5))
def test_flat_log_joint_matches_pyro_replay(spatial_model, seed):
    args, kwargs = _batch(spatial_model)
    torch.manual_seed(seed)
    latents = flat.sample_latents_from_guide(spatial_model.module, args, kwargs)
    reference = _pyro_log_joint(spatial_model.module, args, kwargs, latents)
    result = float(flat.flat_log_joint(spatial_model.module, args, kwargs, latents))
    assert abs(result - reference) <= 1e-4 * abs(reference), (
        f"seed {seed}: flat {result:.6f} vs pyro {reference:.6f}"
    )


def test_flat_log_joint_is_differentiable(spatial_model):
    args, kwargs = _batch(spatial_model)
    torch.manual_seed(0)
    latents = flat.sample_latents_from_guide(spatial_model.module, args, kwargs, requires_grad=True)
    value = flat.flat_log_joint(spatial_model.module, args, kwargs, latents)
    value.backward()
    grads = [v.grad for v in latents.values() if v.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("seed", range(3))
def test_flat_log_joint_gradients_match_pyro(spatial_model, seed):
    """Forward equality is not enough for a training engine: a fused backward with
    recompute-in-backward can be wrong while the forward matches. Per-latent
    gradients must equal pyro's autograd through the replayed trace."""
    import pyro

    args, kwargs = _batch(spatial_model)
    torch.manual_seed(seed)
    latents = flat.sample_latents_from_guide(spatial_model.module, args, kwargs, requires_grad=True)

    flat_value = flat.flat_log_joint(spatial_model.module, args, kwargs, latents)
    names = [n for n, v in latents.items() if v.requires_grad]
    tensors = [latents[n] for n in names]
    flat_grads = torch.autograd.grad(flat_value, tensors, retain_graph=False, allow_unused=True)

    trace = _sample_values_to_from_dict(latents)
    replayed = pyro.poutine.trace(pyro.poutine.replay(spatial_model.module.model, trace=trace)).get_trace(
        *args, **kwargs
    )
    pyro_grads = torch.autograd.grad(replayed.log_prob_sum(), tensors, allow_unused=True)

    for name, fg, pg in zip(names, flat_grads, pyro_grads):
        if fg is None and pg is None:
            continue
        assert fg is not None and pg is not None, name
        scale = pg.abs().max().clamp_min(1e-6)
        assert (fg - pg).abs().max() <= 1e-4 * scale, (
            f"{name}: max grad diff {(fg - pg).abs().max():.3e} vs scale {scale:.3e}"
        )


def test_stable_alpha_caps_the_poisson_limit_but_is_a_noop_in_range():
    """alpha = 1/alpha_g_inverse**2 overflows fp32 as the parameter underflows to
    zero (the model's Exponential prior drives low-overdispersion genes there),
    NaN-ing the log-joint. _stable_alpha caps alpha at the representable Poisson
    limit while leaving every in-range value untouched, so the pyro-replay pins
    above are unaffected (they operate far below the cap).
    """
    from cell2location.accel._flat_joint import _ALPHA_MAX, _stable_alpha

    ones = torch.ones(4)
    # underflowed parameter -> would be inf; capped instead
    agi = torch.tensor([1.0, 0.1, 1e-30, 0.0])
    alpha = _stable_alpha(agi, ones)
    assert torch.isfinite(alpha).all(), "alpha must be finite even at parameter 0"
    assert alpha[0] == 1.0 and abs(alpha[1] - 100.0) < 1e-3, "in-range values untouched"
    assert alpha[2] == _ALPHA_MAX and alpha[3] == _ALPHA_MAX, "extreme values capped"
    # the cap is orders of magnitude above any harness/contract alpha
    assert _ALPHA_MAX >= 1e5
