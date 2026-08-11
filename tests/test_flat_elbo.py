"""Contract for the flat ELBO (engine task #13, log q subtraction).

The flat trainer will optimize ``flat_log_joint - log q`` where log q is computed
from the same eps used for the draw: u = loc + scale*eps, z = transform(u),
log q(z) = Normal(loc, scale).log_prob(u) - log|det J_transform(u)| -- no inverse
transforms anywhere. The reference is pyro's own machinery: replaying AutoNormal
at the same unconstrained values yields the guide ``log_prob_sum`` (its Delta
correction is exactly -log|det J|), and model-replay-minus-guide-replay is the
single-particle Trace_ELBO estimator. Values and per-draw gradients must match.
These tests are the executable specification; they must never be weakened to pass.
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


def _unconstrained_trace(unconstrained):
    from pyro.poutine.trace_struct import Trace

    trace = Trace()
    for name, value in unconstrained.items():
        trace.add_node(name + "_unconstrained", type="sample", value=value, is_observed=False, infer={})
    return trace


def _pyro_traces(module, args, kwargs, unconstrained):
    """Guide and model traces replayed at our unconstrained draw, graphs intact."""
    import pyro

    guide_trace = pyro.poutine.trace(
        pyro.poutine.replay(module.guide, trace=_unconstrained_trace(unconstrained))
    ).get_trace(*args, **kwargs)
    model_trace = pyro.poutine.trace(pyro.poutine.replay(module.model, trace=guide_trace)).get_trace(
        *args, **kwargs
    )
    return guide_trace, model_trace


@pytest.mark.parametrize("seed", range(5))
def test_flat_log_q_matches_pyro_guide_replay(spatial_model, seed):
    args, kwargs = _batch(spatial_model)
    torch.manual_seed(seed)
    unconstrained = flat.sample_unconstrained_from_guide(spatial_model.module)
    guide_trace, _ = _pyro_traces(spatial_model.module, args, kwargs, unconstrained)
    reference = float(guide_trace.log_prob_sum())
    result = float(flat.flat_log_q(spatial_model.module, unconstrained))
    assert abs(result - reference) <= 1e-4 * abs(reference), (
        f"seed {seed}: flat log q {result:.6f} vs pyro {reference:.6f}"
    )


@pytest.mark.parametrize("seed", range(5))
def test_flat_elbo_matches_pyro_particle(spatial_model, seed):
    args, kwargs = _batch(spatial_model)
    torch.manual_seed(seed)
    unconstrained = flat.sample_unconstrained_from_guide(spatial_model.module)
    guide_trace, model_trace = _pyro_traces(spatial_model.module, args, kwargs, unconstrained)
    reference = float(model_trace.log_prob_sum()) - float(guide_trace.log_prob_sum())
    result = float(flat.flat_elbo(spatial_model.module, args, kwargs, unconstrained))
    assert abs(result - reference) <= 1e-4 * abs(reference), (
        f"seed {seed}: flat elbo {result:.6f} vs pyro {reference:.6f}"
    )


def test_constrain_latents_matches_guide_delta_values(spatial_model):
    args, kwargs = _batch(spatial_model)
    torch.manual_seed(0)
    unconstrained = flat.sample_unconstrained_from_guide(spatial_model.module)
    guide_trace, _ = _pyro_traces(spatial_model.module, args, kwargs, unconstrained)
    latents = flat.constrain_latents(spatial_model.module, unconstrained)
    assert set(latents) == set(unconstrained)
    for name, value in latents.items():
        pyro_value = guide_trace.nodes[name]["value"]
        assert torch.allclose(value, pyro_value, rtol=1e-5, atol=1e-7), name


@pytest.mark.parametrize("seed", range(3))
def test_flat_elbo_gradients_match_pyro(spatial_model, seed):
    """The trainer backpropagates through the ELBO, so per-draw gradients (w.r.t.
    the unconstrained sample, where the reparameterized path starts) must equal
    pyro's autograd through the replayed guide and model traces.

    Tolerance is 1e-3 relative, calibrated by an fp64 control (2026-08-10): with
    the whole comparison in float64 every site's relative diff is <= 2.2e-13, so
    the transcription is exact; in float32, near-stationary sites whose gradient
    terms nearly cancel (alpha_g_phi_hyp: 2.08e-4 at seed 2) carry cancellation
    noise above 1e-4. A wrong or missing term shows at O(1) relative -- 1e-3
    keeps >100x separation. Do not tighten without rerunning the fp64 control;
    do not loosen, period."""
    args, kwargs = _batch(spatial_model)
    torch.manual_seed(seed)
    unconstrained = flat.sample_unconstrained_from_guide(spatial_model.module, requires_grad=True)
    names = list(unconstrained)
    tensors = [unconstrained[n] for n in names]

    flat_value = flat.flat_elbo(spatial_model.module, args, kwargs, unconstrained)
    flat_grads = torch.autograd.grad(flat_value, tensors, allow_unused=True)

    guide_trace, model_trace = _pyro_traces(spatial_model.module, args, kwargs, unconstrained)
    reference = model_trace.log_prob_sum() - guide_trace.log_prob_sum()
    pyro_grads = torch.autograd.grad(reference, tensors, allow_unused=True)

    for name, fg, pg in zip(names, flat_grads, pyro_grads):
        if fg is None and pg is None:
            continue
        assert fg is not None and pg is not None, name
        scale = pg.abs().max().clamp_min(1e-6)
        assert (fg - pg).abs().max() <= 1e-3 * scale, (
            f"{name}: max grad diff {(fg - pg).abs().max():.3e} vs scale {scale:.3e}"
        )
