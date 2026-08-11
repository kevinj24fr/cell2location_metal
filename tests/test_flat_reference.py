"""Equivalence contract for the RegressionModel flat log-joint.

Companion to ``test_flat_joint.py``, which pins the spatial model. The reference
signature model needs its own transcription for two reasons: its sites differ,
and unlike the spatial model it **minibatches by default**, so its joint carries
the observation plate's ``n_obs / batch`` scale on the likelihood. That scale is
the single easiest thing to get silently wrong -- it is invisible at full batch
and wrong by a constant factor everywhere else -- so it is pinned here against
pyro replay at three batch sizes, not just one.

As with the spatial contract: these tests are the executable specification and
must never be weakened to pass.
"""

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")

flat = pytest.importorskip("cell2location.accel._flat_joint")
ref_flat = pytest.importorskip(
    "cell2location.accel._flat_reference",
    reason="reference flat engine not built yet; this file is its contract",
)

N_OBS, N_GENES = 120, 60


@pytest.fixture(scope="module")
def reference_model():
    from cell2location.models import RegressionModel

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=N_OBS // 2, n_genes=N_GENES)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    model = RegressionModel(adata)
    model.train(max_epochs=2, batch_size=None, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    return model


def _batch(model, batch_size):
    from scvi.dataloaders import AnnDataLoader

    dl = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=batch_size)
    return model.module._get_fn_args_from_batch(next(iter(dl)))


def _trace_from_dict(latents):
    from pyro.poutine.trace_struct import Trace

    trace = Trace()
    for name, value in latents.items():
        trace.add_node(name, type="sample", value=value, is_observed=False, infer={})
    return trace


def _pyro_log_joint(module, args, kwargs, latents):
    import pyro

    replayed = pyro.poutine.trace(
        pyro.poutine.replay(module.model, trace=_trace_from_dict(latents))
    ).get_trace(*args, **kwargs)
    return float(replayed.log_prob_sum())


# Full batch plus two genuine minibatches: the plate scale is 1.0, 2.0 and 4.0
# respectively, so a missing or misplaced scale cannot pass all three.
BATCH_SIZES = [N_OBS, N_OBS // 2, N_OBS // 4]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seed", range(3))
def test_reference_log_joint_matches_pyro_replay(reference_model, batch_size, seed):
    args, kwargs = _batch(reference_model, batch_size)
    torch.manual_seed(seed)
    latents = flat.sample_latents_from_guide(reference_model.module, args, kwargs)
    reference = _pyro_log_joint(reference_model.module, args, kwargs, latents)
    result = float(ref_flat.reference_log_joint(reference_model.module, args, kwargs, latents))
    assert abs(result - reference) <= 1e-4 * abs(reference), (
        f"batch {batch_size}, seed {seed}: flat {result:.6f} vs pyro {reference:.6f}"
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_reference_log_joint_gradients_match_pyro(reference_model, batch_size):
    """Forward equality is not enough for a training engine -- the gradients are
    what training consumes, and the plate scale multiplies them too."""
    import pyro

    args, kwargs = _batch(reference_model, batch_size)
    torch.manual_seed(0)
    latents = flat.sample_latents_from_guide(
        reference_model.module, args, kwargs, requires_grad=True
    )

    flat_value = ref_flat.reference_log_joint(reference_model.module, args, kwargs, latents)
    names = [n for n, v in latents.items() if v.requires_grad]
    tensors = [latents[n] for n in names]
    flat_grads = torch.autograd.grad(flat_value, tensors, allow_unused=True)

    replayed = pyro.poutine.trace(
        pyro.poutine.replay(reference_model.module.model, trace=_trace_from_dict(latents))
    ).get_trace(*args, **kwargs)
    pyro_grads = torch.autograd.grad(replayed.log_prob_sum(), tensors, allow_unused=True)

    for name, fg, pg in zip(names, flat_grads, pyro_grads):
        if fg is None and pg is None:
            continue
        assert fg is not None and pg is not None, name
        scale = pg.abs().max().clamp_min(1e-6)
        assert (fg - pg).abs().max() <= 1e-4 * scale, (
            f"{name} at batch {batch_size}: max gradient difference "
            f"{(fg - pg).abs().max():.3e} exceeds {1e-4 * scale:.3e}"
        )


def test_reference_log_joint_is_differentiable(reference_model):
    args, kwargs = _batch(reference_model, N_OBS)
    torch.manual_seed(0)
    latents = flat.sample_latents_from_guide(
        reference_model.module, args, kwargs, requires_grad=True
    )
    value = ref_flat.reference_log_joint(reference_model.module, args, kwargs, latents)
    value.backward()
    grads = [v.grad for v in latents.values() if v.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_log_joint_dispatch_is_by_module_type(reference_model):
    """The flat engine's applicability check lives on a mixin BOTH models inherit.
    Nothing may resolve a model to another model's transcription: that would train
    silently against the wrong density. Resolution is by module type, and an
    unknown module resolves to None (caller falls back to pyro)."""
    resolved = flat.log_joint_for(reference_model.module)
    assert resolved is ref_flat.reference_log_joint

    class _Unknown:
        pass

    class _FakeModule:
        model = _Unknown()

    assert flat.log_joint_for(_FakeModule()) is None


def test_extra_categoricals_are_out_of_scope(reference_model):
    """The covariate-effect site (detection_tech_gene_tg) is not transcribed. A
    model configured with extra categoricals must resolve to no flat transcription
    rather than to one that silently drops the site."""
    module = reference_model.module
    model_obj = getattr(module.model, "_orig_mod", module.model)
    original = model_obj.n_extra_categoricals
    try:
        model_obj.n_extra_categoricals = [2]
        assert flat.log_joint_for(module) is None
    finally:
        model_obj.n_extra_categoricals = original
