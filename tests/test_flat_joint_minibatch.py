"""Contract for the spatial log-joint under a subsampled observation plate.

The spatial model has five per-location latents, so its plate scale multiplies
those latents' priors AND the likelihood -- not the likelihood alone, which is
the reference model's much simpler case. Getting that split wrong is invisible
at full batch and wrong by a constant factor everywhere else.

This pins the scaled form against pyro replay through a genuinely subsampled
plate at three batch sizes, before any training engine is built on it.
"""

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")

from cell2location.accel import _flat_joint as flat  # noqa: E402

N_OBS, N_GENES = 120, 40


@pytest.fixture(scope="module")
def spatial_model():
    import pandas as pd

    from cell2location.models import Cell2location

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=N_OBS // 2, n_genes=N_GENES)
    Cell2location.setup_anndata(adata, batch_key="batch")
    sig = pd.DataFrame(np.random.rand(N_GENES, 3) + 0.1,
                       index=adata.var_names, columns=list("abc"))
    model = Cell2location(adata, cell_state_df=sig, N_cells_per_location=8, detection_alpha=20)
    model.train(max_epochs=2, accelerator="cpu",
                enable_progress_bar=False, enable_model_summary=False)
    return model


def _local_sites(module):
    mod = getattr(module.model, "_orig_mod", module.model)
    return set(mod.list_obs_plate_vars()["sites"])


def _full_args(model):
    from scvi.dataloaders import AnnDataLoader

    dl = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=model.adata.n_obs)
    return model.module._get_fn_args_from_batch(next(iter(dl)))


def _subsample(args, rows):
    """The batch's view of the data: x_data, idx and batch_index all row-sliced."""
    x_data, idx, batch_index = args
    return (x_data[rows], idx[rows], batch_index[rows])


def _trace_from_dict(latents):
    from pyro.poutine.trace_struct import Trace

    trace = Trace()
    for name, value in latents.items():
        trace.add_node(name, type="sample", value=value, is_observed=False, infer={})
    return trace


def _pyro_log_joint(module, args, latents):
    import pyro

    replayed = pyro.poutine.trace(
        pyro.poutine.replay(module.model, trace=_trace_from_dict(latents))
    ).get_trace(*args)
    return float(replayed.log_prob_sum())


BATCH_SIZES = [N_OBS, N_OBS // 2, N_OBS // 4]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seed", range(3))
def test_scaled_log_joint_matches_subsampled_pyro(spatial_model, batch_size, seed):
    module = spatial_model.module
    args, kwargs = _full_args(spatial_model)
    local = _local_sites(module)

    torch.manual_seed(seed)
    full_latents = flat.sample_latents_from_guide(module, args, kwargs)

    rows = torch.arange(batch_size)
    batch_args = _subsample(args, rows)
    # Local latents are subsampled with the data; global ones are not.
    batch_latents = {
        name: (value[rows] if name in local else value)
        for name, value in full_latents.items()
    }
    scale = N_OBS / batch_size

    # pyro's own scaling, obtained by replaying the model through a plate whose
    # subsample is exactly these rows.
    reference = _pyro_log_joint(module, batch_args, batch_latents)
    result = float(flat.flat_log_joint(module, batch_args, kwargs, batch_latents,
                                       plate_scale=scale))

    assert abs(result - reference) <= 1e-4 * abs(reference), (
        f"batch {batch_size}, seed {seed}: flat {result:.6f} vs pyro {reference:.6f}"
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_scaled_log_joint_gradients_match(spatial_model, batch_size):
    """Gradients are what training consumes, and the plate scale multiplies the
    local block's gradients too."""
    import pyro

    module = spatial_model.module
    args, kwargs = _full_args(spatial_model)
    local = _local_sites(module)

    torch.manual_seed(0)
    full_latents = flat.sample_latents_from_guide(module, args, kwargs, requires_grad=True)
    rows = torch.arange(batch_size)
    batch_args = _subsample(args, rows)
    batch_latents = {
        name: (value[rows] if name in local else value)
        for name, value in full_latents.items()
    }
    scale = N_OBS / batch_size

    names = [n for n, v in batch_latents.items() if v.requires_grad]
    tensors = [batch_latents[n] for n in names]

    value = flat.flat_log_joint(module, batch_args, kwargs, batch_latents, plate_scale=scale)
    flat_grads = torch.autograd.grad(value, tensors, allow_unused=True, retain_graph=True)

    replayed = pyro.poutine.trace(
        pyro.poutine.replay(module.model, trace=_trace_from_dict(batch_latents))
    ).get_trace(*batch_args)
    pyro_grads = torch.autograd.grad(replayed.log_prob_sum(), tensors, allow_unused=True)

    for name, fg, pg in zip(names, flat_grads, pyro_grads):
        if fg is None and pg is None:
            continue
        assert fg is not None and pg is not None, name
        ref = pg.abs().max().clamp_min(1e-6)
        assert (fg - pg).abs().max() <= 1e-4 * ref, (
            f"{name} at batch {batch_size}: max gradient difference "
            f"{(fg - pg).abs().max():.3e} exceeds {1e-4 * ref:.3e}"
        )
