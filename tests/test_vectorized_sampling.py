"""The vectorized posterior sampler must reproduce the looped sampler's statistics.

For a mean-field AutoNormal guide the joint factorizes over sites, so drawing all
samples at once from each site's transformed Normal is the same distribution the
1000-iteration loop draws from -- just shaped as one batch.
"""

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")

from cell2location.accel._sampling import NotVectorizable, vectorized_posterior_samples


@pytest.fixture(scope="module")
def trained_model():
    from cell2location.models import RegressionModel

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=100, n_genes=60)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    model = RegressionModel(adata)
    model.train(max_epochs=5, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    return model


def _batch_args(model):
    from scvi.dataloaders import AnnDataLoader

    dl = AnnDataLoader(model.adata_manager, shuffle=False, batch_size=model.adata.n_obs)
    return model.module._get_fn_args_from_batch(next(iter(dl)))


def test_vectorized_matches_looped_statistics(trained_model):
    args, kwargs = _batch_args(trained_model)
    torch.manual_seed(0)
    fast = vectorized_posterior_samples(trained_model.module, args, kwargs, num_samples=800)
    torch.manual_seed(0)
    slow = trained_model._get_posterior_samples(args, kwargs, num_samples=200, show_progress=False)

    shared = set(fast) & set(slow)
    assert shared, "no common sites between samplers"
    for site in shared:
        f, s = np.asarray(fast[site]), np.asarray(slow[site])
        assert f.shape[1:] == s.shape[1:], site
        # Same distribution, independent draws: means agree within joint MC error.
        f_mean, s_mean = f.mean(0), s.mean(0)
        f_sd = f.std(0)
        tol = 6 * f_sd / np.sqrt(200) + 1e-3 + 0.02 * np.abs(f_mean)
        assert (np.abs(f_mean - s_mean) <= tol).mean() > 0.98, site


def test_unknown_guide_raises_not_vectorizable(trained_model):
    class _FakeModule:
        guide = object()

    args, kwargs = _batch_args(trained_model)
    with pytest.raises(NotVectorizable):
        vectorized_posterior_samples(_FakeModule(), args, kwargs, num_samples=8)


def test_return_sites_filters(trained_model):
    args, kwargs = _batch_args(trained_model)
    sites = list(vectorized_posterior_samples(trained_model.module, args, kwargs, num_samples=4))
    chosen = sites[:1]
    out = vectorized_posterior_samples(trained_model.module, args, kwargs, num_samples=4,
                                       return_sites=chosen)
    assert list(out) == chosen
