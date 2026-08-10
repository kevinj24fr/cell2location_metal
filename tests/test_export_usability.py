"""export_posterior must work as its own docstring says it does.

The docstring tells users to put ``num_samples`` and ``batch_size`` in
``sample_kwargs``. The ``use_quantiles=True`` branch forwarded those kwargs verbatim
into ``posterior_quantile``, which does not sample and rejects ``num_samples`` -- so
following the documentation raised a TypeError on every device.
"""

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")


@pytest.fixture(scope="module")
def trained_spatial_model():
    from cell2location.models import Cell2location, RegressionModel

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=100, n_genes=60)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    reference = RegressionModel(adata)
    reference.train(max_epochs=2, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    cell_state_df = reference._compute_cluster_averages(key="labels")

    Cell2location.setup_anndata(adata, batch_key="batch")
    model = Cell2location(adata, cell_state_df=cell_state_df, N_cells_per_location=8, detection_alpha=20)
    model.train(max_epochs=2, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
    return model, adata


def test_quantile_export_accepts_documented_sample_kwargs(trained_spatial_model):
    model, adata = trained_spatial_model
    out = model.export_posterior(
        adata,
        use_quantiles=True,
        add_to_obsm=["q05", "q50", "q95"],
        sample_kwargs={"num_samples": 10, "batch_size": adata.n_obs},
    )
    assert "q50_cell_abundance_w_sf" in out.obsm
    assert np.isfinite(np.asarray(out.obsm["q50_cell_abundance_w_sf"])).all()


def test_sample_export_still_works(trained_spatial_model):
    model, adata = trained_spatial_model
    out = model.export_posterior(
        adata,
        sample_kwargs={"num_samples": 10, "batch_size": adata.n_obs},
    )
    assert "means_cell_abundance_w_sf" in out.obsm
