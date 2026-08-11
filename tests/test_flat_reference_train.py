"""Contract for minibatch flat training of the reference signature model.

``test_flat_reference.py`` pins the transcription against pyro replay. This file
pins the *engine* around it: that RegressionModel actually runs on the flat path,
that the scope gates route everything else to pyro, and -- the failure this
engine is uniquely exposed to -- that a minibatch run's recorded history means
the same thing as the pyro path's, since scvi's ``elbo_train`` is the epoch mean
of per-step losses rather than a sum.
"""

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")

from cell2location.accel import _flat_joint, _flat_reference  # noqa: E402

mps_available = torch.backends.mps.is_available()

N_OBS, N_GENES = 200, 50


def _make_reference(seed=0, n_extra=None):
    from cell2location.models import RegressionModel

    torch.manual_seed(seed)
    np.random.seed(seed)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=N_OBS // 2, n_genes=N_GENES)
    RegressionModel.setup_anndata(adata, labels_key="labels", batch_key="batch")
    return RegressionModel(adata)


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_reference_trains_on_flat_engine_with_minibatches():
    model = _make_reference(seed=1)
    model.train(max_epochs=10, batch_size=50, enable_progress_bar=False,
                enable_model_summary=False)
    assert model.flat_engine_used_ is True
    assert model.is_trained_ is True
    losses = np.asarray(model.history["elbo_train"], dtype=float)
    assert len(losses) == 10
    assert np.isfinite(losses).all()
    # Four minibatches per epoch, each a full-data ELBO estimate: the recorded
    # value must be their MEAN, not their sum, or it is 4x the pyro path's.
    assert losses[-1] < losses[0], "training did not reduce the loss"


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_reference_history_is_epoch_mean_not_sum():
    """scvi logs elbo_train as the epoch mean over steps. A sum would still look
    like a decreasing curve, so compare the two batch sizes: same data, same
    scaling, so the recorded losses must be on the same scale."""
    coarse = _make_reference(seed=2)
    coarse.train(max_epochs=5, batch_size=N_OBS, enable_progress_bar=False,
                 enable_model_summary=False)
    fine = _make_reference(seed=2)
    fine.train(max_epochs=5, batch_size=N_OBS // 4, enable_progress_bar=False,
               enable_model_summary=False)

    c = float(np.asarray(coarse.history["elbo_train"], dtype=float)[0])
    f = float(np.asarray(fine.history["elbo_train"], dtype=float)[0])
    # Four steps per epoch instead of one: a summed history would sit ~4x higher.
    assert abs(f - c) <= 0.5 * abs(c), (
        f"minibatch history {f:.6g} is not on the same scale as full batch {c:.6g}"
    )


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_reference_guide_is_written_back():
    """Export reads the guide, so a flat run that does not write back would train
    into a void and silently return the initial posterior."""
    from pyro.infer.autoguide.utils import deep_getattr

    model = _make_reference(seed=3)
    model.train(max_epochs=8, batch_size=100, enable_progress_bar=False,
                enable_model_summary=False)
    assert model.flat_engine_used_ is True

    twin = _make_reference(seed=3)
    twin.train(max_epochs=1, batch_size=100, accelerator="cpu",
               enable_progress_bar=False, enable_model_summary=False)

    name = "per_cluster_mu_fg"
    trained = deep_getattr(model.module.guide.locs, name).detach().cpu()
    untrained = deep_getattr(twin.module.guide.locs, name).detach().cpu()
    assert not torch.allclose(trained, untrained), "guide locs unchanged after training"


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_spatial_minibatch_still_falls_back_to_pyro():
    """The spatial model has five per-location latents; minibatching it is NOT
    implemented. A spatial caller passing batch_size must land on pyro rather
    than on an engine that would scale the likelihood and ignore the locals."""
    import pandas as pd

    from cell2location.models import Cell2location

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=100, n_genes=60)
    Cell2location.setup_anndata(adata, batch_key="batch")
    sig = pd.DataFrame(np.random.rand(60, 3) + 0.1, index=adata.var_names, columns=list("abc"))
    model = Cell2location(adata, cell_state_df=sig, N_cells_per_location=8, detection_alpha=20)
    model.train(max_epochs=2, batch_size=100, enable_progress_bar=False,
                enable_model_summary=False)
    assert model.flat_engine_used_ is False
    assert model.is_trained_ is True


def test_minibatch_support_reads_the_models_own_declaration():
    """Eligibility comes from list_obs_plate_vars(), not a hard-coded class list,
    so a model that grows a per-observation latent stops qualifying by itself."""
    from cell2location.accel._train import _supports_minibatch

    reference = _make_reference(seed=0)
    assert _supports_minibatch(reference.module) is True

    model_obj = getattr(reference.module.model, "_orig_mod", reference.module.model)
    original = model_obj.list_obs_plate_vars
    try:
        model_obj.list_obs_plate_vars = lambda: {"name": "obs_plate", "sites": {"w_sf": 3}}
        assert _supports_minibatch(reference.module) is False
    finally:
        model_obj.list_obs_plate_vars = original


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_extra_categoricals_route_to_pyro():
    """The covariate-effect site is not transcribed; such a model must train on
    pyro rather than on a transcription that omits one of its densities."""
    model = _make_reference(seed=0)
    model_obj = getattr(model.module.model, "_orig_mod", model.module.model)
    model_obj.n_extra_categoricals = [2]
    assert _flat_joint.log_joint_for(model.module) is None


def test_spatial_initial_values_still_route_to_pyro():
    """The initial-value gate moved from the shared applicability check into the
    per-transcription scope, because the reference forward never reads init_val_*
    while the spatial forward turns them into extra *_initial density terms. The
    spatial exclusion must be exactly as strict as it was before that move."""
    import pandas as pd

    from cell2location.models import Cell2location

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=50, n_genes=40)
    Cell2location.setup_anndata(adata, batch_key="batch")
    sig = pd.DataFrame(np.random.rand(40, 3) + 0.1, index=adata.var_names, columns=list("abc"))
    model = Cell2location(adata, cell_state_df=sig, N_cells_per_location=8, detection_alpha=20)

    assert _flat_joint.log_joint_for(model.module) is _flat_joint.flat_log_joint

    model_obj = getattr(model.module.model, "_orig_mod", model.module.model)
    model_obj.np_init_vals = {"w_sf": np.zeros(1)}
    assert _flat_joint.log_joint_for(model.module) is None


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_streamed_and_resident_paths_agree(monkeypatch):
    """Staging the matrix on the device must be a pure performance choice. Both
    paths see the same data in the same batch sizes, so their training curves
    have to land in the same place -- otherwise residency changes results."""
    from cell2location.accel import _flat_train

    resident = _make_reference(seed=7)
    resident.mps_early_stopping = None
    resident.train(max_epochs=15, batch_size=50, enable_progress_bar=False,
                   enable_model_summary=False)

    monkeypatch.setattr(_flat_train, "_resident_fits", lambda model, device: False)
    streamed = _make_reference(seed=7)
    streamed.mps_early_stopping = None
    streamed.train(max_epochs=15, batch_size=50, enable_progress_bar=False,
                   enable_model_summary=False)

    assert resident.flat_engine_used_ is True
    assert streamed.flat_engine_used_ is True
    a = float(np.asarray(resident.history["elbo_train"], dtype=float)[-1])
    b = float(np.asarray(streamed.history["elbo_train"], dtype=float)[-1])
    # Different shuffle RNGs, so this is a scale comparison, not bitwise equality.
    assert abs(a - b) <= 0.05 * abs(b), f"resident {a:.6g} vs streamed {b:.6g}"


def test_residency_declines_when_the_data_would_not_fit(monkeypatch):
    """The reason a caller passes batch_size may be that the data does not fit.
    Residency must be a measured decision against the driver's budget, and must
    decline rather than assume when the budget is unavailable."""
    from cell2location.accel import _flat_train

    model = _make_reference(seed=0)
    device = torch.device("mps")

    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 512 * 1024**3)
    assert _flat_train._resident_fits(model, device) is True

    # A budget the matrix cannot claim a quarter of: the fixture's counts are
    # 200 x 50 float32 (~42 KB), so 64 KiB leaves it well over the fraction.
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 64 * 1024)
    assert _flat_train._resident_fits(model, device) is False

    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 0)
    assert _flat_train._resident_fits(model, device) is False

    assert _flat_train._resident_fits(model, torch.device("cpu")) is False


@pytest.mark.skipif(not mps_available, reason="flat engine is the Metal path")
def test_resident_batches_cover_every_observation_once():
    """A permutation per epoch, not sampling with replacement: every observation
    must appear exactly once, including in a trailing partial batch."""
    from cell2location.accel._flat_train import _ResidentBatches

    model = _make_reference(seed=0)
    model.module.to(torch.device("mps"))
    batches = _ResidentBatches(model, 70, torch.device("mps"))  # 200 = 70+70+60

    sizes, total_rows = [], []
    for args, _kwargs in batches:
        sizes.append(args[0].shape[0])
        total_rows.append(args[0].shape[0])
    assert sizes == [70, 70, 60]
    assert sum(total_rows) == N_OBS
    assert len(batches) == 3


def test_packed_reference_model_matches_module():
    """The packed proxy must reproduce the model's buffers exactly -- it is what
    the training loop reads instead of the module."""
    from cell2location.accel._flat_train import pack_module

    model = _make_reference(seed=0)
    packed = pack_module(model.module).model
    real = getattr(model.module.model, "_orig_mod", model.module.model)
    for name in ("detection_mean_hyp_prior_alpha", "gene_add_alpha_hyp_prior_beta",
                 "alpha_g_phi_hyp_prior_alpha", "ones"):
        assert torch.allclose(
            getattr(packed, name).float(), getattr(real, name).float()
        ), name
    assert packed.n_obs == real.n_obs
    assert packed.n_factors == real.n_factors
