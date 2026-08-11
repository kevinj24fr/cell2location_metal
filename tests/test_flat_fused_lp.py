"""Contract: the flat engine's data likelihood routed through the fused NB Metal
kernel must match the eager flat path (which test_flat_joint.py pins against pyro
replay). GammaPoisson(alpha, alpha/mu) == NB(mu, theta=alpha), so the kernel gets
(x, mu, alpha) directly. The kernel self-verifies against eager at first use; this
test pins the ROUTING -- values and gradients through the whole flat log-joint,
and a probe proving the fused path actually executed (no vacuous pass)."""

import numpy as np
import pytest
import torch

scvi_data = pytest.importorskip("scvi.data")

from cell2location.accel import _flat_joint as flat  # noqa: E402
from cell2location.accel import _flat_train as flat_train  # noqa: E402
from cell2location.accel import _fused_nb  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="fused kernel is MPS-only")


@pytest.fixture(scope="module")
def mps_setup():
    import pandas as pd

    from cell2location.models import Cell2location

    torch.manual_seed(0)
    np.random.seed(0)
    adata = scvi_data.synthetic_iid(n_labels=3, batch_size=100, n_genes=60)
    Cell2location.setup_anndata(adata, batch_key="batch")
    sig = pd.DataFrame(np.random.rand(60, 3) + 0.1, index=adata.var_names, columns=list("abc"))
    model = Cell2location(adata, cell_state_df=sig, N_cells_per_location=8, detection_alpha=20)
    device = torch.device("mps")
    model.module.to(device)
    args, kwargs = flat_train._full_batch_args(model, device)
    with torch.no_grad():
        model.module.guide(*args, **kwargs)
    return model, args, kwargs


def test_fused_routing_matches_eager_flat(mps_setup, monkeypatch):
    model, args, kwargs = mps_setup
    probe = torch.rand(4, 8, device="mps") + 0.5
    if _fused_nb.fused_log_nb_positive(probe.round(), probe, probe) is None:
        pytest.skip("fused NB kernel unavailable on this machine")

    state = flat_train.FlatGuideState.from_guide(model.module.guide)
    torch.manual_seed(9)
    eps = torch.randn_like(state.loc)

    calls = []
    real = _fused_nb.fused_log_nb_positive

    def recording(*a, **k):
        out = real(*a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(_fused_nb, "fused_log_nb_positive", recording)

    _fused_nb.enable_fused_nb()
    loss_fused = flat_train.flat_training_loss(model.module, state, args, kwargs, eps)
    grads_fused = torch.autograd.grad(loss_fused, [state.loc, state.rho])
    assert calls and calls[-1] is True, "flat loss did not route through the fused kernel"
    monkeypatch.undo()

    _fused_nb.disable_fused_nb()
    try:
        loss_eager = flat_train.flat_training_loss(model.module, state, args, kwargs, eps)
        grads_eager = torch.autograd.grad(loss_eager, [state.loc, state.rho])
    finally:
        _fused_nb.enable_fused_nb()

    assert abs(float(loss_fused) - float(loss_eager)) <= 1e-5 * abs(float(loss_eager))
    for gf, ge in zip(grads_fused, grads_eager):
        assert (gf - ge).abs().max() <= 1e-3 * ge.abs().max().clamp_min(1e-6)
