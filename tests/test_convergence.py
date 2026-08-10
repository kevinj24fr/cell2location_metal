"""The relative early-stopping callback: stop when the ELBO plateaus, never before
the floor, and never on noise within tolerance."""

import numpy as np
import pytest
import torch

from cell2location.accel._convergence import RelativeEarlyStopping


class _FakeTrainer:
    def __init__(self):
        self.callback_metrics = {}
        self.should_stop = False
        self.current_epoch = 0


def _run(callback, losses):
    trainer = _FakeTrainer()
    for epoch, loss in enumerate(losses):
        trainer.current_epoch = epoch
        trainer.callback_metrics = {"elbo_train": torch.tensor(float(loss))}
        callback.on_train_epoch_end(trainer, None)
        if trainer.should_stop:
            return epoch
    return None


def test_stops_on_plateau_after_patience():
    cb = RelativeEarlyStopping(rel_tol=1e-4, patience=10, min_epochs=5)
    losses = [1000.0 - i for i in range(20)] + [980.0] * 30
    stopped = _run(cb, losses)
    assert stopped is not None
    assert 29 <= stopped <= 31, f"stopped at {stopped}"


def test_never_stops_while_improving():
    cb = RelativeEarlyStopping(rel_tol=1e-4, patience=10, min_epochs=5)
    losses = [1000.0 * (0.999 ** i) for i in range(200)]
    assert _run(cb, losses) is None


def test_respects_min_epochs_even_on_flat_loss():
    """min_epochs counts completed epochs: with a flat loss the callback wants to
    stop immediately, and the floor holds it to 0-indexed epoch min_epochs - 1."""
    cb = RelativeEarlyStopping(rel_tol=1e-4, patience=3, min_epochs=50)
    stopped = _run(cb, [100.0] * 60)
    assert stopped == 49


def test_noise_within_tolerance_counts_as_plateau():
    rng = np.random.default_rng(0)
    cb = RelativeEarlyStopping(rel_tol=1e-3, patience=10, min_epochs=5)
    noisy_flat = 1000.0 + rng.normal(0, 0.05, 100)  # 5e-5 relative noise
    stopped = _run(cb, noisy_flat.tolist())
    assert stopped is not None and stopped <= 20


def test_missing_metric_is_ignored():
    cb = RelativeEarlyStopping(rel_tol=1e-4, patience=3, min_epochs=1)
    trainer = _FakeTrainer()
    trainer.callback_metrics = {}
    cb.on_train_epoch_end(trainer, None)  # must not raise
    assert not trainer.should_stop


def test_non_finite_losses_neither_stop_nor_become_best():
    """An inf first loss must not poison _best (every later improvement would look
    like plateau), and nan must not count toward patience."""
    cb = RelativeEarlyStopping(rel_tol=1e-4, patience=5, min_epochs=1)
    losses = [float("inf"), float("nan")] + [1000.0 - i for i in range(20)]
    assert _run(cb, losses) is None
    assert cb._best == pytest.approx(981.0)
