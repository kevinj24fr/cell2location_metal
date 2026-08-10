"""Stop training when the ELBO has genuinely plateaued.

Upstream trains a fixed ``max_epochs=30000`` with no stopping criterion; on typical
data the ELBO plateaus far earlier and the remaining steps buy nothing. Lightning's
own ``EarlyStopping`` uses an absolute ``min_delta``, which is meaningless for a
loss of order 1e8 -- this callback uses a *relative* criterion: stop when the best
loss has not improved by ``rel_tol`` (relative) within the last ``patience``
epochs, and never before ``min_epochs``.
"""

import logging

__all__ = ["RelativeEarlyStopping", "EARLY_STOP_ENV_VAR"]

logger = logging.getLogger(__name__)

#: Set to 0 to disable convergence-based early stopping on Metal runs.
EARLY_STOP_ENV_VAR = "CELL2LOCATION_MPS_EARLY_STOP"


class RelativeEarlyStopping:
    """Lightning-compatible callback; duck-typed so tests need no Trainer."""

    def __init__(self, rel_tol: float = 1e-4, patience: int = 500, min_epochs: int = 1000,
                 monitor: str = "elbo_train"):
        self.rel_tol = rel_tol
        self.patience = patience
        self.min_epochs = min_epochs
        self.monitor = monitor
        self._best = None
        self._epochs_since_improvement = 0
        self.stopped_epoch = None

    def on_train_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        loss = float(metric)

        if self._best is None or self._best - loss > self.rel_tol * abs(self._best):
            self._best = loss
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1

        if trainer.current_epoch + 1 < self.min_epochs:
            return
        if self._epochs_since_improvement >= self.patience:
            trainer.should_stop = True
            self.stopped_epoch = trainer.current_epoch
            logger.info(
                "Converged: no relative ELBO improvement above %.1e for %d epochs; "
                "stopping at epoch %d.",
                self.rel_tol, self.patience, trainer.current_epoch,
            )

    def as_callback(self):
        """Wrap into a real Lightning Callback (imported lazily)."""
        from lightning.pytorch.callbacks import Callback

        stopper = self

        class _RelativeEarlyStoppingCallback(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                stopper.on_train_epoch_end(trainer, pl_module)

        return _RelativeEarlyStoppingCallback()
