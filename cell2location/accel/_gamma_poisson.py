"""The training likelihood, routed through the acceleration layer on Metal.

cell2location's observation sites sample from ``pyro.distributions.GammaPoisson``,
whose ``log_prob`` is the negative-binomial log-likelihood in the ``(concentration,
rate)`` parameterisation -- computed with the same ``lgamma``-heavy arithmetic the
:mod:`cell2location.accel` layer guards and (optionally) fuses. Subclassing here is
what connects the main training loop to that layer; without it the fused Metal
kernel only ever accelerates code that calls ``log_nb_positive`` directly.
"""

import torch
from pyro.distributions import GammaPoisson as _PyroGammaPoisson

from ._ops import log_nb_positive

__all__ = ["GammaPoisson"]


class GammaPoisson(_PyroGammaPoisson):
    """``pyro.distributions.GammaPoisson`` with Metal-aware ``log_prob``.

    Off Metal this defers to pyro unchanged, so CPU and CUDA training stay
    bit-identical to upstream. On MPS the likelihood goes through
    :func:`cell2location.accel.log_nb_positive`: broadcast views never reach
    ``lgamma`` (the historic wrong-results shape), and the fused single-pass kernel
    is used when enabled and verified.
    """

    def log_prob(self, value):
        if value.device.type != "mps":
            return super().log_prob(value)
        if self._validate_args:
            self._validate_sample(value)
        theta = self.concentration
        return log_nb_positive(value, mu=theta / self.rate, theta=theta)

    def expand(self, batch_shape, _instance=None):
        # The base implementation hardcodes its own class when no instance is given,
        # and pyro plates expand every distribution -- without this override the
        # subclass (and the Metal routing with it) would be dropped at the sample site.
        new = self._get_checked_instance(GammaPoisson, _instance)
        return super().expand(batch_shape, _instance=new)
