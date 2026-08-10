"""Vectorized posterior sampling for mean-field AutoNormal guides.

scvi's ``_get_posterior_samples`` draws each posterior sample with a full guide
trace in a Python loop -- a thousand traces through pyro's effect handlers, each a
burst of tiny kernels, which is why export runs at CPU speed even on a GPU. For a
mean-field ``AutoNormal`` guide the joint posterior factorizes over sites, so the
loop is equivalent to drawing every sample at once from each site's transformed
Normal: ``transform(loc + scale * eps)`` with a leading sample dimension.

Exactness boundary: this holds for ``AutoNormal`` (independent sites). Guides with
cross-site dependencies (hierarchical messengers, amortised encoders) must keep the
looped path; :class:`NotVectorizable` marks the boundary and callers fall back.
"""

import logging

import torch
from torch.distributions import biject_to

logger = logging.getLogger(__name__)

__all__ = ["NotVectorizable", "vectorized_posterior_samples"]


class NotVectorizable(Exception):
    """The guide's joint is not a product of per-site marginals we can read off."""


def _autonormal_site_params(guide):
    from pyro.infer.autoguide import AutoNormal
    from pyro.infer.autoguide.utils import deep_getattr

    if not isinstance(guide, AutoNormal):
        raise NotVectorizable(f"guide {type(guide).__name__} is not a mean-field AutoNormal")
    if getattr(guide, "prototype_trace", None) is None:
        raise NotVectorizable("guide has no prototype trace; train or call it once first")

    for name, site in guide.prototype_trace.iter_stochastic_nodes():
        if site.get("is_observed"):
            continue
        try:
            loc = deep_getattr(guide.locs, name)
            scale = deep_getattr(guide.scales, name)
        except AttributeError as exc:
            raise NotVectorizable(f"site {name}: {exc}") from exc
        yield name, loc, scale, biject_to(site["fn"].support)


def vectorized_posterior_samples(module, args, kwargs, num_samples=1000, return_sites=None):
    """All posterior samples in one batched draw per site.

    Returns ``{site: np.ndarray}`` with ``num_samples`` leading, matching the
    looped sampler's convention. ``args``/``kwargs`` are accepted for signature
    parity with the looped path; a mean-field guide's parameters do not depend on
    the minibatch, so they are unused here.
    """
    del args, kwargs
    guide = getattr(module, "guide", None)
    if guide is None:
        raise NotVectorizable("module has no guide")

    wanted = set(return_sites) if return_sites is not None else None
    samples = {}
    with torch.no_grad():
        for name, loc, scale, transform in _autonormal_site_params(guide):
            if wanted is not None and name not in wanted:
                continue
            eps = torch.randn((num_samples,) + tuple(loc.shape), device=loc.device, dtype=loc.dtype)
            samples[name] = transform(loc + scale * eps).cpu().numpy()
    if not samples:
        raise NotVectorizable("no guide sites matched the requested return_sites")
    return samples
