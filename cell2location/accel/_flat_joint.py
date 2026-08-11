"""The Cell2location joint log-density as plain tensor code (engine task #13).

A hand transcription of ``LocationModelLinearDependentWMultiExperimentLocation-
BackgroundNormLevelGeneAlphaPyroModel.forward`` with pyro's effect handlers
removed: every ``pyro.sample`` site becomes an explicit log-density term. The
transcription is trusted only because ``tests/test_flat_joint.py`` pins both the
value and the per-latent gradients against pyro's replayed ``log_prob_sum`` --
any edit here must keep that contract green.

Scope (matches the contract fixture): full-batch evaluation (obs-plate scale 1),
``training_wo_observed=False``, dropout off. Minibatch plate scaling and the
initial-value observation branches are deliberately out of scope until needed.
"""

import math

import torch

__all__ = [
    "sample_latents_from_guide",
    "flat_log_joint",
    "sample_unconstrained_from_guide",
    "constrain_latents",
    "flat_log_q",
    "flat_elbo",
]


def _gamma_lp(x, concentration, rate):
    return (
        concentration * torch.log(rate)
        - torch.lgamma(concentration)
        + (concentration - 1.0) * torch.log(x)
        - rate * x
    ).sum()


def _exponential_lp(x, rate):
    return (torch.log(rate) - rate * x).sum()


def _gamma_poisson_lp(x, concentration, rate):
    return (
        concentration * torch.log(rate)
        - (concentration + x) * torch.log1p(rate)
        + torch.lgamma(concentration + x)
        - torch.lgamma(concentration)
        - torch.lgamma(x + 1.0)
    ).sum()


def _nb_lp(x, mu, alpha):
    """Data likelihood: GammaPoisson(alpha, alpha/mu) == NB(mu, theta=alpha).

    Routes through the self-verifying fused Metal kernel when available (the
    eager expression materialises ~a dozen full-size intermediates; the kernel
    reads each operand once). The eager fallback is byte-for-byte the expression
    the CPU contract pins against pyro replay."""
    from ._fused_nb import fused_log_nb_positive

    fused = fused_log_nb_positive(x, mu, alpha)
    if fused is not None:
        return fused.sum()
    return _gamma_poisson_lp(x, alpha, alpha / mu)


def sample_latents_from_guide(module, args, kwargs, requires_grad: bool = False):
    """One draw of every latent from the mean-field guide, as leaf tensors."""
    from ._sampling import _autonormal_site_params

    del args, kwargs
    latents = {}
    with torch.no_grad():
        for name, loc, scale, transform in _autonormal_site_params(module.guide):
            eps = torch.randn(loc.shape, device=loc.device, dtype=loc.dtype)
            value = transform(loc + scale * eps)
            latents[name] = value.detach().clone().requires_grad_(requires_grad)
    return latents


def sample_unconstrained_from_guide(module, requires_grad: bool = False):
    """One unconstrained draw u = loc + scale*eps per guide site, as leaf tensors."""
    from ._sampling import _autonormal_site_params

    unconstrained = {}
    with torch.no_grad():
        for name, loc, scale, _transform in _autonormal_site_params(module.guide):
            eps = torch.randn(loc.shape, device=loc.device, dtype=loc.dtype)
            u = loc + scale * eps
            unconstrained[name] = u.detach().clone().requires_grad_(requires_grad)
    return unconstrained


def constrain_latents(module, unconstrained):
    """Constrained latents z = transform(u), graph intact for autograd."""
    from ._sampling import _autonormal_site_params

    return {
        name: transform(unconstrained[name])
        for name, _loc, _scale, transform in _autonormal_site_params(module.guide)
    }


def flat_log_q(module, unconstrained):
    """log q(z) of the mean-field guide at the draw, from u directly (no inverses):
    sum over sites of Normal(loc, scale).log_prob(u) - log|det J_transform(u)|."""
    from ._sampling import _autonormal_site_params

    log_q = None
    for name, loc, scale, transform in _autonormal_site_params(module.guide):
        u = unconstrained[name]
        normal_lp = (
            -0.5 * ((u - loc) / scale).pow(2) - torch.log(scale) - 0.5 * math.log(2.0 * math.pi)
        ).sum()
        term = normal_lp - transform.log_abs_det_jacobian(u, transform(u)).sum()
        log_q = term if log_q is None else log_q + term
    return log_q


def flat_elbo(module, args, kwargs, unconstrained):
    """Single-particle ELBO estimate at the draw: flat_log_joint - flat_log_q."""
    latents = constrain_latents(module, unconstrained)
    return flat_log_joint(module, args, kwargs, latents) - flat_log_q(module, unconstrained)


#: The pyro model ``flat_log_joint`` below transcribes, by class name.
_SPATIAL_MODEL = (
    "LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel"
)


def log_joint_for(module):
    """The flat transcription matching this module, or None if there isn't one.

    Resolution is by pyro model type, deliberately. ``_flat_train_if_applicable``
    lives on ``AppleSiliconTrainMixin``, which BOTH ``Cell2location`` and
    ``RegressionModel`` inherit, so an engine that assumed one model would train
    the other against the wrong density -- silently, since the shapes broadcast.
    An unknown module resolves to None and the caller falls back to pyro.
    """
    from ._flat_reference import reference_log_joint, supports as _reference_supports

    mod = getattr(module, "model", None)
    if mod is None:
        return None
    mod = getattr(mod, "_orig_mod", mod)  # unwrap torch.compile
    name = type(mod).__name__
    if name == _SPATIAL_MODEL:
        return flat_log_joint if _spatial_supports(mod) else None
    if _reference_supports(mod):
        return reference_log_joint
    return None


def _spatial_supports(mod) -> bool:
    """Scope of the spatial transcription below.

    Excludes models carrying initial values: the spatial ``forward`` emits extra
    ``*_initial`` Gamma terms built from its ``init_val_*`` buffers, which this
    transcription does not carry. (The reference model registers the same buffers
    but never reads them in ``forward`` -- there they only initialize the guide --
    so that model is not excluded on these grounds.)
    """
    return getattr(mod, "np_init_vals", None) is None


def flat_log_joint(module, args, kwargs, latents, plate_scale=1.0):
    """log p(latents, data) for the spatial model, matching pyro replay exactly.

    ``plate_scale`` is the observation plate's n_obs/batch factor. pyro applies
    it to everything inside the plate, which here is five per-location latents'
    priors AND the likelihood -- not the likelihood alone, as it is for the
    reference model whose latents are all global. Terms are accumulated in two
    running sums so the scale multiplies exactly the local block. At the default
    1.0 this is the full-batch expression the contract pins against pyro replay.
    """
    del kwargs
    x_data, idx, batch_index = args
    mod = module.model
    mod = getattr(mod, "_orig_mod", mod)  # unwrap torch.compile
    L = latents

    obs2sample = torch.nn.functional.one_hot(
        batch_index.squeeze(-1).long(), num_classes=mod.n_batch
    ).to(x_data.dtype)
    ones = mod.ones

    lp = x_data.new_zeros(())        # global sites
    local = x_data.new_zeros(())     # sites inside the observation plate

    # --- gene technology scaling m_g ---
    lp = lp + _gamma_lp(
        L["m_g_mean"],
        mod.m_g_mu_mean_var_ratio_hyp * mod.m_g_mu_hyp,
        mod.m_g_mu_mean_var_ratio_hyp,
    )
    lp = lp + _exponential_lp(L["m_g_alpha_e_inv"], mod.m_g_alpha_hyp_mean)
    m_g_alpha_e = ones / L["m_g_alpha_e_inv"].pow(2)
    lp = lp + _gamma_lp(L["m_g"], m_g_alpha_e, m_g_alpha_e / L["m_g_mean"])

    # --- cells and groups per location ---
    local = local + _gamma_lp(
        L["n_s_cells_per_location"],
        mod.N_cells_per_location * mod.N_cells_mean_var_ratio,
        mod.N_cells_mean_var_ratio,
    )
    local = local + _gamma_lp(L["b_s_groups_per_location"], mod.B_groups_per_location, ones)

    shape = mod.ones_1_n_groups * L["b_s_groups_per_location"] / mod.n_groups_tensor
    rate = mod.ones_1_n_groups / (L["n_s_cells_per_location"] / L["b_s_groups_per_location"])
    local = local + _gamma_lp(L["z_sr_groups_factors"], shape, rate)

    lp = lp + _gamma_lp(L["k_r_factors_per_groups"], mod.factors_per_groups, ones)
    lp = lp + _gamma_lp(
        L["x_fr_group2fact"],
        L["k_r_factors_per_groups"] / mod.n_factors_tensor,
        L["k_r_factors_per_groups"],
    )

    w_sf_mu = L["z_sr_groups_factors"] @ L["x_fr_group2fact"]
    local = local + _gamma_lp(
        L["w_sf"], w_sf_mu * mod.w_sf_mean_var_ratio_tensor, mod.w_sf_mean_var_ratio_tensor
    )

    # --- detection efficiency ---
    lp = lp + _gamma_lp(
        L["detection_mean_y_e"],
        ones * mod.detection_mean_hyp_prior_alpha,
        ones * mod.detection_mean_hyp_prior_beta,
    )
    detection_hyp_prior_alpha = mod.ones_n_batch_1 * mod.detection_hyp_prior_alpha
    alpha_s = obs2sample @ detection_hyp_prior_alpha
    local = local + _gamma_lp(
        L["detection_y_s"], alpha_s, alpha_s / (obs2sample @ L["detection_mean_y_e"])
    )

    # --- ambient / additive RNA ---
    lp = lp + _gamma_lp(
        L["s_g_gene_add_alpha_hyp"],
        ones * mod.gene_add_alpha_hyp_prior_alpha,
        ones * mod.gene_add_alpha_hyp_prior_beta,
    )
    lp = lp + _gamma_lp(
        L["s_g_gene_add_mean"], mod.gene_add_mean_hyp_prior_alpha, mod.gene_add_mean_hyp_prior_beta
    )
    lp = lp + _exponential_lp(L["s_g_gene_add_alpha_e_inv"], L["s_g_gene_add_alpha_hyp"])
    s_g_alpha_e = ones / L["s_g_gene_add_alpha_e_inv"].pow(2)
    lp = lp + _gamma_lp(L["s_g_gene_add"], s_g_alpha_e, s_g_alpha_e / L["s_g_gene_add_mean"])

    # --- overdispersion ---
    lp = lp + _gamma_lp(
        L["alpha_g_phi_hyp"],
        ones * mod.alpha_g_phi_hyp_prior_alpha,
        ones * mod.alpha_g_phi_hyp_prior_beta,
    )
    lp = lp + _exponential_lp(L["alpha_g_inverse"], L["alpha_g_phi_hyp"])

    # --- data likelihood ---
    mu = ((L["w_sf"] @ mod.cell_state) * L["m_g"] + (obs2sample @ L["s_g_gene_add"])) * L[
        "detection_y_s"
    ]
    alpha = obs2sample @ (ones / L["alpha_g_inverse"].pow(2))
    local = local + _nb_lp(x_data, mu, alpha)

    return lp + plate_scale * local
