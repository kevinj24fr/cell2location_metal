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

import torch

__all__ = ["sample_latents_from_guide", "flat_log_joint"]


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


def flat_log_joint(module, args, kwargs, latents):
    """log p(latents, data) for the spatial model, matching pyro replay exactly."""
    del kwargs
    x_data, idx, batch_index = args
    mod = module.model
    mod = getattr(mod, "_orig_mod", mod)  # unwrap torch.compile
    L = latents

    obs2sample = torch.nn.functional.one_hot(
        batch_index.squeeze(-1).long(), num_classes=mod.n_batch
    ).to(x_data.dtype)
    ones = mod.ones

    lp = x_data.new_zeros(())

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
    lp = lp + _gamma_lp(
        L["n_s_cells_per_location"],
        mod.N_cells_per_location * mod.N_cells_mean_var_ratio,
        mod.N_cells_mean_var_ratio,
    )
    lp = lp + _gamma_lp(L["b_s_groups_per_location"], mod.B_groups_per_location, ones)

    shape = mod.ones_1_n_groups * L["b_s_groups_per_location"] / mod.n_groups_tensor
    rate = mod.ones_1_n_groups / (L["n_s_cells_per_location"] / L["b_s_groups_per_location"])
    lp = lp + _gamma_lp(L["z_sr_groups_factors"], shape, rate)

    lp = lp + _gamma_lp(L["k_r_factors_per_groups"], mod.factors_per_groups, ones)
    lp = lp + _gamma_lp(
        L["x_fr_group2fact"],
        L["k_r_factors_per_groups"] / mod.n_factors_tensor,
        L["k_r_factors_per_groups"],
    )

    w_sf_mu = L["z_sr_groups_factors"] @ L["x_fr_group2fact"]
    lp = lp + _gamma_lp(
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
    lp = lp + _gamma_lp(
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
    lp = lp + _gamma_poisson_lp(x_data, alpha, alpha / mu)

    return lp
