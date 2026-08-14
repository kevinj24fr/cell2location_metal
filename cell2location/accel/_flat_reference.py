"""Flat log-joint for the reference signature model (``RegressionModel``).

Why this is separate from ``_flat_joint.py``
--------------------------------------------
That module transcribes the *spatial* model. This one transcribes
``RegressionBackgroundDetectionTechPyroModel`` -- step 1 of every cell2location
workflow, which until now trained entirely through pyro's effect handlers while
the spatial model had a flat engine.

The two models differ in a way that matters for the engine, not just for the
arithmetic. The spatial model has five *per-location* latents, which is why its
flat engine is full-batch only: minibatching would have to subsample the guide
in lockstep with the data. This model's ``list_obs_plate_vars()["sites"]`` is
empty -- **every one of its nine latents is global** (per-gene, per-factor,
per-experiment). Only the data lives in the observation plate.

That makes minibatching here a scaling problem rather than an indexing one, and
it has to be handled: ``RegressionModel.train`` defaults to ``batch_size=2500``
because real references are large (a 675k-cell reference is ordinary), so a
full-batch-only transcription would be useless for this model.

pyro's ``plate(size=n_obs, subsample=idx)`` scales the log-density of everything
inside it by ``n_obs / len(idx)``. Only the likelihood is inside. The guide is
untouched -- its sites are all global, so ``flat_log_q`` needs no scale, and the
minibatch ELBO is the usual ``log p(global) + (N/B) sum_batch log p(x|global)
- log q(global)``.

Scope: no extra categorical covariates. When ``n_extra_categoricals`` is set the
model gains a ``detection_tech_gene_tg`` site and a multiplicative term that are
deliberately not transcribed here; ``log_joint_for`` resolves such a module to
``None`` so the caller falls back to pyro rather than silently dropping a site.
"""

import torch

from ._flat_joint import _exponential_lp, _gamma_lp, _nb_lp, _stable_alpha


def reference_log_joint(module, args, kwargs, latents, plate_scale=None):
    """log p(latents, data) for the reference model, matching pyro replay exactly.

    Includes the observation plate's ``n_obs / batch`` scale on the likelihood,
    so this is correct at full batch and at any minibatch size. ``plate_scale``
    defaults to computing that ratio from the batch; the shared minibatch runner
    passes it explicitly so both models' transcriptions take the same argument.
    """
    del kwargs
    x_data, _idx, batch_index, label_index, _extra_categoricals = args
    mod = module.model
    mod = getattr(mod, "_orig_mod", mod)  # unwrap torch.compile
    L = latents

    obs2sample = torch.nn.functional.one_hot(
        batch_index.squeeze(-1).long(), num_classes=mod.n_batch
    ).to(x_data.dtype)
    obs2label = torch.nn.functional.one_hot(
        label_index.squeeze(-1).long(), num_classes=mod.n_factors
    ).to(x_data.dtype)
    ones = mod.ones

    lp = x_data.new_zeros(())

    # --- per-cluster average mRNA count ---
    lp = lp + _gamma_lp(L["per_cluster_mu_fg"], ones, ones)

    # --- cell-specific detection efficiency ---
    lp = lp + _gamma_lp(
        L["detection_mean_y_e"],
        ones * mod.detection_mean_hyp_prior_alpha,
        ones * mod.detection_mean_hyp_prior_beta,
    )
    detection_y_c = obs2sample @ L["detection_mean_y_e"]

    # --- gene-specific additive component (ambient / free-floating RNA) ---
    lp = lp + _gamma_lp(
        L["s_g_gene_add_alpha_hyp"],
        ones * mod.gene_add_alpha_hyp_prior_alpha,
        ones * mod.gene_add_alpha_hyp_prior_beta,
    )
    lp = lp + _gamma_lp(
        L["s_g_gene_add_mean"],
        mod.gene_add_mean_hyp_prior_alpha,
        mod.gene_add_mean_hyp_prior_beta,
    )
    lp = lp + _exponential_lp(L["s_g_gene_add_alpha_e_inv"], L["s_g_gene_add_alpha_hyp"])
    s_g_alpha_e = ones / L["s_g_gene_add_alpha_e_inv"].pow(2)
    lp = lp + _gamma_lp(L["s_g_gene_add"], s_g_alpha_e, s_g_alpha_e / L["s_g_gene_add_mean"])

    # --- gene-specific overdispersion ---
    lp = lp + _gamma_lp(
        L["alpha_g_phi_hyp"],
        ones * mod.alpha_g_phi_hyp_prior_alpha,
        ones * mod.alpha_g_phi_hyp_prior_beta,
    )
    lp = lp + _exponential_lp(L["alpha_g_inverse"], L["alpha_g_phi_hyp"])

    # --- data likelihood, scaled by the observation plate ---
    alpha = _stable_alpha(L["alpha_g_inverse"], ones)
    mu = (obs2label @ L["per_cluster_mu_fg"] + obs2sample @ L["s_g_gene_add"]) * detection_y_c

    if plate_scale is None:
        plate_scale = float(mod.n_obs) / float(x_data.shape[0])
    lp = lp + plate_scale * _nb_lp(x_data, mu, alpha)

    return lp


def supports(mod) -> bool:
    """True when this transcription covers the given pyro model object."""
    if type(mod).__name__ != "RegressionBackgroundDetectionTechPyroModel":
        return False
    # The covariate-effect site is not transcribed; see the module docstring.
    return getattr(mod, "n_extra_categoricals", None) is None
