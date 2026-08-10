"""Optional CoreML export of the amortised guide encoder, targeting the Neural Engine.

**Read this before expecting a speedup.**

The Apple Neural Engine is not a general-purpose accelerator. It is reachable only
through CoreML, it runs inference only (no autograd, no optimiser state), it is
float16 throughout, and it wants static shapes. cell2location's training loop is
stochastic variational inference over a Pyro graph with sampling statements,
data-dependent plates and a full backward pass -- essentially none of which the ANE
can express. **Training will always run on the GPU (Metal/MPS), never on the ANE.**

There is exactly one component of the model that is a genuine ANE candidate: when
``amortised=True``, the guide approximates location-specific parameters with a plain
feed-forward network::

    x  ->  encoder (FCLayers)  ->  hidden  ->  Linear -> loc
                                           \\-> Linear -> scale

That subgraph is static, dense and inference-only, which is precisely the ANE's
sweet spot. Exporting it lets ``export_posterior`` / ``posterior_quantile`` run the
encoder on the Neural Engine while the GPU stays free -- worthwhile when you are
scoring many slides against an already-trained model, and irrelevant otherwise.

Requires ``coremltools`` (``pip install coremltools``), which is macOS-only in
practice. Everything here degrades to a clear error elsewhere.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "coremltools_available",
    "AmortisedSiteEncoder",
    "extract_site_encoder",
    "export_site_encoder",
    "verify_coreml_parity",
    "CoreMLSiteEncoder",
]


def coremltools_available() -> bool:
    try:
        import coremltools  # noqa: F401

        return True
    except ImportError:
        return False


def _require_coremltools():
    if not coremltools_available():
        raise ImportError(
            "coremltools is required for Neural Engine export. Install it with "
            "`pip install coremltools`. It is only meaningfully supported on macOS."
        )
    import coremltools as ct

    return ct


def _deep_getattr(obj, name: str):
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


class AmortisedSiteEncoder(torch.nn.Module):
    """The exportable feed-forward slice of an amortised guide, for one site.

    Composes ``encoder -> (hidden2locs, hidden2scales)`` into a single module with a
    plain ``forward(x) -> (loc, scale_unconstrained)`` signature that CoreML can trace.

    Note that ``scale`` is returned *before* the softplus + init-scale offset that the
    guide applies, so the caller reproduces the guide's exact arithmetic in float32
    rather than baking a float16 softplus into the graph.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        hidden2loc: torch.nn.Module,
        hidden2scale: torch.nn.Module,
        log1p_input: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden2loc = hidden2loc
        self.hidden2scale = hidden2scale
        self.log1p_input = log1p_input

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.log1p_input:
            x = torch.log1p(x)
        hidden = self.encoder(x)
        return self.hidden2loc(hidden), self.hidden2scale(hidden)


def extract_site_encoder(
    guide,
    site_name: str,
    log1p_input: bool = True,
) -> AmortisedSiteEncoder:
    """Pull the encoder subgraph for ``site_name`` out of a trained amortised guide.

    Parameters
    ----------
    guide
        A trained ``AutoAmortisedHierarchicalNormalMessenger`` (i.e.
        ``model.module.guide`` after ``train()`` with ``amortised=True``).
    site_name
        Name of the latent site, e.g. ``"w_sf"`` or ``"detection_y_s"``.
    log1p_input
        Whether the guide applies ``log1p`` to the raw counts before the encoder.
        Matches the default ``data_transform="log1p"``.
    """
    if hasattr(guide, "multiple_encoders") and hasattr(guide.multiple_encoders, site_name):
        encoder = _deep_getattr(guide.multiple_encoders, site_name)
    elif hasattr(guide, "one_encoder"):
        encoder = guide.one_encoder
    else:
        raise AttributeError(
            "The guide has no encoder to export. Neural Engine export only applies to "
            "models trained with amortised=True; a non-amortised guide has no neural network."
        )

    try:
        hidden2loc = _deep_getattr(guide.hidden2locs, site_name)
        hidden2scale = _deep_getattr(guide.hidden2scales, site_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Site {site_name!r} has no hidden2locs/hidden2scales head. Available sites: "
            f"{list(getattr(guide, 'amortised_plate_sites', {}).get('sites', {}).keys())}"
        ) from exc

    module = AmortisedSiteEncoder(encoder, hidden2loc, hidden2scale, log1p_input=log1p_input)
    return module.eval().float().cpu()


def export_site_encoder(
    guide,
    site_name: str,
    n_genes: int,
    batch_sizes: Sequence[int] = (1, 128, 1024, 2048),
    output_path: Optional[str] = None,
    compute_units: str = "ALL",
    precision: str = "float16",
    log1p_input: bool = True,
    minimum_deployment_target: Optional[str] = "macOS14",
):
    """Trace and convert an amortised site encoder to a CoreML package.

    Parameters
    ----------
    n_genes
        Encoder input width. Must equal ``model.adata.n_vars`` for the default
        ``data_transform``.
    batch_sizes
        Enumerated batch shapes to compile. The ANE requires static shapes, so CoreML
        compiles one variant per entry; a request with a batch size outside this list
        falls back to CPU/GPU. Keep the list short -- each entry costs compile time
        and package size.
    compute_units
        ``"ALL"`` lets CoreML schedule across ANE/GPU/CPU (recommended).
        ``"CPU_AND_NE"`` forces the Neural Engine path, useful to confirm the model
        actually lands there rather than silently falling back to GPU.
    precision
        ``"float16"`` is required for ANE residency. ``"float32"`` will run, but on
        GPU/CPU only.

    Returns
    -------
    The converted ``coremltools`` model.
    """
    ct = _require_coremltools()

    module = extract_site_encoder(guide, site_name, log1p_input=log1p_input)

    example = torch.zeros((batch_sizes[0], n_genes), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False)

    if len(batch_sizes) == 1:
        input_shape = ct.Shape(shape=(batch_sizes[0], n_genes))
    else:
        input_shape = ct.EnumeratedShapes(shapes=[(b, n_genes) for b in batch_sizes], default=(batch_sizes[0], n_genes))

    ct_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    if precision != "float16":
        logger.warning("precision=%s will not run on the Neural Engine; ANE is float16-only.", precision)

    convert_kwargs: Dict[str, Any] = dict(
        inputs=[ct.TensorType(name="x", shape=input_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="loc"), ct.TensorType(name="scale_unconstrained")],
        convert_to="mlprogram",
        compute_precision=ct_precision,
        compute_units=getattr(ct.ComputeUnit, compute_units),
    )
    if minimum_deployment_target is not None:
        convert_kwargs["minimum_deployment_target"] = getattr(ct.target, minimum_deployment_target)

    mlmodel = ct.convert(traced, **convert_kwargs)

    if output_path is not None:
        mlmodel.save(output_path)
        logger.info("Saved CoreML encoder for site %r to %s", site_name, output_path)

    return mlmodel


def verify_coreml_parity(
    mlmodel,
    guide,
    site_name: str,
    n_genes: int,
    batch_size: int = 128,
    n_trials: int = 3,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    seed: int = 0,
) -> Dict[str, float]:
    """Compare CoreML output against the PyTorch module on random counts.

    float16 inference genuinely loses precision, so the default tolerances are loose
    by PyTorch standards. What matters is whether the *downstream* cell-abundance
    estimates move, which the returned relative error lets you judge.
    """
    module = extract_site_encoder(guide, site_name)
    generator = torch.Generator().manual_seed(seed)

    max_abs_err = 0.0
    max_rel_err = 0.0

    for _ in range(n_trials):
        counts = torch.poisson(torch.full((batch_size, n_genes), 3.0), generator=generator)
        with torch.no_grad():
            loc_ref, scale_ref = module(counts)

        prediction = mlmodel.predict({"x": counts.numpy().astype(np.float32)})
        loc_ct = torch.from_numpy(np.asarray(prediction["loc"]))

        abs_err = (loc_ct - loc_ref).abs()
        rel_err = abs_err / loc_ref.abs().clamp_min(1e-6)
        max_abs_err = max(max_abs_err, float(abs_err.max()))
        max_rel_err = max(max_rel_err, float(rel_err.max()))

    within_tolerance = max_abs_err <= atol + rtol * abs(max_rel_err)
    return {
        "max_abs_error": max_abs_err,
        "max_rel_error": max_rel_err,
        "within_tolerance": bool(within_tolerance),
    }


class CoreMLSiteEncoder:
    """Callable wrapper presenting a CoreML encoder with a torch-tensor interface."""

    def __init__(self, mlmodel, supported_batch_sizes: Optional[List[int]] = None):
        self.mlmodel = mlmodel
        self.supported_batch_sizes = supported_batch_sizes

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        array = x.detach().cpu().numpy().astype(np.float32)
        if self.supported_batch_sizes is not None and array.shape[0] not in self.supported_batch_sizes:
            raise ValueError(
                f"batch size {array.shape[0]} was not compiled into this CoreML model "
                f"(available: {self.supported_batch_sizes}). Re-export with this size included."
            )
        prediction = self.mlmodel.predict({"x": array})
        loc = torch.from_numpy(np.asarray(prediction["loc"])).to(device)
        scale = torch.from_numpy(np.asarray(prediction["scale_unconstrained"])).to(device)
        return loc, scale
