# Running cell2location on Apple silicon

## Short version

```bash
pip install -e .
python benchmarks/apple_silicon_check.py        # verify your machine first
```

```python
import cell2location
from cell2location.accel import configure, prepare_anndata

configure()                 # allocator watermarks + backend check
prepare_anndata(adata)      # float64 counts -> float32, once

model.train(max_epochs=30000, accelerator="mps")
```

`accelerator="auto"` already resolves to Metal on an Apple silicon Mac, so `"mps"`
is only needed when you want to be explicit.

## What actually runs where

| Hardware | Used for | Why |
| --- | --- | --- |
| **GPU (Metal / MPS)** | All training, all posterior sampling | General-purpose, float32, full autograd |
| **CPU** | A handful of ops with no Metal kernel | Correctness fallback, see below |
| **Neural Engine (ANE)** | Optionally, the amortised encoder at inference time | Inference-only, float16, CoreML-only |

The Neural Engine cannot train this model, and no amount of engineering will change
that. It is reachable only through CoreML, which has no autograd, no optimiser state,
no sampling statements and no dynamic shapes. cell2location trains by stochastic
variational inference over a Pyro graph that needs all four. Anything promising
"cell2location on the ANE" for training is promising something the hardware does not
do.

What the ANE *can* do is the one static feed-forward subgraph in the model: when
`amortised=True`, the guide predicts location-specific parameters with an MLP. See
[Neural Engine export](#neural-engine-export) below.

## The three things that break, and what was done about them

### 1. float64 buffers

MPS has no float64 kernels at all — moving a float64 tensor to the device raises
`TypeError`. cell2location registers most of its hyperparameters straight from NumPy
arrays (`N_cells_per_location`, `cell_state`, `init_vals`, …), and NumPy defaults to
float64, so a plain `.to("mps")` dies before any model code runs.

`AppleSiliconCompatMixin` intercepts the move and downcasts float64 parameters and
buffers *while they are still on the CPU*, where the cast is possible. Integer buffers
are left alone. Nothing changes on CPU or CUDA.

Count matrices are the other half of this problem — the model can be clean while every
minibatch arrives as float64. `prepare_anndata(adata)` fixes that once; `train()` warns
if you forget.

### 2. `lgamma` on broadcast inputs

This is the one to take seriously, because it fails silently.

`torch.lgamma` on MPS has returned incorrect values for non-contiguous / broadcast
inputs ([pytorch#132605](https://github.com/pytorch/pytorch/issues/132605)). The
negative-binomial log-likelihood calls `lgamma` on exactly that: `theta` is routinely a
`(1, n_genes)` view broadcast against an `(n_obs, n_genes)` batch. The failure mode is
not a crash — it is a plausible-looking ELBO that descends normally and produces wrong
cell abundances.

`cell2location.accel.log_nb_positive` never hands a stride-0 view to the kernel. It is
otherwise arithmetically identical to the upstream function, and produces bit-identical
results on CPU and CUDA.

If a particular macOS / PyTorch combination still disagrees with the CPU, three escape
hatches are available without touching model code:

```bash
export CELL2LOCATION_MPS_LGAMMA=contiguous   # default: materialise, then native kernel
export CELL2LOCATION_MPS_LGAMMA=stirling     # pure-composition series, no lgamma kernel
export CELL2LOCATION_MPS_LGAMMA=cpu          # evaluate on CPU, copy back
```

`stirling` is a shifted asymptotic expansion built only from `log`, `reciprocal` and
arithmetic — ops with solid Metal coverage. Its accuracy in float32 is absolute
(~1e-6) rather than relative, which is what a summed log-likelihood needs. The
validation script tests all four modes and tells you which ones pass on your machine.

### 3. Missing random-number generators

`NegativeBinomial.sample()` is a Gamma-Poisson compound needing `_standard_gamma` and
`poisson`, neither of which Metal has historically implemented. Those two calls fall
back to the CPU.

The fallback is *probed at runtime*, not hard-coded, so a PyTorch release that adds the
kernels is picked up automatically with no code change.

This is deliberately narrower than `PYTORCH_ENABLE_MPS_FALLBACK=1`, which a library
cannot set for its users anyway (it is read before `import torch`) and which silently
routes every missing op through the CPU — turning a one-line gap into an invisible
performance cliff.

## Verifying a run while it happens

The validation script checks synthetic data at shapes I chose. Your dataset has its
own dynamic range, sparsity and dispersion values. The numerical guard closes that gap
by checking during the actual run: every N steps it traces the guide once on the GPU,
then replays the *same sampled latents* through the model on both the GPU and the CPU
and compares the model log-joint. The comparison is deterministic — RNG streams differ
between devices even for identical seeds, so anything seed-based would flag healthy
runs — and the model log-joint is exactly the arithmetic the accelerated kernels
compute. A non-finite comparison counts as divergence, never as agreement.

`train_compiled()` on Metal arms the guard automatically (interval 1000) unless you
have configured one yourself.

```bash
export CELL2LOCATION_MPS_GUARD=1        # default interval, 1000 steps
export CELL2LOCATION_MPS_GUARD=250      # or set the interval directly
```

```python
model.mps_numerical_guard_every_n_steps = 500
model.train(max_epochs=30000)

print(model.numerical_guard_.summary())
# {'checks': 60, 'diverged': False, 'max_relative_difference': 3.1e-06, ...}
```

Cost is one CPU forward pass per check — about 30 extra evaluations across a 30k-step
run. For anything headed into a paper, that is a rounding error against the value of
being able to say the GPU and the CPU agreed throughout.

## Memory, and the thing a Mac does better

Apple silicon shares one memory pool between CPU and GPU. The usual framing is
convenience — no host-to-device copies — but the more interesting consequence is
capacity. A workstation GPU gives you 24 GB and no way past it. A Mac Studio can be
configured with 128 or 512 GB, all of it addressable by the GPU.

That changes what is possible, not just what is fast. Minibatching a spatial model is
not free: `N_cells_per_location` and the detection-efficiency priors couple locations,
and the amortised guide approximates what full-batch inference computes exactly. Where
full-batch fits, it is the better estimator — and on a large-memory Mac it fits at
scales a discrete GPU cannot reach at all.

```python
from cell2location.accel import plan_memory

print(plan_memory(adata, n_factors=50))
# Dataset      : 200,000 locations x 18,000 genes x 50 cell types
# Available    : 340.0 GB usable of unified memory
# Full batch   : 216.1 GB estimated peak
# Verdict      : fits -- train with batch_size=None for exact full-batch inference
```

Estimates are deliberately conservative — being told 8192 and hitting an allocation
failure at step 3000 is worse than being told 4096. Pass `memory_budget_gb` to plan for
a machine you are not currently sitting at.

Allocator watermarks:

```python
from cell2location.accel import configure

configure(high_watermark_ratio=0.0)   # no cap: for a large dataset on a big-memory Mac
configure(high_watermark_ratio=0.8)   # leave 20% for everything else
```

Watermarks are read at the first MPS allocation, so `configure()` must run before the
model touches the GPU. Long runs also get an automatic `torch.mps.empty_cache()` every
500 steps.

## The fused likelihood kernel

On by default; `CELL2LOCATION_MPS_FUSED_NB=0` forces the eager path.

The negative-binomial log-probability is the hot loop: evaluated over the full
`(n_obs, n_genes)` matrix on every SVI step, and in eager mode it materialises about a
dozen intermediates of that size. At 4992 × 12000 that is 240 MB each — several
gigabytes of memory traffic to produce 240 MB of answer. The arithmetic is trivial; the
cost is entirely bandwidth, and on unified memory that bandwidth is shared with
everything else on the machine.

`cell2location/accel/_metal/nb_logprob.metal` collapses the whole expression, forward
and backward, into one pass each. It also reimplements `lgamma` and `digamma` in Metal
— the first so the kernel does not depend on the implementation this backend has got
wrong before, the second because MSL has no `digamma` at all.

**It verifies itself before it is trusted**, which is what makes on-by-default safe:
on first use it runs both the forward pass and the gradients against the eager
implementation across every layout it claims to support. If anything disagrees it logs
exactly what failed, disables itself permanently for the process, and eager continues.
The worst case is one wasted check. (Verified on an M2 Ultra: forward and gradients
match eager on all supported layouts; 5.2x over the eager Metal path at 4992 × 12000,
and the training loop reaches it through the ``GammaPoisson`` likelihood sites.)

```python
from cell2location.accel import fused_nb_status, verify_fused_kernel

print(verify_fused_kernel())   # (True, 'forward and gradients match eager ...')
print(fused_nb_status())
```

If it is rejected on your machine, that is a bug worth reporting — please include the
rejection message, which names the layout and the worst-disagreeing element.

## torch.compile

`train_compiled()` compiles on Metal by default (torch >= 2.12); set
`CELL2LOCATION_ALLOW_MPS_COMPILE=0` to force eager instead.

Because a miscompiled graph would produce plausible-looking but wrong losses, compiled
Metal runs arm the numerical guard automatically: the model log-joint is cross-checked
against the CPU during the run, and divergence is reported rather than published.
Measured on an M2 Ultra at 5,000 × 10,000: ~114 ms/epoch compiled+fused against
140 ms/epoch for the eager+fused path, with the guard's worst relative
difference at 2.3e-7. Both figures predate the flat engine and were taken with
the harness before its timing was corrected, so they describe this pyro-based
path against its own contemporaries, not against today's default — `train()`
now runs the flat engine and the harness reports 32.2 ms/epoch at this shape.
Compile without the fused kernel is *slower* than eager —
inductor cannot fuse across Pyro's effect handlers, so the two optimisations are
complements, not substitutes.

## Neural Engine export

Only useful when you have a trained amortised model and want to score many slides
against it. Training is unaffected.

```python
from cell2location.accel.coreml import export_site_encoder, verify_coreml_parity

mlmodel = export_site_encoder(
    guide=model.module.guide,
    site_name="w_sf",
    n_genes=model.adata.n_vars,
    batch_sizes=(1, 128, 1024, 2048),   # ANE needs static shapes
    output_path="w_sf_encoder.mlpackage",
    compute_units="ALL",                # or "CPU_AND_NE" to confirm ANE residency
)

print(verify_coreml_parity(mlmodel, model.module.guide, "w_sf", model.adata.n_vars))
```

Requires `pip install coremltools`. Check parity before trusting the output: the ANE is
float16, so some precision is lost, and what matters is whether your cell-abundance
estimates move — not whether the encoder activations match to 7 digits.

## Validation

```bash
python benchmarks/apple_silicon_check.py                      # everything
python benchmarks/apple_silicon_check.py --skip-train         # ops + parity only
python benchmarks/apple_silicon_check.py --html report.html   # shareable report
python benchmarks/apple_silicon_check.py --json out.json      # machine-readable
```

`--html` writes a single self-contained file — no network, no dependencies — suitable
for attaching to an issue, keeping beside a paper's methods, or diffing against the
same machine three PyTorch releases later.

It checks the environment, probes op coverage, compares MPS against CPU for `lgamma`
(all four modes) and the full NB likelihood, benchmarks kernels at realistic Visium
shapes, and runs a short CPU-vs-MPS training comparison on synthetic data. Exit code is
non-zero on a parity failure, so it can gate CI on a Mac runner.

Section 3 is the one that matters. A wrong `lgamma` will not announce itself anywhere
else.

## Opting out

```bash
export CELL2LOCATION_DISABLE_MPS=1     # accelerator="auto" ignores Metal
```

or pass `accelerator="cpu"` explicitly.

## Environment variables, all together

| Variable | Effect |
| --- | --- |
| `CELL2LOCATION_DISABLE_MPS=1` | `accelerator="auto"` ignores Metal |
| `CELL2LOCATION_MPS_LGAMMA=` | `contiguous` (default) / `stirling` / `cpu` / `native` |
| `CELL2LOCATION_MPS_GUARD=1` | cross-check the model log-joint against the CPU during training |
| `CELL2LOCATION_MPS_FUSED_NB=0` | disable the fused Metal likelihood kernel (on by default) |
| `CELL2LOCATION_ALLOW_MPS_COMPILE=0` | make `train_compiled()` run eager on Metal (compiles by default) |

## Known limits

- Metal is unavailable inside Docker and inside Linux VMs. Run natively.
- float32 only. If your analysis genuinely needs float64 precision, train on CPU.
- Posterior sampling partially runs on the CPU until PyTorch ships the missing RNG
  kernels. `export_posterior` is therefore less accelerated than training.
- The ANE path covers the amortised encoder only, and only at inference time.
- The fused kernel handles two theta layouts — elementwise, and the common
  `(1, n_genes)` row broadcast. Anything else falls back to eager, which is general.
- The fused kernel and the CoreML export have never run on Apple hardware. Both are
  gated: the kernel verifies itself and disables on disagreement, and the CoreML path
  ships `verify_coreml_parity`. Treat a first run of either as an experiment.
