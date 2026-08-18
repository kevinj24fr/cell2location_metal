# cell2location_metal

**cell2location optimized for Apple silicon.** A fork of
[BayraktarLab/cell2location](https://github.com/BayraktarLab/cell2location):
28× CPU speedup on an M2 Ultra for the spatial model, with runtime CPU/MPS
numerical verification and posterior parity against upstream. The statistical
model is unchanged; CPU and CUDA retain upstream behaviour exactly.

## Current performance

| Operation | Baseline | Metal | Speedup |
|---|---|---|---|
| Spatial training (5,000 × 10,000, full batch) | 897.5 ms/epoch (upstream, CPU) | 32.2 ms/epoch | **28×** |
| Spatial training, minibatch (`batch_size=1250`) | 440.5 ms/epoch (pyro path, MPS) | 70.0 ms/epoch | **6.3×** |
| Reference model (10,000 × 10,000, `batch_size=2500`) | 569.6 ms/epoch (pyro path, MPS) | 194.4 ms/epoch | **2.9×** |
| Posterior export (1,000 samples) | 181.2 s (looped sampler) | 11.1 s | **16.3×** |

Baseline arm and protocol differ per row (stated in
parentheses); measurement protocol and each change's gate numbers:
[docs/engine_changelog.md](docs/engine_changelog.md). Reproduce with
`benchmarks/engine_validation.py` and `benchmarks/reference_validation.py`.

## Install

```bash
pip install git+https://github.com/kevinj24fr/cell2location_metal.git
```

## What changed

cell2location's probabilistic model remains the reference implementation. On
supported MPS configurations, training runs on a specialized tensor execution
engine that reproduces the pyro computation without pyro's effect-handler
overhead; the negative-binomial likelihood is executed by a fused Metal kernel.
Unsupported configurations fall back to upstream execution automatically.

```
cell2location API ──> flat inference engine ──> fused Metal NB kernel
                            │
                            ▼
                pyro model == parity / guard oracle
```

- **Flat inference engine** — the model's log-joint, ELBO and mean-field guide
  as plain tensor code, for both the spatial and reference models, minibatched
  where the model allows it.
- **Fused Metal NB kernel** — the likelihood's dozen full-size intermediates
  collapsed into one bandwidth-bound pass, self-verified against the eager
  implementation before it is allowed to run.
- **Vectorized posterior export** — all draws as one batched transform per
  site instead of a thousand sequential guide traces.
- **Convergence-based early stopping** — training stops when the ELBO
  plateaus instead of running upstream's fixed 30,000 epochs.
- **Zero configuration** — `model.train()` picks the device, converts dtypes,
  and stages data; every optimization has an escape hatch
  (see [docs/apple_silicon.md](docs/apple_silicon.md)).

## How correctness is checked

The accelerated engine reproduces upstream's model log-joint and per-latent
gradients against pyro replay (contract tests), and every engine change is
gated on converged-ELBO parity and posterior-summary parity before it can
merge. At runtime, a numerical guard periodically recomputes the training
computation on the CPU under the same sampled latents and compares — silent
GPU divergence is caught on your data, during your run, not on synthetic
shapes.

## Compatibility and fallbacks

- Does not require Apple silicon; CPU/CUDA retain upstream behaviour.
- Does not claim performance generalization beyond the tested M2 Ultra.
- Unsupported accelerated configurations fall back to the upstream pyro path
  automatically.

Every optimization has a kill switch; the environment variables are documented
in [docs/apple_silicon.md](docs/apple_silicon.md). Known open issues,
including an intermittent torch 2.12.x custom-kernel dispatch defect that the
runtime guard detects: [docs/known_issues.md](docs/known_issues.md). Full
optimization history with per-change validation numbers:
[docs/engine_changelog.md](docs/engine_changelog.md).

---

### About cell2location

Comprehensive mapping of tissue cell architecture via integrated single cell and
spatial transcriptomics. All scientific credit belongs to the original authors —
if you use cell2location (through this fork or otherwise) please cite:

Kleshchevnikov, V., Shmatko, A., Dann, E. et al. Cell2location maps fine-grained cell types in spatial transcriptomics. Nat Biotechnol (2022). https://doi.org/10.1038/s41587-021-01139-4
https://www.nature.com/articles/s41587-021-01139-4

## Apple silicon (Metal / MPS)

The Metal support summarized at the top of this README is zero-config:

```python
model.train()                      # accelerator="auto" picks Metal on a Mac
model.train_compiled()             # adds torch.compile; the numerical guard
                                   # cross-checks the loss against CPU as it trains
```

Verify your machine before trusting a run — the failure mode worth caring about is a
silently wrong `lgamma`, not a crash:

```bash
python benchmarks/apple_silicon_check.py
```
Full details, escape hatches and known limits: [docs/apple_silicon.md](docs/apple_silicon.md).

## Documentation

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/.
