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

Please note that cell2locations requires 2 user-provided hyperparameters (N_cells_per_location and detection_alpha) - for detailed guidance on setting these hyperparameters and their impact see [the flow diagram and the note](https://github.com/BayraktarLab/cell2location/blob/master/docs/images/Note_on_selecting_hyperparameters.pdf). Many real datasets (especially human) show within-slide variability in RNA detection sensitivity - requiring you to try both recommended settings of the `detection_alpha` parameter: `detection_alpha=200` for low within-slide technical variability and `detection_alpha=20` for high within-slide technical variability.

Cell2location is a principled Bayesian model that can resolve fine-grained cell types in spatial transcriptomic data and create comprehensive cellular maps of diverse tissues. Cell2location accounts for technical sources of variation and borrows statistical strength across locations, thereby enabling the integration of single cell and spatial transcriptomics with higher sensitivity and resolution than existing tools. This is achieved by estimating which combination of cell types in which cell abundance could have given the mRNA counts in the spatial data, while modelling technical effects (platform/technology effect, contaminating RNA, unexplained variance).

<p align="center">
   <img src="https://github.com/BayraktarLab/cell2location/blob/master/docs/images/Fig1_v2_white_bg.png?raw=True">
</p>
Overview of the spatial mapping approach and the workflow enabled by cell2location. From left to right: Single-cell RNA-seq and spatial transcriptomics profiles are generated from the same tissue (1). Cell2location takes scRNA-seq derived cell type reference signatures and spatial transcriptomics data as input (2, 3). The model then decomposes spatially resolved multi-cell RNA counts matrices into the reference signatures, thereby establishing a spatial mapping of cell types (4).    

## Usage and Tutorials

The tutorial covering the estimation of expresson signatures of reference cell types, spatial mapping with cell2location and the downstream analysis can be found here and tried on [Google Colab](https://colab.research.google.com/github/BayraktarLab/cell2location/blob/master/docs/notebooks/cell2location_tutorial.ipynb): https://cell2location.readthedocs.io/en/latest/

Please report bugs via https://github.com/BayraktarLab/cell2location/issues and ask any usage questions about [cell2location](https://discourse.scverse.org/c/ecosytem/cell2location/42), [scvi-tools](https://discourse.scverse.org/c/help/scvi-tools/7) or [Visium data](https://discourse.scverse.org/c/general/visium/32) in scverse community discourse.

Cell2location package is implemented in a general way (using https://pyro.ai/ and https://scvi-tools.org/) to support multiple related models - both for spatial mapping, estimating reference cell type signatures and downstream analysis.

## Installation

We suggest using a separate conda environment for installing cell2location.

Create conda environment and install `cell2location` package

```bash
conda create -y -n cell2loc_env python=3.10

conda activate cell2loc_env
pip install "cell2location[tutorials] @ git+https://github.com/kevinj24fr/cell2location_metal.git"
```

Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:

```bash
conda activate cell2loc_env
python -m ipykernel install --user --name=cell2loc_env --display-name='Environment (cell2loc_env)'
```

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

Before installing cell2location and it's dependencies, it could be necessary to make sure that you are creating a fully isolated conda environment by telling python to NOT use user site for installing packages by running this line before creating conda environment and every time before activatin conda environment in a new terminal session:

```bash
export PYTHONNOUSERSITE="literallyanyletters"
```


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

## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

## Acknowledgements 

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar
