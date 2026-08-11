# cell2location_metal

**cell2location with first-class Apple silicon support.** A fork of
[BayraktarLab/cell2location](https://github.com/BayraktarLab/cell2location) that makes
the full workflow — reference signatures, spatial mapping, posterior export — run
fast, correctly, and with zero configuration on the Metal (MPS) backend, while
staying behaviourally identical to upstream on CPU and CUDA.

**What this fork adds:**

- **Zero-config Metal training.** `model.train()` picks the GPU, converts dtypes in
  place, keeps the full batch device-resident, and routes the negative-binomial
  likelihood through a fused Metal kernel that verifies itself against the eager
  implementation (forward *and* gradients) before it is allowed to run.
- **A numerical guard.** During training, the model log-joint is periodically
  recomputed on the CPU under the same sampled latents and compared. Silent GPU
  divergence — the failure mode that matters — is caught on your data, during your
  run, not on synthetic shapes.
- **`torch.compile` on Metal.** `train_compiled()` works (torch ≥ 2.12) and arms the
  guard automatically.
- **Measured on an M2 Ultra** (5,000 locations × 10,000 genes, full batch):
  CPU 742 ms/epoch → Metal 140 ms/epoch out of the box (5.2x) → ~114 ms/epoch
  compiled (6.5x), with guard-verified CPU/GPU agreement at 2×10⁻⁷ relative.
- **Verified against upstream's own test suite** (109 passed) plus ~100 fork tests
  covering the Metal layer, and a benchmark/parity harness
  (`benchmarks/apple_silicon_check.py`) that produces a shareable HTML report.

**Install this fork:**

```bash
pip install git+https://github.com/kevinj24fr/cell2location_metal.git
```

Details, escape hatches, and known limits: [docs/apple_silicon.md](docs/apple_silicon.md).
Everything below describes the cell2location method itself, unchanged from upstream.

## Engine changelog (validated)

Every entry here cleared the push gate in `benchmarks/engine_validation.py` before
merging: measurably faster than what it replaced, final-ELBO parity within 0.5%,
numerical guard clean, and posterior summaries matching the replaced path within
Monte-Carlo error. Changes that do not clear the gate do not merge, and are not
listed.

- **Flat likelihood through the fused NB kernel.** The flat engine's data
  likelihood — its single biggest term — now routes through the same
  self-verifying Metal kernel the pyro path uses (GammaPoisson(α, α/μ) ≡
  NB(μ, θ=α), so the kernel receives μ directly and the eager expression's
  ~dozen full-size intermediates disappear). Eager fallback is byte-identical
  to the contract-pinned expression; the runtime guard now cross-checks the
  kernel against eager CPU every guarded run (worst difference 2.3e-7).
  Measured at 5,000×10,000 (M2 Ultra): flat step 54.4 → 24.1 ms (**2.26x**),
  inside the workload's estimated bandwidth floor (~20–30 ms); harness
  protocol 117.9 → 100.1 ms/epoch, ALL GATES PASS.

- **Flat training engine.** Training no longer runs through pyro's effect
  handlers: all 17 sites' log-densities and the GammaPoisson likelihood are
  hand-transcribed tensor code, the mean-field guide lives in two flat tensors
  (unconstrained loc, softplus-unconstrained scale), and each step draws one
  reparameterized sample and optimizes −(log-joint − log q) with unclipped
  Adam — matching what the pyro path actually uses (its ClippedAdam docstring
  is stale; an elementwise clamp tried first destabilized long runs with a
  measured ~2-posterior-sd abundance shift and was removed on that artifact).
  Transcription, ELBO and per-draw gradients are contract-pinned against pyro
  replay; the numerical guard runs natively (same-latents flat log-joint, MPS
  vs CPU). Validation harness at 5,000×10,000 (M2 Ultra): training 274.9 →
  117.9 ms/epoch (**2.33x**), final-ELBO parity, guard clean (worst CPU/GPU
  difference 7.6e-8), export parity unchanged. Same-data trajectory comparison
  against the pyro path at convergence, early stopping active: final-loss
  parity 1.2e-6 by tail median (the tail mean is one heavy-tail single-draw
  outlier — ~1 in 700 epochs, both engines' estimator family — away from
  meaningless), abundance r 0.992 with **100% of values within 1 posterior
  standard deviation** (median drift 0.08 sd), **2.3x wall-clock**. Scope:
  full-batch MPS training with the default AutoNormal guide; minibatch, custom
  optimizers/callbacks, loaded-model warmups and other unhandled arguments fall
  back to the pyro path automatically. Kill switch
  `CELL2LOCATION_MPS_FLAT_ENGINE=0`.

- **Convergence-based early stopping (Metal runs).** Upstream trains a fixed
  30,000 epochs with no stopping criterion; the ELBO plateaus long before that and
  the remainder is a random walk on a flat objective. Training now stops when the
  best ELBO has not improved by 1e-5 (relative) within 2,000 epochs. Validated by
  a same-seed 30k-epoch reference comparison at 5,000×10,000 (M2 Ultra): **3.1x
  wall-clock** (72.6 → 23.5 min), final-ELBO parity, abundance r = 0.990 with
  every value within 1 posterior standard deviation of the full run (median drift
  0.34 sd). A ~7x looser setting exists but drifts beyond the posterior's own
  resolution and is deliberately not the default. Disable with
  `model.mps_early_stopping = None` or `CELL2LOCATION_MPS_EARLY_STOP=0` to
  reproduce upstream's fixed-epoch behaviour exactly.

- **Vectorized posterior export.** The looped sampler ran one full guide trace per
  posterior draw — a thousand sequential traces through pyro's effect handlers. For
  the mean-field `AutoNormal` guides both models use by default, the joint
  factorizes over sites, so all draws are one batched
  `transform(loc + scale · eps)` per site: the same distribution, shaped as a
  batch. Falls back to the loop for non-mean-field guides, minibatched export, or
  observed-site sampling. Validation harness at 5,000 locations × 10,000 genes,
  1,000 samples, M2 Ultra: export 181.2s → 11.1s (**16.3x**); summary parity vs
  the loop within Monte-Carlo error (means 0.3%, quantiles 0.6% median relative);
  final-ELBO parity and numerical guard clean (worst CPU/GPU difference 1.5e-7).

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

A note on the Neural Engine: it cannot train this model. The ANE is reachable only
through CoreML, which has no autograd, no optimiser state and no sampling statements,
while cell2location trains by variational inference over a Pyro graph that needs all
three. Training runs on the GPU. The ANE can optionally run the amortised guide's
encoder network at inference time, which is worthwhile when scoring many slides against
an already-trained model.

Two further things worth knowing:

- **Verify while you train.** `CELL2LOCATION_MPS_GUARD=1` cross-checks the loss against
  the CPU every 1000 steps, so silent divergence is caught on your data rather than on
  synthetic shapes.
- **Unified memory changes what fits.** `cell2location.accel.plan_memory(adata)` tells
  you whether your dataset trains full-batch on this machine — which, on a large-memory
  Mac, it often does at scales a discrete GPU cannot reach.

Full details, escape hatches and known limits: [docs/apple_silicon.md](docs/apple_silicon.md).

## Documentation and API details

User documentation is availlable on https://cell2location.readthedocs.io/en/latest/. 

Cell2location architecture is designed to simplify extended versions of the model that account for additional technical and biologial information. We plan to provide a tutorial showing how to add new model classes but please get in touch if you would like to contribute or build on top our package.

## Acknowledgements 

We thank all paper authors for their contributions:
Vitalii Kleshchevnikov, Artem Shmatko, Emma Dann, Alexander Aivazidis, Hamish W King, Tong Li, Artem Lomakin, Veronika Kedlian, Mika Sarkin Jain, Jun Sung Park, Lauma Ramona, Liz Tuck, Anna Arutyunyan, Roser Vento-Tormo, Moritz Gerstung, Louisa James, Oliver Stegle, Omer Ali Bayraktar

We also thank Pyro developers (Fritz Obermeyer, Martin Jankowiak), Krzysztof Polanski, Luz Garcia Alonso, Carlos Talavera-Lopez, Ni Huang for feedback on the package, Martin Prete for dockerising cell2location and other software support.

## FAQ

See https://github.com/BayraktarLab/cell2location/discussions

## Future development and experimental features
Future developments of cell2location are focused on 1) scalability to 100k-mln+ locations using amortised inference of cell abundance (same ideas as used in VAE), 2) extending cell2location to related spatial analysis tasks that require modification of the model (such as using cell type hierarchy information), and 3) incorporating features presented by more recently proposed methods (such as CAR spatial proximity modelling). We are also experimenting with Numpyro and JAX (https://github.com/vitkl/cell2location_numpyro).

## Tips

### Conda environment for A100 GPUs

```bash
export PYTHONNOUSERSITE="True"
conda create -y -n cell2location_cuda118_torch22 python=3.10
conda activate cell2location_cuda118_torch22

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install scvi-tools==1.1.2

pip install git+https://github.com/BayraktarLab/cell2location.git#egg=cell2location[tutorials,dev]
python -m ipykernel install --user --name=cell2location_cuda118_torch22 --display-name='Environment (cell2location_cuda118_torch22)'
```

### Issues with package version mismatches often originate from python user site rather than conda environment being used to install a subset of packages

Before installing cell2location and it's dependencies, it could be necessary to make sure that you are creating a fully isolated conda environment by telling python to NOT use user site for installing packages by running this line before creating conda environment and every time before activatin conda environment in a new terminal session:

```bash
export PYTHONNOUSERSITE="True"
```

### Useful code for reading and combining multiple Visium sections

Keeping info on distinct sections in a csv file (Google Sheet).

```python
sample_annot = pd.read_csv('./sample_annot.csv')

from glob import glob
sample_annot['path'] = pd.Series(
    glob(f'{sp_data_folder}*'),
    index=[sub('^.+WTSI_', '', sub('_GRCh38-2020-A$', '', i)) for i in glob(f'{sp_data_folder}*')]
)[sample_annot['Sample_ID']].values
import os
sample_annot['file'] = [os.path.basename(i) for i in sample_annot['path']]

sample_annot['Sample_ID'].unique()
```

Reading and concatenating samples.

```python
def read_and_qc(sample_name, file, path=sp_data_folder):
    """
    Read one Visium file and add minimum metadata and QC metrics to adata.obs
    NOTE: var_names is ENSEMBL ID as it should be, you can always plot with sc.pl.scatter(gene_symbols='SYMBOL')
    """
    
    adata = sc.read_visium(path + str(file) +'/',
                           count_file='filtered_feature_bc_matrix.h5',
                           load_images=True)
    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names
    adata.var.rename(columns={'gene_ids': 'ENSEMBL'}, inplace=True)
    adata.var_names = adata.var['ENSEMBL']
    adata.var.drop(columns='ENSEMBL', inplace=True)
    
    # just in case there are non-unique ENSEMBL IDs
    adata.var_names_make_unique()

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
    adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']
    
    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    adata.obs_names = 's' + adata.obs["sample"] \
                          + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'
    
    file = list(adata.uns['spatial'].keys())[0]
    adata.uns['spatial'][sample_name] = adata.uns['spatial'][file].copy()
    del adata.uns['spatial'][file]
    print(adata.uns['spatial'].keys())
    
    return adata

def read_all_and_qc(
    sample_annot, Sample_ID_col, file_col, sp_data_folder, 
    count_file='filtered_feature_bc_matrix.h5',
):
    """
    Read and concatenate all Visium files.
    """
    # read first sample
    adata = read_and_qc(
        sample_annot[Sample_ID_col][0], sample_annot[file_col][0], 
        path=sp_data_folder
    ) 

    # read the remaining samples
    slides = {}
    for i, s in enumerate(sample_annot[Sample_ID_col][1:]):
        adata_1 = read_and_qc(s, sample_annot[file_col][i], path=sp_data_folder) 
        slides[str(s)] = adata_1

    adata_0 = adata.copy()

    # combine individual samples
    #adata = adata.concatenate(list(slides.values()), index_unique=None)
    adata = adata.concatenate(
        list(slides.values()),
        batch_key="sample",
        uns_merge="unique",
        batch_categories=sample_annot[Sample_ID_col], 
        index_unique=None
    )

    sample_annot.index = sample_annot[Sample_ID_col]
    for c in sample_annot.columns:
        sample_annot.loc[:, c] = sample_annot[c].astype(str)
    adata.obs[sample_annot.columns] = sample_annot.reindex(index=adata.obs['sample']).values
    
    return adata
    
adata = read_all_and_qc(
    sample_annot=sample_annot, 
    Sample_ID_col='Sample_ID', 
    file_col='file', 
    sp_data_folder=sp_data_folder, 
    count_file='filtered_feature_bc_matrix.h5',
)

adata_incl_nontissue = read_all_and_qc(
    sample_annot=sample_annot, 
    Sample_ID_col='Sample_ID', 
    file_col='file', 
    sp_data_folder=sp_data_folder, 
    count_file='raw_feature_bc_matrix.h5',
)
```

Since Version 0.9.0 (released on 2023-04-11), the function `AnnData.concatenate()` has been deprecated in favour of `anndata.concat()` as per the official release notes ([Reference](https://anndata.readthedocs.io/en/latest/release-notes/index.html#id4)). Here is the updated code snippet of `read_all_and_qc`:

```python
from anndata import concat

def read_all_and_qc(
    sample_annot, Sample_ID_col, file_col, sp_data_folder, 
    count_file='filtered_feature_bc_matrix.h5',
):
    """
    Read and concatenate all Visium files.
    """

    # read all samples and store them in a list
    adatas = []
    for i, s in enumerate(sample_annot[Sample_ID_col]):
        adata_i = read_and_qc(s, Sample_ID_col[file_col][i], path=sp_data_folder) 
        adatas.append(adata_i)
    # combine individual samples
    adata = concat(
        adatas,
        merge="unique",
        uns_merge="unique",
        label="batch",
        keys=sample_annot[Sample_ID_col].tolist(), 
        index_unique=None
    )

    sample_annot.index = sample_annot[Sample_ID_col]
    for c in sample_annot.columns:
        sample_annot.loc[:, c] = sample_annot[c].astype(str)
    adata.obs[sample_annot.columns] = sample_annot.reindex(index=adata.obs['sample']).values

    return adata

adata = read_all_and_qc(
    sample_annot=sample_annot, 
    Sample_ID_col='Sample_ID', 
    file_col='file', 
    sp_data_folder=sp_data_folder, 
    count_file='filtered_feature_bc_matrix.h5',
)

cell2location.models.Cell2location.setup_anndata(
    adata=adata_vis,
    batch_key="batch")
```
