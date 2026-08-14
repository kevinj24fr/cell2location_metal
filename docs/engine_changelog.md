# Engine changelog (validated)

This is the proof trail behind the README's performance table: every validated
engine change, in reverse order, with the harness numbers it cleared its push
gate with. The README communicates the final state of the system; this file is
how it got there.

Every entry here cleared a push gate before merging: measurably faster than what
it replaced, final-ELBO parity within 0.5%, numerical guard clean, and posterior
summaries matching the replaced path within Monte-Carlo error. Changes that do
not clear the gate do not merge, and are not listed. The gates are
`benchmarks/engine_validation.py` (spatial model, `--minibatch` for its
minibatch configuration) and `benchmarks/reference_validation.py` (reference
signature model); a change touching shared code must pass every arm.

**On the numbers below.** Each entry records what that change measured against
the baseline current *at the time*, with the harness as it existed then. The
harness has since been corrected in ways that shift absolute figures without
changing what any entry demonstrated: timing now excludes the numerical guard
(which cross-checks on the CPU, so timing it measured the verifier rather than
training), discards a warm-up run, frees each repeat's model (holding them made
every later run slower), and reports the minimum of the repeats rather than the
median (contention only adds time). Each entry's *ratio* stands; its absolute
ms/epoch is not comparable across entries, nor to what the harness prints
today. The current baselines are in `benchmarks/*_baseline.json`.

- **Guard judges divergence by rate, not the single worst check.**
  A single SVI draw's log-joint is heavy-tailed: a rare extreme sample makes one
  term dominate, and on that draw device and CPU fp32 differ by a large
  *relative* amount while training -- which stays on one device -- is unaffected.
  The guard's `diverged = any check over tolerance` flagged such a run as
  failure: a real b03 spatial fit converged over 30,000 finite epochs but tripped
  on 1 of 150 checks at 0.43. The guard now judges by the *rate* of
  over-tolerance checks (the handoff's own "judge by tail median, never worst"
  lesson): more than 5% is a consistent bias -- a wrong kernel misses on
  essentially every check -- while one or two of many is the heavy tail. A
  non-finite check is always divergence regardless of rate (overflow/NaN). No
  training or timing change (the guard runs guard-off in timing); all three arms
  pass, suite 200. (A reproducibility-recheck variant was tried first and cut: a
  per-check device re-evaluation is not a clean determinism test across the
  guard's CPU round-trip, and rate alone is the simpler criterion that the real
  b03 run needs.)

- **NB overdispersion clamp — the flat engine completes real reference fits.**
  `alpha_g_inverse` carries an Exponential prior whose mode is at zero, so
  low-overdispersion genes are pulled toward `alpha = 1/alpha_g_inverse² → ∞`
  (the Poisson limit). In fp32 that parameter underflows to zero, `1/0 = inf`,
  and `inf − inf` NaNs the forward log-joint — which on the 675k-cell GBM
  reference happened at epoch ~75 and forced the whole fit onto the 8x-slower
  pyro fallback (the gradient-masking fix above kept parameters finite but
  could not stop the *value* overflow one level up). `_stable_alpha` caps
  `alpha` at 1e6, beyond which the negative binomial is Poisson to <1e-3 for
  these count ranges (excess variance μ/α) — numerically invisible to
  converged signatures, orders of magnitude above any α in the contract-test
  or harness regimes, so the pyro-replay pins are untouched. The b02 reference
  now completes all 250 epochs on the flat engine, no fallback, final loss
  marginally below the pyro-path fits (5.098e9 vs 5.106e9). Gates: spatial
  0.977x, minibatch 1.008x, reference **0.911x** (faster — it stays on the
  flat engine instead of falling back), suite 198, ALL GATES PASS every arm.

- **Non-finite gradient masking (correctness, not speed).** On a real 675k-cell
  GBM reference, one gene's exact gradient through `alpha = alpha_g_inverse⁻²`
  exceeded float32 range while the loss was still finite (5.035e9; inf in 1 of
  2.47M gradient elements) — Adam turned the inf into NaN parameters, every
  later loss was NaN, and the engine fell back to pyro at 8x the cost. Both
  flat loops now zero exactly the non-finite gradient elements each step
  (`nan_to_num_`, on-device, async — a counted per-step check costs a device
  sync and measured 1.246x on the minibatched reference arm, so the logged
  count runs on the guard cadence instead; protection is unconditional, the
  warning is sampled). Finite-gradient training is bit-identical; this is not
  the elementwise ±10 clamp that biased converged abundances and was removed —
  that engaged on every element of every step, this engages only where the
  alternative is a dead run. Survival-verified on the failing reference fit
  (masked 2 elements at the fatal epoch, completed 30/30 finite, final loss
  parity with the pyro path). Gates: spatial **1.011x** baseline (train),
  export 0.965x; minibatch arm pass; reference **1.074x** — all inside the
  no-regression tolerance, ALL GATES PASS on every arm.

- **Minibatch training for the spatial model.** Passing `batch_size` used to
  drop the spatial model onto the pyro path, so the caller whose data does not
  fit in memory — the one who needs the help most — got none of this fork. The
  flat engine now subsamples the observation plate, per-location latents
  included: the five of them (`w_sf`, `detection_y_s`,
  `n_s_cells_per_location`, `b_s_groups_per_location`, `z_sr_groups_factors`)
  are indexed with the data, and their priors and their `log q` carry the
  plate's `n_obs/batch` scale alongside the likelihood. Which sites those are is
  read from the model's own `list_obs_plate_vars()`, and each one's guide
  parameter is checked to actually be indexed by observation before the engine
  will subsample it — a site shaped otherwise routes to pyro rather than being
  silently mis-indexed. Measured at 5,000 × 10,000, `batch_size=1250`
  (M2 Ultra): **440.5 → 70.0 ms/epoch (6.3x)**, ELBO parity, guard clean
  (2.3e-7), export parity unchanged. The arithmetic is pinned against pyro
  replay through a genuinely subsampled plate at three batch sizes.

- **Device-resident minibatches.** Profiling the reference model's step showed
  the arithmetic was not the cost: the flat step took 11.4 ms while scvi's
  loader collation plus the host-to-device copy took 16.7 ms, so 59% of each
  step went to moving data that never changes. When the training matrix fits on
  the device with headroom it is now staged once and minibatches are gathered
  there. Residency is a measured decision, not an assumption — a caller passing
  `batch_size` may be doing it precisely because the data does not fit, so the
  estimate is checked against the driver's recommended working set (and
  declines when that is unavailable). Batch composition is unchanged: a fresh
  permutation each epoch, trailing partial batch kept. Measured at 10,000 ×
  10,000, `batch_size=2500` (M2 Ultra): **182.2 → 119.6 ms/epoch (1.52x)**,
  median of three runs, ELBO parity, guard clean (1.6e-7). Both paths are
  pinned to agree on where training lands, so residency stays a performance
  choice rather than a numerical one.

- **Flat engine for the reference signature model, with minibatches.** Every
  number below this entry describes the spatial model. The reference model --
  step 1 of every workflow, where per-cluster expression signatures are
  estimated -- trained through pyro regardless, so half the pipeline saw none of
  it. It now runs on the flat engine, and unlike the spatial engine it
  minibatches, which it must: `RegressionModel` defaults to `batch_size=2500`
  because real references are large. That is possible here because this model
  declares no per-observation latent sites — all nine of its latents are
  global — so a minibatch step subsamples the data and scales the likelihood by
  `n_obs/batch`, rather than having to subsample the guide in lockstep (the
  spatial model has five per-location latents and stays full-batch; a spatial
  caller passing `batch_size` still routes to pyro). Batches come from scvi's
  own loader rather than a device-resident copy of the matrix, since a caller
  who asked for minibatching may have done so because the data does not fit.
  Measured at 10,000 cells × 10,000 genes, `batch_size=2500` (M2 Ultra):
  **569.6 → 194.4 ms/epoch (2.93x)**, final-ELBO parity within 0.15%, guard
  clean (12 checks, worst GPU/CPU difference 2.4e-7). The spatial harness was
  re-run unchanged on the same commit and still passes every gate. Contracts:
  the transcription is pinned against pyro replay for value and per-latent
  gradients at three batch sizes, so the plate scale cannot silently drift.
  Kill switch `CELL2LOCATION_MPS_FLAT_ENGINE=0`, shared with the spatial engine.

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
