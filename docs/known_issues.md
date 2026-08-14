# Known issues

## Fused NB kernel: intermittent corruption under GPU load (torch 2.12.x)

Status: open, under investigation (2026-08).

On macOS with torch 2.12.1, kernels dispatched through
`torch.mps.compile_shader` can **partially execute** when a Metal
command-buffer boundary falls near the dispatch: in controlled experiments,
entire threadgroups' writes were missing from the output buffer even after
`torch.mps.synchronize()`. For this fork that means the fused negative-binomial
kernel can intermittently return a wrong log-likelihood (observed rate
0.2–3% of evaluations, strongly load-dependent; identical inputs, wrong
output). The defect is in the dispatch machinery, not the kernel's
arithmetic — the same kernel is bitwise-correct on the other >99% of calls,
and barrier-based mitigations (event ordering, pre-dispatch synchronize) were
tested and do not close it.

Practical guidance:

- **The runtime numerical guard detects this.** It cross-checks the training
  computation between MPS and CPU on your data during your run; a corrupted
  evaluation shows up as a guard divergence. Treat any guard warning as real.
- `CELL2LOCATION_MPS_FUSED_NB=0` disables the fused kernel; the eager path
  produced zero corrupted results across every soak test and is bitwise
  deterministic. Cost: roughly 2x on the spatial training epoch.
- (Resolved 2026-08-14.) The flat engine used to die on some real reference
  datasets — one gene's exact gradient through `alpha = x⁻²` exceeds float32
  range while the loss is still finite, and Adam turns the inf into NaN
  parameters. The trainer now zeroes exactly the non-finite gradient elements
  (warning logged); finite-gradient training is bit-identical. An earlier note
  here attributed the NaN to missing epsilon floors in the eager likelihood —
  that was wrong; the value was finite, the gradient was not.

This section will be updated when the defect is fixed upstream or worked
around; the investigation's reproduction scripts live outside the package.
