---
step_id: 11-epoch-ccf-matrix
phase: C
status: in_progress  # 11a–c on main (#95/#98→#101); 11d fusion-doc + optional pipeline rows remain
github_issue: https://github.com/UCSC-Transients/dark-hunter_rv/issues/94
branches:
  - step/11-epoch-ccf-matrix
  - step/11-epoch-ccf-matrix-cli
merged: 2026-08-02 via #95/#98 → #101 (c6801f7)
depends_on: [01-benchmark-cool-precision]
blocks: [03-method-fusion-coverage]
# Soft: fusion can consume epoch_ccf fills; matrix work can start before fusion
master_todo_id: epoch-ccf-matrix
related_legacy_plans:
  - docs/plans/steps/05-short-pair-epoch-ccf.md
repo_docs_to_update:
  - docs/rv_methods_evaluation.md
  - docs/validation_playbook.md
  - docs/plans/ATTACK_ORDER.md
---

# Step 11: Epoch–epoch CCF matrix (relative RV path)

## Goal / science outcome

Build a **spectrum–spectrum cross-correlation** path that does **not** require an absolute RV zeropoint. For a star with \(N\) epochs:

1. Form the full relative-RV matrix \(\Delta v_{ij}\) (and \(\sigma_{ij}\)) by CCF of epoch \(i\) vs epoch \(j\), including **auto-correlation** \(i=j\) (should be ~0; QC).
2. Combine with the sparse vector of **absolute** RVs \(A_i \pm \sigma_{A,i}\) from mask / template / strong_lines / fusion.
3. Infer best absolute RVs \(\hat{A}_i\) for epochs missing absolutes (propagate from any anchors). If **no** absolute exists, keep \(\Delta v_{ij}\) as a scientifically useful relative solution (fixable zeropoint).

Especially valuable for **low S/N** epochs where template/mask/strong fail but pairwise CCF against a better epoch still peaks.

## Scope (in) / non-goals (out)

**In:**

- Log-λ (or velocity-grid) continuum-normalized CCF between every pair of epochs of the same star.
- Symmetric matrix storage; upper triangle compute with \(\Delta v_{ji} = -\Delta v_{ij}\).
- Auto-correlation diagonal for QC (peak at 0, width → resolution / vsini proxy).
- Estimator combining \(\{\Delta v_{ij},\sigma_{ij}\}\) + \(\{A_i,\sigma_{A,i}\}\) (GLS / weighted least squares on epoch parameters).
- Pipeline or post-process method tag e.g. `epoch_ccf` / `epoch_ccf_filled`.
- Validation vs short pairs (Δt≈0 ⇒ \(\Delta v \approx 0\)) and vs abs−abs differences when both epochs have absolutes.

**Out:**

- Replacing mask as the primary absolute calibrator when mask is good.
- Joint SB2 two-component matrix (step 07; may reuse machinery later).
- Cross-star CCF.

## Prerequisites

- Multi-epoch spectra on disk; continuum path consistent with production (`split` / blaze as frozen).
- At least the ability to load two normalized orders or a merged 1D product for CCF (reuse mask/template prep where possible).
- Absolute RVs optional but preferred for fill tests (Phase 1 baselines).

## Mathematical sketch

**Relative observations**

\[
\Delta v_{ij}^{\mathrm{obs}} = v_i - v_j + \epsilon_{ij}, \quad
\mathrm{Var}(\epsilon_{ij}) = \sigma_{ij}^2
\]

(auto-corr: expect \(\Delta v_{ii} \approx 0\)).

**Absolute observations** (when present)

\[
A_i^{\mathrm{obs}} = v_i + \eta_i, \quad \mathrm{Var}(\eta_i) = \sigma_{A,i}^2
\]

**Parameters:** epoch velocities \(v_0,\ldots,v_{N-1}\) (or \(v_0\) free + relatives if no absolute).

**Estimator (v1):** weighted least squares stacking all pairwise and absolute rows; if no absolutes, fix \(v_0 = 0\) (or median-zero) and report relatives only.

**Errors:** formal covariance from WLS; inflate using short-pair / abs-consistency scatter.

## Implementation tasks

### 11a — Pairwise CCF engine

- [x] API: `epoch_pair_ccf(spec_i, spec_j) → {dv_kms, err_kms, peak, width, qc}`
- [x] Shared wavelength prep (orders used, telluric mask, continuum)
- [x] Auto-correlation path + diagonal QC flags
- [x] Unit tests: synthetic Doppler shift recovery; antisymmetric \(\Delta v_{ji}\)

### 11b — Matrix builder

- [x] For one Gaia ID: compute \(i \le j\) pairs (parallelizable)
- [x] Persist `epoch_ccf_matrix.npz` / CSV (long form: i, j, dv, err, qc)
- [x] CLI: `validation/epoch_ccf_matrix.py --gaia-id ...`

### 11c — Absolute + relative combiner

- [x] Load abs RVs from diagnostics / summary / fusion columns
- [x] WLS (or GLS) fill; output \(\hat{A}_i\), \(\sigma_i\), `n_abs_anchors`, `float_zeropoint`
- [x] Tests: one absolute + known relatives → recovers all; zero absolute → relatives only

### 11d — Product wiring

- [x] Optional pipeline / post-process rows: `epoch_ccf_rel`, `epoch_ccf_abs_fill` (matrix CLI CSV columns; **not** default adopted RV)
- [x] Opt-in enrich hook: `--enrich-diagnostics-glob` / `--enrich-out-dir` attaches columns (+ optional method rows) when matrix fill exists; **not** default adopted RV
- [x] Document interaction with step 03 fusion (epoch fill as optional prior / post-fusion salvage, not cascade replacement) in `docs/rv_methods_evaluation.md`
- [x] Playbook recipe for low-S/N multi-epoch stars
- [ ] Human accept: enable `epoch_ccf_*` as default adopted RV (still open — leave step `in_progress`)

## Parallel subagent split

| Subagent | Work | Sync point |
|----------|------|------------|
| **11-engine** | Pair CCF + synthetic Doppler tests | Matrix schema |
| **11-matrix** | Star-level pair loop, I/O, pair-parallel job | Combiner input format |
| **11-combine** | WLS fill + abs/rel unit tests | Engine dv/err contract |
| **11-validate** | Short-pair Δ≈0; abs−abs vs matrix; low-S/N case study | After 11a–c |

## Key files (proposed)

- `darkhunter_rv/epoch_ccf.py` (engine + combiner)
- `validation/epoch_ccf_matrix.py`
- `tests/test_epoch_ccf.py`
- Diagnostics / summary columns TBD

## Commands (target)

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m validation.epoch_ccf_matrix \
  --gaia-id <id> \
  --data-root /Users/rfoley/darkhunter/rvs/data \
  --abs-diagnostics-glob 'output/Gaia_DR3_<id>_diagnostics.csv' \
  --out-dir validation_output/epoch_ccf/<id>
PYTHONPATH=. python -m pytest tests/test_epoch_ccf.py -q
```

## Acceptance criteria

- Synthetic: recover injected \(\Delta v\) within formal errors.
- Real multi-epoch star: matrix antisymmetric; diagonal ~0; when ≥1 abs anchor, filled abs agree with independent abs on held-out epochs within inflated σ.
- Low-S/N epoch with failed mask/template still gets finite fill when paired against a high-S/N epoch (documented example).
- Zero-anchor mode writes relative-only product without crashing.

## Tests / validation

- Synthetic Doppler grid
- Short-pair cohort (links step 05a)
- Compare \(\Delta v_{ij}\) to \(A_i - A_j\) when both abs finite

## Relationship to step 05

| 05 (legacy) | 11 (this step) |
|-------------|----------------|
| Epoch CCF as **QC monitor** | Epoch CCF as **RV measurement / fill** |
| Compare to adopted ΔRV | Produce relatives + abs fill |
| Blocked on plots (04) in old plan | Starts after abs baselines exist; parallel design earlier |

Keep **05a short-pair calibration** as a validation consumer of 11.

## Propagation checklist (on merge)

- [ ] Create GitHub issue; set `github_issue` frontmatter
- [ ] INDEX.md row for step 11
- [ ] Update ATTACK_ORDER Phase 4 checkboxes
- [ ] Soft-update step 05: 05b superseded by 11

## Open decisions

- CCF on merged 1D vs per-order then IVW Δv?
- Lag estimator: reuse `gauss_offset` / mask CCF machinery vs FFT template-style?
- Fusion policy: run epoch fill **before** fusion (salvage inputs) or **after** (only fill adopted holes)?
- BERV / barycentric: ensure both spectra in same frame before CCF.
