---
step_id: 06-strong-line-line-list
phase: C
status: complete
github_issue: https://github.com/UCSC-Transients/dark-hunter_rv/issues/43
related_issues:
  - https://github.com/UCSC-Transients/dark-hunter_rv/issues/91
  - https://github.com/UCSC-Transients/dark-hunter_rv/issues/92
  - https://github.com/UCSC-Transients/dark-hunter_rv/issues/93
pull_request: https://github.com/UCSC-Transients/dark-hunter_rv/pull/90
merged: 2026-08-02
branches:
  - step/06-strong-line-line-list
  - step/10-template-fft-precision  # Voigt+Lorentz API shipped in PR #88
  - step/06-strong-line-teff-sweep   # product IVW path in PR #90
depends_on: [05-short-pair-epoch-ccf]
blocks: [07-sb2-search]
master_todo_id: strong-line-line-list
related_legacy_plans:
  - template_grid_and_hβ_rv_dff5dfce.plan.md
  - three_rv_methods_e1b72701.plan.md
repo_docs_to_update:
  - docs/broad_line_method.md
---

# Step 06: Strong-line rest wavelengths and multi-line Voigt+Lorentz

## Goal / science outcome

Empirically chosen strong lines per Teff/S/N; extend Voigt+Lorentz beyond Hβ-only; ship one debiased, quality-weighted `strong_lines` RV per exposure.

## Scope (in) / non-goals (out)

**In:** Line list study; generalized Voigt+Lorentz per line; inclusion QC; per-line mask offsets + qualities; IVW combine into one `strong_lines` row.

**Out:** Reviving Gaussian multi-line centroids as product RV; SB2 / mask retile.

## Prerequisites

- Hβ path in `rv_core.py` / `measure_strong_line_voigt_lorentz` (PR #88)
- Overlap report Teff strata

## Implementation tasks

- [x] Survey candidates from `STRONG_LINES` / literature (`docs/broad_line_method.md`)
- [x] Balmer Teff sweep vs mask (`validation/strong_line_teff_sweep.py` on 114 stems) — **#43**
  - Hα uncovered on APF; product/free always Hβ; oracle non-Hβ wins 8/114 with tiny residual gain → **stay Hβ-primary** for Balmer preference
- [x] Metal / secondary candidate sweep (`validation/strong_line_candidate_sweep.py`)
  - **Keep:** Mg I b₂/b₃, Ca I 6122, Ca I 6162, Ca I 4227
  - **Exclude:** Ca H&K, red IR (8498/8807); hold Ca II 8662 (fringe)
- [x] Refactor `measure_strong_line_voigt_lorentz(rest=...)` from Hβ code (PR #88)
- [x] **#91** Wire keep metals into `product_strong_line_rests` / pipeline
- [x] **#92** Inclusion gates: depth, width, err, telluric, continuum `median(flux/eflux)` near line
- [x] **#93** Debias + quality file + IVW: `w = Q_line × (S/N_near_line)²`
  - File: `calibration/strong_line_offsets.txt` (offset + quality; CaI6122 = 1)
  - Pipeline: `read_strong_line_calibration` → `combine_strong_line_rvs`
- [x] Tests: `tests/test_strong_lines_product.py`, Teff/candidate sweep tests, Hβ synthetic

## Open decisions (locked)

- **APF Balmer preference:** Hβ primary; Hγ/Hδ fallback; Hα last (no APF coverage).
- **Product RV:** multi-line IVW after inclusion (not single-best-line-only). Candidate order: Hβ → MgIb2 → CaI6122 → CaI6162 → MgIb3 → CaI4227 → Hγ → Hδ → Hα.
- **Quality:** from MAD of **debiased** residuals vs mask on inclusion-gated campaign rows; separate from per-exposure S/N.

## Calibrated qualities (campaign, pending merge)

| Line | offset (km/s) | Q |
|------|-------------:|--:|
| CaI6122 | 1.274 | 1.000 |
| CaI6162 | 1.687 | 0.895 |
| MgIb2 | 1.553 | 0.612 |
| MgIb3 | 2.658 | 0.546 |
| Hβ | 0.525 | 0.322 |
| CaI4227 | 2.696 | 0.178 |
| Hγ / Hδ / Hα | 0 | 0.20 / 0.20 / 0.15 (placeholders) |

## Key files

- `darkhunter_rv/strong_lines.py`
- `darkhunter_rv/pipeline.py` (strong_lines IVW row)
- `darkhunter_rv/rv_core.py`
- `calibration/strong_line_offsets.txt`
- `docs/broad_line_method.md`
- `validation/strong_line_teff_sweep.py`
- `validation/strong_line_candidate_sweep.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m validation.strong_line_teff_sweep \
  --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \
  --overlap-csv validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/overlap/overlap_enriched_per_exposure.csv \
  --data-root /Users/rfoley/darkhunter/rvs/data \
  --out-dir validation_output/strong_line_teff_sweep \
  --continuum-mode sinc_blaze
PYTHONPATH=. python -m pytest \
  tests/test_strong_lines_product.py \
  tests/test_h_beta_rv.py \
  tests/test_strong_line_teff_sweep.py \
  tests/test_strong_line_candidate_sweep.py -q
```

## Acceptance criteria

- Documented line list with Teff / instrument applicability
- Additional lines beyond Hβ tested on real spectra; keep/exclude decisions recorded
- One `strong_lines` diagnostics row per exposure via inclusion + Q×SNR² IVW
- No regression on hot-star Hβ path
- PR #90 CI green and merged

## Tests / validation

- Unit: inclusion, local flux/eflux S/N, debias-before-Q, file→IVW weights
- 114-stem Balmer Teff sweep (`validation_output/strong_line_teff_sweep/`)
- 114-stem metal candidate sweep (`validation_output/strong_line_candidate_sweep/`)

## Current status (2026-08-02)

- **Merged** PR [#90](https://github.com/UCSC-Transients/dark-hunter_rv/pull/90) (`dbdcdc0`).
- Issues #43 / #91 / #92 / #93 closed via PR.

## Propagation checklist (on merge)

- [x] Master todo `strong-line-line-list` → completed
- [x] Set this step `status: complete`; INDEX Merged column → 2026-08-02 + #90
- [ ] Update `three_rv_methods` plan if still open
- [x] Close #43, #91, #92, #93 (via PR)
