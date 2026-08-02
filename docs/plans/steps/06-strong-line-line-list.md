---
step_id: 06-strong-line-line-list
phase: C
status: complete
github_issue: https://github.com/UCSC-Transients/dark-hunter_rv/issues/43
branches:
  - step/06-strong-line-line-list
  - step/10-template-fft-precision  # API shipped in PR #88
  - step/06-strong-line-teff-sweep
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

Empirically chosen strong lines per Teff/S/N; extend `measure_h_beta_rv` API beyond Hβ-only for the `strong_lines` method.

## Scope (in) / non-goals (out)

**In:** Line list study; generalized Voigt+Lorentz per line; pipeline still exposes one `strong_lines` row.

**Out:** Reviving Gaussian multi-line centroids as product RV.

## Prerequisites

- Hβ path in `rv_core.py`
- Overlap report Teff strata

## Implementation tasks

- [x] Survey candidates from `STRONG_LINES` / literature (`docs/broad_line_method.md`)
- [x] Validation sweep: recovery vs mask/template per Teff bin (`validation/strong_line_teff_sweep.py` on 114 stems)
  - Hα uncovered on APF; product/free always Hβ; oracle non-Hβ wins 8/114 with tiny residual gain → **stay Hβ-primary**
- [x] Refactor `measure_strong_line_voigt_lorentz(rest=...)` from Hβ code
- [x] Wire best line(s) into pipeline `strong_lines` row (Teff-ordered single best)
- [x] Update tests in `tests/test_h_beta_rv.py` (Hα synthetic + Teff order)

## Open decisions (locked)

- **Single best line per exposure** (not joint multi-line) for first ship.
- **APF production:** Hβ primary; Hγ/Hδ fallback when Hβ fails; Hα last (no APF coverage).

## Key files

- `darkhunter_rv/rv_core.py`
- `darkhunter_rv/continuum.py` (`STRONG_LINES`)
- `validation/h_beta_profile_method_report.py`
- `validation/strong_line_teff_sweep.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m validation.strong_line_teff_sweep \
  --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \
  --overlap-csv validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/overlap/overlap_enriched_per_exposure.csv \
  --data-root /Users/rfoley/darkhunter/rvs/data \
  --out-dir validation_output/strong_line_teff_sweep \
  --continuum-mode sinc_blaze
PYTHONPATH=. python -m pytest tests/test_h_beta_rv.py tests/test_strong_line_teff_sweep.py -q
```

## Acceptance criteria

- Documented line list with Teff applicability
- At least one additional line beyond Hβ tested on real spectra OR explicit decision to stay Hβ-only with rationale
- No regression on hot-star Hβ performance

## Tests / validation

- Synthetic line recovery tests per rest wavelength
- Overlap residuals for strong_lines vs mask on cool stars
- 114-stem force-fit sweep vs mask (`SWEEP_SUMMARY.md`)

## Propagation checklist (on merge)

- [ ] Master todo `strong-line-line-list` → completed
- [ ] Update `three_rv_methods` plan
- [ ] Close #43
