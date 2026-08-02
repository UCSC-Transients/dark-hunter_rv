---
step_id: 07-sb2-search
phase: D
status: in_progress
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/44
branches:
  - step/07a-sb2-detection
  - step/07b-sb2-reporting
  - step/07c-sb2-orbit-optional
depends_on: [06-strong-line-line-list]
blocks: []
master_todo_id: sb2-search
related_legacy_plans:
  - rv_pipeline_roadmap_3a7b3787.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
---

# Step 07: SB2 search and reporting

## Goal / science outcome

Detect and report double-lined systems where appropriate; optional two-lined orbit extension.

## Scope (in) / non-goals (out)

**In:** CCF asymmetry / dual-Gaussian / profile flags; `sb2_candidate`, primary/secondary RV columns when resolved; Gaia NSS cross-link.

**Out (07c optional):** Full SB2 Keplerian MCMC in `fit_apf_rv_keplerian.py`.

## Prerequisites

- Mask CCF diagnostics (`gauss_ok`, peak shape)
- Multi-epoch data for known binaries

## Implementation tasks

### 07a (`step/07a-sb2-detection`)

- [x] Prototype dual-Gaussian CCF fit or bisector metric in `rv_core.py`
  - Implemented as `BiGaussCcfResult` / `estimate_ccf_bi_gauss_from_arrays` / `estimate_ccf_secondary_seeded` in `ccf_rv_estimators.py`; scored via `darkhunter_rv.sb2` (median-CCF gate). Not inlined into `rv_core.py`.
- [x] Per-chunk SB2 scores in diagnostics
  - Per-order CCF rows in `sb2_orders.csv`; exposure-level bi-Gauss score + `sb2_candidate` in `sb2_epochs.csv` / `sb2_report.json`.
  - **Pipeline fuse:** when mask CCF already computed, `process_spectrum` writes `sb2_candidate` + `sb2_rv1_kms` / `sb2_rv2_kms` / `sb2_delta_chi2` onto `*_diagnostics.csv` (opt-out `--no-sb2-score`).

### 07b (`step/07b-sb2-reporting`)

- [x] Exposure-level `sb2_candidate` flag + columns in CSV/summary
  - `python -m validation.sb2_search` writes `sb2_epochs.csv` (`sb2_candidate`, `rv1_kms`, `rv2_kms`, `delta_chi2`, ...) and `sb2_report.json`.
  - Pipeline `*_diagnostics.csv` also carries `sb2_candidate` / primary–secondary RV columns when mask CCF runs.
- [ ] Validation report: fraction flagged vs Gaia NSS SB2
  - Recipe documented in `docs/validation_playbook.md` (Gaia NSS cohort fraction); **full table deferred** (TODO: `validation/sb2_nss_cohort_report.py`).

### 07c (`step/07c-sb2-orbit-optional`, defer if needed)

- [x] Two-lined Keplerian likelihood sketch or separate module
  - Kept from WIP: `darkhunter_rv/sb2_rv_fit.py` + `validation/sb2_orbit_fit.py` (independent + joint variants). Optional path; single-lined `fit_apf_rv_keplerian.py` unchanged.

## Key files

- `darkhunter_rv/ccf_rv_estimators.py` (`BiGaussCcfResult`, seeded secondary)
- `darkhunter_rv/sb2.py` (detection + template separation)
- `darkhunter_rv/sb2_rv_fit.py` / `validation/sb2_orbit_fit.py` (optional 07c)
- `validation/sb2_search.py`, `validation/sb2_fit_diagnostics_report.py`
- `fit_apf_rv_keplerian.py` (single-lined; unchanged)

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
# Per-star SB2 search (mask CCF + optional two-template fit):
python -m validation.sb2_search \
  --gaia-id 77413727493690112 \
  --spec-root /Users/rfoley/darkhunter/rvs/data \
  --out-dir validation_output/sb2_77413727493690112
# Optional orbit on sb2_epochs.csv:
python -m validation.sb2_orbit_fit --sb2-dir validation_output/sb2_77413727493690112
```

## Acceptance criteria

- SB2 flag does not fire on high-S/N cool calibration stars (low false positive)
- Known asymmetric CCF test case flagged
- Documented limitations (single-lined orbit fitter unchanged unless 07c done)

## Tests / validation

- Synthetic double-lined injection test
- Manual check on suspect epochs from overlap discordance

## Propagation checklist (on merge)

- [ ] Master todo `sb2-search` → completed
- [ ] Update `rv_pipeline_roadmap` phase 3 binary section

## Open decisions

- Spectral decomposition per epoch vs time-series only? (WIP does multi-epoch template fit + per-epoch separated spectra.)
- 07c in scope for this step or separate future step? **Kept:** orbit modules tracked; full MCMC into `fit_apf_rv_keplerian.py` still out of scope.
- [x] Pipeline `*_diagnostics.csv` `sb2_candidate` fuse (low-cost default from mask CCF; `--no-sb2-score` to skip).
- Gaia NSS cohort fraction table: deferred (playbook recipe + TODO).
