---
step_id: 08-external-rv-crosscheck
phase: E
status: complete  # full CLI+report shipped; ≥10-star APF overlap still data-limited (documented PARTIAL)
github_issue: https://github.com/astrofoley/dark-hunter_rv/issues/45
branches:
  - step/08-external-rv-crosscheck
depends_on: [00-literature-rv-master]
# soft: 07-sb2-search (required for 08-full only; lite OK pre-SB2)
blocks: []
master_todo_id: external-rv-crosscheck
related_legacy_plans:
  - pipeline_legacy_diagnostics_e98cdb98.plan.md
repo_docs_to_update:
  - docs/validation_playbook.md
---

# Step 08: External RV cross-check (literature + catalogs)

## Goal / science outcome

Systematic comparison of pipeline adopted RVs and orbit fits to published literature (primary: El-Badry 2024) and LAMOST/RAVE.

## Scope (in) / non-goals (out)

**In:** `validation/compare_literature_rvs.py`; join on `gaia_dr3_id` + nearest BJD; orbit-fit overlay from master CSV; LAMOST/RAVE extension.

**Out:** Rebuilding literature master (step 00).

## Prerequisites

- `calibration/literature_rv_master.csv`
- Pipeline summaries and diagnostics for overlapping Gaia IDs

## Implementation tasks

- [x] CLI: load master CSV + pipeline diagnostics (lite; summaries optional later)
- [x] Per-epoch ΔRV vs published err; per-star bias/RMS tables (lite)
- [x] Optional: wire literature points in `fit_apf_rv_keplerian.py` plots from master CSV (`--literature-master`)
- [x] Extend to `external_rvs` from star summaries (LAMOST/RAVE)
- [x] Playbook recipes and example output paths (lite + full)

## Key files

- `calibration/literature_rv_master.csv`
- `validation/compare_literature_rvs.py` (new)
- `validation/diagnose_legacy_campaign.py` (reference joins)
- `fit_apf_rv_keplerian.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
# Lite (mask + template vs El-Badry master; no LAMOST/RAVE)
python -m validation.compare_literature_rvs \
  --master calibration/literature_rv_master.csv \
  --diagnostics-glob \
    '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/validation_output/template_fft_baseline/pipeline_cool_vsini12_mhfix/*_diagnostics.csv' \
  --report-dir validation_output/literature_crosscheck_lite \
  --copy-key-table calibration/literature_crosscheck_lite/per_star_bias_rms.csv
```

**Lite note (2026-08-01):** soft-dep on step 07; CLI + playbook + nearest-BJD join done.

**Full note (2026-08-02):** Using `output/Gaia_DR3_*_diagnostics.csv` reaches **n_stars=8** (all master IDs that have local spectra). Remaining 15 master systems lack spectra under `/Users/rfoley/darkhunter/rvs/data` — n≥10 needs ingest or master expansion. Empty-`teff` diagnostics no longer crash the loader.

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m validation.compare_literature_rvs \
  --master calibration/literature_rv_master.csv \
  --diagnostics-glob \
    '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/validation_output/template_fft_baseline/pipeline_blaze_split/*_diagnostics.csv' \
  --summaries-glob \
    '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_*_summary.txt' \
  --report-dir validation_output/literature_crosscheck_full \
  --copy-key-table calibration/literature_crosscheck_full/per_star_bias_rms.csv
```

## Acceptance criteria

**Lite path (pre-SB2):** overlapping Gaia IDs from existing diagnostics; report under `validation_output/literature_crosscheck_lite/` + tracked key table `calibration/literature_crosscheck_lite/per_star_bias_rms.csv`.

**Full path:**

- Report covers ≥10 El-Badry 2024 stars with both APF and literature epochs
- Published M_star, M2, P_orb joined for orbit-fit QA table
- LAMOST/RAVE rows compared where present in summaries

## Tests / validation

- [x] Unit test: nearest BJD join logic + LAMOST/RAVE summary load (`tests/validation/test_compare_literature_rvs.py`)
- [x] Compare Gaia NS1 literature vs pipeline epoch table (report artifacts)
- [x] Unit test: `--literature-master` overlay (`tests/test_fit_apf_rv_keplerian.py`)

## Propagation checklist (on merge)

- [ ] Master todo `external-rv-crosscheck` → completed (orchestrator/INDEX owned elsewhere; mark on merge)
- [x] Playbook + step doc CLI path for full report

## Open decisions

- Match adopted RV vs mask-only vs all methods in comparison?
