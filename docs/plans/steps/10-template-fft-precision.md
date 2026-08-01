---
step_id: 10-template-fft-precision
phase: C
status: in_progress
github_issue: https://github.com/UCSC-Transients/dark-hunter_rv/issues/87
branches:
  - step/10-template-fft-precision
depends_on: [01-benchmark-cool-precision]
blocks: [03-method-fusion-coverage]
master_todo_id: template-fft-precision
related_legacy_plans:
  - template_grid_and_hβ_rv_dff5dfce.plan.md
  - three_rv_methods_e1b72701.plan.md
  - rv_methods_evaluation_plan_fcd09d94.plan.md
repo_docs_to_update:
  - docs/rv_methods_evaluation.md
  - docs/validation_playbook.md
  - docs/operations.md
---

# Step 10: Template FFT precision (per-method lane)

## Goal / science outcome

Improve **template_fft** per-epoch RV accuracy and precision on the same footing as the completed **mask_ccf** lane: debiased IVW stack, campaign metrics (median σ_RV north star), overlap vs mask, and deployable production config. Template fitting is a **separate measurement path** — PHOENIX bank search, lag-window seeding, vsini grid, and continuum/LSF choices — not chunk-layout reuse from mask.

## Scope (in) / non-goals (out)

**In:**

- Baseline overlap report (mask − template residuals vs Teff / log₁₀ S/N) on `--run-all-methods` diagnostics.
- Template-specific knobs in `darkhunter_rv/config.py` and `rv_core.py`: mask-seeded lag window, `FFT_COARSE_TOP_K`, cool/hot |RV| caps, two-phase bank search, vsini proxy grid.
- Per-chunk template QC and stack parity with mask (sigma-clip, min chunk count, debias keys).
- Optional template debias / method offsets (`method_rv_offsets.txt`, `compute_method_rv_offsets.py`) after raw stack is stable.
- Targeted diagnostics: `validation/diagnose_template_fft_star.py`, `plot_legacy_outlier_orders --overlap-csv`.

**Out:**

- Mask chunk tiling (step 02a — done for mask; template uses same **layout YAML** but different per-chunk measurement).
- Method fusion / adopted-RV policy (step 03) — after mask + template + strong-line lanes are individually tuned.
- Strong-line Voigt+Lorentz extension (step 06).

## Prerequisites

- **Frozen mask reference** for comparison: deploy `subchunks_8` + rebuilt `bias_statistics.txt` (see step 02a closure).
- Diagnostics from pipeline with `--run-all-methods` on campaign cohort (`validation_output/chunk_campaign/spectrum_list.txt` or production `output/`).
- Mask lane artifacts: step 09 (`gauss_offset`), chunk campaign (`adaptive_stack_comparison.csv`).

## North star (same as mask lane)

**Primary:** `median_sigma_rv_kms` — median per-exposure calibrated RV uncertainty after per-chunk debias and intrinsic-scatter IVW stacking (`validation/chunk_calibration.summarize_sigma_rv_metrics`).

**Secondary:** mask−template median residual vs Teff/S/N; `p90_sigma_rv_kms`; APF–APF relative gate on template-only stacks.

## Implementation tasks

### 10a — Baseline and overlap snapshot

- [x] Run `subchunks_8` production layout on bias-train + campaign list (or reuse campaign `template_fft` rows if layout matches).
- [x] `validation/rv_method_overlap_report` + `rv_method_diagnostics_report` on fixed diagnostics glob → `validation_output/template_fft_baseline/`.
- [x] Document mask−template MAD / median residual in applicability overlap (`method_regions`).
- [x] PDF triage for high-|mask−template| exposures (`plot_legacy_outlier_orders --overlap-csv`).

**Frozen reference (2026-07, keep `split`):**
- Continuum: `--continuum-mode split`; layout `calibration/chunk_layouts/subchunks_8.yaml`
- Cohort: `validation_output/chunk_campaign/spectrum_list.txt` (114 stems)
- Artifacts: `validation_output/template_fft_baseline/pipeline_blaze_split/`, `overlap_blaze_split/`, `phase0_blaze_split/`
- Overlap (mask+template valid, n=114): median(mask−template) ≈ −2.2 km/s, MAD ≈ **11.7 km/s**, mean |Δ| ≈ 15.7 km/s
- Phase0 p75 (`phase0_blaze_split`): **10/29 (34.5%) `B_template_outlier`**; per-chunk median |tpl−mask| ≈ 26.6 km/s
- Older larger overlap (`phase0_high_snr`): 32/121 class-B — continuum compare reference only; **A/B vs `phase0_blaze_split`**

### 10b — Template measurement knobs

- [x] Triage class-B (`phase0_blaze_split`): warm Teff; many fixed keys with **vsini≈70** (rejected-proxy path).
- [x] **Bugfix:** HiRes PHOENIX `_parse_lte_hires_filename` dropped minus on dash-split `[M/H]` (`-1.0`→`+1.0`) → empty banks for metal-poor stars (commit on #87 branch).
- [x] Cool rejected/nonfinite vsini grid → 12 km/s (`VSINI_PROXY_REJECTED_GRID_KMS_COOL` / `_NONFINITE_…_COOL`); hot keeps 75/45.
- [x] Class-B + 114-stem campaign before/after (arm `ab_cool_vsini12_mhfix`).
  - Class-B (10 stems): median |Δ| **15.15 → 1.08**; class-B **10 → 3** ([CLASS_B_COMPARISON.md](../../validation_output/template_fft_baseline/ab_cool_vsini12_mhfix/CLASS_B_COMPARISON.md)).
  - Full 114 confirm: in progress under `pipeline_cool_vsini12_mhfix/` (resume after stall).
- [x] Sensitivity: seed / caps / top-K / peak-pick deferred — vsini+MH arm sufficient for class-B collapse.
- [x] vsini grid / PHOENIX bank — MH parse fix + cool rejected=12 addressed primary failure mode.
- [ ] Reject rules: `ccf_flat_like`, per-chunk QC parity with mask — only if 114 confirm regresses.

### 10c — Template debias and deploy

- [ ] Template-specific `bias_statistics` or method-offset table (mask remains truth reference unless fusion says otherwise).
- [ ] Rebuild bias on `calibration/bias_train.txt` with frozen layout + template path.
- [ ] Record production defaults in `docs/operations.md`; refit catalog on ziggy.

## Key files

- `darkhunter_rv/rv_core.py` — `estimate_rv_fft_with_ccf`, velocity window, two-phase search
- `darkhunter_rv/config.py` — template FFT caps, coarse top-K, mask-seed width
- `darkhunter_rv/pipeline.py` — template chunk loop, stack, diagnostics rows
- `validation/rv_method_overlap_report.py`, `validation/rv_method_diagnostics_report.py`
- `validation/diagnose_template_fft_star.py`
- `validation/compute_method_rv_offsets.py`

## Commands

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
CAMPAIGN=validation_output/chunk_campaign
CHUNK_LAYOUT=calibration/chunk_layouts/subchunks_8.yaml

# Baseline overlap (mask vs template on existing diagnostics)
PYTHONPATH=. python3 -m validation.rv_method_overlap_report \
  --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \
  --gaia-summary-dir output \
  --out-dir validation_output/template_fft_baseline/overlap

PYTHONPATH=. python3 -m validation.rv_method_diagnostics_report \
  --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \
  --legacy-summary-dir output \
  --out-dir validation_output/template_fft_baseline/teff_residuals

# Single-star deep dive
PYTHONPATH=. python3 -m validation.diagnose_template_fft_star \
  --spectrum /Users/rfoley/darkhunter/rvs/data/Gaia_DR3_<id>_epoch_1.txt \
  --out-dir validation_output/template_fft_baseline/diag_<id>

# Outlier order PDFs (mask vs template disagree)
PYTHONPATH=. python3 -m validation.plot_legacy_outlier_orders \
  --overlap-csv validation_output/template_fft_baseline/overlap/overlap_enriched_per_exposure.csv \
  --threshold-kms 50 \
  --out-dir validation_output/template_fft_baseline/outlier_pdfs
```

## Acceptance criteria

- Published baseline: median σ_RV (template stack), mask−template residual vs Teff/S/N, overlap fraction.
- At least one template knob change documented with before/after on campaign cohort (median σ_RV or residual MAD).
- Template production config committed (config + optional offsets file); playbook recipe added.
- Explicit statement: template lane ready for step 03 fusion inputs (or list remaining gaps).

## Tests / validation

- Existing: `tests/test_weighted_template_fft.py`, pipeline integration tests.
- Add template-campaign tests if a benchmark module is created (mirror step 09).

## Propagation checklist (on merge)

- [ ] Master plan todo `template-fft-precision` → completed or in_progress
- [ ] INDEX.md step 10 row
- [ ] `docs/rv_methods_evaluation.md` — template validity and tuning section

## Open decisions (locked for #87)

- Template debias: **`method_rv_offsets.txt` (mask truth)** after raw A/B; no separate template `bias_statistics` in first pass.
- Hot-star seed: triage first; only widen/disable if diagnostics show truncation (class-B are cool Teff≲6150 — seed less likely).
- Primary metric: **↓ class-B rate + ↓ mask−template MAD** (not σ_RV alone).
- Triage hypothesis (phase0_blaze_split): class-B stems prefer **vsini≈70** templates (`VSINI_PROXY_REJECTED_GRID_KMS=75` path); D_agree often vsini≈12. First A/B: cooler rejected-vsini grid for cool stars.

## Relationship to other steps

| Step | Role |
|------|------|
| 01 | Mask scatter benchmark + Phase A gates (mask reference) |
| 02a | Chunk layout `subchunks_8` (shared YAML; template uses same splits) |
| 09 | Mask CCF estimator (`gauss_offset`) — **not** template estimators |
| 03 | Fusion after lanes 09 + 10 + 06 are baselined |
| 08 | External validation uses adopted RV; literature master from step 00 |
