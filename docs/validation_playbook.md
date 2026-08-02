# RV Validation Playbook

**See also:** [operations.md](operations.md) for **`run_calibration_setup`** / **`run_production_remaining`** and [rv_methods_evaluation.md](rv_methods_evaluation.md) for adopted-RV rules.

## Environment

- From the repo root, Python needs the package on the path. In **bash/zsh**: `export PYTHONPATH=.` then run `python3 validation/...`, or one shot: `env PYTHONPATH=. python3 validation/...`.
- **tcsh/csh** does not support `VAR=value command`; use `setenv PYTHONPATH .` then `python3 validation/...`, or `env PYTHONPATH=. python3 validation/...`.

## Commands

- **Adopted-RV match plot** (step 04): with `--plots` or `--plots-focus`, pipeline writes `{stem}_adopted_rv_match.png` — continuum-normalized orders, stellar mask + strong-line markers at debiased adopted RV (prefers fusion `rv_calibrated_kms` when accepted; else cascade).
- **Full calibration (bias + method offsets + manifest):** `python -m validation.run_calibration_setup` (see [operations.md](operations.md)).
- Build bias set only:
  - `python3 validation/build_bias_set.py --input-dir output --out-dir validation_output/bias`
- Method consistency:
  - `python3 validation/evaluate_method_consistency.py --diag-glob "output/*_diagnostics.csv" --out-dir validation_output/consistency`
- Broad-line benchmark:
  - `python3 validation/benchmark_broad_lines.py --out-dir validation_output/broad_line`
- Cool high-S/N mask precision (step 01; 0.1 km/s goal):
  - `python -m validation.benchmark_cool_precision --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/benchmark_cool_precision`
  - **0.1 km/s interpretation:** north star is single-epoch calibrated mask **σ_RV** (`subchunks_8` campaign median **0.0189 km/s** — met). Also track Phase A APF–APF relative gate (median still ~0.3 km/s on overlap). `chunk_scatter_kms` from the cool benchmark is a raw pre-stack diagnostic (typically ≫ 0.1) and is **not** the epoch-precision pass/fail. Mask lane deploy: `calibration/mask_lane_deploy.md`.
- **Short-pair QC** (step 05a; Δt≈0 absolute + epoch-CCF scatter / σ_ij inflation):
  - `PYTHONPATH=. python -m validation.find_short_pairs --diagnostics-glob '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_*_epoch_*_diagnostics.csv' --data-root /Users/rfoley/darkhunter/rvs/data --epoch-ccf-root validation_output/epoch_ccf --out-dir validation_output/short_pair_qc --max-delta-days 1`
  - Optional `--same-calendar-night`; `--compute-epoch-ccf` when step-11 pairs CSV missing; `--abs-violation-kms` / `--n-sigma` for flags.
  - Artifacts: `validation_output/short_pair_qc/short_pairs.csv`, `SHORT_PAIR_QC.md`, `short_pair_sigma_scale.json`; tracked summary `calibration/short_pair_sigma_scale.json` (+ `.csv`).
  - Feed recommended scale into matrix: `--sigma-ij-scale` on `validation.epoch_ccf_matrix` (or `inflate_sigma_ij`).
- **Epoch–epoch CCF matrix** (step 11; relative RVs + optional abs fill; **not** default adopted RV):
  - `PYTHONPATH=. python -m validation.epoch_ccf_matrix --gaia-id <id> --data-root /Users/rfoley/darkhunter/rvs/data --abs-diagnostics-glob 'output/Gaia_DR3_<id>_epoch_*_diagnostics.csv' --out-dir validation_output/epoch_ccf/<id>`
  - Artifacts: `epoch_ccf_pairs.csv`, `epoch_ccf_matrix.npz`, `epoch_ccf_abs_fill.csv` (`epoch_ccf_rel` / `epoch_ccf_abs_fill` columns), `epoch_ccf_meta.json`. Diagonal auto-corr should be ~0; when abs anchors exist, see `epoch_ccf_vs_abs_delta.csv`.
  - Low-S/N salvage: run without requiring every epoch to have mask/template; pairs vs a high-S/N epoch still fill via WLS when ≥1 abs anchor (or relative-only if none).
- **Phase A baseline** (overlap inventory + calibration gates; regression vs `calibration/phase_a_baseline/reference_manifest.json`):
  - `python -m validation.rv_phase_a_baseline --master calibration/literature_rv_master.csv --summary-dir output --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/rv_phase_a_baseline`
  - Absolute gate (APF vs literature, |ΔRV| < 1 km/s): use `--no-bias-correction-applied` after a `--no-bias` pipeline rerun on overlap stars.
  - Outputs: `overlap_stars.csv`, `pair_candidates.csv`, gate summaries, `plots/` (see `calibration/phase_a_baseline/README.md`).
- **Chunk residuals** (mask-applicable cool stars; per-object and sample bias plots):
  - `python -m validation.plot_chunk_residuals --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/chunk_residuals`
  - `--overlap-only` limits to phase-A overlap stars that pass the mask region cut.
  - Per object: `*_residuals_by_spectrum.png`, `*_chunk_weighted_mean.png`; sample: `sample_per_object_chunk_bias.png`.
  - Per-spectrum clip (default): 7σ LOO + ±20 km/s (`--chunk-outlier-sigma 7`, `--chunk-max-delta-kms 20`).
  - Per-object weighted mean: only chunks with ≥3 surviving measurements (`--min-chunk-measurements 3`).
  - Full-sample clip on per-object chunk biases (default): 5σ LOO + ±10 km/s (`--sample-outlier-sigma 5`, `--sample-max-delta-kms 10`). Excluded points shown as gray ×.
- **Apply chunk calibration + relative reassessment:**
  - `python -m validation.reassess_relative_calibration --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --bias-csv validation_output/chunk_residuals/per_object_chunk_bias.csv --out-dir validation_output/chunk_calibration_assessment`
- **Chunk bias regression (stellar params + bias curve):**
  - `python -m validation.chunk_bias_regression --bias-csv validation_output/chunk_residuals/per_object_chunk_bias.csv --summary-dir output --out-dir validation_output/chunk_bias_regression --reassess-relative-gate`
  - Outputs: `regression_adjusted_chunk_bias.csv`, `regression_chunk_bias_for_calibration.csv`, `CHUNK_OPTIMIZATION_ADVICE.md`
- **Evaluate chunk layouts (offline merge / compare N+1 edges):**
  - `python -m validation.evaluate_chunk_layout --layouts calibration/chunk_layouts/*.yaml --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/chunk_layout_eval`
  - Sub-chunk layouts require pipeline rerun with `--subchunks N` before bias loop.
- **Parametric grid search (rough N: subchunks 1,2,4 + merge widths 1,2,4):**
  - `python -m validation.chunk_grid_search --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/chunk_grid_search`
- **Full chunk campaign (pipeline + cache + edge presets + stages B/C):**
  - `python -m validation.chunk_campaign --run-pipeline --out-dir validation_output/chunk_campaign`
  - Uses `measurement_cache.csv` to avoid re-measuring identical chunks; per-layout diagnostics under `validation_output/chunk_campaign/diagnostics/<layout>/`
- Error model calibration:
  - `python3 validation/calibrate_error_model.py --diag-glob "output/*_diagnostics.csv" --out-dir validation_output/error_model`
- Full campaign report:
  - `python3 validation/run_campaign.py --orders-dir output --diag-glob "output/*_diagnostics.csv" --out-dir validation_output/campaign`
- Legacy vs new pipeline (APF + Gaia):
  - `env PYTHONPATH=. python3 validation/diagnose_legacy_campaign.py --data-dir ../data --legacy-output-dir ../output --new-output-dir validation_output/pipeline_rerun --report-dir validation_output/diagnose_legacy --spectrum-glob 'Gaia_DR3_1702*.txt' --run-pipeline --query-gaia --dump-gaia-json -- --instrument APF --run-all-methods --log-level ERROR`
  - Omit `--no-bias` to apply repo [`bias_statistics.txt`](bias_statistics.txt) (match legacy if it was debiased). Add `--no-bias` only for an explicit no-debias comparison.
  - Pipeline flags go **after a lone `--`**.
  - **Multi-star:** use a broad glob (e.g. `Gaia_DR3_*.txt`) and **`--multi-star`**; reports go under `report-dir/<source_id>/`. Optional `--min-epochs 10` and `--write-combined-csv`.
- **Literature RV cross-check (step 08; El-Badry master vs mask/template + LAMOST/RAVE):**
  - Lite: `python -m validation.compare_literature_rvs --master calibration/literature_rv_master.csv --diagnostics-glob '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/validation_output/template_fft_baseline/pipeline_blaze_split/*_diagnostics.csv' --report-dir validation_output/literature_crosscheck_lite --copy-key-table calibration/literature_crosscheck_lite/per_star_bias_rms.csv`
  - Full (LAMOST/RAVE from summaries): `python -m validation.compare_literature_rvs --master calibration/literature_rv_master.csv --diagnostics-glob '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/validation_output/template_fft_baseline/pipeline_blaze_split/*_diagnostics.csv' --summaries-glob '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_*_summary.txt' --report-dir validation_output/literature_crosscheck_full --copy-key-table calibration/literature_crosscheck_full/per_star_bias_rms.csv`
  - Optional: add `strong_lines` via `--methods mask_ccf,template_fft,strong_lines`.
  - Orbit-plot overlay from master CSV: `python fit_apf_rv_keplerian.py --summary /Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_<id>_summary.txt --literature-master calibration/literature_rv_master.csv`
  - Outputs (gitignored): `validation_output/literature_crosscheck_full/{REPORT.md,epoch_pairs.csv,per_star_bias_rms.csv,orbit_qa.csv,external_*.csv}`; tracked key tables under `calibration/literature_crosscheck_full/`.
  - Nearest-BJD join on `gaia_dr3_id` for pipeline↔literature and external↔literature (LAMOST_LRS / LAMOST_MRS / RAVE_DR6 by default).
  - Prefer `output/Gaia_DR3_*_diagnostics.csv` for max lit overlap (n_stars=8 with current data∩master).
- **Trust-weight A/B (step 02b; offline re-stack):**
  - `PYTHONPATH=. python -m validation.trust_weight_ab_report --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' --out-dir validation_output/trust_ab_post103 --max-files 200`
  - Keep `--trust-weights` opt-in; review summary.json before enabling default.
- Interpretation plots + text (after diagnose, or on existing CSVs):
  - `env PYTHONPATH=. python3 validation/legacy_interpretation_report.py --report-dir validation_output/diagnose_legacy --pipeline-summary validation_output/pipeline_rerun/Gaia_DR3_<id>_summary.txt --legacy-summary ../output/<id>_summary.txt`

## Key outputs

- `bias/bias_statistics.txt`, `bias/bias_by_chunk.csv`
- `consistency/method_pair_offsets.csv`, `consistency/method_trends.csv`
- `broad_line/broad_line_summary.csv`, `docs/broad_line_method.md`
- `error_model/systematic_floors.csv`, `error_model/coverage_report.csv`
- `campaign/validation_report.json`
- `diagnose_legacy/exposure_comparison.csv` (includes `gaia_source_id` when multi-star), `method_exposure_summary.csv`, `method_pair_stats.csv`, `by_teff_bin.csv`, optional `gaia_query_*.json`, `interpretation_summary.txt`, `rv_vs_mjd.png`, `methods_heatmap.png`, `delta_method_vs_teff.png`, `delta_rv_histogram.png`, etc.

## Suggested acceptance thresholds

- Cool-star method pair median offsets: `|offset| < 0.1 km/s`
- Chunk rejection rates should be stable by night/instrument (no pathological swings)
- Calibrated 1-sigma coverage target: roughly `0.60-0.75`
- Calibrated 2-sigma coverage target: roughly `0.90-0.98`

## Interpretation notes

- If method offsets are coherent with Teff or mask-line count, use those features in post-hoc method trust regions.
- If error coverage is under-dispersed, increase systematic floor terms per method/instrument/chunk family.
- For broad-line stars, use the benchmark recommendation from `docs/broad_line_method.md`.

## SB2 search (step 07)

Mask-CCF bi-Gaussian / primary-seeded secondary gate, optional two-template separation and orbit fit.

- **Pipeline fuse (default when mask CCF runs):** `process_spectrum` median-stacks already-computed `order_mask_ccf` and writes exposure-level `sb2_candidate`, `sb2_rv1_kms`, `sb2_rv2_kms`, `sb2_delta_chi2`, … on every `*_diagnostics.csv` row. Opt out: `--no-sb2-score`.
- Per-star search (full report / template fit): `python -m validation.sb2_search --gaia-id <id> --spec-root /Users/rfoley/darkhunter/rvs/data --out-dir validation_output/sb2_<id>`
  - Outputs: `sb2_epochs.csv` (`sb2_candidate`, `rv1_kms`, `rv2_kms`, `delta_chi2`, …), `sb2_orders.csv`, `sb2_report.json`; on detect/`--force-fit`: `sb2_fit.json` + separated spectra.
- Fit diagnostics: `python -m validation.sb2_fit_diagnostics_report --sb2-dir validation_output/sb2_<id>`
- Optional SB2 orbit (07c): `python -m validation.sb2_orbit_fit --sb2-dir validation_output/sb2_<id>` (uses `darkhunter_rv.sb2_rv_fit`; single-lined Keplerian fitter unchanged).
- Unit tests: `python -m pytest tests/test_sb2.py tests/test_sb2_rv_fit.py tests/test_plot_sb2_decomposition_orders.py -m "not slow"`
- Limits: expect false positives on noisy/asymmetric single-lined CCFs; cool high-S/N calib stars should rarely flag.

### Gaia NSS cohort fraction table

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m validation.sb2_nss_cohort_report \
  --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \
  --nss-ids-csv calibration/nss_sb2_source_ids_stub.csv \
  --out-dir validation_output/sb2_nss_cohort
```

Replace the stub CSV with a dump of NSS two-body / SB2 ``source_id`` values for real
``frac_flagged_among_nss`` / ``frac_flagged_among_non_nss``. Outputs:
``exposure_sb2_flags.csv``, ``per_star.csv``, ``fraction_table.csv``.

Requires pipeline fuse so diagnostics carry ``sb2_candidate``.
