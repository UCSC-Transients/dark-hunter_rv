# Broad-line / strong-lines method

Product path: **joint Voigt+Lorentz** via `rv_core.measure_strong_line_voigt_lorentz(rest=...)`
(diagnostics method `strong_lines`). Hβ wrapper: `measure_h_beta_rv`.

Exposure RV is an **inverse-variance stack** of included lines after per-line mask offsets
(`darkhunter_rv.strong_lines.combine_strong_line_rvs`; issues #91–#93).

## Product line list (#91)

Order in `strong_line_rests_for_teff` / `product_strong_line_rests`:

1. Hβ 4861.3
2. Mg I b₂ 5172.68
3. Ca I 6122.22
4. Ca I 6162.17
5. Mg I b₃ 5183.60
6. Ca I 4226.73
7. Hγ 4340.5
8. Hδ 4101.7
9. Hα 6562.8 (APF uncovered — last)

**Excluded:** Ca II H&K, Ca II IR / Mg I 8807 (red fringe), Fe I 5269/5328, Na D, Mg I b₁.

## Inclusion gates (#92)

`strong_line_passes_inclusion` (config knobs in `darkhunter_rv.config`):

| gate | default |
|------|--------:|
| min depth | 0.05 |
| min S/N (flux/eflux median in continuum wings near the line) | 8.0 |
| width (Voigt σ → km/s) | 3–250 |
| max formal err | 40 km/s |
| max telluric fraction in ±40 Å | 0.08 |

## Debias + weights (#93)

File: `calibration/strong_line_offsets.txt` — columns `line_name offset_kms quality`.

**Calibration (must debias before ranking quality):**

1. `offset_L = median(RV_L − mask)`
2. debiased residual = `(RV_L − offset_L) − mask`
3. `quality_L = mad_ref / MAD(|debiased residual|)` (Ca I 6122 = 1)

Quality is how well the line recovers the correct RV **after** removing its systematic
offset — comparable across lines. It is not S/N.

**Exposure stack:**

\[
w = Q_{\mathrm{line}} \times (\mathrm{S/N}_{\mathrm{at\,line}})^{2}
\]

S/N at the line is per-exposure continuum ``median(flux/eflux)`` in the wings within
±25 Å of the rest wavelength (excluding the core). That tracks **instrument sensitivity
and stellar color** at each line; normalized-flux scatter is only a fallback.

Combined RV → one `strong_lines` row (`qc_reason=ivw_n=…`).

### Testing notes

- Unit tests: quality from debiased residuals; `w = Q·SNR²` separates species quality from S/N.
- Campaign: file qualities match `estimate_strong_line_offsets_and_qualities` on the sweep CSV.
- Inclusion: unit tests + approximate depth/err rates on the sweep CSV (not a full 114 re-pipeline).

## Metal / secondary candidate survey (114 stems)

Force-fit recommended + secondary lines vs mask with detection QC
(`python -m validation.strong_line_candidate_sweep`). Full table:
`validation_output/strong_line_candidate_sweep/CANDIDATE_SWEEP_SUMMARY.md`.

### Keep (optical; now wired)

| line | rest Å | detect | helpful | median \|Δ\| mask |
|------|-------:|-------:|--------:|------------------:|
| Ca I 6122 | 6122.22 | 100% | 100% | **1.30** |
| Mg I b₂ | 5172.68 | 89% | 95% | **1.57** |
| Ca I 6162 | 6162.17 | 99% | 100% | **1.70** |
| Mg I b₃ | 5183.60 | 100% | 97% | 2.72 |
| Ca I 4227 | 4226.73 | 76% | 77% | 4.75 |

### Hold / exclude

| line | rest Å | verdict | reason |
|------|-------:|---------|--------|
| Ca II 8662 | 8662.14 | **hold_red_fringe** | Numerically OK but λ≳8000 Å |
| Ca II 8498 / Mg I 8807 | red | **exclude_red_risk** | poor residuals / fringe |
| Fe I 5269 / 5328 | mid | exclude / marginal | unhelpful vs Hβ |
| Ca II K / H | blue | **exclude_undetected** | mis-ID / activity |

## Balmer-only validation sweep (114-stem campaign, 2026-08)

Force-fit all Balmer lines vs mask (`python -m validation.strong_line_teff_sweep`).

Report: `validation_output/strong_line_teff_sweep/SWEEP_SUMMARY.md`.

## Notes

Legacy Gaussian multi-line centroids (`measure_strong_line_centroids`) are **not** the product RV.
Synthetic RMS comparisons of profile shapes live in `validation/h_beta_profile_method_report.py`.
