# Broad-line / strong-lines method

Product path: **joint Voigt+Lorentz** via `rv_core.measure_strong_line_voigt_lorentz(rest=...)`
(diagnostics method `strong_lines`). Hβ wrapper: `measure_h_beta_rv`.

## Line list (step 06 / #43)

**Product Balmer list (rest Å):** Hα 6562.8, Hβ 4861.3, Hγ 4340.5, Hδ 4101.7  
(also continuum `STRONG_LINES` includes 3970.1, 3889.0 for masking only).

Teff preference (`strong_line_rests_for_teff`): **Hβ → Hγ → Hδ → Hα** for all Teff on APF.

- Hβ short-circuit: good Hβ fit (score &lt; 20 km/s err) stops the candidate loop.
- Hα is listed last: the APF echellogram does **not** cover 6563 Å (114/114
  `no_order_coverage` in `validation_output/strong_line_teff_sweep/`).

## Metal / secondary candidate survey (114 stems)

Force-fit recommended + secondary lines vs mask with detection QC
(`python -m validation.strong_line_candidate_sweep`). Full table:
`validation_output/strong_line_candidate_sweep/CANDIDATE_SWEEP_SUMMARY.md`.

Detection: core depth ≥ 0.05, σ ≤ 40 km/s, telluric fraction ≤ 0.08.  
Helpful: detected and |RV−mask| ≤ 15 km/s. Hβ reference median |Δ| ≈ **1.55** km/s.

### Keep (optical; promoteable as cool/warm fallbacks after Hβ)

| line | rest Å | detect | helpful | median \|Δ\| mask |
|------|-------:|-------:|--------:|------------------:|
| Ca I 6122 | 6122.22 | 100% | 100% | **1.30** |
| Mg I b₂ | 5172.68 | 89% | 95% | **1.57** |
| Ca I 6162 | 6162.17 | 99% | 100% | **1.70** |
| Mg I b₃ | 5183.60 | 100% | 97% | 2.72 |
| Mg I b₁ | 5167.32 | 94% | 95% | 3.73 (blend with b₂ — prefer b₂/b₃) |
| Ca I 4227 | 4226.73 | 76% | 77% | 4.75 |

### Hold / exclude

| line | rest Å | verdict | reason |
|------|-------:|---------|--------|
| Ca II 8662 | 8662.14 | **hold_red_fringe** | Looks good numerically (97% det, med \|Δ\| 3.8) but λ≳8000 Å APF fringe/water risk — do not ship yet |
| Ca II 8498 | 8498.02 | **exclude_red_risk** | med \|Δ\| **19** km/s |
| Mg I 8807 | 8806.76 | **exclude_red_risk** | 65% det; med \|Δ\| 14; red |
| Fe I 5328 | 5328.04 | **marginal** | helpful often but med \|Δ\| **12** (≫ Hβ) |
| Fe I 5269 | 5269.54 | **exclude_unhelpful** | 13% helpful; med \|Δ\| 21 |
| Ca II K | 3933.66 | **exclude_undetected** | 34% det; 0% helpful; med \|Δ\| ~145 (mis-ID / blue continuum) |
| Ca II H | 3968.47 | **exclude_undetected** | 11% det; epoch RV std ~10 km/s (activity / blend with Hε) |

**Na D** remains excluded (ISM + sky). No candidates near strong sky lines (OI 5577/6300, Hg 4358) were promoted.

### Decision

- Stay **Hβ-primary** in production.
- Next wiring candidate (optional): **Ca I 6122 / 6162** and **Mg I b₂/b₃** as cool-star fallbacks when Hβ fails — not yet in `strong_line_rests_for_teff`.
- Do **not** add Ca H&K or red IR lines without a dedicated fringe/activity study.

## Balmer-only validation sweep (114-stem campaign, 2026-08)

Force-fit all Balmer lines vs mask (`python -m validation.strong_line_teff_sweep`):

| result | value |
|--------|------:|
| Product line (Hβ short-circuit) | Hβ **114/114** |
| Free selection (no short-circuit) | Hβ **114/114** |
| Oracle non-Hβ beats Hβ vs mask | **8/114** (7%); median gain ≈ 2.5 km/s |
| Cohort median \|prod−mask\| | 2.16 km/s (oracle 2.01) |
| Median \|Hγ/Hδ−mask\| | tens–hundreds km/s (unusable as primary) |

Report: `validation_output/strong_line_teff_sweep/SWEEP_SUMMARY.md`.

## Notes

Legacy Gaussian multi-line centroids (`measure_strong_line_centroids`) are **not** the product RV.
Synthetic RMS comparisons of profile shapes live in `validation/h_beta_profile_method_report.py`.
