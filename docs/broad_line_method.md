# Broad-line / strong-lines method

Product path: **joint Voigt+Lorentz** via `rv_core.measure_strong_line_voigt_lorentz(rest=...)`
(diagnostics method `strong_lines`). Hβ wrapper: `measure_h_beta_rv`.

## Line list (step 06 / #43)

Candidates (rest Å): Hα 6562.8, Hβ 4861.3, Hγ 4340.5, Hδ 4101.7
(also continuum `STRONG_LINES` includes 3970.1, 3889.0 for masking only).

Teff preference (`strong_line_rests_for_teff`): **Hβ → Hγ → Hδ → Hα** for all Teff on APF.

- Hβ short-circuit: good Hβ fit (score &lt; 20 km/s err) stops the candidate loop.
- Hα is listed last: the APF echellogram does **not** cover 6563 Å (114/114
  `no_order_coverage` in `validation_output/strong_line_teff_sweep/`).

## Validation sweep (114-stem campaign, 2026-08)

Force-fit all Balmer lines vs mask (`python -m validation.strong_line_teff_sweep`):

| result | value |
|--------|------:|
| Product line (Hβ short-circuit) | Hβ **114/114** |
| Free selection (no short-circuit) | Hβ **114/114** |
| Oracle non-Hβ beats Hβ vs mask | **8/114** (7%); median gain ≈ 2.5 km/s |
| Cohort median \|prod−mask\| | 2.16 km/s (oracle 2.01) |
| Median \|Hγ/Hδ−mask\| | tens–hundreds km/s (unusable as primary) |

**Decision:** stay Hβ-primary on APF; Hγ/Hδ are fallback-only when Hβ fails. No production
change to pick non-Hβ by residual. Report: `validation_output/strong_line_teff_sweep/SWEEP_SUMMARY.md`.

## Notes

Legacy Gaussian multi-line centroids (`measure_strong_line_centroids`) are **not** the product RV.
Synthetic RMS comparisons of profile shapes live in `validation/h_beta_profile_method_report.py`.
