# Broad-line / strong-lines method

Product path: **joint Voigt+Lorentz** via `rv_core.measure_strong_line_voigt_lorentz(rest=...)`
(diagnostics method `strong_lines`). Hβ wrapper: `measure_h_beta_rv`.

## Line list (step 06 / #43)

Candidates (rest Å): Hα 6562.8, Hβ 4861.3, Hγ 4340.5, Hδ 4101.7
(also continuum `STRONG_LINES` includes 3970.1, 3889.0 for masking only).

Teff preference (`strong_line_rests_for_teff`):

- Teff ≥ 5500 K (`METHOD_REGION_STRONG_LINES_MIN_TEFF_K`): **Hβ → Hγ → Hα → Hδ**
- Cooler: **Hα → Hβ → Hγ → Hδ**

Pipeline picks the single best successful fit (formal error + preference order). Good Hβ
(score &lt; 20 km/s err) short-circuits so Hβ remains primary when it works.

## Notes

Legacy Gaussian multi-line centroids (`measure_strong_line_centroids`) are **not** the product RV.
Synthetic RMS comparisons of profile shapes live in `validation/h_beta_profile_method_report.py`.
