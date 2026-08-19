# The Joker RV orbit fits

Parent issue: [#111](https://github.com/UCSC-Transients/dark-hunter_rv/issues/111). Children: [#112](https://github.com/UCSC-Transients/dark-hunter_rv/issues/112) NSS/Thiele-Innes priors, [#113](https://github.com/UCSC-Transients/dark-hunter_rv/issues/113) sampler/JSON, [#114](https://github.com/UCSC-Transients/dark-hunter_rv/issues/114) plots, [#115](https://github.com/UCSC-Transients/dark-hunter_rv/issues/115) website/ops.

## Default vs chi2

Default orbit path is [The Joker](https://thejoker.readthedocs.io/en/latest/index.html) (`fit_joker_rv.py`): rejection sampling, then pymc MCMC when the posterior is unimodal.

Chi2 least-squares remains in `fit_apf_rv_keplerian.py`. Invoke it with `python fit_joker_rv.py --rvchi2 ...` or `RV_FITTER=rvchi2` on batch/refit scripts.

## Four prior variants

Shared: `P_min=20 d`, `P_max=2000 d` (expanded if Gaia P ± 5σ is outside), `sigma_K0 = max(mean(RV), 30 km/s)`, `sigma_v = max(max(|RV|), 100 km/s)`.

| id | Priors |
|----|--------|
| `rv_only` | Joker defaults |
| `period` | Gaia `P ± σ_P` |
| `ecc` | TruncatedNormal `e` on (0, 1) |
| `full` | P, e, ω, and M0 when T0 exists. Catalog values first; Campbell (i, ω, Ω, a) from Thiele-Innes A,B,F,G if needed. Gaia `t_periastron` is days from J2016.0. |

## Artifacts

Per star under `rv_fit_reports/`:

- `<id>_joker_fit.json` — medians, 16/84, masses
- `<id>_joker_<variant>.hdf5` — posterior samples
- `<id>_keplerian_fit.png` / `_keplerian_residuals.png` — 10 faint samples + thick median orbit
- `<id>_joker_corner.png` — four stacked corners (RV-only top, full bottom)

Website masses: **M2** Gaia astrometry; **M2 sini** RV-only + M1; **M2 at i** RV-only + M1 + astrometric i; **M2 RV+astrometry** full variant + M1 + astrometric i.

## Commands

n≥4, all variants:

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python fit_joker_rv.py --all --use-gaia-nss --min-points 4 \
  --variants rv_only,period,ecc,full \
  --output-dir output --reports-dir rv_fit_reports
```

n=2–3, full priors only:

```bash
python fit_joker_rv.py --all --use-gaia-nss --min-points 2 --max-points 3 \
  --variants full --output-dir output --reports-dir rv_fit_reports
```

Ziggy screen examples: see [operations.md](operations.md).
