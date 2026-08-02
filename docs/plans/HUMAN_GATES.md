# Human decision checklist (post soft-residuals)

Program code lanes for mask / template / strong / fusion / epoch-CCF / SB2 tooling are on `main` through [#107](https://github.com/UCSC-Transients/dark-hunter_rv/pull/107).

**Human answers (2026-08-02):**

| # | Decision | Applied | Issue |
|---|----------|---------|-------|
| 1 | `epoch_ccf_*` in adopt path + always-run systematics | **Yes** — cascade after `strong_lines`; always run multi-epoch matrix; flag abs vs relative ΔRV discord | reopen/extend [#94](https://github.com/UCSC-Transients/dark-hunter_rv/issues/94) as needed |
| 2 | Trust weights default on | **Yes** (`order_chunk_qc.yaml` + `DEFAULT_TRUST_WEIGHTS`; `--no-trust-weights` to disable) | — |
| 3 | Pre-final bias rebuild + ziggy | **Not yet** — when ziggy access is better | [#57](https://github.com/UCSC-Transients/dark-hunter_rv/issues/57), [#39](https://github.com/UCSC-Transients/dark-hunter_rv/issues/39) **open** |
| 4 | Lit n≥10 | **Wait** — find more El-Badry spectra (do not treat n=8 as final) | [#45](https://github.com/UCSC-Transients/dark-hunter_rv/issues/45) — reopen when ingesting |
| 5 | Full `output/` SB2 refit for flag rates | **Not yet** — define SB2 flag metric first | [#44](https://github.com/UCSC-Transients/dark-hunter_rv/issues/44) |

## Epoch CCF policy (gate #1)

Cascade: `mask_ccf → template_fft → strong_lines → epoch_ccf_abs_fill` (abs-anchored only).

Always run for multi-epoch stars:

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m validation.run_epoch_ccf_multi_epoch \
  --data-root /Users/rfoley/darkhunter/rvs/data \
  --out-root validation_output/epoch_ccf \
  --abs-diagnostics-root output \
  --enrich-diagnostics-root output
```

Discord flags land in `epoch_ccf_vs_abs_delta.csv` (`epoch_ccf_abs_rel_discordant`); flag only — no auto-override of cascade.

Pipeline can attach a precomputed fill with `--epoch-ccf-fill-csv path/to/epoch_ccf_abs_fill.csv`.

## Still deferred

1. **Bias rebuild + ziggy** — `bash scripts/rebuild_mask_bias.sh` then ziggy refit (`calibration/mask_lane_deploy.md`).
2. Lit n≥10 after more spectra.
3. SB2 campaign refit after flag-metric choice.

## Quick verify

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
PYTHONPATH=. python -m pytest tests/test_method_evaluation.py tests/test_epoch_ccf.py tests/validation/test_trust_weights.py -q
```
